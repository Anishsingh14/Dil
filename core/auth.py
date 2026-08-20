import secrets
import bcrypt
from typing import Optional
from datetime import datetime, timezone, timedelta

from db.models import APIKey, Developer
from db.session import get_session
from core.config import settings


KEY_PREFIX_LENGTH = 12
RAW_KEY_LENGTH = 32
KEY_PREFIX = "sk_live_"
TEST_KEY_PREFIX = "sk_test_"
VALID_PREFIXES = (KEY_PREFIX, TEST_KEY_PREFIX)
MAX_KEY_LENGTH = 72
BCRYPT_ROUNDS = 12


def generate_raw_key() -> str:
    """Generate a cryptographically random API key with sk_live_ prefix."""
    token = secrets.token_urlsafe(RAW_KEY_LENGTH)
    return f"{KEY_PREFIX}{token}"[:MAX_KEY_LENGTH]


def extract_key_prefix(raw_key: str) -> str:
    """Extract the first 12 characters of the raw key for indexing."""
    if not raw_key.startswith(VALID_PREFIXES):
        raise ValueError("Invalid API key format")
    return raw_key[:KEY_PREFIX_LENGTH]


def hash_api_key(raw_key: str) -> str:
    """Hash an API key using bcrypt with work factor 12."""
    salt = bcrypt.gensalt(BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(raw_key.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_api_key(raw_key: str, hashed_key: str) -> bool:
    """Verify an API key against its bcrypt hash (constant-time)."""
    return bcrypt.checkpw(raw_key.encode("utf-8"), hashed_key.encode("utf-8"))


async def create_api_key(
    developer_id: int,
    session,
    expires_in_days: Optional[int] = None,
) -> tuple[str, APIKey]:
    """Create a new API key for a developer. Returns (raw_key, api_key_model)."""
    raw_key = generate_raw_key()
    key_prefix = extract_key_prefix(raw_key)
    hashed_key = hash_api_key(raw_key)

    expires_at = None
    if expires_in_days is None:
        expires_in_days = settings.API_KEY_EXPIRATION_DAYS
    if expires_in_days > 0:
        expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

    api_key = APIKey(
        developer_id=developer_id,
        key_prefix=key_prefix,
        hashed_key=hashed_key,
        is_active=True,
        expires_at=expires_at,
    )
    session.add(api_key)
    await session.flush()
    await session.refresh(api_key)

    return raw_key, api_key


def is_api_key_expired(api_key: APIKey) -> bool:
    """Check if an API key has expired."""
    if api_key.expires_at is None:
        return False
    # Ensure both datetimes are offset-aware for comparison
    expires_at = api_key.expires_at
    if expires_at.tzinfo is None:
        # Assume UTC if no timezone info
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) > expires_at


async def revoke_api_key(api_key_id: int, session) -> bool:
    """Revoke an API key by setting is_active to False."""
    from sqlalchemy import select
    result = await session.execute(
        select(APIKey).where(APIKey.id == api_key_id)
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        return False
    api_key.is_active = False
    await session.flush()
    return True


async def verify_api_key_and_get_developer(
    raw_key: str,
    session,
) -> Optional[Developer]:
    """Verify an API key and return the associated developer if valid and active."""
    if not raw_key.startswith(VALID_PREFIXES):
        return None

    key_prefix = extract_key_prefix(raw_key)

    from sqlalchemy import select

    result = await session.execute(
        select(APIKey).where(
            APIKey.key_prefix == key_prefix,
            APIKey.is_active == True,
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        return None

    if is_api_key_expired(api_key):
        return None

    if not verify_api_key(raw_key, api_key.hashed_key):
        return None

    result = await session.execute(
        select(Developer).where(
            Developer.id == api_key.developer_id,
            Developer.is_active == True,
        )
    )
    developer = result.scalar_one_or_none()

    if not developer:
        return None

    api_key.request_count += 1
    api_key.last_used_at = datetime.now(timezone.utc)
    await session.flush()

    return developer


from datetime import datetime, timezone