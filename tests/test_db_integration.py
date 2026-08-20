import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, Developer, APIKey
from core.auth import (
    generate_raw_key,
    hash_api_key,
    verify_api_key,
    create_api_key,
    verify_api_key_and_get_developer,
)


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session(engine):
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


class TestDatabaseIntegration:
    async def test_create_developer(self, session):
        developer = Developer(
            email="test@example.com",
            name="Test Developer",
            organization="Test Org",
        )
        session.add(developer)
        await session.flush()
        await session.refresh(developer)

        assert developer.id is not None
        assert developer.email == "test@example.com"
        assert developer.name == "Test Developer"
        assert developer.organization == "Test Org"
        assert developer.is_active is True
        assert developer.created_at is not None

    async def test_create_api_key(self, session):
        developer = Developer(
            email="test@example.com",
            name="Test Developer",
        )
        session.add(developer)
        await session.flush()
        await session.refresh(developer)

        raw_key, api_key = await create_api_key(developer.id, session)

        assert raw_key.startswith("sk_live_")
        assert api_key.developer_id == developer.id
        assert api_key.key_prefix == raw_key[:12]
        assert api_key.is_active is True
        assert api_key.request_count == 0
        assert verify_api_key(raw_key, api_key.hashed_key)

    async def test_verify_api_key_success(self, session):
        developer = Developer(
            email="test@example.com",
            name="Test Developer",
        )
        session.add(developer)
        await session.flush()
        await session.refresh(developer)

        raw_key, _ = await create_api_key(developer.id, session)

        verified_developer = await verify_api_key_and_get_developer(raw_key, session)

        assert verified_developer is not None
        assert verified_developer.id == developer.id
        assert verified_developer.email == "test@example.com"

    async def test_verify_api_key_invalid_key(self, session):
        developer = Developer(
            email="test@example.com",
            name="Test Developer",
        )
        session.add(developer)
        await session.flush()
        await session.refresh(developer)

        await create_api_key(developer.id, session)

        verified_developer = await verify_api_key_and_get_developer("sk_live_invalidkey", session)

        assert verified_developer is None

    async def test_verify_api_key_wrong_prefix(self, session):
        verified_developer = await verify_api_key_and_get_developer("sk_test_invalid", session)
        assert verified_developer is None

    async def test_verify_api_key_inactive_developer(self, session):
        developer = Developer(
            email="test@example.com",
            name="Test Developer",
            is_active=False,
        )
        session.add(developer)
        await session.flush()
        await session.refresh(developer)

        raw_key, _ = await create_api_key(developer.id, session)

        verified_developer = await verify_api_key_and_get_developer(raw_key, session)

        assert verified_developer is None

    async def test_verify_api_key_inactive_key(self, session):
        developer = Developer(
            email="test@example.com",
            name="Test Developer",
        )
        session.add(developer)
        await session.flush()
        await session.refresh(developer)

        raw_key, api_key = await create_api_key(developer.id, session)
        api_key.is_active = False
        await session.flush()

        verified_developer = await verify_api_key_and_get_developer(raw_key, session)

        assert verified_developer is None

    async def test_request_count_increments(self, session):
        developer = Developer(
            email="test@example.com",
            name="Test Developer",
        )
        session.add(developer)
        await session.flush()
        await session.refresh(developer)

        raw_key, api_key = await create_api_key(developer.id, session)
        initial_count = api_key.request_count

        await verify_api_key_and_get_developer(raw_key, session)
        await verify_api_key_and_get_developer(raw_key, session)

        from sqlalchemy import select
        result = await session.execute(
            select(APIKey).where(APIKey.id == api_key.id)
        )
        updated_key = result.scalar_one()

        assert updated_key.request_count == initial_count + 2

    async def test_last_used_at_updates(self, session):
        developer = Developer(
            email="test@example.com",
            name="Test Developer",
        )
        session.add(developer)
        await session.flush()
        await session.refresh(developer)

        raw_key, api_key = await create_api_key(developer.id, session)
        assert api_key.last_used_at is None

        await verify_api_key_and_get_developer(raw_key, session)

        from sqlalchemy import select
        result = await session.execute(
            select(APIKey).where(APIKey.id == api_key.id)
        )
        updated_key = result.scalar_one()

        assert updated_key.last_used_at is not None