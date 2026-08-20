import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from db.session import get_db
from db.models import Base, Developer, APIKey
from core.auth import hash_api_key


TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestAsyncSession = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db() -> AsyncSession:
    async with TestAsyncSession() as session:
        yield session


from app.main import app
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
async def setup_database():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def valid_api_key(setup_database):
    async with TestAsyncSession() as session:
        developer = Developer(
            email="test@example.com",
            name="Test Developer",
        )
        session.add(developer)
        await session.flush()
        await session.refresh(developer)

        raw_key = "sk_test_validkey12345678901234567890"
        hashed_key = hash_api_key(raw_key)
        api_key = APIKey(
            developer_id=developer.id,
            key_prefix=raw_key[:12],
            hashed_key=hashed_key,
            is_active=True,
        )
        session.add(api_key)
        await session.commit()
        return raw_key


@pytest.mark.asyncio
async def test_healthz_endpoint(setup_database):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert "database" in data
        assert "tabular_model" in data
        assert "imaging_model" in data
        assert "device" in data
        assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_predict_tabular_requires_auth(setup_database):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "age": 58,
            "sex": 1,
            "cp": 2,
            "trestbps": 140,
            "chol": 240,
            "fbs": 0,
            "restecg": 1,
            "thalach": 155,
            "exang": 0,
            "oldpeak": 1.6,
            "slope": 2,
            "ca": 0,
            "thal": 3,
        }
        response = await client.post("/api/v1/predict-tabular", json=payload)
        assert response.status_code == 401
        data = response.json()
        detail = data.get("detail", data)
        assert detail["status"] == "error"
        assert detail["error_code"] == "INVALID_API_KEY"


@pytest.mark.asyncio
async def test_predict_image_requires_auth(setup_database):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test.png", b"fake image data", "image/png")}
        data = {"modality": "chest_xray"}
        response = await client.post("/api/v1/predict-image", files=files, data=data)
        assert response.status_code == 401
        data = response.json()
        detail = data.get("detail", data)
        assert detail["status"] == "error"
        assert detail["error_code"] == "INVALID_API_KEY"


@pytest.mark.asyncio
async def test_predict_image_rejects_invalid_file_type(valid_api_key):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test.tiff", b"fake image data", "image/tiff")}
        data = {"modality": "chest_xray"}
        headers = {"X-API-Key": valid_api_key}
        response = await client.post("/api/v1/predict-image", files=files, data=data, headers=headers)
        assert response.status_code == 415
        data = response.json()
        detail = data.get("detail", data)
        assert detail["status"] == "error"
        assert detail["error_code"] == "UNSUPPORTED_FILE_TYPE"


@pytest.mark.asyncio
async def test_cors_headers(setup_database):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz", headers={"Origin": "http://localhost:8501"})
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers