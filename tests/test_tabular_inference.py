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
async def test_api_key(setup_database):
    async with TestAsyncSession() as session:
        developer = Developer(
            email="test@example.com",
            name="Test Developer",
        )
        session.add(developer)
        await session.flush()
        await session.refresh(developer)

        raw_key = "sk_test_testkey123456789012345678901234"
        hashed_key = hash_api_key(raw_key)
        api_key = APIKey(
            developer_id=developer.id,
            key_prefix=raw_key[:12],
            hashed_key=hashed_key,
            is_active=True,
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)
        return raw_key


@pytest.mark.asyncio
async def test_predict_tabular_success(test_api_key, setup_database):
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
        headers = {"X-API-Key": test_api_key}
        response = await client.post("/api/v1/predict-tabular", json=payload, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "risk_score" in data
        assert 0.0 <= data["risk_score"] <= 1.0
        assert data["risk_level"] in ["Low Risk", "Moderate Risk", "High Risk"]
        assert data["model_version"] == "tabular-xgb-v1.1.0"
        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], int)
        assert data["latency_ms"] >= 0
        assert "timestamp" in data


@pytest.mark.asyncio
async def test_predict_tabular_validation_error(test_api_key, setup_database):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "age": 150,  # invalid: > 120
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
        headers = {"X-API-Key": test_api_key}
        response = await client.post("/api/v1/predict-tabular", json=payload, headers=headers)
        assert response.status_code == 422
        data = response.json()
        assert data["status"] == "error"
        assert data["error_code"] == "VALIDATION_ERROR"
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0


@pytest.mark.asyncio
async def test_healthz_shows_model_loaded(setup_database):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert data["tabular_model"] == "loaded"
        assert data["device"] in ("cpu", "cuda:0")