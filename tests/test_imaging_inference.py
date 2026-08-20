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
async def test_healthz_shows_imaging_model_loaded(valid_api_key):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert data["imaging_model"] == "loaded"
        assert data["device"] in ("cpu", "cuda:0")


@pytest.mark.asyncio
async def test_predict_image_success(valid_api_key):
    import io
    from PIL import Image
    
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test.png", img_bytes.getvalue(), "image/png")}
        data = {"modality": "chest_xray"}
        headers = {"X-API-Key": valid_api_key}
        response = await client.post("/api/v1/predict-image", files=files, data=data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "risk_score" in data
        assert 0.0 <= data["risk_score"] <= 1.0
        assert data["risk_level"] in ["Low Risk", "Moderate Risk", "High Risk"]
        assert data["modality"] == "chest_xray"
        assert "findings" in data
        assert "st_segment_abnormality_detected" in data["findings"]
        assert "structural_risk_pattern_confidence" in data["findings"]
        assert data["model_version"] == "imaging-densenet121-lstm-v1.1.0"
        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], int)
        assert data["latency_ms"] >= 0
        assert "timestamp" in data


@pytest.mark.asyncio
async def test_predict_image_dicom_valid(valid_api_key):
    import pydicom
    from pydicom.dataset import Dataset
    from pydicom.uid import ExplicitVRLittleEndian
    import numpy as np
    import io
    
    ds = Dataset()
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    ds.file_meta.MediaStorageSOPInstanceUID = '1.2.3.4.5.6.7.8.9'
    
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    ds.Modality = "CR"
    ds.Rows = 224
    ds.Columns = 224
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.WindowCenter = "100"
    ds.WindowWidth = "200"
    
    pixel_data = np.random.randint(0, 4095, (224, 224), dtype=np.uint16)
    ds.PixelData = pixel_data.tobytes()
    
    dicom_bytes = io.BytesIO()
    ds.save_as(dicom_bytes, write_like_original=False)
    dicom_bytes.seek(0)
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test.dicom", dicom_bytes.getvalue(), "application/dicom")}
        data = {"modality": "chest_xray"}
        headers = {"X-API-Key": valid_api_key}
        response = await client.post("/api/v1/predict-image", files=files, data=data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "risk_score" in data
        assert 0.0 <= data["risk_score"] <= 1.0


@pytest.mark.asyncio
async def test_predict_image_unsupported_file_type(valid_api_key):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test.tiff", b"fake image data", "image/tiff")}
        data = {"modality": "chest_xray"}
        headers = {"X-API-Key": valid_api_key}
        response = await client.post("/api/v1/predict-image", files=files, data=data, headers=headers)
        assert response.status_code == 415
        detail = response.json().get("detail", response.json())
        assert detail["status"] == "error"
        assert detail["error_code"] == "UNSUPPORTED_FILE_TYPE"