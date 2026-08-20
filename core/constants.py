"""Application constants - replacing magic numbers with named constants."""

from enum import Enum
from typing import Final

# Risk thresholds
RISK_THRESHOLD_LOW: Final[float] = 0.33
RISK_THRESHOLD_MODERATE: Final[float] = 0.66
RISK_THRESHOLD_HIGH: Final[float] = 1.0

RISK_LEVEL_LOW: Final[str] = "Low Risk"
RISK_LEVEL_MODERATE: Final[str] = "Moderate Risk"
RISK_LEVEL_HIGH: Final[str] = "High Risk"

# Image processing constants
IMAGE_SIZE: Final[int] = 224
IMAGE_CHANNELS: Final[int] = 3
IMAGE_NET_MEAN: Final[list[float]] = [0.485, 0.456, 0.406]
IMAGE_NET_STD: Final[list[float]] = [0.229, 0.224, 0.225]

# Image quality thresholds
QUALITY_MIN_INTENSITY: Final[float] = 30.0
QUALITY_MAX_INTENSITY: Final[float] = 225.0
QUALITY_MIN_CONTRAST: Final[float] = 15.0
QUALITY_MIN_SHARPNESS: Final[float] = 100.0
QUALITY_MIN_DYNAMIC_RANGE: Final[float] = 50.0
QUALITY_MAX_SATURATION_LOW: Final[float] = 0.05
QUALITY_MAX_SATURATION_HIGH: Final[float] = 0.05
QUALITY_MAX_NOISE: Final[float] = 30.0
QUALITY_MIN_ASPECT_RATIO: Final[float] = 0.5
QUALITY_MAX_ASPECT_RATIO: Final[float] = 2.0

# DICOM quality
DICOM_MIN_BITS_STORED: Final[int] = 8
DICOM_MAX_BITS_STORED: Final[int] = 16

# Model constants
TABULAR_FEATURE_COUNT: Final[int] = 13
TABULAR_MODEL_VERSION: Final[str] = "tabular-xgb-v1.1.0"
IMAGING_MODEL_VERSION: Final[str] = "imaging-densenet121-lstm-v1.1.0"
DENSENET_FEATURE_DIM: Final[int] = 1024
LSTM_HIDDEN_SIZE: Final[int] = 256
LSTM_LAYERS: Final[int] = 1
DROPOUT_RATE: Final[float] = 0.3

# API constants
API_KEY_PREFIX_LENGTH: Final[int] = 12
API_KEY_RAW_LENGTH: Final[int] = 32
API_KEY_MAX_LENGTH: Final[int] = 72
API_KEY_PREFIX: str = "sk_live_"
TEST_API_KEY_PREFIX: str = "sk_test_"
VALID_API_KEY_PREFIXES: tuple[str, ...] = (API_KEY_PREFIX, TEST_API_KEY_PREFIX)
BCRYPT_ROUNDS: Final[int] = 12
MAX_KEY_LENGTH: Final[int] = 72
DEFAULT_API_KEY_EXPIRATION_DAYS: Final[int] = 365
DEFAULT_RATE_LIMIT_REQUESTS: Final[int] = 60
DEFAULT_RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60
DEFAULT_JWT_SECRET_KEY: Final[str] = "your-super-secret-jwt-key-change-in-production"
DEFAULT_JWT_ALGORITHM: Final[str] = "HS256"
DEFAULT_JWT_EXPIRATION_HOURS: Final[int] = 24
DEFAULT_API_KEY_EXPIRATION_DAYS: Final[int] = 365
DEFAULT_RATE_LIMIT_REQUESTS: Final[int] = 60
DEFAULT_RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60
DEFAULT_JWT_SECRET_KEY: Final[str] = "your-super-secret-jwt-key-change-in-production"
DEFAULT_JWT_ALGORITHM: Final[str] = "HS256"
DEFAULT_JWT_EXPIRATION_HOURS: Final[int] = 24

# Rate limiting
DEFAULT_RATE_LIMIT: Final[int] = 60
DEFAULT_RATE_WINDOW: Final[int] = 60

# File upload limits
MAX_FILE_SIZE_MB: Final[int] = 15
MAX_SERIES_IMAGES: Final[int] = 50
ALLOWED_IMAGE_EXTENSIONS: Final[set[str]] = {".png", ".jpg", ".jpeg", ".dicom", ".dcm"}
ALLOWED_IMAGE_MIME_TYPES: Final[set[str]] = {
    "image/png", "image/jpeg", "application/dicom"
}

# Database
DEFAULT_SQLITE_PATH: str = "sqlite+aiosqlite:///./dil_dev.db"
DEFAULT_DB_POOL_SIZE: Final[int] = 20
DEFAULT_DB_MAX_OVERFLOW: Final[int] = 10
DEFAULT_DB_POOL_PRE_PING: Final[bool] = True
DEFAULT_DB_ECHO: Final[bool] = False

# Cache
MODEL_CACHE_TTL_SECONDS: Final[int] = 3600

# Graceful shutdown
DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT: Final[int] = 30

# CORS
DEFAULT_CORS_ORIGINS: list[str] = ["http://localhost:8501"]

# API
API_VERSION: str = "v1"
API_PREFIX: str = f"/api/{API_VERSION}"
API_TITLE: str = "Dil API"
API_DESCRIPTION: str = "Multi-Modal Cardiovascular Risk Diagnostic Platform"
API_VERSION_STR: str = "1.0.0"

# Model paths
DEFAULT_TABULAR_MODEL_PATH: str = "./models/tabular_xgb.pkl"
DEFAULT_IMAGING_MODEL_PATH: str = "./models/imaging_densenet121_lstm.pth"
DEFAULT_SCALER_PATH: str = "./models/scaler.joblib"

# Training
DEFAULT_EPOCHS: int = 100
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_RANDOM_STATE: int = 42

# Redis
DEFAULT_REDIS_URL: str = "redis://localhost:6379/0"

# Logging
DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_LOG_FORMAT: str = "json"

# External
CHEXPERT_SAS_URL_ENV: str = "CHEXPERT_SAS_URL"

# File extensions
DICOM_EXTENSIONS: Final[set[str]] = {".dicom", ".dcm"}
IMAGE_EXTENSIONS: Final[set[str]] = {".png", ".jpg", ".jpeg"}

# Image quality checks
QUALITY_CHECKS: Final[list[str]] = [
    "exposure_ok",
    "contrast_ok",
    "sharpness_ok",
    "dynamic_range_ok",
    "saturation_ok",
    "noise_ok",
]

# Health check
HEALTH_CHECK_DEPTHS: Final[set[str]] = {"liveness", "readiness"}
DEFAULT_HEALTH_CHECK_DEPTH: str = "readiness"

# Error codes
class ErrorCode(str, Enum):
    INVALID_API_KEY = "INVALID_API_KEY"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    NO_FILES_PROVIDED = "NO_FILES_PROVIDED"
    TOO_MANY_FILES = "TOO_MANY_FILES"
    TLS_REQUIRED = "TLS_REQUIRED"
    INTERNAL_ERROR = "INTERNAL_ERROR"

# Modality
class Modality(str, Enum):
    CHEST_XRAY = "chest_xray"
    CARDIAC_MRI = "cardiac_mri"

# Risk level
class RiskLevel(str, Enum):
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        if score < RISK_THRESHOLD_LOW:
            return cls.LOW
        elif score < RISK_THRESHOLD_MODERATE:
            return cls.MODERATE
        return cls.HIGH

# File extensions
class FileExtension(str, Enum):
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    DICOM = ".dicom"
    DCM = ".dcm"

    @classmethod
    def is_allowed(cls, ext: str) -> bool:
        return ext.lower() in ALLOWED_IMAGE_EXTENSIONS