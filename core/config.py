from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Database
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./dil_dev.db")
    DB_ECHO: bool = Field(default=False)
    DB_POOL_SIZE: int = Field(default=20)
    DB_MAX_OVERFLOW: int = Field(default=10)
    DB_POOL_PRE_PING: bool = Field(default=True)

    # API Security
    JWT_SECRET_KEY: str = Field(default="your-super-secret-jwt-key-change-in-production")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION_HOURS: int = Field(default=24)
    API_KEY_EXPIRATION_DAYS: int = Field(default=365)

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=60)
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60)

    # Redis (for distributed rate limiting)
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_ENABLED: bool = Field(default=False)

    # Model Configuration
    TABULAR_MODEL_PATH: str = Field(default="./models/tabular_xgb.pkl")
    IMAGING_MODEL_PATH: str = Field(default="./models/imaging_densenet121_lstm.pth")
    SCALER_PATH: str = Field(default="./models/scaler.joblib")

    # Device Configuration
    FORCE_CPU: bool = Field(default=False)

    # Logging
    LOG_LEVEL: str = Field(default="INFO")

    # CORS
    CORS_ALLOWED_ORIGINS: str = Field(default="http://localhost:8501")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()