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

    # API Security
    JWT_SECRET_KEY: str = Field(default="your-super-secret-jwt-key-change-in-production")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION_HOURS: int = Field(default=24)

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=60)
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60)

    # Model Configuration
    TABULAR_MODEL_PATH: str = Field(default="./models/tabular_mlp.pth")
    IMAGING_MODEL_PATH: str = Field(default="./models/imaging_densenet121_lstm.pth")
    SCALER_PATH: str = Field(default="./models/scaler.joblib")

    # Device Configuration
    FORCE_CPU: bool = Field(default=False)

    # Logging
    LOG_LEVEL: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()