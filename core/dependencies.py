"""Dependency injection container and providers."""

from functools import lru_cache
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from fastapi import Depends, Request

from core.config import settings
from core.constants import (
    DEFAULT_SQLITE_PATH,
    DEFAULT_DB_POOL_SIZE,
    DEFAULT_DB_MAX_OVERFLOW,
    DEFAULT_DB_POOL_PRE_PING,
    DEFAULT_DB_ECHO,
)
from db.models import Base
from db.session import async_session_maker as session_maker
from models.tabular_inference import TabularInferenceEngine, get_tabular_engine as _get_tabular_engine
from models.imaging_inference import ImagingInferenceEngine, get_imaging_engine as _get_imaging_engine
from core.auth import verify_api_key_and_get_developer
from db.models import Developer


class DIContainer:
    """Dependency injection container for managing service instances."""
    
    def __init__(self):
        self._tabular_engine: Optional[TabularInferenceEngine] = None
        self._imaging_engine: Optional[ImagingInferenceEngine] = None
        self._engine = None
        self._session_factory = None
    
    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_async_engine(
                settings.DATABASE_URL,
                echo=settings.DB_ECHO,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
                pool_pre_ping=settings.DB_POOL_PRE_PING,
                future=True,
            )
        return self._engine
    
    @property
    def session_factory(self):
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._session_factory
    
    @property
    def tabular_engine(self) -> TabularInferenceEngine:
        if self._tabular_engine is None:
            self._tabular_engine = TabularInferenceEngine()
        return self._tabular_engine
    
    @property
    def imaging_engine(self) -> ImagingInferenceEngine:
        if self._imaging_engine is None:
            self._imaging_engine = ImagingInferenceEngine()
        return self._imaging_engine
    
    async def initialize(self) -> None:
        """Initialize all services."""
        # Initialize database
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Initialize models
        await self.tabular_engine.load()
        await self.imaging_engine.load()
    
    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._engine:
            await self._engine.dispose()


# Global container instance
_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Get the global DI container."""
    global _container
    if _container is None:
        _container = DIContainer()
    return _container


async def init_container() -> None:
    """Initialize the global container."""
    container = get_container()
    await container.initialize()


async def close_container() -> None:
    """Shutdown the global container."""
    global _container
    if _container is not None:
        await _container.shutdown()
        _container = None


# FastAPI dependency providers
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    container = get_container()
    async with container.session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_current_developer(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> Developer:
    """Get current authenticated developer."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "error_code": "INVALID_API_KEY",
                "detail": "The provided X-API-Key is missing, malformed, or inactive.",
            },
        )
    developer = await verify_api_key_and_get_developer(api_key, db)
    if not developer:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "error_code": "INVALID_API_KEY",
                "detail": "The provided X-API-Key is missing, malformed, or inactive.",
            },
        )
    return developer


def get_tabular_inference_engine() -> TabularInferenceEngine:
    """Get tabular inference engine."""
    return get_container().tabular_engine


def get_imaging_inference_engine() -> ImagingInferenceEngine:
    """Get imaging inference engine."""
    return get_container().imaging_engine


# Backwards compatibility
def get_tabular_engine() -> TabularInferenceEngine:
    return get_tabular_inference_engine()


def get_imaging_engine() -> ImagingInferenceEngine:
    return get_imaging_inference_engine()


async def initialize_tabular_model() -> None:
    """Initialize tabular model (for backwards compatibility)."""
    engine = get_tabular_inference_engine()
    if not engine.is_loaded():
        engine.load()


async def initialize_imaging_model() -> None:
    """Initialize imaging model (for backwards compatibility)."""
    engine = get_imaging_inference_engine()
    if not engine.is_loaded():
        engine.load()


@asynccontextmanager
async def lifespan(app) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    await init_container()
    yield
    await close_container()