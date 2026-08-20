from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from db.models import Base
from core.config import settings


def create_engine():
    """Create async engine with appropriate pooling for the database dialect."""
    url = settings.DATABASE_URL
    
    if url.startswith("mysql") or url.startswith("mysql+aiomysql"):
        # MySQL with aiomysql - use proper pooling
        return create_async_engine(
            url,
            echo=settings.DB_ECHO,
            future=True,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_pre_ping=settings.DB_POOL_PRE_PING,
            pool_recycle=3600,
        )
    elif url.startswith("postgresql") or url.startswith("postgresql+asyncpg"):
        # PostgreSQL with asyncpg
        return create_async_engine(
            url,
            echo=settings.DB_ECHO,
            future=True,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_pre_ping=settings.DB_POOL_PRE_PING,
        )
    else:
        # SQLite - use NullPool to avoid threading issues
        return create_async_engine(
            url,
            echo=settings.DB_ECHO,
            future=True,
            poolclass=NullPool,
        )


engine = create_engine()

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with get_session() as session:
        yield session