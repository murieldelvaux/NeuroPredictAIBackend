from __future__ import annotations

from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings


def _normalize_database_url(raw_url: str) -> str:
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if raw_url.startswith("postgres://"):
        return raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
    return raw_url


@lru_cache(maxsize=1)
def get_database_url() -> str:
    return _normalize_database_url(settings.database_url)

# Adicione essa função antes do `engine`
def _get_connect_args(url: str) -> dict:
    """Retorna connect_args com SSL para Neon (asyncpg não aceita ssl na URL)."""
    if "neon.tech" in url or "neondb" in url:
        return {"ssl": "require"}
    return {}


from sqlalchemy.pool import NullPool

# Engine com NullPool (ideal para Neon Postgres serverless pooler e conexões asyncpg)
engine = create_async_engine(
    get_database_url(),
    echo=settings.database_echo,
    future=True,
    connect_args=_get_connect_args(get_database_url()),
    poolclass=NullPool,
)


async_session_maker = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)
