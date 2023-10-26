import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.settings import settings

SQLALCHEMY_DB_URL = settings.get_db_url()


engine = create_async_engine(SQLALCHEMY_DB_URL,
                             json_serializer=lambda dct: json.dumps(dct, ensure_ascii=False),
                             json_deserializer=lambda s: json.loads(s))
async_session_factory = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Базовый класс для ORM моделей SQLAlchemy"""
    pass


async def get_db() -> AsyncIterator[AsyncSession]:
    """Функция для получения объекта сессии БД"""
    async with async_session_factory() as db:
        yield db

get_db_ctx = asynccontextmanager(get_db)  # нужно для других целей(для тестирования), в проде можно убрать
