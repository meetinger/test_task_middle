import asyncio
import json
from typing import Generator, AsyncIterator, Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from httpx import AsyncClient

import app.db.crud.users_crud as user_crud
from app.core.security import TokenUtils
from app.core.settings import settings
from app.db.database import get_db, Base
from app.db.models import User
from app.main import app
from app.schemas.users_schemas import UserCreateSchema


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncIterator[AsyncSession]:
    """Сессия БД для тестирования"""
    async_engine = create_async_engine(settings.get_db_url(test=True),
                                       json_serializer=lambda dct: json.dumps(dct, ensure_ascii=False),
                                       json_deserializer=lambda s: json.loads(s))
    async_session_factory = async_sessionmaker(autocommit=False, autoflush=False, bind=async_engine,
                                               expire_on_commit=False)

    async with async_session_factory() as db:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        yield db

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await async_engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(monkeypatch, db_session: AsyncSession) -> Generator[AsyncClient, Any, None]:
    """Клиент для тестирования"""

    async def _get_db() -> AsyncIterator[AsyncSession]:
        yield db_session

    # перегрузка зависимостей
    app.dependency_overrides[get_db] = _get_db

    async with AsyncClient(app=app, base_url=f'http://localhost:{settings.HTTP_PORT}') as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def user(db_session):
    """Фикстура пользователя"""

    async def _user(user_in: UserCreateSchema) -> User:
        return await user_crud.create_user(user=user_in, db=db_session)

    return _user


@pytest.fixture(scope="function")
def user_token():
    """Фикстура токена пользователя"""

    def _user_token(user: User) -> dict:
        return TokenUtils.create_token_pair({'sub': user.username})

    return _user_token


# датасет для тестирования
DATASET = {
    'users': {
        '1': UserCreateSchema(username='user1', email='user1@gmail.com', password='password1'),
        '2': UserCreateSchema(username='user2', email='user2@gmail.com', password='password2')
    }
}
