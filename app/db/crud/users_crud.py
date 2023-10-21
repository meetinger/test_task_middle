import datetime as dt
import uuid
from typing import NoReturn

from fastapi import status, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


from app.db.models import User
from app.schemas.users_schemas import UserCreateSchema, UserUpdateSchema


async def check_duplicates(user: UserCreateSchema, db: AsyncSession) -> None | NoReturn:
    username_exist_query = select(User.id).where(User.username == user.username)
    email_exist_query = select(User.id).where(User.email == user.email)

    if user.username and (await db.execute(username_exist_query)).scalars().first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")

    if user.email and (await db.execute(email_exist_query)).scalars().first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")


async def create_user(user: UserCreateSchema, db: AsyncSession) -> User | NoReturn:
    """Создание пользователя"""

    await check_duplicates(user, db)

    user_db = User(username=user.username, email=user.email)

    db.add(user_db)

    await db.commit()
    await db.refresh(user_db)

    return user_db


async def get_user(user_id: uuid.UUID, db: AsyncSession) -> User | NoReturn:
    """Получение пользователя"""

    user_db = await db.get(User, user_id)
    if user_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found')
    return user_db  # у SQLAlchemy проблемы с typehints, поэтому если ругается - не обращайте внимания


async def update_user(user: UserUpdateSchema, db: AsyncSession) -> User | NoReturn:
    """Обновление пользователя"""

    await check_duplicates(user, db)

    user_db = await get_user(user.id, db)

    if user.email is not None:
        user_db.email = user.email

    if user.username is not None:
        user_db.username = user.username

    await db.commit()
    return user_db


async def get_users(page: int, page_size: int, db: AsyncSession) -> list[User, ...]:
    """Получение списка пользователей с пагинацией"""

    query = select(User).offset(page*page_size).limit(page_size)

    result = await db.execute(query)

    users_db = list(result.scalars())

    return users_db


async def delete_user(user_id: uuid.UUID, db: AsyncSession) -> True | NoReturn:
    """Удаление пользователя"""

    user_db = await get_user(user_id, db)
    await db.delete(user_db)
    return True
