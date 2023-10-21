import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

import app.db.crud.users_crud as user_crud
from app.db.database import get_db
from app.schemas.users_schemas import UserFullSchema, UserCreateSchema, UserUpdateSchema

users_router = APIRouter(tags=['users'])


@users_router.get('/{user_id:uuid}', response_model=UserFullSchema)
async def get_user(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Эндпоинт получения пользователя (READ)"""

    user_db = await user_crud.get_user(user_id, db=db)
    return user_db


# @users_router.get('/', response_model=UserFullSchema)
# async def get_users(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
#     """Эндпоинт получения пользователей (READ)"""
#
#     user_db = await user_crud.get_user(user_id, db=db)
#     return user_db


@users_router.post('/', response_model=UserFullSchema)
async def create_user(user: UserCreateSchema, db: AsyncSession = Depends(get_db)):
    """Эндпоинт создания пользователя (CREATE)"""

    user_db = await user_crud.create_user(user, db=db)
    return user_db


@users_router.put('/', response_model=UserFullSchema)
async def update_user(user: UserUpdateSchema, db: AsyncSession = Depends(get_db)):
    """Эндпоинт обновления пользователя (UPDATE)"""

    user_db = await user_crud.update_user(user, db=db)
    return user_db


@users_router.delete('/{user_id:uuid}', response_model=UserFullSchema)
async def delete_user(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Эндпоинт удаления пользователя (DELETE)"""

    user_db = await user_crud.delete_user(user_id, db=db)
    return user_db
