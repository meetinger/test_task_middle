import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

import app.db.crud.users_crud as user_crud
from app.db.database import get_db
from app.schemas.common_shemas import EntityList
from app.schemas.users_schemas import UserFullSchema, UserCreateSchema, UserUpdateSchema

users_router = APIRouter(tags=['users'])


@users_router.get('/{user_id:uuid}', response_model=UserFullSchema)
async def get_user(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Эндпоинт получения пользователя (READ)"""

    user_db = await user_crud.get_user(user_id, db=db)
    return user_db


@users_router.get('/', response_model=EntityList)
async def get_users(page: int = Query(default=0, description='Номер страницы'),
                    page_size: int = Query(default=10, description='Размер страницы(максимум 50)'), db: AsyncSession = Depends(get_db)):
    """Эндпоинт получения пользователей (READ) с пагинацией"""
    if page_size > 50:
        page_size = 50
    users_db = await user_crud.get_users(page, page_size, db=db)
    return EntityList[UserFullSchema](entities=users_db)


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


@users_router.get('/analytics')
async def user_analytics(email_domain: str = Query(default='gmail.com', description='Домен электронной почты'),
                         db: AsyncSession = Depends(get_db)):
    """Эндпоинт с данными из различных функций из 2-ого задания"""

    last_registered_users = await user_crud.get_count_last_7d_registered_users(db)
    top_username_users = await user_crud.get_top5_users_with_longest_usernames(db)
    email_domain_users = await user_crud.get_users_by_email_domain(email_domain, db=db)

    return {
        'last_registered_users': last_registered_users,
        'top_username_users': top_username_users,
        'email_domain_users': email_domain_users
    }
