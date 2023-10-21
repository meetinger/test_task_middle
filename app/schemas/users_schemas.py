import datetime
import uuid
from typing import Generic

from pydantic import BaseModel, Field, EmailStr

from app.schemas.common_shemas import Entity


class UserCreateSchema(BaseModel, Generic[Entity]):
    """Схема для создания пользователя"""
    username: str | None = Field(description='Имя пользователя', default=None)
    email: EmailStr | None = Field(description='Email пользователя', default=None)

    class Config:
        from_attributes = True


class UserUpdateSchema(UserCreateSchema):
    """Схема для обновления пользователя"""
    id: uuid.UUID = Field(description='ID пользователя')


class UserFullSchema(UserUpdateSchema):
    """Схема пользователя"""
    registration_date: datetime.datetime = Field(description='Дата регистрации пользователя')
