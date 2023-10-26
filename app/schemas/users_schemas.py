import datetime
import uuid
from typing import Generic, TypeVar, TypeAlias

from pydantic import BaseModel, Field, EmailStr

from app.schemas.common_shemas import Entity

# в pydantic v2 нельзя удалять поля при наследовании, поэтому наследование я не использовал

UserSchema = TypeVar('UserSchema', bound=[Entity])  # дженерик для схем пользователя


class UserOutSchema(BaseModel, Generic[UserSchema]):
    """Выходная схема пользователя"""

    id: uuid.UUID = Field(description='ID пользователя')
    username: str | None = Field(description='Имя пользователя', default=None)
    email: EmailStr | None = Field(description='Email пользователя', default=None)
    registration_date: datetime.datetime = Field(description='Дата регистрации пользователя')

    class Config:
        from_attributes = True


class UserCreateSchema(BaseModel, Generic[UserSchema]):
    """Схема для создания пользователя"""

    username: str = Field(description='Имя пользователя')
    email: EmailStr = Field(description='Email пользователя')
    password: str = Field(description='Пароль пользователя')

    class Config:
        from_attributes = True


class UserUpdateSchema(BaseModel, Generic[UserSchema]):
    """Схема для обновления пользователя"""

    id: uuid.UUID = Field(description='ID пользователя')
    username: str | None = Field(description='Имя пользователя', default=None)
    email: EmailStr | None = Field(description='Email пользователя', default=None)
    password: str | None = Field(description='Пароль пользователя', default=None)

    class Config:
        from_attributes = True


class UserAuthSchema(BaseModel, Generic[UserSchema]):
    """Схема для аутентификации пользователя"""

    username: str = Field(description='Имя пользователя')
    password: str = Field(description='Пароль пользователя')


class Tokens(BaseModel, Generic[Entity]):
    """Токены"""

    access_token: str | None = Field(description='Access токен', default=None)
    refresh_token: str | None = Field(description='Refresh токен', default=None)

