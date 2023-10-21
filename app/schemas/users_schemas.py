import datetime
import uuid

from pydantic import BaseModel, Field, EmailStr


class UserCreateSchema(BaseModel):
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
