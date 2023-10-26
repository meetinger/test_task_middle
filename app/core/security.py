from fastapi import Header, status, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

import logging
from datetime import timedelta, datetime
from typing import Literal, NoReturn

from jose import jwt, JWTError
from passlib.context import CryptContext

from app.core.settings import settings
from app.db.crud import users_crud as user_crud
from app.db.database import get_db
from app.db.models import User
from app.schemas.users_schemas import Tokens, UserAuthSchema

crypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

logger = logging.getLogger(__name__)


class PasswordUtils:
    """Класс для работы с паролями"""

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Проверка пароля"""

        return crypt_context.verify(plain_password, hashed_password)

    @staticmethod
    def hash_password(password: str) -> str:
        """Хеширование пароля"""

        return crypt_context.hash(password)


class TokenUtils:
    """Класс для работы с JWT-токенами"""

    @staticmethod
    def create_token(data: dict, token_type: Literal['access_token', 'refresh_token']) -> str:
        """Создание токена"""

        data_to_encode = data.copy()
        expire_in = datetime.utcnow() + timedelta(
            minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES if token_type == 'refresh_token'
            else settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        data_to_encode['token_type'] = token_type
        data_to_encode['exp'] = expire_in
        encoded_jwt = jwt.encode(data_to_encode, key=settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt

    @staticmethod
    def decode_token(token: str) -> dict | None:
        """Расшифровка токена"""

        return jwt.decode(token, key=settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

    @staticmethod
    def create_token_pair(data: dict) -> dict:
        """Создание пары access-refresh токенов"""

        return {
            'access_token': TokenUtils.create_token(data, token_type='access_token'),
            'refresh_token': TokenUtils.create_token(data, token_type='refresh_token'),
            'token_type': 'Bearer'
        }

