from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError
from sqlalchemy.ext.asyncio import AsyncSession

import app.db.crud.users_crud as user_crud
from app.core.security import PasswordUtils, TokenUtils
from app.db.database import get_db
from app.db.models import User
from app.schemas.users_schemas import Tokens, UserAuthSchema

auth_router = APIRouter(tags=['auth'])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/get_token")


def create_tokens(user_db: User) -> Tokens:
    """Создание пары токенов"""

    tokens = TokenUtils.create_token_pair({'sub': user_db.username})
    return Tokens(access_token=tokens['access_token'], refresh_token=tokens['refresh_token'])


async def authenticate_user(user: UserAuthSchema, db: AsyncSession) -> User | bool:
    """Аутентификация пользователя"""

    user_db = await user_crud.get_user_by_username(user.username, db=db)

    if PasswordUtils.verify_password(plain_password=user.password, hashed_password=user_db.password_hash):
        return user_db

    return False


async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User | NoReturn:
    """Получить текущего пользователя из токена"""

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"Authorization": "Bearer"},
    )
    try:
        payload = TokenUtils.decode_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await user_crud.get_user_by_username(username, db=db)
    return user


@auth_router.post('/token', response_model=Tokens)
async def get_tokens(user: UserAuthSchema, db: AsyncSession = Depends(get_db)) -> Tokens:
    """Получить токены"""

    user_db = await user_crud.get_user_by_username(user.username, db=db)

    if PasswordUtils.verify_password(plain_password=user.password, hashed_password=user_db.password_hash):
        tokens = create_tokens(user_db)
        return tokens

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate credentials')


@auth_router.post('/refresh_token', response_model=Tokens)
async def refresh_tokens(tokens: Tokens, db: AsyncSession = Depends(get_db)) -> Tokens:
    """Обновить токены"""

    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
    )
    try:
        payload = TokenUtils.decode_token(tokens.refresh_token)
        username = payload['sub']
        token_type = payload['token_type']
        if token_type != 'refresh_token':
            raise HTTPException(status_code=403, detail='Wrong token type')
    except ExpiredSignatureError:
        raise HTTPException(status_code=403, detail='Token expired')
    except Exception as e:
        raise credentials_exception
    user_db = await user_crud.get_user_by_username(username=username, db=db)
    try:
        tokens = create_tokens(user_db)
        return tokens
    except Exception as e:
        raise HTTPException(status_code=500, detail='Error while token pair generation')
