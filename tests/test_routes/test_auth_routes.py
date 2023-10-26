from pprint import pprint
from unittest.mock import patch

from fastapi import status
from httpx import AsyncClient

from app.core.settings import settings
from tests.conftest import DATASET


class TestGetTokens:
    async def test_get_tokens_ok(self, client: AsyncClient, user, user_token):
        """Тест получения токенов если всё правильно"""

        user_db = await user(DATASET['users']['1'])
        resp = await client.post('v1/auth/token', json=DATASET['users']['1'].model_dump())
        assert resp.status_code == status.HTTP_200_OK
        resp_json = resp.json()
        assert 'access_token' in resp_json and 'refresh_token' in resp_json

    async def test_get_tokens_wrong_password(self, client: AsyncClient, user, user_token):
        """Тест получения токенов если всё неправильный пароль"""

        user_db = await user(DATASET['users']['1'])
        req_json = DATASET['users']['1'].model_dump()
        req_json['password'] = 'wrong_password'
        resp = await client.post('v1/auth/token', json=req_json)
        assert resp.status_code == status.HTTP_401_UNAUTHORIZED


class TestRefreshTokens:
    async def test_refresh_tokens_ok(self, client: AsyncClient, user, user_token):
        """Тест обновления токенов если всё правильно"""

        user_db = await user(DATASET['users']['1'])
        tokens = user_token(user_db)
        resp = await client.post('v1/auth/refresh_token', json=tokens)
        assert resp.status_code == status.HTTP_200_OK
        resp_json = resp.json()
        assert 'access_token' in resp_json and 'refresh_token' in resp_json

    async def test_refresh_tokens_wrong_token_type(self, client: AsyncClient, user, user_token):
        """Тест обновления токенов неправильный тип токена"""

        user_db = await user(DATASET['users']['1'])
        tokens = user_token(user_db)
        tokens['refresh_token'] = tokens['access_token']
        resp = await client.post('v1/auth/refresh_token', json=tokens)
        assert resp.status_code == status.HTTP_403_FORBIDDEN

    @patch.object(settings, 'REFRESH_TOKEN_EXPIRE_MINUTES', -1)
    async def test_refresh_tokens_expired(self, client: AsyncClient, user, user_token):
        """Тест обновления токенов если токен просрочен"""

        user_db = await user(DATASET['users']['1'])
        tokens = user_token(user_db)
        resp = await client.post('v1/auth/refresh_token', json=tokens)
        assert resp.status_code == status.HTTP_403_FORBIDDEN
