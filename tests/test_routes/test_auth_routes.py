from pprint import pprint

from fastapi import status
from httpx import AsyncClient

from tests.conftest import DATASET


class TestGetTokens:
    async def test_get_tokens_ok(self, client: AsyncClient, user, user_token):
        user_db = await user(DATASET['users']['1'])
        resp = await client.post('v1/auth/token', json=DATASET['users']['1'].model_dump())
        assert resp.status_code == status.HTTP_200_OK
        resp_json = resp.json()
        assert 'access_token' in resp_json and 'refresh_token' in resp_json

    async def test_get_tokens_wrong_password(self, client: AsyncClient, user, user_token):
        user_db = await user(DATASET['users']['1'])
        req_json = DATASET['users']['1'].model_dump()
        req_json['password'] = 'wrong_password'
        resp = await client.post('v1/auth/token', json=req_json)
        assert resp.status_code == status.HTTP_401_UNAUTHORIZED


class TestRefreshTokens:
    async def test_refresh_tokens_ok(self, client: AsyncClient, user, user_token):
        user_db = await user(DATASET['users']['1'])
        tokens = user_token(user_db)
        resp = await client.post('v1/auth/refresh_token', json=tokens)
        assert resp.status_code == status.HTTP_200_OK
        resp_json = resp.json()
        assert 'access_token' in resp_json and 'refresh_token' in resp_json
