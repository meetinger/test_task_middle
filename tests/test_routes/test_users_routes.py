from pprint import pprint

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from tests.conftest import DATASET


async def test_get_user_ok(client: AsyncClient, user, user_token):
    user_db = await user(DATASET['users']['1'])
    tokens = user_token(user_db)

    resp = await client.get(f'v1/users/{user_db.id}', headers={'Authorization': f'Bearer {tokens["access_token"]}'})

    assert resp.status_code == status.HTTP_200_OK


async def test_get_users_ok(client: AsyncClient, user, user_token):
    user_db = await user(DATASET['users']['1'])
    tokens = user_token(user_db)

    resp = await client.get(f'v1/users/', headers={'Authorization': f'Bearer {tokens["access_token"]}'})

    assert resp.status_code == status.HTTP_200_OK


async def test_create_users_ok(client: AsyncClient, user):
    resp = await client.post(f'v1/users/', json=DATASET['users']['1'].model_dump())

    assert resp.status_code == status.HTTP_200_OK


async def test_update_user_ok(client: AsyncClient, user, user_token):
    user_db = await user(DATASET['users']['1'])
    tokens = user_token(user_db)

    req_body = DATASET['users']['2'].model_dump()
    req_body['id'] = str(user_db.id)

    resp = await client.put(f'v1/users/', headers={'Authorization': f'Bearer {tokens["access_token"]}'}, json=req_body)

    assert resp.status_code == status.HTTP_200_OK


async def test_delete_user_ok(client: AsyncClient, user, user_token):
    user_db = await user(DATASET['users']['1'])
    tokens = user_token(user_db)

    resp = await client.delete(f'v1/users/{user_db.id}', headers={'Authorization': f'Bearer {tokens["access_token"]}'})

    assert resp.status_code == status.HTTP_200_OK


async def test_analytics_user_ok(client: AsyncClient, user, user_token):
    user_db = await user(DATASET['users']['1'])
    tokens = user_token(user_db)

    resp = await client.get(f'v1/users/analytics', headers={'Authorization': f'Bearer {tokens["access_token"]}'})

    assert resp.status_code == status.HTTP_200_OK
