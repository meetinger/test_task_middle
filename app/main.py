import uvicorn
from fastapi import FastAPI

from app.core.settings import settings
from app.routes.v1.router_v1 import router_v1

app = FastAPI()

app.include_router(router_v1, prefix='/v1')


if __name__ == '__main__':
    print('Starting server...')
    uvicorn.run('main:app', host="0.0.0.0", port=int(settings.HTTP_PORT), reload=True)
