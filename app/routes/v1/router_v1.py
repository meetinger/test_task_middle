from fastapi import APIRouter

from app.routes.v1.auth_routes import auth_router
from app.routes.v1.users_routes import users_router

router_v1 = APIRouter()
router_v1.include_router(users_router, prefix="/users")
router_v1.include_router(auth_router, prefix="/auth")
