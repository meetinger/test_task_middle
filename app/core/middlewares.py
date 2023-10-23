from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


# class LogActivityMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next):
#         access_token = request.headers.get('Authorization').replace('Bearer ', '')
#
#
#         # if current_user is None and request.url.path != _LOGIN_URL:
#         #     return RedirectResponse(url=_LOGIN_URL, status_code=status.HTTP_302_FOUND)
#
#         response = await call_next(request)
#         if request.url.path != _LOGIN_URL:
#             response.set_cookie(key="access_token", value=create_access_token_from_login(current_user))
#         return response
