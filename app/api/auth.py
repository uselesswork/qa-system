"""登录 API 与通用用户/CSRF 依赖。"""

from threading import Lock

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel

from src.auth.service import AuthContext, AuthError, AuthService, SESSION_LIFETIME
from src.db.session import create_database_engine, create_session_factory


router = APIRouter(prefix="/auth", tags=["auth"])
_auth_lock = Lock()
COOKIE_NAME = "qa_session"


class Credentials(BaseModel):
    username: str
    password: str


def get_auth_service(request: Request) -> AuthService:
    service = getattr(request.app.state, "auth_service", None)
    if service is not None:
        return service
    with _auth_lock:
        service = getattr(request.app.state, "auth_service", None)
        if service is not None:
            return service
        try:
            engine = create_database_engine()
            service = AuthService(create_session_factory(engine))
            request.app.state.auth_service = service
            request.app.state.auth_engine = engine
            return service
        except Exception as exc:
            raise HTTPException(503, "账号数据库不可用。") from exc


def require_user(request: Request,
                 service: AuthService = Depends(get_auth_service)) -> AuthContext:
    try:
        return service.authenticate(request.cookies.get(COOKIE_NAME))
    except AuthError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


def require_csrf(user: AuthContext = Depends(require_user),
                 csrf_token: str = Header(default="", alias="X-CSRF-Token")) -> AuthContext:
    from secrets import compare_digest
    if not csrf_token or not compare_digest(csrf_token, user.csrf_token):
        raise HTTPException(403, "缺少或无效的 CSRF 令牌。")
    return user


@router.post("/register", status_code=201)
def register(body: Credentials, service: AuthService = Depends(get_auth_service)):
    try:
        return service.register(body.username, body.password)
    except AuthError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.post("/login")
def login(body: Credentials, request: Request, response: Response,
          service: AuthService = Depends(get_auth_service)):
    try:
        token, user = service.login(body.username, body.password)
    except AuthError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    response.set_cookie(
        COOKIE_NAME, token, max_age=int(SESSION_LIFETIME.total_seconds()),
        httponly=True, secure=request.url.scheme == "https", samesite="strict", path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return {"id": user.user_id, "username": user.username, "csrf_token": user.csrf_token}


@router.get("/me")
def me(response: Response, user: AuthContext = Depends(require_user)):
    response.headers["Cache-Control"] = "no-store"
    return {"id": user.user_id, "username": user.username, "csrf_token": user.csrf_token}


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response,
           user: AuthContext = Depends(require_csrf),
           service: AuthService = Depends(get_auth_service)):
    service.logout(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(COOKIE_NAME, path="/")
