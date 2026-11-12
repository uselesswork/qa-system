"""Argon2 密码哈希与数据库中的短期登录会话。"""

import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256

from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.db import AuthSession, User


PASSWORD_HASH = PasswordHash.recommended()
DUMMY_PASSWORD_HASH = PASSWORD_HASH.hash("unused-authentication-placeholder")
SESSION_LIFETIME = timedelta(days=1)
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,32}$")


class AuthError(Exception):
    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    username: str
    csrf_token: str


class AuthService:
    def __init__(self, sessions: sessionmaker):
        self.sessions = sessions

    def register(self, username: str, password: str) -> dict:
        username = username.strip().lower()
        if not USERNAME_PATTERN.fullmatch(username):
            raise AuthError("用户名需为 3～32 位英文字母、数字或下划线。", 400)
        if not 12 <= len(password) <= 128:
            raise AuthError("密码长度需为 12～128 个字符。", 400)
        password_hash = PASSWORD_HASH.hash(password)
        try:
            with self.sessions.begin() as session:
                user = User(username=username, password_hash=password_hash)
                session.add(user)
                session.flush()
                return {"id": user.id, "username": user.username}
        except IntegrityError as exc:
            raise AuthError("用户名已存在。", 409) from exc

    def login(self, username: str, password: str) -> tuple[str, AuthContext]:
        with self.sessions() as session:
            user = session.scalar(select(User).where(User.username == username.strip().lower()))
            valid = PASSWORD_HASH.verify(
                password, user.password_hash if user is not None else DUMMY_PASSWORD_HASH
            )
            if user is None or not valid:
                raise AuthError("用户名或密码错误。")
            user_id, saved_username = user.id, user.username
        token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(24)
        with self.sessions.begin() as session:
            session.add(AuthSession(
                token_hash=sha256(token.encode()).hexdigest(),
                user_id=user_id,
                csrf_token=csrf_token,
                expires_at=datetime.now(timezone.utc).replace(tzinfo=None) + SESSION_LIFETIME,
            ))
        return token, AuthContext(user_id, saved_username, csrf_token)

    def authenticate(self, token: str | None) -> AuthContext:
        if not token:
            raise AuthError("请先登录。")
        token_hash = sha256(token.encode()).hexdigest()
        with self.sessions() as session:
            auth = session.get(AuthSession, token_hash)
            if auth is None or auth.expires_at <= datetime.now(timezone.utc).replace(tzinfo=None):
                raise AuthError("登录已失效，请重新登录。")
            user = session.get(User, auth.user_id)
            if user is None:
                raise AuthError("账号不存在。")
            return AuthContext(user.id, user.username, auth.csrf_token)

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with self.sessions.begin() as session:
            auth = session.get(AuthSession, sha256(token.encode()).hexdigest())
            if auth is not None:
                session.delete(auth)
