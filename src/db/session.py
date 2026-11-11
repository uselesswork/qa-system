"""数据库连接只在调用时创建，导入模块不会连接 MySQL。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_database_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("缺少 DATABASE_URL：请在项目根目录 .env 中配置数据库连接。")
    return url


def create_database_engine(url: str | None = None) -> Engine:
    return create_engine(url or get_database_url(), pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
