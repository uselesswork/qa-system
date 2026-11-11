"""Alembic 从本地环境读取连接串，仓库不保存数据库密码。"""

from alembic import context

from src.db import Base
from src.db.session import create_database_engine, get_database_url


target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_database_engine()
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection, target_metadata=target_metadata,
                compare_type=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
