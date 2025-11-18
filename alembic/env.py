"""Alembic 마이그레이션 환경 설정"""

from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context
from src.core.config import Config

# Alembic Config 객체
config = context.config

# 로깅 설정
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 메타데이터 가져오기 (SQLAlchemy 모델이 있다면)
# from src.core.models import Base
# target_metadata = Base.metadata
target_metadata = None

# DB 연결 문자열 생성
def get_url() -> str:
    """Config에서 DB 연결 문자열 생성"""
    cfg = Config.from_env()
    return (
        f"postgresql://{cfg.postgres_user}:{cfg.postgres_password}"
        f"@{cfg.postgres_host}:{cfg.postgres_port}/{cfg.postgres_db}"
    )


def run_migrations_offline() -> None:
    """오프라인 모드에서 마이그레이션 실행"""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """온라인 모드에서 마이그레이션 실행"""
    connectable = create_engine(
        get_url(),
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

