from dataclasses import dataclass
from typing import Optional

from .enums import EmbeddingModel, LexicalSearchBackend


@dataclass
class Config:
    """애플리케이션 설정"""

    # PostgreSQL 설정
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "semantica"
    postgres_password: str = "semantica"
    postgres_db: str = "semantica_codegraph"

    # MeiliSearch 설정
    meilisearch_url: str = "http://localhost:7700"
    meilisearch_master_key: Optional[str] = None

    # Zoekt 설정
    zoekt_url: str = "http://localhost:6070"
    zoekt_timeout: int = 30  # HTTP 요청 타임아웃 (초)

    # 검색 백엔드 선택
    lexical_search_backend: LexicalSearchBackend = LexicalSearchBackend.MEILISEARCH

    # 임베딩 모델 설정
    embedding_model: EmbeddingModel = EmbeddingModel.ALL_MINI_LM_L6_V2
    embedding_api_key: Optional[str] = None  # OpenAI, Cohere 등 API 키
    embedding_dimension: Optional[int] = None  # 모델별 벡터 차원 (None이면 모델 기본값)

    @classmethod
    def from_env(cls) -> "Config":
        """환경변수에서 설정 로드"""
        import os

        return cls(
            postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
            postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
            postgres_user=os.getenv("POSTGRES_USER", "semantica"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", "semantica"),
            postgres_db=os.getenv("POSTGRES_DB", "semantica_codegraph"),
            meilisearch_url=os.getenv("MEILISEARCH_URL", "http://localhost:7700"),
            meilisearch_master_key=os.getenv("MEILISEARCH_MASTER_KEY"),
            zoekt_url=os.getenv("ZOEKT_URL", "http://localhost:6070"),
            zoekt_timeout=int(os.getenv("ZOEKT_TIMEOUT", "30")),
            lexical_search_backend=LexicalSearchBackend(
                os.getenv("LEXICAL_SEARCH_BACKEND", LexicalSearchBackend.MEILISEARCH.value)
            ),
            embedding_model=EmbeddingModel(
                os.getenv("EMBEDDING_MODEL", EmbeddingModel.ALL_MINI_LM_L6_V2.value)
            ),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY"),
            embedding_dimension=(
                int(os.getenv("EMBEDDING_DIMENSION")) if os.getenv("EMBEDDING_DIMENSION") else None
            ),
        )

