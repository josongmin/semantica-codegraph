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
    embedding_model: EmbeddingModel = EmbeddingModel.CODESTRAL_EMBED
    embedding_api_key: Optional[str] = None  # Mistral, OpenAI, Cohere API 키
    embedding_dimension: Optional[int] = None  # 모델별 벡터 차원 (None이면 모델 기본값)
    mistral_api_base: str = "https://api.mistral.ai/v1"  # Mistral API 베이스 URL

    # 퍼지 매칭 설정
    fuzzy_matching_enabled: bool = True  # 퍼지 매칭 활성화 여부
    fuzzy_threshold: float = 0.82  # 유사도 임계값 (0.0~1.0, 높을수록 엄격)
    fuzzy_max_candidates: int = 100  # 퍼지 매칭 대상 최대 심볼 수
    fuzzy_cache_size: int = 50000  # 퍼지 매칭 결과 캐시 크기 (LRU)
    fuzzy_results_per_token: int = 5  # 토큰당 반환할 퍼지 매칭 결과 수
    fuzzy_min_token_length: int = 3  # 퍼지 매칭할 최소 토큰 길이

    # 성능 최적화 설정
    db_connection_pool_size: int = 10  # DB 커넥션 풀 크기
    db_connection_pool_max: int = 20  # DB 커넥션 풀 최대 크기
    parallel_search_enabled: bool = True  # 검색 병렬화 활성화
    parallel_indexing_enabled: bool = True  # 인덱싱 병렬화 활성화
    max_workers: int = 8  # 병렬 처리 워커 수 (0이면 CPU 코어 수)

    @classmethod
    def from_env(cls) -> "Config":
        """환경변수에서 설정 로드 (.env 파일 자동 로드)"""
        import os
        
        # .env 파일 로드 (존재하는 경우)
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            # python-dotenv가 설치되지 않은 경우 무시
            pass

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
                os.getenv("EMBEDDING_MODEL", EmbeddingModel.CODESTRAL_EMBED.value)
            ),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY"),
            embedding_dimension=(
                int(os.getenv("EMBEDDING_DIMENSION")) if os.getenv("EMBEDDING_DIMENSION") else None
            ),
            mistral_api_base=os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1"),
            fuzzy_matching_enabled=os.getenv("FUZZY_MATCHING_ENABLED", "true").lower() == "true",
            fuzzy_threshold=float(os.getenv("FUZZY_THRESHOLD", "0.82")),
            fuzzy_max_candidates=int(os.getenv("FUZZY_MAX_CANDIDATES", "100")),
            fuzzy_cache_size=int(os.getenv("FUZZY_CACHE_SIZE", "50000")),
            fuzzy_results_per_token=int(os.getenv("FUZZY_RESULTS_PER_TOKEN", "5")),
            fuzzy_min_token_length=int(os.getenv("FUZZY_MIN_TOKEN_LENGTH", "3")),
            db_connection_pool_size=int(os.getenv("DB_CONNECTION_POOL_SIZE", "10")),
            db_connection_pool_max=int(os.getenv("DB_CONNECTION_POOL_MAX", "20")),
            parallel_search_enabled=os.getenv("PARALLEL_SEARCH_ENABLED", "true").lower() == "true",
            parallel_indexing_enabled=os.getenv("PARALLEL_INDEXING_ENABLED", "true").lower() == "true",
            max_workers=int(os.getenv("MAX_WORKERS", "8")),
        )

