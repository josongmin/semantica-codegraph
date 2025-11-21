from dataclasses import dataclass

from .enums import EmbeddingModel, LexicalSearchBackend, VectorStoreBackend


@dataclass
class Config:
    """애플리케이션 설정"""

    # PostgreSQL 설정
    postgres_host: str = "localhost"
    postgres_port: int = 7711
    postgres_user: str = "semantica"
    postgres_password: str = "semantica"
    postgres_db: str = "semantica_codegraph"

    # DDL 최적화: 런타임 테이블 초기화 스킵 (권장)
    # True로 설정하면 마이그레이션으로 스키마 생성 필수
    skip_table_init: bool = False  # 프로덕션에서는 True 권장

    # MeiliSearch 설정
    meilisearch_url: str = "http://localhost:7712"
    meilisearch_master_key: str | None = None

    # Zoekt 설정
    zoekt_url: str = "http://localhost:7713"
    zoekt_timeout: int = 30  # HTTP 요청 타임아웃 (초)

    # Qdrant 설정
    qdrant_host: str = "localhost"
    qdrant_port: int = 7714
    qdrant_grpc_port: int = 7715
    qdrant_use_grpc: bool = True  # gRPC 사용 여부 (더 빠름)
    qdrant_api_key: str | None = None  # Qdrant Cloud API 키 (옵션)
    qdrant_timeout: int = 30  # HTTP 요청 타임아웃 (초)

    # 검색 백엔드 선택
    vector_store_backend: VectorStoreBackend = VectorStoreBackend.QDRANT
    lexical_search_backend: LexicalSearchBackend = LexicalSearchBackend.MEILISEARCH

    # 임베딩 모델 설정
    embedding_model: EmbeddingModel = EmbeddingModel.CODESTRAL_EMBED
    embedding_api_key: str | None = None  # Mistral, OpenAI, Cohere API 키
    embedding_dimension: int | None = None  # 모델별 벡터 차원 (None이면 모델 기본값)
    mistral_api_base: str = "https://api.mistral.ai/v1"  # Mistral API 베이스 URL
    embedding_api_timeout: int = 30  # 임베딩 API 타임아웃 (초)

    # Chunker 설정
    chunker_max_tokens: int = 7000  # 청크 최대 토큰 수 (임베딩 API 제한보다 낮게 설정)
    chunker_max_lines: int = 100  # 청크 최대 라인 수
    chunker_overlap_lines: int = 5  # 청크 간 오버랩 라인 수
    chunker_enable_file_summary: bool = True  # 조건부 파일 요약 청크 생성
    chunker_min_symbols_for_summary: int = 5  # 파일 요약 생성 최소 심볼 개수

    # Fusion 전략 설정
    fusion_strategy: str = "weighted_sum"  # "weighted_sum" | "rrf" | "combsum"
    fusion_rrf_k: int = 60  # RRF 상수 (일반적으로 60)
    fusion_combsum_use_weights: bool = True  # CombSum 가중치 사용 여부

    # 퍼지 매칭 설정
    fuzzy_matching_enabled: bool = True  # 퍼지 매칭 활성화 여부
    fuzzy_threshold: float = 0.82  # 유사도 임계값 (0.0~1.0, 높을수록 엄격)
    fuzzy_max_candidates: int = 100  # 퍼지 매칭 대상 최대 심볼 수
    fuzzy_cache_size: int = 50000  # 퍼지 매칭 결과 캐시 크기 (LRU)
    fuzzy_results_per_token: int = 5  # 토큰당 반환할 퍼지 매칭 결과 수
    fuzzy_min_token_length: int = 3  # 퍼지 매칭할 최소 토큰 길이

    # 성능 최적화 설정
    db_connection_pool_size: int = 2  # DB 커넥션 풀 크기 (프로파일링 추가로 인한 조정)
    db_connection_pool_max: int = 5  # DB 커넥션 풀 최대 크기
    parallel_search_enabled: bool = True  # 검색 병렬화 활성화
    parallel_indexing_enabled: bool = True  # 인덱싱 병렬화 활성화
    max_workers: int = 8  # 병렬 처리 워커 수 (0이면 CPU 코어 수)

    # 검색 K 설정 (명확한 분리)
    retrieve_k: int = 100  # 후보 풀 크기 (각 백엔드가 가져오는 총량)
    rerank_k: int = 20  # Reranker에 보낼 후보 수 (LLM의 경우)
    final_k: int = 5  # 최종 반환 개수

    # 백엔드별 fetch 비율 (retrieve_k 기준)
    lexical_fetch_ratio: float = 0.5  # 50개 (precision 중심)
    semantic_fetch_ratio: float = 0.8  # 80개 (recall 보장, 넓게)
    graph_fetch_ratio: float = 0.4  # 40개
    fuzzy_fetch_ratio: float = 0.3  # 30개

    # Feature 추출 설정
    enable_file_metadata_features: bool = True  # 파일 메타데이터 feature
    enable_graph_features: bool = True  # PageRank, call graph feature

    # 점수 정규화 설정
    enable_score_normalization: bool = True  # 점수 정규화 활성화 여부
    lexical_score_max: float = 10.0  # Lexical(BM25) 점수 최대값 (정규화용)
    fuzzy_score_max: float = 1.0  # Fuzzy 점수 최대값 (정규화용)
    score_normalization_method: str = "rank"  # "minmax" | "rank" | "zscore"

    # Graph scoring 설정
    graph_depth_decay: float = 0.5  # Depth마다 점수 감소 계수 (0~1)
    graph_edge_weights: dict[str, float] | None = None  # Edge 타입별 가중치

    # Fuzzy scoring 설정
    fuzzy_stopwords: list[str] | None = None  # 제외할 불용어 리스트
    fuzzy_max_chunks_per_node: int = 3  # 노드당 반환할 최대 청크 수

    # Reranker 설정
    reranker_type: str = "basic"  # "basic" | "hybrid" | "morph" | "two-stage"
    reranker_debug_mode: bool = False  # 디버그 모드 (explanation 생성)

    # Two-Stage Reranker 설정
    two_stage_feature_reranker: str = "hybrid"  # "basic" | "hybrid" (1단계 reranker)
    two_stage_top_m: int = 20  # LLM에 보낼 후보 수
    two_stage_alpha: float = 0.7  # LLM 가중치 (0~1)
    two_stage_fallback: bool = True  # LLM 실패 시 feature 점수만 사용

    # LLM Scoring 설정
    llm_api_key: str | None = None  # LLM API 키 (Mistral)
    llm_model: str = "codestral-latest"  # LLM 모델명
    llm_temperature: float = 0.0  # LLM 온도 (0=deterministic)
    llm_max_tokens: int = 10  # LLM 최대 토큰 (점수만 반환)

    # Morph Rerank API 설정
    morph_api_key: str | None = None  # Morph API 키
    morph_api_base: str = "https://api-v2.morphstudio.com/v1"  # Morph API 베이스 URL
    morph_model: str = "morph-rerank-v3"  # Morph 모델명
    morph_top_k: int = 10  # 재순위화 후 반환할 최대 결과 수

    # OpenTelemetry 설정
    otel_enabled: bool = False  # OpenTelemetry 활성화
    otel_endpoint: str = "http://localhost:4317"  # OTLP gRPC endpoint
    otel_sample_rate: float = 1.0  # 샘플링 비율 (0.0~1.0, 프로덕션은 0.1 권장)
    otel_service_name: str = "semantica-codegraph"  # 서비스 이름
    environment: str = "development"  # 환경 (development, staging, production)

    def __post_init__(self):
        """기본값 초기화"""
        # Edge 타입별 기본 가중치 (중요도 순)
        if self.graph_edge_weights is None:
            self.graph_edge_weights = {
                "calls": 1.0,  # 호출 관계가 가장 중요
                "uses": 0.8,  # 사용 관계
                "inherits": 0.9,  # 상속 관계
                "implements": 0.9,  # 인터페이스 구현
                "imports": 0.6,  # 임포트 관계
                "defines": 0.7,  # 정의 관계
                "belongs_to": 0.5,  # 소속 관계
                "overrides": 0.85,  # 오버라이드
            }

        # Fuzzy 검색 불용어 기본값 (일반적인 get/set/is 등)
        if self.fuzzy_stopwords is None:
            self.fuzzy_stopwords = [
                "get",
                "set",
                "is",
                "has",
                "can",
                "should",
                "will",
                "do",
                "make",
                "create",
                "update",
                "delete",
                "find",
                "to",
                "from",
                "with",
                "for",
                "by",
                "on",
                "at",
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "of",
            ]

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
            postgres_port=int(os.getenv("POSTGRES_PORT", "7711")),
            postgres_user=os.getenv("POSTGRES_USER", "semantica"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", "semantica"),
            postgres_db=os.getenv("POSTGRES_DB", "semantica_codegraph"),
            meilisearch_url=os.getenv("MEILISEARCH_URL", "http://localhost:7712"),
            meilisearch_master_key=os.getenv("MEILISEARCH_MASTER_KEY"),
            zoekt_url=os.getenv("ZOEKT_URL", "http://localhost:7713"),
            zoekt_timeout=int(os.getenv("ZOEKT_TIMEOUT", "30")),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "7714")),
            qdrant_grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "7715")),
            qdrant_use_grpc=os.getenv("QDRANT_USE_GRPC", "true").lower() == "true",
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_timeout=int(os.getenv("QDRANT_TIMEOUT", "30")),
            vector_store_backend=VectorStoreBackend(
                os.getenv("VECTOR_STORE_BACKEND", VectorStoreBackend.QDRANT.value)
            ),
            lexical_search_backend=LexicalSearchBackend(
                os.getenv("LEXICAL_SEARCH_BACKEND", LexicalSearchBackend.MEILISEARCH.value)
            ),
            embedding_model=EmbeddingModel(
                os.getenv("EMBEDDING_MODEL", EmbeddingModel.CODESTRAL_EMBED.value)
            ),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            embedding_dimension=(int(dim) if (dim := os.getenv("EMBEDDING_DIMENSION")) else None),
            mistral_api_base=os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1"),
            chunker_max_tokens=int(os.getenv("CHUNKER_MAX_TOKENS", "7000")),
            chunker_max_lines=int(os.getenv("CHUNKER_MAX_LINES", "100")),
            chunker_overlap_lines=int(os.getenv("CHUNKER_OVERLAP_LINES", "5")),
            chunker_enable_file_summary=os.getenv("CHUNKER_ENABLE_FILE_SUMMARY", "true").lower()
            == "true",
            chunker_min_symbols_for_summary=int(os.getenv("CHUNKER_MIN_SYMBOLS_FOR_SUMMARY", "5")),
            fusion_strategy=os.getenv("FUSION_STRATEGY", "weighted_sum"),
            fusion_rrf_k=int(os.getenv("FUSION_RRF_K", "60")),
            fusion_combsum_use_weights=os.getenv("FUSION_COMBSUM_USE_WEIGHTS", "true").lower()
            == "true",
            fuzzy_matching_enabled=os.getenv("FUZZY_MATCHING_ENABLED", "true").lower() == "true",
            fuzzy_threshold=float(os.getenv("FUZZY_THRESHOLD", "0.82")),
            fuzzy_max_candidates=int(os.getenv("FUZZY_MAX_CANDIDATES", "100")),
            fuzzy_cache_size=int(os.getenv("FUZZY_CACHE_SIZE", "50000")),
            fuzzy_results_per_token=int(os.getenv("FUZZY_RESULTS_PER_TOKEN", "5")),
            fuzzy_min_token_length=int(os.getenv("FUZZY_MIN_TOKEN_LENGTH", "3")),
            db_connection_pool_size=int(os.getenv("DB_CONNECTION_POOL_SIZE", "10")),
            db_connection_pool_max=int(os.getenv("DB_CONNECTION_POOL_MAX", "20")),
            parallel_search_enabled=os.getenv("PARALLEL_SEARCH_ENABLED", "true").lower() == "true",
            parallel_indexing_enabled=os.getenv("PARALLEL_INDEXING_ENABLED", "true").lower()
            == "true",
            max_workers=int(os.getenv("MAX_WORKERS", "8")),
            retrieve_k=int(os.getenv("RETRIEVE_K", "100")),
            rerank_k=int(os.getenv("RERANK_K", "20")),
            final_k=int(os.getenv("FINAL_K", "5")),
            lexical_fetch_ratio=float(os.getenv("LEXICAL_FETCH_RATIO", "0.5")),
            semantic_fetch_ratio=float(os.getenv("SEMANTIC_FETCH_RATIO", "0.8")),
            graph_fetch_ratio=float(os.getenv("GRAPH_FETCH_RATIO", "0.4")),
            fuzzy_fetch_ratio=float(os.getenv("FUZZY_FETCH_RATIO", "0.3")),
            enable_file_metadata_features=os.getenv("ENABLE_FILE_METADATA_FEATURES", "true").lower()
            == "true",
            enable_graph_features=os.getenv("ENABLE_GRAPH_FEATURES", "true").lower() == "true",
            enable_score_normalization=os.getenv("ENABLE_SCORE_NORMALIZATION", "true").lower()
            == "true",
            lexical_score_max=float(os.getenv("LEXICAL_SCORE_MAX", "10.0")),
            fuzzy_score_max=float(os.getenv("FUZZY_SCORE_MAX", "1.0")),
            reranker_type=os.getenv("RERANKER_TYPE", "basic"),
            reranker_debug_mode=os.getenv("RERANKER_DEBUG_MODE", "false").lower() == "true",
            two_stage_feature_reranker=os.getenv("TWO_STAGE_FEATURE_RERANKER", "hybrid"),
            two_stage_top_m=int(os.getenv("TWO_STAGE_TOP_M", "20")),
            two_stage_alpha=float(os.getenv("TWO_STAGE_ALPHA", "0.7")),
            two_stage_fallback=os.getenv("TWO_STAGE_FALLBACK", "true").lower() == "true",
            llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("MISTRAL_API_KEY"),
            llm_model=os.getenv("LLM_MODEL", "codestral-latest"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "10")),
            morph_api_key=os.getenv("MORPH_API_KEY"),
            morph_api_base=os.getenv("MORPH_API_BASE", "https://api-v2.morphstudio.com/v1"),
            morph_model=os.getenv("MORPH_MODEL", "morph-rerank-v3"),
            morph_top_k=int(os.getenv("MORPH_TOP_K", "10")),
            otel_enabled=os.getenv("OTEL_ENABLED", "false").lower() == "true",
            otel_endpoint=os.getenv("OTEL_ENDPOINT", "http://localhost:4317"),
            otel_sample_rate=float(os.getenv("OTEL_SAMPLE_RATE", "1.0")),
            otel_service_name=os.getenv("OTEL_SERVICE_NAME", "semantica-codegraph"),
            environment=os.getenv("ENVIRONMENT", "development"),
        )
