"""의존성 주입 및 포트 초기화"""

import logging
from typing import Any

from meilisearch import Client

from ..search.adapters.lexical.meili_adapter import MeiliSearchAdapter
from ..search.adapters.lexical.zoekt_adapter import ZoektAdapter
from ..search.ports.lexical_search_port import LexicalSearchPort
from .config import Config
from .enums import LexicalSearchBackend

logger = logging.getLogger(__name__)


class Bootstrap:
    """포트 인스턴스 생성 및 의존성 주입"""

    def __init__(self, config: Config):
        self.config = config
        self._connection_string = self._build_connection_string()

        # 인스턴스 캐시 (lazy loading)
        self._repo_store: Any = None
        self._graph_store: Any = None
        self._chunk_store: Any = None
        self._embedding_service: Any = None
        self._embedding_service_small: Any = None  # Phase 1: 3-small
        self._embedding_service_large: Any = None  # Phase 1: 3-large
        self._embedding_store: Any = None
        self._qdrant_store: Any = None  # Qdrant vector store
        self._semantic_node_store: Any = None  # Phase 1: semantic nodes
        self._lexical_search: Any = None
        self._ir_builder: Any = None
        self._chunker: Any = None
        self._scanner: Any = None
        self._pipeline: Any = None
        self._semantic_search: Any = None
        self._graph_search: Any = None
        self._fuzzy_search: Any = None
        self._symbol_search: Any = None
        self._fusion_strategy: Any = None
        self._hybrid_retriever: Any = None
        self._ranker: Any = None
        self._context_packer: Any = None
        self._route_store: Any = None
        self._query_log_store: Any = None  # Phase 2

    def _build_connection_string(self) -> str:
        """PostgreSQL 연결 문자열 생성"""
        return (
            f"host={self.config.postgres_host} "
            f"port={self.config.postgres_port} "
            f"dbname={self.config.postgres_db} "
            f"user={self.config.postgres_user} "
            f"password={self.config.postgres_password}"
        )

    @property
    def repo_store(self):
        """저장소 메타데이터 스토어"""
        if self._repo_store is None:
            from .repo_store import RepoMetadataStore

            self._repo_store = RepoMetadataStore(self._connection_string)
        return self._repo_store

    @property
    def graph_store(self):
        """코드 그래프 스토어"""
        if self._graph_store is None:
            from ..graph.store_postgres import PostgresGraphStore

            self._graph_store = PostgresGraphStore(
                connection_string=self._connection_string,
                pool_size=self.config.db_connection_pool_size,
                pool_max=self.config.db_connection_pool_max,
            )
        return self._graph_store

    @property
    def chunk_store(self):
        """청크 스토어"""
        if self._chunk_store is None:
            from ..chunking.store import PostgresChunkStore

            self._chunk_store = PostgresChunkStore(self._connection_string)
        return self._chunk_store

    @property
    def embedding_service(self):
        """임베딩 서비스 (기본, 코드 청크용)"""
        if self._embedding_service is None:
            from ..embedding.service import EmbeddingService

            self._embedding_service = EmbeddingService(
                model=self.config.embedding_model,
                api_key=self.config.embedding_api_key,
                api_base=self.config.mistral_api_base,
                dimension=self.config.embedding_dimension,
                timeout=self.config.embedding_api_timeout,
            )
        return self._embedding_service

    @property
    def embedding_service_small(self):
        """임베딩 서비스 (3-small, semantic nodes용)"""
        if self._embedding_service_small is None:
            import os

            from ..core.enums import EmbeddingModel
            from ..embedding.service import EmbeddingService

            # OpenAI 전용이므로 OPENAI_API_KEY 직접 사용
            openai_key = os.getenv("OPENAI_API_KEY") or self.config.embedding_api_key

            self._embedding_service_small = EmbeddingService(
                model=EmbeddingModel.OPENAI_3_SMALL,
                api_key=openai_key,
                timeout=self.config.embedding_api_timeout,
            )
        return self._embedding_service_small

    @property
    def embedding_service_large(self):
        """임베딩 서비스 (3-large, Phase 2 중요 노드용)"""
        if self._embedding_service_large is None:
            import os

            from ..core.enums import EmbeddingModel
            from ..embedding.service import EmbeddingService

            # OpenAI 전용이므로 OPENAI_API_KEY 직접 사용
            openai_key = os.getenv("OPENAI_API_KEY") or self.config.embedding_api_key

            self._embedding_service_large = EmbeddingService(
                model=EmbeddingModel.OPENAI_3_LARGE,
                api_key=openai_key,
                timeout=self.config.embedding_api_timeout,
            )
        return self._embedding_service_large

    @property
    def embedding_store(self):
        """
        임베딩 스토어 (자동 선택)

        VECTOR_STORE_BACKEND에 따라 자동 선택:
        - pgvector: PgVectorStore
        - qdrant: QdrantStore
        """
        from ..core.enums import VectorStoreBackend

        backend = self.config.vector_store_backend

        if backend == VectorStoreBackend.QDRANT:
            return self.qdrant_store
        elif backend == VectorStoreBackend.PGVECTOR:
            # PgVector
            if self._embedding_store is None:
                from ..embedding.store_pgvector import PgVectorStore

                self._embedding_store = PgVectorStore(
                    connection_string=self._connection_string,
                    embedding_dimension=self.embedding_service.get_dimension(),
                    model_name=self.config.embedding_model.value,
                    pool_size=self.config.db_connection_pool_size,
                    pool_max=self.config.db_connection_pool_max,
                    skip_table_init=self.config.skip_table_init,
                )
            return self._embedding_store
        else:
            raise ValueError(f"Unknown vector store backend: {backend}")

    @property
    def qdrant_store(self):
        """임베딩 스토어 (Qdrant)"""
        if self._qdrant_store is None:
            from ..embedding.store_qdrant import QdrantStore

            self._qdrant_store = QdrantStore(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                grpc_port=self.config.qdrant_grpc_port,
                embedding_dimension=self.embedding_service.get_dimension(),
                model_name=self.config.embedding_model.value,
                use_grpc=self.config.qdrant_use_grpc,
                api_key=self.config.qdrant_api_key,
                timeout=self.config.qdrant_timeout,
                chunk_store=self.chunk_store,
            )
        return self._qdrant_store

    @property
    def lexical_search(self) -> LexicalSearchPort:
        """Lexical 검색 포트"""
        if self._lexical_search is None:
            backend = self.config.lexical_search_backend

            if backend == LexicalSearchBackend.MEILISEARCH:
                client = Client(
                    self.config.meilisearch_url,
                    api_key=self.config.meilisearch_master_key,
                )
                self._lexical_search = MeiliSearchAdapter(client)
            elif backend == LexicalSearchBackend.ZOEKT:
                # Zoekt는 chunk 매핑을 위해 ChunkStore 필요
                self._lexical_search = ZoektAdapter(
                    self.config.zoekt_url,
                    chunk_store=self.chunk_store,
                    timeout=self.config.zoekt_timeout,
                )
            else:
                raise ValueError(f"Unknown lexical search backend: {backend}")

        return self._lexical_search  # type: ignore[no-any-return]

    @property
    def ir_builder(self):
        """IR 빌더"""
        if self._ir_builder is None:
            from ..graph.ir_builder import IRBuilder

            self._ir_builder = IRBuilder()
        return self._ir_builder

    @property
    def chunker(self):
        """청커"""
        if self._chunker is None:
            from ..chunking.chunker import Chunker

            self._chunker = Chunker(
                max_lines=self.config.chunker_max_lines,
                overlap_lines=self.config.chunker_overlap_lines,
                max_tokens=self.config.chunker_max_tokens,
                enable_file_summary=self.config.chunker_enable_file_summary,
                min_symbols_for_summary=self.config.chunker_min_symbols_for_summary,
            )
        return self._chunker

    @property
    def scanner(self):
        """저장소 스캐너"""
        if self._scanner is None:
            from ..indexer.repo_scanner import RepoScanner

            self._scanner = RepoScanner()
        return self._scanner

    @property
    def pipeline(self):
        """인덱싱 파이프라인"""
        if self._pipeline is None:
            from ..indexer.pipeline import IndexingPipeline
            from ..parser.cache import ParseCache

            # ParseCache 초기화
            parse_cache = ParseCache()

            self._pipeline = IndexingPipeline(
                repo_store=self.repo_store,
                graph_store=self.graph_store,
                chunk_store=self.chunk_store,
                embedding_service=self.embedding_service,
                embedding_store=self.embedding_store,
                lexical_search=self.lexical_search,
                ir_builder=self.ir_builder,
                chunker=self.chunker,
                scanner=self.scanner,
                parse_cache=parse_cache,
                route_store=self.route_store,
                semantic_node_store=self.semantic_node_store,  # Phase 1
                embedding_service_small=self.embedding_service_small,  # Phase 1
                embedding_service_large=self.embedding_service_large,  # Phase 2용
            )
            # Config 전달 (병렬 처리 옵션용)
            self._pipeline.config = self.config  # type: ignore[attr-defined]
        return self._pipeline

    @property
    def semantic_search(self):
        """의미론적 검색"""
        if self._semantic_search is None:
            from ..search.adapters.semantic.pgvector_adapter import PgVectorSemanticSearch

            self._semantic_search = PgVectorSemanticSearch(
                embedding_service=self.embedding_service, embedding_store=self.embedding_store
            )
        return self._semantic_search

    @property
    def graph_search(self):
        """그래프 검색"""
        if self._graph_search is None:
            from ..search.adapters.graph.postgres_graph_adapter import PostgresGraphSearch

            self._graph_search = PostgresGraphSearch(graph_store=self.graph_store)
        return self._graph_search

    @property
    def fuzzy_search(self):
        """퍼지 검색"""
        if self._fuzzy_search is None:
            from ..search.adapters.fuzzy.symbol_fuzzy_matcher import SymbolFuzzyMatcher

            self._fuzzy_search = SymbolFuzzyMatcher(
                graph_store=self.graph_store, config=self.config
            )
        return self._fuzzy_search

    @property
    def symbol_search(self):
        """심볼 검색"""
        if self._symbol_search is None:
            from ..search.adapters.symbol.postgres_symbol_search import PostgresSymbolSearch

            self._symbol_search = PostgresSymbolSearch(graph_store=self.graph_store)
        return self._symbol_search

    @property
    def fusion_strategy(self):
        """Fusion 전략"""
        if self._fusion_strategy is None:
            strategy_name = self.config.fusion_strategy

            if strategy_name == "weighted_sum":
                from ..search.adapters.fusion import WeightedFusion

                self._fusion_strategy = WeightedFusion()
            elif strategy_name == "rrf":
                from ..search.adapters.fusion import ReciprocalRankFusion

                self._fusion_strategy = ReciprocalRankFusion(k=self.config.fusion_rrf_k)
            elif strategy_name == "combsum":
                from ..search.adapters.fusion import CombSumFusion

                self._fusion_strategy = CombSumFusion(
                    use_weights=self.config.fusion_combsum_use_weights
                )
            else:
                raise ValueError(
                    f"Unknown fusion strategy: {strategy_name}. "
                    f"Available: 'weighted_sum', 'rrf', 'combsum'"
                )

        return self._fusion_strategy

    @property
    def hybrid_retriever(self):
        """하이브리드 리트리버"""
        if self._hybrid_retriever is None:
            from ..search.adapters.retriever.hybrid_retriever import HybridRetriever

            self._hybrid_retriever = HybridRetriever(
                lexical_search=self.lexical_search,
                semantic_search=self.semantic_search,
                graph_search=self.graph_search,
                fuzzy_search=self.fuzzy_search,
                chunk_store=self.chunk_store,
                config=self.config,
                fusion_strategy=self.fusion_strategy,  # Fusion 전략 주입
                query_log_store=self.query_log_store,  # Phase 2: Query logging
            )
        return self._hybrid_retriever

    @property
    def reranker(self):
        """리랭커"""
        if self._ranker is None:
            reranker_type = self.config.reranker_type

            if reranker_type == "two-stage":
                # Two-Stage Reranker (Feature + LLM + Fusion)
                from ..search.adapters.ranking.llm_reranker import LLMReranker
                from ..search.adapters.ranking.llm_service import LLMScoringService
                from ..search.adapters.ranking.two_stage_reranker import TwoStageReranker

                # LLM API 키 확인
                if not self.config.llm_api_key:
                    raise ValueError(
                        "LLM_API_KEY or MISTRAL_API_KEY environment variable is required "
                        "for 'two-stage' reranker type"
                    )

                # 1단계: Feature-based reranker 선택
                from ..search.adapters.ranking.hybrid_reranker import HybridReranker
                from ..search.adapters.ranking.reranker import Reranker

                feature_reranker_type = self.config.two_stage_feature_reranker
                feature_reranker: HybridReranker | Reranker
                if feature_reranker_type == "hybrid":
                    feature_reranker = HybridReranker(debug_mode=self.config.reranker_debug_mode)
                else:  # "basic"
                    feature_reranker = Reranker()

                # LLM Scoring Service 생성
                llm_service = LLMScoringService(
                    api_key=self.config.llm_api_key,
                    model=self.config.llm_model,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                )

                # LLM Reranker 생성
                llm_reranker = LLMReranker(llm_service=llm_service)

                # Two-Stage Reranker 생성
                self._ranker = TwoStageReranker(
                    feature_reranker=feature_reranker,
                    llm_reranker=llm_reranker,
                    top_m=self.config.two_stage_top_m,
                    alpha=self.config.two_stage_alpha,
                    fallback_to_feature=self.config.two_stage_fallback,
                )
                logger.info(
                    f"Using TwoStageReranker "
                    f"(feature={feature_reranker_type}, top_m={self.config.two_stage_top_m}, "
                    f"alpha={self.config.two_stage_alpha}, llm_model={self.config.llm_model})"
                )

            elif reranker_type == "hybrid":
                from ..search.adapters.ranking.hybrid_reranker import HybridReranker

                self._ranker = HybridReranker(debug_mode=self.config.reranker_debug_mode)
                logger.info("Using HybridReranker")

            elif reranker_type == "morph":
                from ..search.adapters.ranking.morph_reranker import MorphReranker

                if not self.config.morph_api_key:
                    raise ValueError(
                        "MORPH_API_KEY environment variable is required for 'morph' reranker type"
                    )

                self._ranker = MorphReranker(
                    api_key=self.config.morph_api_key,
                    api_base=self.config.morph_api_base,
                    model=self.config.morph_model,
                    top_k=self.config.morph_top_k,
                )
                logger.info(f"Using MorphReranker (model={self.config.morph_model})")

            else:  # "basic" or unknown
                from ..search.adapters.ranking.reranker import Reranker

                self._ranker = Reranker()
                logger.info("Using basic Reranker")

        return self._ranker

    @property
    def context_packer(self):
        """컨텍스트 패커"""
        if self._context_packer is None:
            from ..context.packer import ContextPacker

            self._context_packer = ContextPacker(
                chunk_store=self.chunk_store, graph_store=self.graph_store
            )
        return self._context_packer

    @property
    def route_store(self):
        """Route 인덱스 스토어"""
        if self._route_store is None:
            from ..indexer.route_store import RouteStore

            self._route_store = RouteStore(
                connection_string=self._connection_string,
                pool_size=2,
                pool_max=5,
            )
        return self._route_store

    @property
    def semantic_node_store(self):
        """Semantic Node 스토어 (Phase 1)"""
        if self._semantic_node_store is None:
            from ..indexer.semantic_node_store import SemanticNodeStore

            self._semantic_node_store = SemanticNodeStore(
                connection_string=self._connection_string,
                pool_size=2,
                pool_max=10,
            )
        return self._semantic_node_store

    @property
    def query_log_store(self):
        """Query Log 스토어 (Phase 2)"""
        if self._query_log_store is None:
            from ..search.query_log_store import QueryLogStore

            self._query_log_store = QueryLogStore(
                connection_string=self._connection_string,
                pool_size=2,
                pool_max=5,
            )
        return self._query_log_store


def create_bootstrap(config: Config | None = None) -> Bootstrap:
    """
    Bootstrap 인스턴스 생성

    Args:
        config: 애플리케이션 설정 (None이면 환경변수에서 로드)

    Returns:
        Bootstrap 인스턴스

    Usage:
        >>> bootstrap = create_bootstrap()
        >>> result = bootstrap.pipeline.index_repository("/path/to/repo")
    """
    if config is None:
        config = Config.from_env()
    return Bootstrap(config)
