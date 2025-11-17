"""의존성 주입 및 포트 초기화"""

from typing import Optional

from meilisearch import Client

from .config import Config
from .enums import LexicalSearchBackend
from ..search.ports.lexical_search_port import LexicalSearchPort
from ..search.lexical.meili_adapter import MeiliSearchAdapter
from ..search.lexical.zoekt_adapter import ZoektAdapter


class Bootstrap:
    """포트 인스턴스 생성 및 의존성 주입"""

    def __init__(self, config: Config):
        self.config = config
        self._connection_string = self._build_connection_string()
        
        # 인스턴스 캐시 (lazy loading)
        self._repo_store = None
        self._graph_store = None
        self._chunk_store = None
        self._embedding_service = None
        self._embedding_store = None
        self._lexical_search = None
        self._ir_builder = None
        self._chunker = None
        self._scanner = None
        self._pipeline = None
        self._semantic_search = None
        self._graph_search = None
        self._hybrid_retriever = None
        self._ranker = None
        self._context_packer = None
    
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
            self._graph_store = PostgresGraphStore(self._connection_string)
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
        """임베딩 서비스"""
        if self._embedding_service is None:
            from ..embedding.service import EmbeddingService
            self._embedding_service = EmbeddingService(
                model=self.config.embedding_model,
                api_key=self.config.embedding_api_key,
                api_base=self.config.mistral_api_base,
                dimension=self.config.embedding_dimension
            )
        return self._embedding_service
    
    @property
    def embedding_store(self):
        """임베딩 스토어"""
        if self._embedding_store is None:
            from ..embedding.store_pgvector import PgVectorStore
            self._embedding_store = PgVectorStore(
                connection_string=self._connection_string,
                embedding_dimension=self.embedding_service.get_dimension(),
                model_name=self.config.embedding_model.value
            )
        return self._embedding_store
    
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
        
        return self._lexical_search
    
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
            self._chunker = Chunker()
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
            self._pipeline = IndexingPipeline(
                repo_store=self.repo_store,
                graph_store=self.graph_store,
                chunk_store=self.chunk_store,
                embedding_service=self.embedding_service,
                embedding_store=self.embedding_store,
                lexical_search=self.lexical_search,
                ir_builder=self.ir_builder,
                chunker=self.chunker,
                scanner=self.scanner
            )
        return self._pipeline
    
    @property
    def semantic_search(self):
        """의미론적 검색"""
        if self._semantic_search is None:
            from ..search.semantic.pgvector_adapter import PgVectorSemanticSearch
            self._semantic_search = PgVectorSemanticSearch(
                embedding_service=self.embedding_service,
                embedding_store=self.embedding_store
            )
        return self._semantic_search
    
    @property
    def graph_search(self):
        """그래프 검색"""
        if self._graph_search is None:
            from ..search.graph.postgres_graph_adapter import PostgresGraphSearch
            self._graph_search = PostgresGraphSearch(
                graph_store=self.graph_store
            )
        return self._graph_search
    
    @property
    def hybrid_retriever(self):
        """하이브리드 리트리버"""
        if self._hybrid_retriever is None:
            from ..search.retriever.hybrid_retriever import HybridRetriever
            self._hybrid_retriever = HybridRetriever(
                lexical_search=self.lexical_search,
                semantic_search=self.semantic_search,
                graph_search=self.graph_search
            )
        return self._hybrid_retriever
    
    @property
    def ranker(self):
        """랭커"""
        if self._ranker is None:
            from ..search.ranking.ranker import Ranker
            self._ranker = Ranker()
        return self._ranker
    
    @property
    def context_packer(self):
        """컨텍스트 패커"""
        if self._context_packer is None:
            from ..context.packer import ContextPacker
            self._context_packer = ContextPacker(
                chunk_store=self.chunk_store,
                graph_store=self.graph_store
            )
        return self._context_packer


def create_bootstrap(config: Optional[Config] = None) -> Bootstrap:
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
