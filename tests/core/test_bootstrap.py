"""Bootstrap 테스트"""


import pytest

from src.core.bootstrap import Bootstrap
from src.core.config import Config
from src.core.enums import EmbeddingModel, LexicalSearchBackend


@pytest.fixture
def test_config():
    """테스트용 설정"""
    return Config(
        postgres_host="localhost",
        postgres_port=7711,
        postgres_user="semantica",
        postgres_password="semantica",
        postgres_db="semantica_codegraph",
        embedding_api_key="test_key",
        embedding_model=EmbeddingModel.CODESTRAL_EMBED,
        lexical_search_backend=LexicalSearchBackend.MEILISEARCH,
        meilisearch_url="http://localhost:7700",
        meilisearch_master_key="test_key"
    )


@pytest.fixture
def bootstrap(test_config):
    """Bootstrap 인스턴스"""
    return Bootstrap(test_config)


def test_bootstrap_initialization(bootstrap):
    """Bootstrap 초기화 테스트"""
    assert bootstrap is not None
    assert bootstrap.config is not None
    assert bootstrap._connection_string is not None


def test_build_connection_string(bootstrap):
    """연결 문자열 생성 테스트"""
    conn_str = bootstrap._build_connection_string()

    assert "host=" in conn_str
    assert "port=" in conn_str
    assert "dbname=" in conn_str
    assert "user=" in conn_str
    assert "password=" in conn_str


def test_repo_store_lazy_loading(bootstrap):
    """RepoStore lazy loading 테스트"""
    # 처음 접근 시 생성
    store1 = bootstrap.repo_store
    assert store1 is not None

    # 두 번째 접근 시 같은 인스턴스 반환
    store2 = bootstrap.repo_store
    assert store1 is store2


def test_graph_store_lazy_loading(bootstrap):
    """GraphStore lazy loading 테스트"""
    store1 = bootstrap.graph_store
    assert store1 is not None

    store2 = bootstrap.graph_store
    assert store1 is store2


def test_chunk_store_lazy_loading(bootstrap):
    """ChunkStore lazy loading 테스트"""
    store1 = bootstrap.chunk_store
    assert store1 is not None

    store2 = bootstrap.chunk_store
    assert store1 is store2


def test_embedding_service_lazy_loading(bootstrap):
    """EmbeddingService lazy loading 테스트"""
    service1 = bootstrap.embedding_service
    assert service1 is not None

    service2 = bootstrap.embedding_service
    assert service1 is service2


def test_embedding_store_lazy_loading(bootstrap):
    """EmbeddingStore lazy loading 테스트"""
    # embedding_service가 먼저 필요함
    _ = bootstrap.embedding_service

    store1 = bootstrap.embedding_store
    assert store1 is not None

    store2 = bootstrap.embedding_store
    assert store1 is store2


def test_lexical_search_lazy_loading(bootstrap):
    """LexicalSearch lazy loading 테스트"""
    search1 = bootstrap.lexical_search
    assert search1 is not None

    search2 = bootstrap.lexical_search
    assert search1 is search2


def test_ir_builder_lazy_loading(bootstrap):
    """IRBuilder lazy loading 테스트"""
    builder1 = bootstrap.ir_builder
    assert builder1 is not None

    builder2 = bootstrap.ir_builder
    assert builder1 is builder2


def test_chunker_lazy_loading(bootstrap):
    """Chunker lazy loading 테스트"""
    chunker1 = bootstrap.chunker
    assert chunker1 is not None

    chunker2 = bootstrap.chunker
    assert chunker1 is chunker2


def test_scanner_lazy_loading(bootstrap):
    """RepoScanner lazy loading 테스트"""
    scanner1 = bootstrap.scanner
    assert scanner1 is not None

    scanner2 = bootstrap.scanner
    assert scanner1 is scanner2


def test_pipeline_lazy_loading(bootstrap):
    """IndexingPipeline lazy loading 테스트"""
    pipeline1 = bootstrap.pipeline
    assert pipeline1 is not None

    pipeline2 = bootstrap.pipeline
    assert pipeline1 is pipeline2


def test_semantic_search_lazy_loading(bootstrap):
    """SemanticSearch lazy loading 테스트"""
    # embedding_service와 embedding_store가 필요
    _ = bootstrap.embedding_service
    _ = bootstrap.embedding_store

    search1 = bootstrap.semantic_search
    assert search1 is not None

    search2 = bootstrap.semantic_search
    assert search1 is search2


def test_graph_search_lazy_loading(bootstrap):
    """GraphSearch lazy loading 테스트"""
    # graph_store가 필요
    _ = bootstrap.graph_store

    search1 = bootstrap.graph_search
    assert search1 is not None

    search2 = bootstrap.graph_search
    assert search1 is search2


def test_fuzzy_search_lazy_loading(bootstrap):
    """FuzzySearch lazy loading 테스트"""
    # graph_store가 필요
    _ = bootstrap.graph_store

    search1 = bootstrap.fuzzy_search
    assert search1 is not None

    search2 = bootstrap.fuzzy_search
    assert search1 is search2


def test_ranker_lazy_loading(bootstrap):
    """Ranker lazy loading 테스트"""
    ranker1 = bootstrap.ranker
    assert ranker1 is not None

    ranker2 = bootstrap.ranker
    assert ranker1 is ranker2


def test_context_packer_lazy_loading(bootstrap):
    """ContextPacker lazy loading 테스트"""
    # chunk_store와 graph_store가 필요
    _ = bootstrap.chunk_store
    _ = bootstrap.graph_store

    packer1 = bootstrap.context_packer
    assert packer1 is not None

    packer2 = bootstrap.context_packer
    assert packer1 is packer2


def test_hybrid_retriever_lazy_loading(bootstrap):
    """HybridRetriever lazy loading 테스트"""
    # 모든 검색 서비스가 필요
    _ = bootstrap.lexical_search
    _ = bootstrap.semantic_search
    _ = bootstrap.graph_search
    _ = bootstrap.fuzzy_search
    _ = bootstrap.chunk_store

    retriever1 = bootstrap.hybrid_retriever
    assert retriever1 is not None

    retriever2 = bootstrap.hybrid_retriever
    assert retriever1 is retriever2


def test_meilisearch_adapter_creation(bootstrap):
    """MeiliSearch 어댑터 생성 테스트"""
    # MeiliSearch가 백엔드로 설정된 경우
    if bootstrap.config.lexical_search_backend == "meilisearch":
        search = bootstrap.lexical_search
        assert search is not None
        # MeiliSearchAdapter 인스턴스인지 확인
        assert hasattr(search, 'client') or hasattr(search, '_get_index_name')


def test_zoekt_adapter_creation():
    """Zoekt 어댑터 생성 테스트"""
    config = Config(
        postgres_host="localhost",
        postgres_port=7711,
        postgres_user="semantica",
        postgres_password="semantica",
        postgres_db="semantica_codegraph",
        lexical_search_backend=LexicalSearchBackend.ZOEKT,
        zoekt_url="http://localhost:7713"
    )

    bootstrap = Bootstrap(config)

    if config.lexical_search_backend == LexicalSearchBackend.ZOEKT:
        search = bootstrap.lexical_search
        assert search is not None


def test_bootstrap_with_minimal_config():
    """최소 설정으로 Bootstrap 생성 테스트"""
    config = Config(
        postgres_host="localhost",
        postgres_port=7711,
        postgres_user="semantica",
        postgres_password="semantica",
        postgres_db="semantica_codegraph"
    )

    bootstrap = Bootstrap(config)
    assert bootstrap is not None

    # 필수 서비스는 생성 가능해야 함
    assert bootstrap.repo_store is not None
    assert bootstrap.graph_store is not None
    assert bootstrap.chunk_store is not None


def test_bootstrap_dependency_order(bootstrap):
    """의존성 순서 테스트"""
    # 의존성이 있는 순서대로 접근
    _ = bootstrap.repo_store
    _ = bootstrap.graph_store
    _ = bootstrap.chunk_store
    _ = bootstrap.embedding_service
    _ = bootstrap.embedding_store
    _ = bootstrap.lexical_search
    _ = bootstrap.semantic_search
    _ = bootstrap.graph_search
    _ = bootstrap.fuzzy_search
    _ = bootstrap.hybrid_retriever
    _ = bootstrap.context_packer

    # 모든 서비스가 정상적으로 생성되었는지 확인
    assert bootstrap.pipeline is not None

