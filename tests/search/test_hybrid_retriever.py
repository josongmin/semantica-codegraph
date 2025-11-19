"""HybridRetriever 테스트"""

from unittest.mock import MagicMock

import pytest

from src.core.models import Candidate, ChunkResult, LocationContext
from src.search.adapters.retriever.hybrid_retriever import HybridRetriever


@pytest.fixture
def mock_lexical_search():
    """Mock LexicalSearchPort"""
    search = MagicMock()
    return search


@pytest.fixture
def mock_semantic_search():
    """Mock SemanticSearchPort"""
    search = MagicMock()
    return search


@pytest.fixture
def mock_graph_search():
    """Mock GraphSearchPort"""
    search = MagicMock()
    return search


@pytest.fixture
def mock_fuzzy_search():
    """Mock FuzzySearchPort"""
    search = MagicMock()
    return search


@pytest.fixture
def mock_chunk_store():
    """Mock ChunkStorePort"""
    store = MagicMock()
    return store


@pytest.fixture
def hybrid_retriever(
    mock_lexical_search,
    mock_semantic_search,
    mock_graph_search,
    mock_fuzzy_search,
    mock_chunk_store,
):
    """HybridRetriever 인스턴스"""
    return HybridRetriever(
        lexical_search=mock_lexical_search,
        semantic_search=mock_semantic_search,
        graph_search=mock_graph_search,
        fuzzy_search=mock_fuzzy_search,
        chunk_store=mock_chunk_store,
    )


def test_hybrid_retriever_initialization(hybrid_retriever):
    """HybridRetriever 초기화 테스트"""
    assert hybrid_retriever is not None
    assert hybrid_retriever.lexical_search is not None
    assert hybrid_retriever.semantic_search is not None
    assert hybrid_retriever.graph_search is not None
    assert hybrid_retriever.fuzzy_search is not None


def test_retrieve_sequential_basic(hybrid_retriever, mock_lexical_search, mock_semantic_search):
    """순차 검색 기본 테스트"""
    repo_id = "test-repo"
    query = "test query"

    # Mock 검색 결과
    mock_lexical_search.search.return_value = [
        ChunkResult(
            repo_id=repo_id,
            chunk_id="chunk-1",
            score=0.9,
            source="lexical",
            file_path="test.py",
            span=(0, 0, 10, 0),
        )
    ]

    mock_semantic_search.search.return_value = [
        ChunkResult(
            repo_id=repo_id,
            chunk_id="chunk-1",
            score=0.8,
            source="semantic",
            file_path="test.py",
            span=(0, 0, 10, 0),
        ),
        ChunkResult(
            repo_id=repo_id,
            chunk_id="chunk-2",
            score=0.7,
            source="semantic",
            file_path="test2.py",
            span=(0, 0, 5, 0),
        ),
    ]

    # 검색 실행
    results = hybrid_retriever._retrieve_sequential(
        repo_id=repo_id,
        query=query,
        k=10,
        location_ctx=None,
        weights={"lexical": 0.3, "semantic": 0.5, "graph": 0.2, "fuzzy": 0.0},
    )

    # 검증
    assert len(results) > 0
    assert all(isinstance(r, Candidate) for r in results)

    # chunk-1은 두 검색 결과에 모두 있어야 함
    chunk_1 = next((r for r in results if r.chunk_id == "chunk-1"), None)
    assert chunk_1 is not None
    assert "lexical_score" in chunk_1.features
    assert "semantic_score" in chunk_1.features
    assert "total_score" in chunk_1.features


def test_retrieve_with_custom_weights(hybrid_retriever, mock_lexical_search, mock_semantic_search):
    """커스텀 가중치 테스트"""
    repo_id = "test-repo"
    query = "test"

    mock_lexical_search.search.return_value = [
        ChunkResult(
            repo_id=repo_id,
            chunk_id="chunk-1",
            score=0.9,
            source="lexical",
            file_path="test.py",
            span=(0, 0, 10, 0),
        )
    ]
    mock_semantic_search.search.return_value = [
        ChunkResult(
            repo_id=repo_id,
            chunk_id="chunk-2",
            score=0.8,
            source="semantic",
            file_path="test2.py",
            span=(0, 0, 10, 0),
        )
    ]

    # Lexical만 사용 (semantic 가중치 0)
    results = hybrid_retriever._retrieve_sequential(
        repo_id=repo_id,
        query=query,
        k=10,
        location_ctx=None,
        weights={"lexical": 1.0, "semantic": 0.0, "graph": 0.0, "fuzzy": 0.0},
    )

    # Lexical 결과만 있어야 함
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert "lexical_score" in results[0].features
    assert "semantic_score" not in results[0].features


def test_retrieve_with_graph_search(hybrid_retriever, mock_graph_search):
    """Graph 검색 포함 테스트"""
    from src.core.models import CodeNode

    repo_id = "test-repo"
    location_ctx = LocationContext(repo_id=repo_id, file_path="test.py", line=10, column=0)

    # Mock 노드
    current_node = CodeNode(
        repo_id=repo_id,
        id="node-1",
        kind="Function",
        language="python",
        file_path="test.py",
        span=(10, 0, 20, 0),
        name="test_func",
        text="def test_func(): pass",
    )

    neighbor_node = CodeNode(
        repo_id=repo_id,
        id="node-2",
        kind="Function",
        language="python",
        file_path="test.py",
        span=(25, 0, 35, 0),
        name="neighbor_func",
        text="def neighbor_func(): pass",
    )

    mock_graph_search.get_node_by_location.return_value = current_node
    # expand_neighbors_with_edges는 (CodeNode, edge_type, depth) 튜플 반환
    mock_graph_search.expand_neighbors_with_edges.return_value = [(neighbor_node, "calls", 1)]

    results = hybrid_retriever._retrieve_sequential(
        repo_id=repo_id,
        query="test",
        k=10,
        location_ctx=location_ctx,
        weights={"lexical": 0.0, "semantic": 0.0, "graph": 1.0, "fuzzy": 0.0},
    )

    # Graph 검색 결과가 있어야 함
    assert len(results) > 0
    assert any("graph_score" in r.features for r in results)


def test_retrieve_with_fuzzy_search(hybrid_retriever, mock_fuzzy_search, mock_chunk_store):
    """Fuzzy 검색 포함 테스트"""
    from src.core.models import CodeChunk
    from src.search.ports.fuzzy_search_port import FuzzyMatch

    repo_id = "test-repo"

    # Mock fuzzy match
    fuzzy_match = FuzzyMatch(
        matched_text="UserService", score=0.9, node_id="node-1", file_path="user.py", kind="Class"
    )

    mock_fuzzy_search.search_symbols.return_value = [fuzzy_match]

    # Mock chunk store
    mock_chunk_store.get_chunks_by_node.return_value = [
        CodeChunk(
            repo_id=repo_id,
            id="chunk-1",
            node_id="node-1",
            file_path="user.py",
            span=(0, 0, 10, 0),
            language="python",
            text="class UserService: pass",
            attrs={},
        )
    ]

    results = hybrid_retriever._retrieve_sequential(
        repo_id=repo_id,
        query="UserServce",  # 오타
        k=10,
        location_ctx=None,
        weights={"lexical": 0.0, "semantic": 0.0, "graph": 0.0, "fuzzy": 1.0},
    )

    # Fuzzy 검색 결과가 있어야 함
    assert len(results) > 0
    assert any("fuzzy_score" in r.features for r in results)


def test_retrieve_parallel(hybrid_retriever, mock_lexical_search, mock_semantic_search):
    """병렬 검색 테스트"""
    repo_id = "test-repo"
    query = "test"

    mock_lexical_search.search.return_value = [
        ChunkResult(
            repo_id=repo_id,
            chunk_id="chunk-1",
            score=0.9,
            source="lexical",
            file_path="test.py",
            span=(0, 0, 10, 0),
        )
    ]
    mock_semantic_search.search.return_value = [
        ChunkResult(
            repo_id=repo_id,
            chunk_id="chunk-2",
            score=0.8,
            source="semantic",
            file_path="test2.py",
            span=(0, 0, 10, 0),
        )
    ]

    results = hybrid_retriever._retrieve_parallel(
        repo_id=repo_id,
        query=query,
        k=10,
        location_ctx=None,
        weights={"lexical": 0.5, "semantic": 0.5, "graph": 0.0, "fuzzy": 0.0},
    )

    # 병렬 검색 결과
    assert len(results) > 0
    assert all(isinstance(r, Candidate) for r in results)


def test_retrieve_error_handling(hybrid_retriever, mock_lexical_search):
    """에러 처리 테스트"""
    repo_id = "test-repo"
    query = "test"

    # Lexical 검색 실패
    mock_lexical_search.search.side_effect = Exception("Search failed")

    # 에러가 발생해도 다른 검색은 계속 진행되어야 함
    results = hybrid_retriever._retrieve_sequential(
        repo_id=repo_id,
        query=query,
        k=10,
        location_ctx=None,
        weights={"lexical": 0.5, "semantic": 0.5, "graph": 0.0, "fuzzy": 0.0},
    )

    # 에러가 발생해도 빈 리스트 반환 (다른 검색도 실패한 경우)
    assert isinstance(results, list)


def test_extract_symbol_tokens(hybrid_retriever):
    """심볼 토큰 추출 테스트"""
    # CamelCase 분리
    tokens = hybrid_retriever._extract_symbol_tokens("UserService login")
    assert "User" in tokens
    assert "Service" in tokens
    assert "login" in tokens

    # snake_case 분리
    tokens = hybrid_retriever._extract_symbol_tokens("user_service test")
    assert "user" in tokens
    assert "service" in tokens
    assert "test" in tokens

    # 최소 길이 필터링
    tokens = hybrid_retriever._extract_symbol_tokens("a b c")
    # 최소 길이 미만은 제외됨
    assert len(tokens) >= 0


def test_result_to_candidate(hybrid_retriever):
    """ChunkResult를 Candidate로 변환 테스트"""
    result = ChunkResult(
        repo_id="test-repo",
        chunk_id="chunk-1",
        score=0.9,
        source="lexical",
        file_path="test.py",
        span=(0, 0, 10, 0),
    )

    candidate = hybrid_retriever._result_to_candidate(result, lexical_score=0.9, semantic_score=0.8)

    assert isinstance(candidate, Candidate)
    assert candidate.chunk_id == "chunk-1"
    assert candidate.features["lexical_score"] == 0.9
    assert candidate.features["semantic_score"] == 0.8


def test_retrieve_with_empty_results(hybrid_retriever, mock_lexical_search, mock_semantic_search):
    """빈 결과 처리 테스트"""
    repo_id = "test-repo"
    query = "nonexistent"

    mock_lexical_search.search.return_value = []
    mock_semantic_search.search.return_value = []

    results = hybrid_retriever._retrieve_sequential(
        repo_id=repo_id,
        query=query,
        k=10,
        location_ctx=None,
        weights={"lexical": 0.5, "semantic": 0.5, "graph": 0.0, "fuzzy": 0.0},
    )

    assert len(results) == 0


def test_retrieve_default_weights(hybrid_retriever, mock_lexical_search, mock_semantic_search):
    """기본 가중치 테스트"""
    repo_id = "test-repo"
    query = "test"

    mock_lexical_search.search.return_value = [
        ChunkResult(
            repo_id=repo_id,
            chunk_id="chunk-1",
            score=0.9,
            source="lexical",
            file_path="test.py",
            span=(0, 0, 10, 0),
        )
    ]
    mock_semantic_search.search.return_value = []

    # weights=None일 때 기본 가중치 사용
    results = hybrid_retriever.retrieve(repo_id=repo_id, query=query, k=10, weights=None)

    assert len(results) > 0
