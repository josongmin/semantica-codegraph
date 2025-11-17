"""임베딩 통합 테스트: 생성 → 저장 → 검색"""

import os
import pytest

from src.core.config import Config
from src.core.enums import EmbeddingModel
from src.core.models import CodeChunk
from src.embedding.service import EmbeddingService
from src.embedding.store_pgvector import PgVectorStore
from src.chunking.store import PostgresChunkStore


def _get_test_config():
    """테스트용 설정"""
    config = Config.from_env()
    
    # 필수 환경 확인
    if not config.embedding_api_key or config.embedding_api_key == "your_mistral_api_key_here":
        pytest.skip("EMBEDDING_API_KEY not set or invalid")
    
    # PostgreSQL 연결 확인
    try:
        import psycopg2
        conn_str = (
            f"host={config.postgres_host} "
            f"port={config.postgres_port} "
            f"dbname={config.postgres_db} "
            f"user={config.postgres_user} "
            f"password={config.postgres_password}"
        )
        psycopg2.connect(conn_str).close()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")
    
    return config, conn_str


@pytest.fixture
def test_chunks():
    """테스트용 CodeChunk 리스트"""
    return [
        CodeChunk(
            repo_id="test-repo",
            id="chunk-1",
            node_id="node-1",
            file_path="calculator.py",
            span=(0, 0, 5, 0),
            language="python",
            text="def add(a, b):\n    return a + b",
            attrs={"docstring": "두 수를 더합니다"}
        ),
        CodeChunk(
            repo_id="test-repo",
            id="chunk-2",
            node_id="node-2",
            file_path="calculator.py",
            span=(6, 0, 10, 0),
            language="python",
            text="def multiply(a, b):\n    return a * b",
            attrs={"docstring": "두 수를 곱합니다"}
        ),
        CodeChunk(
            repo_id="test-repo",
            id="chunk-3",
            node_id="node-3",
            file_path="utils.py",
            span=(0, 0, 3, 0),
            language="python",
            text="def format_number(n):\n    return str(n)",
            attrs={"docstring": "숫자를 문자열로 변환"}
        ),
    ]


@pytest.fixture
def embedding_service():
    """EmbeddingService 인스턴스"""
    config, _ = _get_test_config()
    
    service = EmbeddingService(
        model=config.embedding_model,
        api_key=config.embedding_api_key,
        api_base=config.mistral_api_base,
        dimension=config.embedding_dimension
    )
    return service


@pytest.fixture
def chunk_store():
    """PostgresChunkStore 인스턴스"""
    config, conn_str = _get_test_config()
    return PostgresChunkStore(connection_string=conn_str)


@pytest.fixture
def vector_store():
    """PgVectorStore 인스턴스"""
    config, conn_str = _get_test_config()
    
    service = EmbeddingService(
        model=config.embedding_model,
        api_key=config.embedding_api_key,
        api_base=config.mistral_api_base,
        dimension=config.embedding_dimension
    )
    
    store = PgVectorStore(
        connection_string=conn_str,
        embedding_dimension=service.get_dimension(),
        model_name=config.embedding_model.value
    )
    return store


def test_embed_and_search_integration(test_chunks, embedding_service, chunk_store, vector_store):
    """
    전체 플로우 테스트:
    1. 청크 저장
    2. 청크 임베딩 생성
    3. 벡터 저장
    4. 검색 쿼리로 유사 청크 찾기
    """
    # 1. 청크 먼저 저장 (검색 시 JOIN에 필요)
    chunk_store.save_chunks(test_chunks)
    
    # 2. 임베딩 생성
    vectors = embedding_service.embed_chunks(test_chunks)
    
    assert len(vectors) == len(test_chunks)
    assert all(len(v) == embedding_service.get_dimension() for v in vectors)
    
    # 3. 벡터 저장
    chunk_ids = [chunk.id for chunk in test_chunks]
    vector_store.save_embeddings("test-repo", chunk_ids, vectors)
    
    # 3. 검색: "더하기" 관련 쿼리
    query_text = "두 숫자를 더하는 함수"
    query_vector = embedding_service.embed_text(query_text)
    
    results = vector_store.search_by_vector(
        repo_id="test-repo",
        vector=query_vector,
        k=3
    )
    
    assert len(results) > 0, "검색 결과가 있어야 함"
    
    # 첫 번째 결과는 "add" 함수여야 함 (가장 유사)
    top_result = results[0]
    assert top_result.chunk_id in chunk_ids
    assert top_result.score > 0.0, "유사도 점수는 0보다 커야 함"
    assert top_result.score <= 1.0, "코사인 유사도는 1 이하여야 함"
    
    # 4. 필터 검색: 특정 파일만
    filtered_results = vector_store.search_by_vector(
        repo_id="test-repo",
        vector=query_vector,
        k=3,
        filters={"file_path": "calculator.py"}
    )
    
    assert all(
        r.file_path == "calculator.py" for r in filtered_results
    ), "필터링된 결과는 calculator.py만 포함해야 함"


def test_semantic_search_accuracy(test_chunks, embedding_service, chunk_store, vector_store):
    """
    의미론적 검색 정확도 테스트:
    - "곱하기" 쿼리 → multiply 함수가 상위에
    - "문자열 변환" 쿼리 → format_number 함수가 상위에
    """
    # 청크 저장
    chunk_store.save_chunks(test_chunks)
    
    # 임베딩 저장
    chunk_ids = [chunk.id for chunk in test_chunks]
    vectors = embedding_service.embed_chunks(test_chunks)
    vector_store.save_embeddings("test-repo", chunk_ids, vectors)
    
    # 테스트 1: "곱하기" 쿼리
    query1 = "두 수를 곱하는 함수"
    vector1 = embedding_service.embed_text(query1)
    results1 = vector_store.search_by_vector("test-repo", vector1, k=1)
    
    assert len(results1) > 0
    # multiply 함수가 가장 유사해야 함
    top1 = results1[0]
    assert top1.chunk_id == "chunk-2", "multiply 함수가 상위에 있어야 함"
    
    # 테스트 2: "문자열 변환" 쿼리
    query2 = "숫자를 문자열로 바꾸는 함수"
    vector2 = embedding_service.embed_text(query2)
    results2 = vector_store.search_by_vector("test-repo", vector2, k=1)
    
    assert len(results2) > 0
    # format_number 함수가 가장 유사해야 함
    top2 = results2[0]
    assert top2.chunk_id == "chunk-3", "format_number 함수가 상위에 있어야 함"


def test_batch_embedding_performance(test_chunks, embedding_service):
    """배치 임베딩 성능 테스트"""
    # 단일 vs 배치 비교
    single_vectors = [embedding_service.embed_chunk(chunk) for chunk in test_chunks]
    batch_vectors = embedding_service.embed_chunks(test_chunks)
    
    assert len(single_vectors) == len(batch_vectors)
    assert all(
        len(sv) == len(bv) == embedding_service.get_dimension()
        for sv, bv in zip(single_vectors, batch_vectors)
    )


def test_vector_dimension_consistency(embedding_service, vector_store):
    """벡터 차원 일관성 테스트"""
    test_text = "def test(): pass"
    vector = embedding_service.embed_text(test_text)
    
    expected_dim = embedding_service.get_dimension()
    assert len(vector) == expected_dim
    
    # 저장소 차원과 일치해야 함
    assert vector_store.embedding_dimension == expected_dim

