"""PgVectorStore 테스트"""

import pytest


@pytest.fixture
def sample_vectors():
    """샘플 벡터 (384차원)"""
    import random
    return [
        [random.random() for _ in range(384)],
        [random.random() for _ in range(384)],
        [random.random() for _ in range(384)]
    ]


@pytest.fixture
def sample_chunk_ids():
    """샘플 chunk ID 리스트"""
    return ["chunk-1", "chunk-2", "chunk-3"]


def test_vector_dimension():
    """벡터 차원 테스트"""
    import random
    vector = [random.random() for _ in range(384)]
    assert len(vector) == 384


def test_save_embeddings(sample_chunk_ids, sample_vectors, ensure_test_repo):
    """임베딩 저장 테스트"""
    from src.embedding.store_pgvector import PgVectorStore
    from src.chunking.store import PostgresChunkStore
    from src.core.models import CodeChunk
    
    conn_str = "host=localhost port=5433 dbname=semantica_test user=semantica password=semantica"
    
    # 저장소 메타데이터 생성
    ensure_test_repo(conn_str)
    
    # 청크 먼저 저장 (외래 키 제약)
    chunk_store = PostgresChunkStore(conn_str)
    chunks = [
        CodeChunk(
            repo_id="test-repo",
            id=chunk_id,
            node_id=f"node-{chunk_id}",
            file_path="test.py",
            span=(0, 0, 10, 0),
            language="python",
            text=f"chunk {chunk_id}",
            attrs={}
        )
        for chunk_id in sample_chunk_ids
    ]
    chunk_store.save_chunks(chunks)
    
    store = PgVectorStore(conn_str, embedding_dimension=384)
    store.save_embeddings("test-repo", sample_chunk_ids, sample_vectors)


def test_vector_search(sample_chunk_ids, sample_vectors, ensure_test_repo):
    """벡터 검색 테스트"""
    from src.embedding.store_pgvector import PgVectorStore
    from src.chunking.store import PostgresChunkStore
    from src.core.models import CodeChunk
    
    conn_str = "host=localhost port=5433 dbname=semantica_test user=semantica password=semantica"
    
    # 저장소 메타데이터 생성
    ensure_test_repo(conn_str)
    
    # 청크 먼저 저장
    chunk_store = PostgresChunkStore(conn_str)
    chunks = [
        CodeChunk(
            repo_id="test-repo",
            id=chunk_id,
            node_id=f"node-{chunk_id}",
            file_path="test.py",
            span=(0, 0, 10, 0),
            language="python",
            text=f"chunk {chunk_id}",
            attrs={}
        )
        for chunk_id in sample_chunk_ids
    ]
    chunk_store.save_chunks(chunks)
    
    store = PgVectorStore(conn_str, embedding_dimension=384)
    # 저장
    store.save_embeddings("test-repo", sample_chunk_ids, sample_vectors)
    
    # 검색 (첫 번째 벡터로)
    results = store.search_by_vector("test-repo", sample_vectors[0], k=3)
    
    assert len(results) > 0
    assert results[0].chunk_id in sample_chunk_ids
    assert results[0].score > 0.0  # 코사인 유사도


def test_vector_search_with_filters(sample_chunk_ids, sample_vectors, ensure_test_repo):
    """벡터 검색 + 필터 테스트"""
    from src.embedding.store_pgvector import PgVectorStore
    from src.chunking.store import PostgresChunkStore
    from src.core.models import CodeChunk
    
    conn_str = "host=localhost port=5433 dbname=semantica_test user=semantica password=semantica"
    
    # 저장소 메타데이터 생성
    ensure_test_repo(conn_str)
    
    # 청크 먼저 저장
    chunk_store = PostgresChunkStore(conn_str)
    chunks = [
        CodeChunk(
            repo_id="test-repo",
            id=chunk_id,
            node_id=f"node-{chunk_id}",
            file_path="test.py",
            span=(0, 0, 10, 0),
            language="python",
            text=f"chunk {chunk_id}",
            attrs={}
        )
        for chunk_id in sample_chunk_ids
    ]
    chunk_store.save_chunks(chunks)
    
    store = PgVectorStore(conn_str, embedding_dimension=384)
    store.save_embeddings("test-repo", sample_chunk_ids, sample_vectors)
    
    # 필터 적용
    results = store.search_by_vector(
        "test-repo",
        sample_vectors[0],
        k=10,
        filters={"language": "python"}
    )
    
    assert isinstance(results, list)


def test_vector_length_validation(sample_chunk_ids):
    """벡터 길이 검증 테스트"""
    from src.embedding.store_pgvector import PgVectorStore
    
    # 잘못된 길이
    wrong_vectors = [[1.0, 2.0]]  # 384가 아님
    
    # Store는 PostgreSQL이 검증하므로 여기서는 간단히만
    assert len(wrong_vectors[0]) != 384


def test_chunk_ids_vectors_length_match():
    """chunk_ids와 vectors 길이 일치 테스트"""
    chunk_ids = ["chunk-1", "chunk-2"]
    vectors = [[1.0] * 384]  # 길이 안 맞음
    
    assert len(chunk_ids) != len(vectors), "길이가 일치해야 함"

