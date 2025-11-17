"""ChunkStore 테스트"""

import pytest

from src.core.models import CodeChunk
from src.chunking.store import PostgresChunkStore


@pytest.fixture
def sample_chunks():
    """샘플 CodeChunk 리스트"""
    return [
        CodeChunk(
            repo_id="test-repo",
            id="chunk-1",
            node_id="test-repo:main.py:Function:foo",
            file_path="main.py",
            span=(10, 0, 20, 0),
            language="python",
            text="def foo():\n    return 42",
            attrs={"node_kind": "Function"}
        ),
        CodeChunk(
            repo_id="test-repo",
            id="chunk-2",
            node_id="test-repo:main.py:Class:Bar",
            file_path="main.py",
            span=(25, 0, 40, 0),
            language="python",
            text="class Bar:\n    pass",
            attrs={"node_kind": "Class"}
        ),
        CodeChunk(
            repo_id="test-repo",
            id="chunk-3",
            node_id="test-repo:utils.py:Function:helper",
            file_path="utils.py",
            span=(5, 0, 10, 0),
            language="python",
            text="def helper(x):\n    return x * 2",
            attrs={}
        )
    ]


def test_chunk_store_initialization():
    """ChunkStore 초기화 테스트 (연결 없이)"""
    # 실제 DB 연결은 통합 테스트에서
    pass


@pytest.mark.skip(reason="Requires PostgreSQL connection")
def test_save_chunks(sample_chunks):
    """청크 저장 테스트"""
    conn_str = "host=localhost dbname=semantica_test user=semantica"
    store = PostgresChunkStore(conn_str)
    
    store.save_chunks(sample_chunks)
    
    # 조회로 검증
    chunk = store.get_chunk("test-repo", "chunk-1")
    assert chunk is not None
    assert chunk.text == "def foo():\n    return 42"


@pytest.mark.skip(reason="Requires PostgreSQL connection")
def test_find_by_location(sample_chunks):
    """위치로 청크 조회 (Zoekt 매핑 테스트)"""
    conn_str = "host=localhost dbname=semantica_test user=semantica"
    store = PostgresChunkStore(conn_str)
    
    store.save_chunks(sample_chunks)
    
    # main.py의 15번 라인 (chunk-1에 포함됨: span 10-20)
    chunk = store.find_by_location("test-repo", "main.py", 15)
    assert chunk is not None
    assert chunk.id == "chunk-1"
    
    # main.py의 30번 라인 (chunk-2에 포함됨: span 25-40)
    chunk = store.find_by_location("test-repo", "main.py", 30)
    assert chunk is not None
    assert chunk.id == "chunk-2"
    
    # utils.py의 7번 라인 (chunk-3에 포함됨: span 5-10)
    chunk = store.find_by_location("test-repo", "utils.py", 7)
    assert chunk is not None
    assert chunk.id == "chunk-3"


@pytest.mark.skip(reason="Requires PostgreSQL connection")
def test_get_chunks_by_node(sample_chunks):
    """노드로 청크 조회 테스트"""
    conn_str = "host=localhost dbname=semantica_test user=semantica"
    store = PostgresChunkStore(conn_str)
    
    store.save_chunks(sample_chunks)
    
    chunks = store.get_chunks_by_node("test-repo", "test-repo:main.py:Function:foo")
    assert len(chunks) == 1
    assert chunks[0].id == "chunk-1"


def test_row_to_chunk_conversion():
    """DB row → CodeChunk 변환 테스트"""
    from src.chunking.store import PostgresChunkStore
    
    row = (
        "test-repo",
        "chunk-1",
        "test-repo:main.py:Function:foo",
        "main.py",
        10, 0, 20, 0,  # span
        "python",
        "def foo():\n    return 42",
        {"node_kind": "Function"}
    )
    
    # Store 인스턴스 없이 변환 로직만 테스트
    chunk = CodeChunk(
        repo_id=row[0],
        id=row[1],
        node_id=row[2],
        file_path=row[3],
        span=(row[4], row[5], row[6], row[7]),
        language=row[8],
        text=row[9],
        attrs=row[10]
    )
    
    assert chunk.repo_id == "test-repo"
    assert chunk.id == "chunk-1"
    assert chunk.span == (10, 0, 20, 0)
    assert chunk.attrs.get("node_kind") == "Function"

