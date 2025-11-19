"""Chunker 테스트"""

import pytest

from src.chunking.chunker import Chunker
from src.core.models import CodeNode


@pytest.fixture
def sample_nodes():
    """샘플 CodeNode 리스트"""
    return [
        # 작은 함수
        CodeNode(
            repo_id="test-repo",
            id="test:file.py:Function:small",
            kind="Function",
            language="python",
            file_path="file.py",
            span=(10, 0, 15, 0),  # 5줄
            name="small",
            text="def small():\n    return 42\n",
            attrs={},
        ),
        # 큰 함수 (200줄)
        CodeNode(
            repo_id="test-repo",
            id="test:file.py:Function:large",
            kind="Function",
            language="python",
            file_path="file.py",
            span=(20, 0, 220, 0),  # 200줄
            name="large",
            text="\n".join([f"line {i}" for i in range(200)]),
            attrs={},
        ),
        # 클래스
        CodeNode(
            repo_id="test-repo",
            id="test:file.py:Class:MyClass",
            kind="Class",
            language="python",
            file_path="file.py",
            span=(250, 0, 280, 0),  # 30줄
            name="MyClass",
            text="class MyClass:\n    pass",
            attrs={},
        ),
        # File 노드 (스킵되어야 함)
        CodeNode(
            repo_id="test-repo",
            id="test:file.py:File:file.py",
            kind="File",
            language="python",
            file_path="file.py",
            span=(0, 0, 300, 0),
            name="file.py",
            text="# entire file",
            attrs={},
        ),
    ]


def test_chunker_initialization():
    """Chunker 초기화 테스트"""
    chunker = Chunker()
    assert chunker is not None
    assert chunker.max_lines == 100
    assert chunker.strategy == "node_based"


def test_node_based_chunking(sample_nodes):
    """Node 기반 청킹 테스트 (1 Node = 1 Chunk)"""
    chunker = Chunker(strategy="node_based")
    chunks = chunker.chunk(sample_nodes)

    # File 노드 제외하고 3개 청크 생성
    assert len(chunks) == 3, "File 제외 3개 노드 → 3개 청크"

    # 각 청크가 node_id를 가져야 함
    assert all(chunk.node_id for chunk in chunks)

    # chunk_id 생성
    assert all(chunk.id.startswith("chunk:") for chunk in chunks)


def test_chunk_id_generation(sample_nodes):
    """Chunk ID 생성 테스트"""
    chunker = Chunker()
    chunks = chunker.chunk(sample_nodes)

    # ID는 고유해야 함
    chunk_ids = [c.id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Chunk ID는 고유해야 함"


def test_chunk_attrs_inheritance(sample_nodes):
    """Chunk가 Node attrs를 상속하는지 테스트"""
    # Node에 attrs 추가
    sample_nodes[0].attrs = {"docstring": "test doc", "custom": "value"}

    chunker = Chunker()
    chunks = chunker.chunk(sample_nodes)

    # 첫 번째 청크 (small 함수)
    chunk = chunks[0]
    assert "node_kind" in chunk.attrs
    assert chunk.attrs["node_kind"] == "Function"
    assert chunk.attrs["node_name"] == "small"
    # Node attrs도 상속
    assert chunk.attrs.get("docstring") == "test doc"
    assert chunk.attrs.get("custom") == "value"


def test_file_node_skipped(sample_nodes):
    """File 노드는 청킹에서 제외되어야 함"""
    chunker = Chunker()
    chunks = chunker.chunk(sample_nodes)

    # File 노드는 청크로 변환 안 됨
    file_chunks = [c for c in chunks if c.attrs.get("node_kind") == "File"]
    assert len(file_chunks) == 0, "File 노드는 제외되어야 함"


def test_size_based_chunking_small_nodes(sample_nodes):
    """크기 기반 청킹: 작은 노드는 그대로"""
    chunker = Chunker(strategy="size_based", max_lines=100)
    chunks = chunker.chunk(sample_nodes)

    # small 함수 (5줄) → 분할 안 됨
    small_chunks = [c for c in chunks if "small" in c.node_id]
    assert len(small_chunks) == 1


def test_size_based_chunking_large_nodes(sample_nodes):
    """크기 기반 청킹: 큰 노드는 분할"""
    chunker = Chunker(strategy="size_based", max_lines=100)
    chunks = chunker.chunk(sample_nodes)

    # large 함수 (200줄) → 분할됨
    large_chunks = [c for c in chunks if "large" in c.node_id]
    assert len(large_chunks) > 1, "큰 노드는 여러 청크로 분할되어야 함"


def test_split_node_overlap():
    """노드 분할 시 오버랩 테스트"""
    chunker = Chunker(max_lines=50, overlap_lines=5)

    large_node = CodeNode(
        repo_id="test",
        id="test:file:Function:big",
        kind="Function",
        language="python",
        file_path="file.py",
        span=(0, 0, 150, 0),  # 150줄
        name="big",
        text="\n".join([f"line {i}" for i in range(150)]),
        attrs={},
    )

    split_chunks = chunker._split_node(large_node)

    # 150줄을 50줄씩 → 3-4개 청크
    assert len(split_chunks) >= 3

    # 오버랩 확인 (선택적)
    # 각 청크가 is_split=True
    assert all(c.attrs.get("is_split") for c in split_chunks)


def test_hierarchical_chunking():
    """계층적 청킹 테스트"""
    nodes = [
        # Class
        CodeNode(
            repo_id="test",
            id="test:file:Class:MyClass",
            kind="Class",
            language="python",
            file_path="file.py",
            span=(0, 0, 50, 0),
            name="MyClass",
            text="class MyClass:\n    ...",
            attrs={},
        ),
        # Method 1
        CodeNode(
            repo_id="test",
            id="test:file:Method:MyClass.foo",
            kind="Method",
            language="python",
            file_path="file.py",
            span=(10, 4, 20, 0),
            name="MyClass.foo",
            text="def foo(self):\n    pass",
            attrs={"parent_class": "MyClass"},
        ),
        # Method 2
        CodeNode(
            repo_id="test",
            id="test:file:Method:MyClass.bar",
            kind="Method",
            language="python",
            file_path="file.py",
            span=(25, 4, 35, 0),
            name="MyClass.bar",
            text="def bar(self):\n    pass",
            attrs={"parent_class": "MyClass"},
        ),
    ]

    chunker = Chunker(strategy="hierarchical")
    chunks = chunker.chunk(nodes)

    # Class 1개 + Method 2개 = 3개 청크
    assert len(chunks) == 3

    # Class 청크 확인
    class_chunk = next((c for c in chunks if c.attrs["node_kind"] == "Class"), None)
    assert class_chunk is not None

    # Method 청크 확인
    method_chunks = [c for c in chunks if c.attrs["node_kind"] == "Method"]
    assert len(method_chunks) == 2


def test_chunk_text_preservation(sample_nodes):
    """청크가 노드 텍스트를 보존하는지 테스트"""
    chunker = Chunker()
    chunks = chunker.chunk(sample_nodes)

    # small 함수 청크
    small_chunk = next((c for c in chunks if "small" in c.node_id), None)
    assert small_chunk is not None
    assert "def small():" in small_chunk.text
    assert "return 42" in small_chunk.text


def test_chunk_span_preservation(sample_nodes):
    """청크가 노드 span을 보존하는지 테스트"""
    chunker = Chunker()
    chunks = chunker.chunk(sample_nodes)

    # small 함수 청크
    small_chunk = next((c for c in chunks if "small" in c.node_id), None)
    assert small_chunk is not None
    assert small_chunk.span == (10, 0, 15, 0)


def test_empty_nodes():
    """빈 노드 리스트 처리"""
    chunker = Chunker()
    chunks = chunker.chunk([])
    assert len(chunks) == 0


def test_line_count_calculation():
    """라인 수 계산 테스트"""
    chunker = Chunker()

    node = CodeNode(
        repo_id="test",
        id="test:id",
        kind="Function",
        language="python",
        file_path="file.py",
        span=(10, 0, 20, 0),  # 10줄
        name="test",
        text="test",
        attrs={},
    )

    line_count = chunker._get_line_count(node)
    assert line_count == 11  # 10부터 20까지 = 11줄


def test_token_estimation():
    """토큰 수 추정 테스트"""
    chunker = Chunker()

    text = "def foo(): return 42"  # 4 words → ~5 tokens
    token_count = chunker._estimate_token_count(text)

    assert token_count > 0
    assert token_count >= 4  # 최소 4개
    assert token_count < 10  # 과대 추정 방지


def test_token_based_chunking():
    """토큰 기반 청킹 테스트"""
    # 매우 긴 텍스트 (토큰 제한 초과) - 여러 줄로 구성
    lines = [
        f"    line_{i} = {' '.join([f'word{j}' for j in range(i * 50, (i + 1) * 50)])}"
        for i in range(200)
    ]  # 200줄, 각 줄에 50개의 단어
    long_text = "\n".join(lines)

    large_node = CodeNode(
        repo_id="test",
        id="test:file:Function:huge",
        kind="Function",
        language="python",
        file_path="file.py",
        span=(0, 0, 200, 0),
        name="huge",
        text=long_text,
        attrs={},
    )

    # max_tokens=7000으로 청킹
    chunker = Chunker(max_tokens=7000)
    chunks = chunker.chunk([large_node])

    # 여러 청크로 분할되어야 함
    assert len(chunks) > 1, "큰 노드는 토큰 제한에 따라 분할되어야 함"

    # 각 청크의 토큰 수가 제한 이하여야 함
    for chunk in chunks:
        token_count = chunker._count_tokens(chunk.text)
        assert token_count <= 7000, f"청크 토큰 수({token_count})가 제한(7000)을 초과"

    # 모든 청크에 token_count가 기록되어야 함
    assert all("token_count" in chunk.attrs for chunk in chunks)


def test_token_count_accurate():
    """tiktoken을 사용한 정확한 토큰 카운팅 테스트"""
    chunker = Chunker()

    # 간단한 코드 샘플
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

    token_count = chunker._count_tokens(code)

    # 토큰 수가 합리적인 범위 내에 있어야 함
    assert token_count > 0
    assert token_count < 100  # 간단한 함수이므로 100 토큰 미만


def test_token_limit_disabled():
    """max_tokens=None일 때 토큰 제한 비활성화 테스트"""
    long_text = " ".join([f"word{i}" for i in range(5000)])

    large_node = CodeNode(
        repo_id="test",
        id="test:file:Function:big",
        kind="Function",
        language="python",
        file_path="file.py",
        span=(0, 0, 500, 0),
        name="big",
        text=long_text,
        attrs={},
    )

    # max_tokens=None으로 청킹 (토큰 제한 없음)
    chunker = Chunker(max_tokens=None, max_lines=1000)
    chunks = chunker.chunk([large_node])

    # 분할되지 않아야 함 (라인 수가 max_lines 이하이므로)
    assert len(chunks) == 1


def test_chunker_initialization_with_max_tokens():
    """max_tokens 파라미터로 Chunker 초기화 테스트"""
    chunker = Chunker(max_tokens=5000)
    assert chunker.max_tokens == 5000

    # 기본값 테스트
    default_chunker = Chunker()
    assert default_chunker.max_tokens == 7000
