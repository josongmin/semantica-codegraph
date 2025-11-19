"""ContextPacker 테스트"""

from unittest.mock import MagicMock

import pytest

from src.context.packer import ContextPacker
from src.core.models import Candidate, CodeChunk, PackedContext, PackedSnippet


@pytest.fixture
def mock_stores():
    """Mock 스토어"""
    chunk_store = MagicMock()
    graph_store = MagicMock()
    return chunk_store, graph_store


@pytest.fixture
def sample_candidates():
    """샘플 Candidate 리스트"""
    return [
        Candidate(
            repo_id="test",
            chunk_id="chunk-1",
            features={"final_score": 0.9, "semantic_score": 0.9},
            file_path="main.py",
            span=(0, 0, 10, 0),
        ),
        Candidate(
            repo_id="test",
            chunk_id="chunk-2",
            features={"final_score": 0.7, "lexical_score": 0.7},
            file_path="utils.py",
            span=(0, 0, 5, 0),
        ),
        Candidate(
            repo_id="test",
            chunk_id="chunk-3",
            features={"final_score": 0.5, "graph_score": 0.5},
            file_path="helpers.py",
            span=(0, 0, 8, 0),
        ),
    ]


def test_packer_basic(mock_stores, sample_candidates):
    """기본 패킹 테스트"""
    chunk_store, graph_store = mock_stores

    # Primary chunk 설정
    chunk_store.get_chunk.return_value = CodeChunk(
        repo_id="test",
        id="chunk-1",
        node_id="node-1",
        file_path="main.py",
        span=(0, 0, 10, 0),
        language="python",
        text="def hello():\n    return 'world'",
        attrs={},
    )

    packer = ContextPacker(chunk_store, graph_store)

    context = packer.pack(candidates=sample_candidates, max_tokens=1000)

    assert context.primary is not None
    assert context.primary.role == "primary"
    assert context.primary.meta["chunk_id"] == "chunk-1"
    assert "hello" in context.primary.text


def test_packer_with_supporting(mock_stores, sample_candidates):
    """Supporting snippets 포함 테스트"""
    chunk_store, graph_store = mock_stores

    # Mock 청크 반환
    def get_chunk_side_effect(repo_id, chunk_id):
        chunks = {
            "chunk-1": CodeChunk(
                repo_id="test",
                id="chunk-1",
                node_id="node-1",
                file_path="main.py",
                span=(0, 0, 10, 0),
                language="python",
                text="def hello():\n    return 'world'",
                attrs={},
            ),
            "chunk-2": CodeChunk(
                repo_id="test",
                id="chunk-2",
                node_id="node-2",
                file_path="utils.py",
                span=(0, 0, 5, 0),
                language="python",
                text="def util():\n    pass",
                attrs={},
            ),
            "chunk-3": CodeChunk(
                repo_id="test",
                id="chunk-3",
                node_id="node-3",
                file_path="helpers.py",
                span=(0, 0, 8, 0),
                language="python",
                text="def helper():\n    return True",
                attrs={},
            ),
        }
        return chunks.get(chunk_id)

    chunk_store.get_chunk.side_effect = get_chunk_side_effect

    packer = ContextPacker(chunk_store, graph_store)

    context = packer.pack(candidates=sample_candidates, max_tokens=1000)

    assert context.primary is not None
    assert len(context.supporting) > 0

    # Supporting snippets는 "related" role
    for snippet in context.supporting:
        assert snippet.role in ["related", "caller", "callee"]


def test_packer_token_limit(mock_stores, sample_candidates):
    """토큰 제한 테스트"""
    chunk_store, graph_store = mock_stores

    # 긴 텍스트
    long_text = "def long_function():\n" + "    pass\n" * 100  # 매우 긴 함수

    chunk_store.get_chunk.return_value = CodeChunk(
        repo_id="test",
        id="chunk-1",
        node_id="node-1",
        file_path="main.py",
        span=(0, 0, 100, 0),
        language="python",
        text=long_text,
        attrs={},
    )

    packer = ContextPacker(chunk_store, graph_store)

    # 작은 토큰 제한
    context = packer.pack(candidates=sample_candidates, max_tokens=200)

    # Primary만 포함되고 supporting은 거의 없어야 함
    assert context.primary is not None
    assert len(context.supporting) == 0  # 토큰 부족으로 supporting 없음


def test_packer_empty_candidates(mock_stores):
    """빈 candidates 처리 테스트"""
    chunk_store, graph_store = mock_stores

    packer = ContextPacker(chunk_store, graph_store)

    with pytest.raises(ValueError, match="No candidates"):
        packer.pack([], max_tokens=1000)


def test_packer_primary_not_found(mock_stores, sample_candidates):
    """Primary chunk를 찾을 수 없는 경우"""
    chunk_store, graph_store = mock_stores

    # chunk_store가 None 반환
    chunk_store.get_chunk.return_value = None

    packer = ContextPacker(chunk_store, graph_store)

    with pytest.raises(ValueError, match="Primary chunk not found"):
        packer.pack(sample_candidates, max_tokens=1000)


def test_packer_metadata_preservation(mock_stores, sample_candidates):
    """메타데이터 보존 테스트"""
    chunk_store, graph_store = mock_stores

    chunk_store.get_chunk.return_value = CodeChunk(
        repo_id="test",
        id="chunk-1",
        node_id="node-1",
        file_path="main.py",
        span=(0, 0, 10, 0),
        language="python",
        text="def hello(): pass",
        attrs={"docstring": "Hello function"},
    )

    packer = ContextPacker(chunk_store, graph_store)

    context = packer.pack(sample_candidates, max_tokens=1000)

    # Meta 정보 확인
    assert "chunk_id" in context.primary.meta
    assert "node_id" in context.primary.meta
    assert "features" in context.primary.meta
    assert context.primary.meta["chunk_id"] == "chunk-1"


def test_to_prompt_markdown(mock_stores):
    """to_prompt() - Markdown 형식 테스트"""
    chunk_store, graph_store = mock_stores
    packer = ContextPacker(chunk_store, graph_store)

    # PackedContext 생성
    primary = PackedSnippet(
        repo_id="test",
        file_path="main.py",
        span=(0, 0, 5, 0),
        role="primary",
        text="def hello():\n    return 'world'",
        meta={"chunk_id": "chunk-1"},
    )

    supporting = [
        PackedSnippet(
            repo_id="test",
            file_path="utils.py",
            span=(0, 0, 3, 0),
            role="caller",
            text="def caller():\n    hello()",
            meta={"chunk_id": "chunk-2"},
        ),
        PackedSnippet(
            repo_id="test",
            file_path="helpers.py",
            span=(0, 0, 4, 0),
            role="callee",
            text="def helper():\n    pass",
            meta={"chunk_id": "chunk-3"},
        ),
    ]

    context = PackedContext(primary=primary, supporting=supporting)

    # Markdown 형식 프롬프트 생성
    prompt = packer.to_prompt(context, query="hello 함수", format="markdown")

    # 검증
    assert "## 검색 쿼리" in prompt
    assert "hello 함수" in prompt
    assert "## 코드 컨텍스트" in prompt
    assert "### Primary Code" in prompt
    assert "main.py" in prompt
    assert "def hello():" in prompt
    assert "### Supporting Code" in prompt
    assert "호출하는 코드" in prompt
    assert "호출되는 코드" in prompt
    assert "```python" in prompt
    assert "utils.py" in prompt
    assert "helpers.py" in prompt


def test_to_prompt_plain(mock_stores):
    """to_prompt() - Plain 텍스트 형식 테스트"""
    chunk_store, graph_store = mock_stores
    packer = ContextPacker(chunk_store, graph_store)

    # PackedContext 생성
    primary = PackedSnippet(
        repo_id="test",
        file_path="main.py",
        span=(0, 0, 5, 0),
        role="primary",
        text="def hello():\n    return 'world'",
        meta={},
    )

    context = PackedContext(primary=primary, supporting=[])

    # Plain 형식 프롬프트 생성
    prompt = packer.to_prompt(context, query="hello", format="plain")

    # 검증
    assert "검색 쿼리: hello" in prompt
    assert "코드 컨텍스트" in prompt
    assert "[Primary]" in prompt
    assert "main.py" in prompt
    assert "def hello():" in prompt
    assert "=" * 80 in prompt


def test_to_prompt_without_query(mock_stores):
    """to_prompt() - 쿼리 없이 테스트"""
    chunk_store, graph_store = mock_stores
    packer = ContextPacker(chunk_store, graph_store)

    primary = PackedSnippet(
        repo_id="test",
        file_path="main.py",
        span=(0, 0, 5, 0),
        role="primary",
        text="def hello(): pass",
        meta={},
    )

    context = PackedContext(primary=primary, supporting=[])

    # 쿼리 없이 프롬프트 생성
    prompt = packer.to_prompt(context, format="markdown")

    # 쿼리 섹션이 없어야 함
    assert "## 검색 쿼리" not in prompt
    assert "## 코드 컨텍스트" in prompt


def test_detect_language(mock_stores):
    """언어 감지 테스트"""
    chunk_store, graph_store = mock_stores
    packer = ContextPacker(chunk_store, graph_store)

    # 다양한 파일 확장자 테스트
    assert packer._detect_language("main.py") == "python"
    assert packer._detect_language("app.ts") == "typescript"
    assert packer._detect_language("component.tsx") == "typescript"
    assert packer._detect_language("script.js") == "javascript"
    assert packer._detect_language("App.jsx") == "javascript"
    assert packer._detect_language("Main.java") == "java"
    assert packer._detect_language("main.go") == "go"
    assert packer._detect_language("lib.rs") == "rust"
    assert packer._detect_language("config.yaml") == "yaml"
    assert packer._detect_language("unknown.xyz") == "text"  # 알 수 없는 확장자
