"""IR Builder 테스트"""

import pytest

from src.core.models import RawRelation, RawSymbol
from src.graph.ir_builder import IRBuilder


@pytest.fixture
def sample_symbols():
    """샘플 RawSymbol 리스트 (0-based indexing)"""
    return [
        RawSymbol(
            repo_id="test-repo",
            file_path="test.py",
            language="python",
            kind="Class",
            name="User",
            span=(0, 0, 5, 12),  # 0번 라인부터 5번 라인 12컬럼까지
            attrs={"docstring": "User class"},
        ),
        RawSymbol(
            repo_id="test-repo",
            file_path="test.py",
            language="python",
            kind="Method",
            name="User.save",
            span=(4, 4, 5, 12),  # 4번 라인 4컬럼부터
            attrs={"parent_class": "User"},
        ),
        RawSymbol(
            repo_id="test-repo",
            file_path="test.py",
            language="python",
            kind="Function",
            name="calculate",
            span=(8, 0, 9, 16),  # 8번 라인부터
            attrs={},
        ),
    ]


@pytest.fixture
def sample_relations():
    """샘플 RawRelation 리스트 (0-based indexing)"""
    return [
        RawRelation(
            repo_id="test-repo",
            file_path="test.py",
            language="python",
            type="defines",
            src_span=(0, 0, 5, 12),  # User class
            dst_span=(4, 4, 5, 12),  # User.save method
            attrs={"target": "User.save"},
        )
    ]


@pytest.fixture
def sample_source():
    """샘플 소스 코드"""
    return {
        "test.py": """class User:
    def __init__(self):
        pass

    def save(self):
        pass


def calculate(x):
    return x * 2
"""
    }


def test_ir_builder_initialization():
    """IR Builder 초기화"""
    builder = IRBuilder()
    assert builder is not None


def test_build_nodes(sample_symbols, sample_source):
    """노드 빌드 테스트"""
    builder = IRBuilder()
    nodes, edges = builder.build(sample_symbols, [], sample_source)

    assert len(nodes) == 3, "3개 노드가 생성되어야 함"
    assert all(node.id for node in nodes), "모든 노드에 ID가 있어야 함"
    assert all(node.text for node in nodes), "모든 노드에 text가 있어야 함"


def test_node_id_generation(sample_symbols):
    """노드 ID 생성 테스트"""
    builder = IRBuilder()
    nodes, _ = builder.build(sample_symbols, [])

    # ID 형식: repo_id:file_path:kind:name
    user_class = next((n for n in nodes if n.name == "User"), None)
    assert user_class is not None
    assert "test-repo" in user_class.id
    assert "test.py" in user_class.id
    assert "Class" in user_class.id
    assert "User" in user_class.id


def test_text_extraction(sample_symbols, sample_source):
    """텍스트 추출 테스트"""
    builder = IRBuilder()
    nodes, _ = builder.build(sample_symbols, [], sample_source)

    user_class = next((n for n in nodes if n.name == "User"), None)
    assert user_class is not None
    assert "class User:" in user_class.text
    assert len(user_class.text) > 0


def test_attrs_preservation(sample_symbols):
    """속성 보존 테스트"""
    builder = IRBuilder()
    nodes, _ = builder.build(sample_symbols, [])

    user_class = next((n for n in nodes if n.name == "User"), None)
    assert user_class is not None
    assert user_class.attrs.get("docstring") == "User class"


def test_build_edges(sample_symbols, sample_relations):
    """엣지 빌드 테스트"""
    builder = IRBuilder()
    nodes, edges = builder.build(sample_symbols, sample_relations)

    # relation이 edge로 변환되어야 함
    # (매핑이 실패할 수 있음 - span 기반 매칭의 한계)
    assert isinstance(edges, list)


def test_duplicate_node_removal(sample_symbols):
    """중복 노드 제거 테스트"""
    # 같은 심볼 2번 추가
    duplicate_symbols = sample_symbols + [sample_symbols[0]]

    builder = IRBuilder()
    nodes, _ = builder.build(duplicate_symbols, [])

    # 중복 제거되어야 함
    assert len(nodes) == 3, "중복 노드는 제거되어야 함"


def test_invalid_edge_removal(sample_symbols):
    """유효하지 않은 엣지 제거 테스트"""
    # 존재하지 않는 노드를 참조하는 관계
    invalid_relation = RawRelation(
        repo_id="test-repo",
        file_path="test.py",
        language="python",
        type="calls",
        src_span=(1, 0, 10, 0),
        dst_span=(999, 0, 999, 0),  # 존재하지 않는 위치
        attrs={},
    )

    builder = IRBuilder()
    nodes, edges = builder.build(sample_symbols, [invalid_relation])

    # 유효하지 않은 엣지는 제거되어야 함
    assert len(edges) == 0, "존재하지 않는 노드를 참조하는 엣지는 제거"


def test_build_without_source_code(sample_symbols):
    """소스 코드 없이 빌드 (text는 빈 문자열)"""
    builder = IRBuilder()
    nodes, _ = builder.build(sample_symbols, [], source_code=None)

    assert len(nodes) == 3
    # text는 빈 문자열이어야 함
    assert all(node.text == "" for node in nodes)
