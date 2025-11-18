"""HybridParser 테스트"""

from pathlib import Path

import pytest

from src.parser.hybrid_parser import HybridParser
from src.parser.python_parser import PythonTreeSitterParser
from src.parser.scip_parser import ScipParser


@pytest.fixture
def sample_python_file():
    """샘플 Python 파일 경로"""
    return Path(__file__).parent.parent / "fixtures" / "sample_python.py"


@pytest.fixture
def file_meta(sample_python_file):
    """파일 메타데이터"""
    return {
        "repo_id": "test-repo",
        "path": "sample_python.py",
        "abs_path": str(sample_python_file),
        "language": "python",
        "repo_root": str(sample_python_file.parent.parent)
    }


def test_hybrid_parser_initialization():
    """HybridParser 초기화 테스트"""
    ts_parser = PythonTreeSitterParser()
    scip_parser = ScipParser()

    hybrid = HybridParser(ts_parser, scip_parser)
    assert hybrid is not None
    assert hybrid.tree_sitter is not None
    assert hybrid.scip is not None


def test_hybrid_without_scip(file_meta):
    """SCIP 없이 하이브리드 파서 사용 (Tree-sitter만)"""
    ts_parser = PythonTreeSitterParser()
    hybrid = HybridParser(ts_parser, scip_parser=None)

    symbols, relations = hybrid.parse_file(file_meta)

    # Tree-sitter 결과만 반환되어야 함
    assert len(symbols) > 0
    assert all(s.span != (0, 0, 0, 0) for s in symbols), "Tree-sitter span이 있어야 함"


def test_symbol_key_generation():
    """심볼 키 생성 테스트"""
    from src.core.models import RawSymbol

    ts_parser = PythonTreeSitterParser()
    hybrid = HybridParser(ts_parser)

    symbol = RawSymbol(
        repo_id="test",
        file_path="test.py",
        language="python",
        kind="Function",
        name="foo",
        span=(1, 0, 5, 0),
        attrs={}
    )

    key = hybrid._make_symbol_key(symbol)
    assert key == ("test.py", "foo", "Function")


def test_span_overlapping():
    """Span 겹침 확인 테스트"""
    ts_parser = PythonTreeSitterParser()
    hybrid = HybridParser(ts_parser)

    span1 = (10, 0, 20, 0)
    span2 = (15, 0, 25, 0)
    span3 = (30, 0, 40, 0)

    assert hybrid._is_span_overlapping(span1, span2) is True
    assert hybrid._is_span_overlapping(span1, span3) is False


def test_merge_symbols_prefers_tree_sitter_span():
    """심볼 병합 시 Tree-sitter span 우선 확인"""
    from src.core.models import RawSymbol

    ts_parser = PythonTreeSitterParser()
    hybrid = HybridParser(ts_parser, prefer_tree_sitter_span=True)

    ts_sym = RawSymbol(
        repo_id="test",
        file_path="test.py",
        language="python",
        kind="Function",
        name="foo",
        span=(10, 0, 20, 0),  # Tree-sitter span
        attrs={"ts_info": "value"}
    )

    scip_sym = RawSymbol(
        repo_id="test",
        file_path="test.py",
        language="python",
        kind="Function",
        name="foo",
        span=(11, 0, 21, 0),  # SCIP span (약간 다름)
        attrs={"type": "int -> int", "scip_info": "value"}
    )

    merged = hybrid._merge_symbols([ts_sym], [scip_sym])

    assert len(merged) == 1
    result = merged[0]

    # Tree-sitter span 유지
    assert result.span == (10, 0, 20, 0)

    # SCIP attrs 추가
    assert "type" in result.attrs
    assert result.attrs["type"] == "int -> int"

    # Tree-sitter attrs도 유지
    assert "ts_info" in result.attrs


def test_merge_relations_prefers_scip():
    """관계 병합 시 SCIP 우선 확인"""
    from src.core.models import RawRelation

    ts_parser = PythonTreeSitterParser()
    hybrid = HybridParser(ts_parser)

    ts_rel = RawRelation(
        repo_id="test",
        file_path="test.py",
        language="python",
        type="calls",
        src_span=(10, 0, 10, 5),
        dst_span=(20, 0, 20, 5),
        attrs={"target": "bar"}
    )

    scip_rel = RawRelation(
        repo_id="test",
        file_path="test.py",
        language="python",
        type="calls",
        src_span=(10, 0, 10, 5),
        dst_span=(20, 0, 20, 5),
        attrs={"target": "bar", "type_checked": True}
    )

    merged = hybrid._merge_relations([ts_rel], [scip_rel])

    # SCIP 관계 우선
    assert len(merged) == 1
    result = merged[0]
    assert "type_checked" in result.attrs

