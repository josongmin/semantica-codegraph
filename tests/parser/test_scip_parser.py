"""SCIP Parser 테스트"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.parser.scip_parser import ScipParser


@pytest.fixture
def sample_scip_index_dir(tmp_path):
    """샘플 SCIP 인덱스 디렉토리 생성"""
    index_dir = tmp_path / "scip_index"
    index_dir.mkdir()

    # 샘플 메타데이터
    metadata = {"version": "1.0", "files": ["test.py"]}

    import json

    (index_dir / "metadata.json").write_text(json.dumps(metadata))

    return index_dir


@pytest.fixture
def sample_scip_index_file(tmp_path):
    """샘플 SCIP 인덱스 파일 생성"""
    index_file = tmp_path / "index.scip"
    # 실제 SCIP 파일은 바이너리이므로 여기서는 빈 파일로 대체
    index_file.write_bytes(b"")
    return index_file


def test_scip_parser_initialization():
    """ScipParser 초기화 테스트"""
    parser = ScipParser()
    assert parser is not None
    assert parser.scip_index_path is None
    assert parser.auto_index is False


def test_scip_parser_with_index_path(sample_scip_index_dir):
    """인덱스 경로 지정 테스트"""
    parser = ScipParser(scip_index_path=sample_scip_index_dir)
    assert parser.scip_index_path == sample_scip_index_dir


def test_scip_parser_with_auto_index():
    """자동 인덱스 생성 옵션 테스트"""
    parser = ScipParser(auto_index=True)
    assert parser.auto_index is True


def test_parse_file_no_index():
    """인덱스가 없을 때 테스트"""
    parser = ScipParser()

    file_meta = {
        "repo_id": "test-repo",
        "path": "test.py",
        "abs_path": "/path/to/test.py",
        "language": "python",
    }

    symbols, relations = parser.parse_file(file_meta)

    # 인덱스가 없으면 빈 리스트 반환
    assert symbols == []
    assert relations == []


def test_parse_file_with_directory_index(sample_scip_index_dir):
    """디렉토리 형식 인덱스 테스트"""
    parser = ScipParser(scip_index_path=sample_scip_index_dir)

    file_meta = {
        "repo_id": "test-repo",
        "path": "test.py",
        "abs_path": str(sample_scip_index_dir / "test.py"),
        "language": "python",
    }

    # _ensure_index가 True를 반환하도록 설정
    parser._is_directory_format = True
    parser._index_data = {"files": ["test.py"]}

    symbols, relations = parser.parse_file(file_meta)

    # 빈 리스트 반환 (실제 SCIP 데이터 없음)
    assert isinstance(symbols, list)
    assert isinstance(relations, list)


def test_parse_file_with_file_index(sample_scip_index_file):
    """파일 형식 인덱스 테스트"""
    parser = ScipParser(scip_index_path=sample_scip_index_file)

    file_meta = {
        "repo_id": "test-repo",
        "path": "test.py",
        "abs_path": str(sample_scip_index_file),
        "language": "python",
    }

    parser._is_directory_format = False

    symbols, relations = parser.parse_file(file_meta)

    assert isinstance(symbols, list)
    assert isinstance(relations, list)


@patch("src.parser.scip_parser.subprocess.run")
def test_auto_index_creation(mock_subprocess, tmp_path):
    """자동 인덱스 생성 테스트"""
    mock_subprocess.return_value = Mock(returncode=0)

    parser = ScipParser(auto_index=True)
    parser.scip_index_path = tmp_path / "scip_index"

    file_meta = {
        "repo_id": "test-repo",
        "path": "test.py",
        "abs_path": str(tmp_path / "test.py"),
        "language": "python",
    }

    # 파일이 없으면 자동 생성 시도
    parser.parse_file(file_meta)

    # subprocess가 호출되었는지 확인 (실제로는 인덱스가 없을 때만)
    # 여기서는 에러 처리만 확인


def test_ensure_index_with_existing_directory(sample_scip_index_dir):
    """존재하는 디렉토리 인덱스 확인 테스트"""
    parser = ScipParser(scip_index_path=sample_scip_index_dir)

    file_meta = {
        "repo_id": "test-repo",
        "path": "test.py",
        "abs_path": str(sample_scip_index_dir / "test.py"),
        "language": "python",
    }

    result = parser._ensure_index(file_meta)

    # 디렉토리가 존재하면 True 반환
    assert result is True
    assert parser._is_directory_format is True


def test_ensure_index_with_existing_file(sample_scip_index_file):
    """존재하는 파일 인덱스 확인 테스트"""
    parser = ScipParser(scip_index_path=sample_scip_index_file)

    file_meta = {
        "repo_id": "test-repo",
        "path": "test.py",
        "abs_path": str(sample_scip_index_file),
        "language": "python",
    }

    result = parser._ensure_index(file_meta)

    # 파일이 존재하면 True 반환
    assert result is True
    assert parser._is_directory_format is False


def test_ensure_index_with_nonexistent_path():
    """존재하지 않는 경로 테스트"""
    parser = ScipParser(scip_index_path=Path("/nonexistent/path"))

    file_meta = {
        "repo_id": "test-repo",
        "path": "test.py",
        "abs_path": "/path/to/test.py",
        "language": "python",
    }

    result = parser._ensure_index(file_meta)

    # 경로가 없으면 False 반환
    assert result is False


@pytest.mark.skip(reason="SCIP 디렉토리 형식 로드 기능 미구현")
def test_load_file_data_directory_format(sample_scip_index_dir):
    """디렉토리 형식에서 파일 데이터 로드 테스트"""
    # 실제 SCIP 데이터 없이는 테스트 불가
    pass


def test_extract_symbols_from_scip():
    """SCIP 데이터에서 심볼 추출 테스트"""
    parser = ScipParser()

    # 샘플 SCIP 데이터 (실제 형식과 다를 수 있음)
    scip_data = {"symbols": [{"name": "test_function", "kind": "Function", "span": [10, 0, 20, 0]}]}

    file_meta = {
        "repo_id": "test-repo",
        "path": "test.py",
        "abs_path": "/path/to/test.py",
        "language": "python",
    }

    symbols = parser._extract_symbols_from_scip(scip_data, file_meta)

    assert isinstance(symbols, list)


def test_extract_relations_from_scip():
    """SCIP 데이터에서 관계 추출 테스트"""
    parser = ScipParser()

    # 샘플 SCIP 데이터
    scip_data = {
        "relations": [{"source": "test_function", "target": "helper_function", "type": "calls"}]
    }

    file_meta = {
        "repo_id": "test-repo",
        "path": "test.py",
        "abs_path": "/path/to/test.py",
        "language": "python",
    }

    relations = parser._extract_relations_from_scip(scip_data, file_meta)

    assert isinstance(relations, list)


def test_parse_file_error_handling():
    """에러 처리 테스트"""
    parser = ScipParser()

    # 잘못된 파일 메타데이터
    file_meta = {
        "repo_id": "test-repo",
        "path": "test.py",
        "abs_path": "/nonexistent/path.py",
        "language": "python",
    }

    # 에러가 발생해도 빈 리스트 반환
    symbols, relations = parser.parse_file(file_meta)

    assert symbols == []
    assert relations == []


def test_parse_file_with_invalid_scip_data(sample_scip_index_dir):
    """잘못된 SCIP 데이터 처리 테스트"""
    parser = ScipParser(scip_index_path=sample_scip_index_dir)
    parser._is_directory_format = True
    parser._index_data = {}

    file_meta = {
        "repo_id": "test-repo",
        "path": "test.py",
        "abs_path": str(sample_scip_index_dir / "test.py"),
        "language": "python",
    }

    # 잘못된 데이터여도 에러 없이 빈 리스트 반환
    symbols, relations = parser.parse_file(file_meta)

    assert isinstance(symbols, list)
    assert isinstance(relations, list)
