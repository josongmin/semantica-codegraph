"""IndexingPipeline 로직 테스트 (DB 불필요)"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.indexer.pipeline import IndexingPipeline
from src.core.models import FileMetadata


def test_generate_repo_id():
    """Repo ID 생성 테스트"""
    pipeline = MagicMock(spec=IndexingPipeline)
    pipeline._generate_repo_id = IndexingPipeline._generate_repo_id.__get__(pipeline)
    
    repo_id = pipeline._generate_repo_id("/path/to/my-repo")
    assert repo_id == "my-repo"
    
    repo_id = pipeline._generate_repo_id("/Users/john/semantica-codegraph")
    assert repo_id == "semantica-codegraph"


def test_read_file(tmp_path):
    """파일 읽기 테스트"""
    pipeline = MagicMock(spec=IndexingPipeline)
    pipeline._read_file = IndexingPipeline._read_file.__get__(pipeline)
    
    # 테스트 파일 생성
    test_file = tmp_path / "test.py"
    test_file.write_text("def hello(): pass")
    
    content = pipeline._read_file(str(test_file))
    assert content == "def hello(): pass"


def test_read_nonexistent_file():
    """존재하지 않는 파일 읽기 테스트"""
    pipeline = MagicMock(spec=IndexingPipeline)
    pipeline._read_file = IndexingPipeline._read_file.__get__(pipeline)
    
    content = pipeline._read_file("/nonexistent/file.py")
    assert content == ""


def test_parse_file_no_parser():
    """파서가 없는 언어 테스트"""
    pipeline = MagicMock(spec=IndexingPipeline)
    pipeline._parse_file = IndexingPipeline._parse_file.__get__(pipeline)
    
    file_meta = FileMetadata(
        file_path="test.unknown",
        abs_path="/path/test.unknown",
        language="unknown"
    )
    
    symbols, relations = pipeline._parse_file("test-repo", file_meta)
    assert symbols == []
    assert relations == []

