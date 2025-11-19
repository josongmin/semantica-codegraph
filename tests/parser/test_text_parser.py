"""TextParser 테스트"""

import pytest

from src.core.enums import NodeKind
from src.parser.text_parser import TextParser


class TestTextParser:
    """TextParser 기본 기능 테스트"""

    @pytest.fixture
    def parser(self):
        return TextParser()

    @pytest.fixture
    def markdown_file(self, tmp_path):
        """임시 마크다운 파일 생성"""
        md_file = tmp_path / "README.md"
        md_file.write_text(
            """# Test Document

This is a test markdown file.

## Section 1
Content here.
""",
            encoding="utf-8",
        )
        return md_file

    @pytest.fixture
    def json_file(self, tmp_path):
        """임시 JSON 파일 생성"""
        json_file = tmp_path / "config.json"
        json_file.write_text('{"key": "value", "count": 42}', encoding="utf-8")
        return json_file

    def test_parse_markdown_file(self, parser, markdown_file):
        """마크다운 파일을 Document 노드로 변환"""
        file_meta = {
            "repo_id": "test_repo",
            "path": "README.md",
            "abs_path": str(markdown_file),
            "language": "markdown",
        }

        symbols, relations = parser.parse_file(file_meta)

        # 단일 Document 노드 생성
        assert len(symbols) == 1
        assert len(relations) == 0  # 관계 없음

        symbol = symbols[0]
        assert symbol.kind == NodeKind.Document.value
        assert symbol.name == "README"
        assert symbol.file_path == "README.md"
        assert "# Test Document" in symbol.attrs["text"]
        assert symbol.attrs["file_type"] == ".md"

    def test_parse_json_file(self, parser, json_file):
        """JSON 파일을 Document 노드로 변환"""
        file_meta = {
            "repo_id": "test_repo",
            "path": "config.json",
            "abs_path": str(json_file),
            "language": "json",
        }

        symbols, relations = parser.parse_file(file_meta)

        assert len(symbols) == 1
        assert len(relations) == 0

        symbol = symbols[0]
        assert symbol.kind == NodeKind.Document.value
        assert symbol.name == "config"
        assert '"key": "value"' in symbol.attrs["text"]

    def test_encoding_error_handling(self, parser, tmp_path):
        """인코딩 에러 처리 (latin-1, cp949 등)"""
        # UTF-8이 아닌 파일 생성
        bad_file = tmp_path / "bad_encoding.txt"
        bad_file.write_bytes(b"\xff\xfe\x00\x00")  # 잘못된 인코딩

        file_meta = {
            "repo_id": "test_repo",
            "file_path": "bad_encoding.txt",
            "abs_path": str(bad_file),
            "language": "text",
        }

        # errors="ignore"로 처리되어야 함
        symbols, relations = parser.parse_file(file_meta)

        # 에러 발생하지 않고 처리됨
        assert len(symbols) == 1 or len(symbols) == 0  # 읽기 실패 시 []

    def test_file_not_found(self, parser):
        """존재하지 않는 파일 처리"""
        file_meta = {
            "repo_id": "test_repo",
            "file_path": "nonexistent.md",
            "abs_path": "/nonexistent/path/file.md",
            "language": "markdown",
        }

        symbols, relations = parser.parse_file(file_meta)

        # 빈 결과 반환
        assert len(symbols) == 0
        assert len(relations) == 0

    def test_empty_file(self, parser, tmp_path):
        """빈 파일 처리"""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        file_meta = {
            "repo_id": "test_repo",
            "path": "empty.txt",
            "abs_path": str(empty_file),
            "language": "text",
        }

        symbols, relations = parser.parse_file(file_meta)

        assert len(symbols) == 1
        symbol = symbols[0]
        assert symbol.attrs["text"] == ""
        assert symbol.span == (0, 0, 0, 0)

    def test_large_file_content(self, parser, tmp_path):
        """대용량 파일 내용 처리"""
        large_file = tmp_path / "large.md"
        content = "# Header\n\n" + ("Line content\n" * 1000)
        large_file.write_text(content, encoding="utf-8")

        file_meta = {
            "repo_id": "test_repo",
            "path": "large.md",
            "abs_path": str(large_file),
            "language": "markdown",
        }

        symbols, relations = parser.parse_file(file_meta)

        assert len(symbols) == 1
        symbol = symbols[0]
        assert len(symbol.attrs["text"]) > 10000
        assert symbol.span[2] == 1002  # 1002 lines total

    def test_special_characters(self, parser, tmp_path):
        """특수 문자 처리"""
        special_file = tmp_path / "special.txt"
        content = "특수문자: 한글, 日本語, 中文\n🎉 Emoji\n<html>&nbsp;</html>"
        special_file.write_text(content, encoding="utf-8")

        file_meta = {
            "repo_id": "test_repo",
            "path": "special.txt",
            "abs_path": str(special_file),
            "language": "text",
        }

        symbols, relations = parser.parse_file(file_meta)

        assert len(symbols) == 1
        symbol = symbols[0]
        assert "한글" in symbol.attrs["text"]
        assert "🎉" in symbol.attrs["text"]
        assert "&nbsp;" in symbol.attrs["text"]
