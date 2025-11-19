"""텍스트 파일 인덱싱 통합 테스트"""

import pytest

from src.chunking.chunker import Chunker
from src.core.enums import NodeKind
from src.core.models import RepoConfig
from src.graph.ir_builder import IRBuilder
from src.indexer.repo_scanner import RepoScanner


@pytest.fixture
def test_repo_with_docs(tmp_path):
    """코드 + 문서가 있는 테스트 repo"""
    repo = tmp_path / "mixed_repo"
    repo.mkdir()

    # Python 코드
    (repo / "main.py").write_text(
        """
def hello():
    '''Say hello'''
    print("Hello, World!")

class App:
    def run(self):
        hello()
"""
    )

    # 문서 파일
    (repo / "README.md").write_text(
        """# Mixed Repo

This project contains both code and documentation.

## Features
- Feature 1
- Feature 2

## Usage
Run `python main.py`
"""
    )

    (repo / "CHANGELOG.md").write_text(
        """# Changelog

## v1.0.0
- Initial release
"""
    )

    (repo / "config.json").write_text(
        """
{
  "app_name": "mixed_repo",
  "version": "1.0.0"
}
"""
    )

    return repo


class TestTextIndexingPipeline:
    """전체 파이프라인 통합 테스트"""

    def test_scan_includes_both_code_and_docs(self, test_repo_with_docs):
        """스캔 시 코드와 문서 모두 포함"""
        scanner = RepoScanner()
        files = scanner.scan(str(test_repo_with_docs))

        file_paths = {f.file_path for f in files}
        languages = {f.language for f in files}

        # 코드 파일
        assert "main.py" in file_paths
        assert "python" in languages

        # 문서 파일
        assert "README.md" in file_paths
        assert "CHANGELOG.md" in file_paths
        assert "config.json" in file_paths
        assert "markdown" in languages
        assert "json" in languages

    def test_parser_creates_different_node_kinds(self, test_repo_with_docs):
        """파서가 코드/문서를 다른 노드 타입으로 생성"""
        from src.parser import create_parser

        # Python 파일 파싱
        py_parser = create_parser("python")
        py_symbols, py_relations = py_parser.parse_file(
            {
                "repo_id": "test",
                "path": "main.py",
                "abs_path": str(test_repo_with_docs / "main.py"),
                "language": "python",
            }
        )

        # 함수, 클래스 노드 생성
        py_kinds = {s.kind for s in py_symbols}
        assert NodeKind.Function in py_kinds or NodeKind.Class in py_kinds

        # Markdown 파일 파싱
        md_parser = create_parser("markdown")
        md_symbols, md_relations = md_parser.parse_file(
            {
                "repo_id": "test",
                "path": "README.md",
                "abs_path": str(test_repo_with_docs / "README.md"),
                "language": "markdown",
            }
        )

        # Document 노드 생성
        assert len(md_symbols) == 1
        assert md_symbols[0].kind == NodeKind.Document
        assert len(md_relations) == 0  # 문서는 관계 없음

    def test_ir_builder_handles_mixed_nodes(self, test_repo_with_docs):
        """IRBuilder가 코드/문서 노드를 모두 처리"""
        from src.parser import create_parser

        # 코드 심볼
        py_parser = create_parser("python")
        py_symbols, py_relations = py_parser.parse_file(
            {
                "repo_id": "test",
                "path": "main.py",
                "abs_path": str(test_repo_with_docs / "main.py"),
                "language": "python",
            }
        )

        # 문서 심볼
        md_parser = create_parser("markdown")
        md_symbols, md_relations = md_parser.parse_file(
            {
                "repo_id": "test",
                "path": "README.md",
                "abs_path": str(test_repo_with_docs / "README.md"),
                "language": "markdown",
            }
        )

        # 통합
        all_symbols = py_symbols + md_symbols
        all_relations = py_relations + md_relations

        # IRBuilder 실행
        ir_builder = IRBuilder()

        source_code = {
            "main.py": (test_repo_with_docs / "main.py").read_text(),
            "README.md": (test_repo_with_docs / "README.md").read_text(),
        }

        nodes, edges = ir_builder.build(all_symbols, all_relations, source_code)

        # 모든 노드 생성
        assert len(nodes) > 0

        # 코드 노드와 문서 노드 구분
        code_node_kinds = {NodeKind.Function.value, NodeKind.Class.value, NodeKind.Method.value}
        code_nodes = [n for n in nodes if n.kind in code_node_kinds]
        doc_nodes = [n for n in nodes if n.kind == NodeKind.Document.value]

        assert len(code_nodes) > 0
        assert len(doc_nodes) > 0

        # 문서 노드는 엣지 없음 (코드 노드만 엣지 생성)
        # 실제 엣지는 코드-코드 관계만 있어야 함

    def test_chunker_handles_document_nodes(self, test_repo_with_docs):
        """Chunker가 Document 노드를 청크로 변환"""
        from src.parser import create_parser

        md_parser = create_parser("markdown")
        md_symbols, _ = md_parser.parse_file(
            {
                "repo_id": "test",
                "path": "README.md",
                "abs_path": str(test_repo_with_docs / "README.md"),
                "language": "markdown",
            }
        )

        ir_builder = IRBuilder()
        nodes, _ = ir_builder.build(
            md_symbols, [], {"README.md": (test_repo_with_docs / "README.md").read_text()}
        )

        chunker = Chunker()
        chunks = chunker.chunk(nodes)

        # 청크 생성 확인
        assert len(chunks) > 0

        # Document 노드가 청크로 변환됨
        chunk = chunks[0]
        assert "Mixed Repo" in chunk.text
        assert chunk.language == "markdown"

    def test_index_text_files_false_excludes_docs(self, test_repo_with_docs):
        """index_text_files=False 시 문서 제외"""
        scanner = RepoScanner()

        config = RepoConfig()
        config.index_text_files = False

        files = scanner.scan(str(test_repo_with_docs), config)

        file_paths = {f.file_path for f in files}
        languages = {f.language for f in files}

        # 코드 파일만 포함
        assert "main.py" in file_paths
        assert "python" in languages

        # 문서 파일 제외
        assert "README.md" not in file_paths
        assert "CHANGELOG.md" not in file_paths
        assert "config.json" not in file_paths
        assert "markdown" not in languages
        assert "json" not in languages


class TestTextIndexingEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_markdown_file(self, tmp_path):
        """빈 마크다운 파일"""
        from src.parser import create_parser

        empty_md = tmp_path / "empty.md"
        empty_md.write_text("")

        parser = create_parser("markdown")
        symbols, relations = parser.parse_file(
            {
                "repo_id": "test",
                "path": "empty.md",
                "abs_path": str(empty_md),
                "language": "markdown",
            }
        )

        assert len(symbols) == 1
        assert symbols[0].attrs["text"] == ""

    def test_very_large_markdown(self, tmp_path):
        """매우 큰 마크다운 (청킹 필요)"""
        from src.parser import create_parser

        large_md = tmp_path / "large.md"
        content = "# Header\n\n" + ("Paragraph text.\n\n" * 500)
        large_md.write_text(content)

        parser = create_parser("markdown")
        symbols, _ = parser.parse_file(
            {
                "repo_id": "test",
                "path": "large.md",
                "abs_path": str(large_md),
                "language": "markdown",
            }
        )

        ir_builder = IRBuilder()
        nodes, _ = ir_builder.build(symbols, [], {"large.md": content})

        chunker = Chunker()
        chunks = chunker.chunk(nodes)

        # Phase 1: 단일 청크로 생성
        # Phase 2: 여러 청크로 분할 (토큰 기반)
        assert len(chunks) >= 1

    def test_mixed_encoding_resilience(self, tmp_path):
        """다양한 인코딩 혼재"""
        from src.parser import create_parser

        # UTF-8
        utf8_file = tmp_path / "utf8.txt"
        utf8_file.write_text("UTF-8 텍스트", encoding="utf-8")

        parser = create_parser("text")
        symbols, _ = parser.parse_file(
            {"repo_id": "test", "path": "utf8.txt", "abs_path": str(utf8_file), "language": "text"}
        )

        assert len(symbols) == 1
        assert "UTF-8" in symbols[0].attrs["text"]
