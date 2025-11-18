"""RepoScanner 테스트"""

from pathlib import Path

import pytest

from src.core.models import RepoConfig
from src.indexer.repo_scanner import RepoScanner


@pytest.fixture
def temp_repo(tmp_path):
    """임시 저장소 생성"""
    # Python 파일
    (tmp_path / "main.py").write_text("def main(): pass")
    (tmp_path / "utils.py").write_text("def util(): pass")

    # TypeScript 파일
    (tmp_path / "app.ts").write_text("function app() {}")

    # 제외해야 할 파일
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "main.cpython-310.pyc").write_text("bytecode")

    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("// lib")

    # 숨김 파일
    (tmp_path / ".hidden.py").write_text("# hidden")

    # 하위 디렉토리
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("class Module: pass")

    return tmp_path


def test_scanner_basic(temp_repo):
    """기본 스캔 테스트"""
    scanner = RepoScanner()
    files = scanner.scan(str(temp_repo))

    # Python 파일 3개 + TypeScript 파일 1개
    assert len(files) == 4

    # 파일 경로 확인
    file_paths = [f.file_path for f in files]
    assert "main.py" in file_paths
    assert "utils.py" in file_paths
    assert "app.ts" in file_paths
    assert "src/module.py" in file_paths or "src\\module.py" in file_paths


def test_scanner_language_detection(temp_repo):
    """언어 감지 테스트"""
    scanner = RepoScanner()
    files = scanner.scan(str(temp_repo))

    # 언어별 분류
    languages = {f.language for f in files}
    assert "python" in languages
    assert "typescript" in languages


def test_scanner_exclude_patterns(temp_repo):
    """제외 패턴 테스트"""
    scanner = RepoScanner()
    files = scanner.scan(str(temp_repo))

    # 제외된 파일 확인
    file_paths = [f.file_path for f in files]
    assert not any("__pycache__" in p for p in file_paths)
    assert not any("node_modules" in p for p in file_paths)
    assert not any(".hidden" in p for p in file_paths)


def test_scanner_language_filter(temp_repo):
    """언어 필터 테스트"""
    scanner = RepoScanner()
    config = RepoConfig(languages=["python"])
    files = scanner.scan(str(temp_repo), config=config)

    # Python 파일만
    assert all(f.language == "python" for f in files)
    assert len(files) == 3


def test_scanner_custom_exclude(temp_repo):
    """커스텀 제외 패턴 테스트"""
    scanner = RepoScanner()
    config = RepoConfig(
        exclude_patterns=["**/src/**"]
    )
    files = scanner.scan(str(temp_repo), config=config)

    # src 디렉토리 제외
    file_paths = [f.file_path for f in files]
    assert not any("src" in p for p in file_paths)


def test_scanner_sorted_output(temp_repo):
    """정렬된 출력 테스트"""
    scanner = RepoScanner()
    files = scanner.scan(str(temp_repo))

    # 파일 경로 기준 정렬 확인
    file_paths = [f.file_path for f in files]
    assert file_paths == sorted(file_paths)


def test_scanner_abs_path(temp_repo):
    """절대 경로 테스트"""
    scanner = RepoScanner()
    files = scanner.scan(str(temp_repo))

    for file in files:
        # abs_path가 실제 존재하는 파일인지 확인
        assert Path(file.abs_path).exists()
        assert Path(file.abs_path).is_file()


def test_scanner_empty_directory(tmp_path):
    """빈 디렉토리 테스트"""
    scanner = RepoScanner()
    files = scanner.scan(str(tmp_path))

    assert len(files) == 0


def test_scanner_nonexistent_path():
    """존재하지 않는 경로 테스트"""
    scanner = RepoScanner()

    with pytest.raises(FileNotFoundError):
        scanner.scan("/nonexistent/path")


def test_scanner_file_not_directory(tmp_path):
    """파일 경로 전달 시 에러"""
    file_path = tmp_path / "test.py"
    file_path.write_text("# test")

    scanner = RepoScanner()

    with pytest.raises(NotADirectoryError):
        scanner.scan(str(file_path))

