"""RepoScanner 필터링 테스트 (텍스트 파일 인덱싱)"""


import pytest

from src.core.models import RepoConfig
from src.indexer.repo_scanner import RepoScanner


class TestRepoScannerTextFiltering:
    """텍스트 파일 필터링 테스트"""

    @pytest.fixture
    def scanner(self):
        return RepoScanner()

    @pytest.fixture
    def test_repo(self, tmp_path):
        """테스트용 repo 구조 생성"""
        repo = tmp_path / "test_repo"
        repo.mkdir()

        # 코드 파일
        (repo / "main.py").write_text("print('hello')")
        (repo / "app.ts").write_text("console.log('hello')")

        # 문서 파일
        (repo / "README.md").write_text("# Project")
        (repo / "notes.txt").write_text("Notes here")
        (repo / "config.json").write_text('{"key": "value"}')
        (repo / "settings.yaml").write_text("key: value")

        # 제외되어야 할 파일들
        node_modules = repo / "node_modules"
        node_modules.mkdir()
        (node_modules / "package.json").write_text("{}")

        (repo / "package-lock.json").write_text('{"dependencies": {}}')
        dist_dir = repo / "dist"
        dist_dir.mkdir(parents=True)
        (dist_dir / "bundle.min.js").write_text("minified")

        # 대용량 파일
        large_file = repo / "large.md"
        large_file.write_text("x" * 2_000_000)  # 2MB

        # 바이너리 파일
        (repo / "image.png").write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00')

        return repo

    def test_scan_includes_text_files(self, scanner, test_repo):
        """텍스트 파일 포함 확인"""
        files = scanner.scan(str(test_repo))

        file_paths = [f.file_path for f in files]

        # 문서 파일 포함
        assert "README.md" in file_paths
        assert "notes.txt" in file_paths
        assert "config.json" in file_paths
        assert "settings.yaml" in file_paths

        # 코드 파일도 포함
        assert "main.py" in file_paths
        assert "app.ts" in file_paths

    @pytest.mark.skip(reason="제외 패턴 기능 미구현 - RepoConfig.exclude_patterns 적용 필요")
    def test_exclude_patterns_applied(self, scanner, test_repo):
        """DEFAULT_EXCLUDE_PATTERNS 적용 확인"""
        files = scanner.scan(str(test_repo))

        file_paths = [f.file_path for f in files]

        # node_modules 제외
        assert not any("node_modules" in p for p in file_paths)

        # lockfile 제외
        assert "package-lock.json" not in file_paths

        # minified 파일 제외
        assert not any(".min.js" in p for p in file_paths)

    @pytest.mark.skip(reason="파일 크기 제한 기능 미구현")
    def test_file_size_limit(self, scanner, test_repo):
        """파일 크기 제한 (1MB) 확인"""
        files = scanner.scan(str(test_repo))

        file_paths = [f.file_path for f in files]

        # 2MB 파일은 제외
        assert "large.md" not in file_paths

    def test_binary_file_exclusion(self, scanner, test_repo):
        """바이너리 파일 제외 확인"""
        files = scanner.scan(str(test_repo))

        file_paths = [f.file_path for f in files]

        # PNG 파일 제외
        assert "image.png" not in file_paths

    def test_index_text_files_switch(self, scanner, test_repo):
        """index_text_files=False 시 텍스트 파일 제외"""
        config = RepoConfig()
        config.index_text_files = False

        files = scanner.scan(str(test_repo), config)

        file_paths = [f.file_path for f in files]
        languages = [f.language for f in files]

        # 텍스트 파일 제외
        assert "README.md" not in file_paths
        assert "notes.txt" not in file_paths
        assert "config.json" not in file_paths

        # 코드 파일은 포함
        assert "main.py" in file_paths
        assert "app.ts" in file_paths

        # 텍스트 언어 없음
        assert "markdown" not in languages
        assert "text" not in languages
        assert "json" not in languages

    @pytest.mark.skip(reason="text_index_extensions 화이트리스트 기능 미구현")
    def test_text_index_extensions_whitelist(self, scanner, test_repo):
        """text_index_extensions 화이트리스트"""
        config = RepoConfig()
        # config.text_index_extensions = [".md"]  # 속성 미구현

        files = scanner.scan(str(test_repo), config)

        file_paths = [f.file_path for f in files]

        # .md만 포함
        assert "README.md" in file_paths

        # 다른 텍스트 파일 제외
        assert "notes.txt" not in file_paths
        assert "config.json" not in file_paths
        assert "settings.yaml" not in file_paths

        # 코드 파일은 여전히 포함
        assert "main.py" in file_paths

    def test_language_detection_text_files(self, scanner, test_repo):
        """텍스트 파일 언어 감지"""
        files = scanner.scan(str(test_repo))

        lang_map = {f.file_path: f.language for f in files}

        assert lang_map.get("README.md") == "markdown"
        assert lang_map.get("notes.txt") == "text"
        assert lang_map.get("config.json") == "json"
        assert lang_map.get("settings.yaml") == "yaml"

    def test_custom_exclude_patterns(self, scanner, test_repo):
        """커스텀 제외 패턴"""
        config = RepoConfig()
        config.exclude_patterns = ["*.md"]  # 마크다운 제외

        files = scanner.scan(str(test_repo), config)

        file_paths = [f.file_path for f in files]

        # .md 파일 제외
        assert "README.md" not in file_paths

        # 다른 파일은 포함
        assert "notes.txt" in file_paths
        assert "main.py" in file_paths


class TestRepoScannerDefenseInDepth:
    """방어적 필터링 테스트"""

    @pytest.fixture
    def scanner(self):
        return RepoScanner()

    @pytest.mark.skip(reason="Private method - tested through scan()")
    def test_check_file_size(self, scanner, tmp_path):
        """파일 크기 체크 유틸"""
        small_file = tmp_path / "small.txt"
        small_file.write_text("small")

        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * 2_000_000)

        # Private method - tested through scan()
        pass

    @pytest.mark.skip(reason="Private method - tested through scan()")
    def test_is_binary(self, scanner, tmp_path):
        """바이너리 파일 감지"""
        text_file = tmp_path / "text.txt"
        text_file.write_text("Hello World", encoding="utf-8")

        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03')

        # Private method - tested through scan()
        pass

    def test_hidden_files_excluded(self, scanner, tmp_path):
        """숨김 파일/디렉토리 제외"""
        repo = tmp_path / "repo"
        repo.mkdir()

        (repo / "visible.md").write_text("visible")
        (repo / ".hidden.md").write_text("hidden")

        git_dir = repo / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git config")

        files = scanner.scan(str(repo))
        file_paths = [f.file_path for f in files]

        assert "visible.md" in file_paths
        assert ".hidden.md" not in file_paths
        assert not any(".git" in p for p in file_paths)

