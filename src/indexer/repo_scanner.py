"""저장소 파일 스캐너"""

import fnmatch
import logging
import os
from pathlib import Path

from ..core.models import FileMetadata, RepoConfig

logger = logging.getLogger(__name__)


class RepoScanner:
    """
    저장소 파일 스캐너

    역할:
    - 저장소 내 파일 목록 스캔
    - 언어 감지 (확장자 기반)
    - 제외 패턴 적용
    """

    # 지원 언어 (확장자 매핑)
    LANGUAGE_EXTENSIONS = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        # 텍스트/문서 파일
        ".md": "markdown",
        ".txt": "text",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
    }

    def scan(
        self,
        root_path: str,
        config: RepoConfig | None = None
    ) -> list[FileMetadata]:
        """
        저장소 스캔

        Args:
            root_path: 저장소 루트 경로
            config: 저장소 설정 (None이면 기본값)

        Returns:
            FileMetadata 리스트
        """
        if config is None:
            config = RepoConfig()

        root = Path(root_path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Repository path not found: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        logger.info(f"Scanning repository: {root}")

        files = []
        for file_path in self._walk_directory(root):
            # 상대 경로
            rel_path = file_path.relative_to(root)
            rel_path_str = str(rel_path)

            # 제외 패턴 체크
            if self._should_exclude(rel_path_str, config.exclude_patterns):
                logger.debug(f"Excluded: {rel_path_str}")
                continue

            # 언어 감지
            language = self._detect_language(file_path)
            if language is None:
                logger.debug(f"Unknown language: {rel_path_str}")
                continue

            # 언어 필터 적용
            if config.languages and language not in config.languages:
                logger.debug(f"Filtered out {language}: {rel_path_str}")
                continue

            # 텍스트 파일 필터 (index_text_files=False일 때)
            text_file_languages = {"markdown", "text", "json", "yaml", "toml", "xml"}
            if not config.index_text_files and language in text_file_languages:
                logger.debug(f"Text file excluded (index_text_files=False): {rel_path_str}")
                continue

            # FileMetadata 생성
            files.append(
                FileMetadata(
                    file_path=rel_path_str,
                    abs_path=str(file_path),
                    language=language
                )
            )

        logger.info(f"Found {len(files)} files")
        return sorted(files, key=lambda f: f.file_path)

    def _walk_directory(self, root: Path):
        """디렉토리 재귀 탐색"""
        for dirpath, dirnames, filenames in os.walk(root):
            # 숨김 디렉토리 제외
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            for filename in filenames:
                # 숨김 파일 제외
                if filename.startswith("."):
                    continue

                yield Path(dirpath) / filename

    def _should_exclude(self, file_path: str, patterns: list[str]) -> bool:
        """제외 패턴 매칭"""
        for pattern in patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
            # Unix 스타일 패턴 지원
            if fnmatch.fnmatch(f"**/{file_path}", pattern):
                return True
        return False

    def _detect_language(self, file_path: Path) -> str | None:
        """확장자 기반 언어 감지"""
        suffix = file_path.suffix.lower()
        return self.LANGUAGE_EXTENSIONS.get(suffix)

