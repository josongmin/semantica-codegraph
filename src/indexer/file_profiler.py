"""파일 프로파일링 - 파일 역할 태깅"""

import logging
import re
from pathlib import Path

from ..core.models import FileProfile, RepoId

logger = logging.getLogger(__name__)


class FileProfiler:
    """
    파일 역할 분석 및 프로파일 생성

    역할:
    - 파일 타입 태깅 (API, Service, Model, Test 등)
    - API 프레임워크 감지
    - Endpoint 정보 추출
    - Import 분석
    """

    # API 관련 패턴
    API_PATTERNS = {
        "fastapi": {
            "decorators": [
                r"@router\.(get|post|put|delete|patch)",
                r"@app\.(get|post|put|delete|patch)",
            ],
            "imports": ["from fastapi import", "import fastapi", "from fastapi.routing import"],
        },
        "django": {
            "decorators": [r"@api_view", r"class.*APIView", r"class.*ViewSet"],
            "imports": ["from django.views", "from rest_framework"],
        },
        "flask": {
            "decorators": [r"@app\.route", r"@blueprint\.route"],
            "imports": ["from flask import", "import flask"],
        },
        "express": {
            "patterns": [r"app\.(get|post|put|delete)", r"router\.(get|post|put|delete)"],
            "imports": ["require('express')", "import express", "from 'express'"],
        },
        "spring": {
            "decorators": [
                r"@(Get|Post|Put|Delete)Mapping",
                r"@RestController",
                r"@RequestMapping",
            ],
            "imports": ["import org.springframework.web"],
        },
    }

    # 파일 역할 키워드
    ROLE_KEYWORDS = {
        "api": ["api", "routes", "router", "endpoints", "handlers", "controllers"],
        "service": ["service", "business", "logic", "use_case", "usecase"],
        "model": ["model", "schema", "entity", "dto", "data"],
        "config": ["config", "configuration", "settings", "env"],
        "test": ["test", "tests", "__tests__", "spec", ".test.", ".spec."],
    }

    def profile_file(
        self, repo_id: RepoId, file_path: str, abs_path: str, framework: str | None = None
    ) -> FileProfile:
        """
        단일 파일 프로파일 생성

        Args:
            repo_id: 저장소 ID
            file_path: 상대 경로
            abs_path: 절대 경로
            framework: 저장소 프레임워크 (RepoProfile에서 전달)

        Returns:
            FileProfile
        """
        profile = FileProfile(repo_id=repo_id, file_path=file_path)

        try:
            # 파일 읽기
            content = Path(abs_path).read_text(errors="ignore")
            lines = content.splitlines()
            profile.line_count = len(lines)

            # 1. 경로 기반 역할 태깅
            self._tag_by_path(profile, file_path)

            # 2. 내용 기반 역할 태깅
            self._tag_by_content(profile, content, framework)

            # 3. Import 분석
            profile.imports = self._extract_imports(content)
            profile.external_deps, profile.internal_deps = self._classify_imports(profile.imports)

            # 4. API 프레임워크 감지 (API 파일인 경우)
            if profile.is_api_file or profile.is_router:
                profile.api_framework = self._detect_file_framework(content)
                profile.api_patterns = self._detect_file_api_patterns(content)
                profile.endpoints = self._extract_endpoints(content, profile.api_framework)

            # 5. 함수/클래스 개수 (간단한 정규식)
            profile.function_count = len(
                re.findall(r"^\s*(def|function|async def)\s+\w+", content, re.MULTILINE)
            )
            profile.class_count = len(re.findall(r"^\s*class\s+\w+", content, re.MULTILINE))

        except Exception as e:
            logger.warning(f"파일 프로파일링 실패: {file_path}, {e}")

        return profile

    def _tag_by_path(self, profile: FileProfile, file_path: str):
        """파일 경로 기반 역할 태깅"""

        path_lower = file_path.lower()

        # 테스트 파일
        if any(kw in path_lower for kw in self.ROLE_KEYWORDS["test"]):
            profile.is_test_file = True
            return  # 테스트면 다른 태그 안붙임

        # API 파일
        if any(kw in path_lower for kw in self.ROLE_KEYWORDS["api"]):
            profile.is_api_file = True
            if "route" in path_lower or "router" in path_lower:
                profile.is_router = True
            if "controller" in path_lower:
                profile.is_controller = True

        # 서비스 파일
        if any(kw in path_lower for kw in self.ROLE_KEYWORDS["service"]):
            profile.is_service = True

        # 모델 파일
        if any(kw in path_lower for kw in self.ROLE_KEYWORDS["model"]):
            profile.is_model = True
            if "schema" in path_lower:
                profile.is_schema = True

        # 설정 파일
        if any(kw in path_lower for kw in self.ROLE_KEYWORDS["config"]):
            profile.is_config = True

        # 엔트리포인트 (파일명 기반)
        filename = Path(file_path).name
        if filename in ["main.py", "app.py", "server.py", "index.ts", "index.js", "app.ts"]:
            profile.is_entry_point = True

    def _tag_by_content(self, profile: FileProfile, content: str, framework: str | None):
        """파일 내용 기반 역할 태깅"""

        # 이미 경로로 태깅되었으면 스킵
        if profile.is_test_file:
            return

        # FastAPI 패턴
        if framework == "fastapi" or "from fastapi import" in content:
            if re.search(r"@(router|app)\.(get|post|put|delete)", content):
                profile.is_api_file = True
                profile.is_router = True

        # Django 패턴
        elif (framework == "django" or "from django" in content) and (
            "@api_view" in content or "APIView" in content
        ):
            profile.is_api_file = True

        # Express 패턴
        elif (framework == "express" or "express" in content) and re.search(
            r"(app|router)\.(get|post|put|delete)\(", content
        ):
            profile.is_api_file = True
            profile.is_router = True

        # Pydantic 모델
        if "BaseModel" in content and "from pydantic import" in content:
            profile.is_schema = True
            profile.is_model = True

    def _extract_imports(self, content: str) -> list[str]:
        """Import 문 추출"""

        imports = []

        # Python import
        for match in re.finditer(r"^(?:from|import)\s+([\w.]+)", content, re.MULTILINE):
            imports.append(match.group(1))

        # TypeScript/JavaScript import
        for match in re.finditer(r"import.*from\s+['\"]([^'\"]+)['\"]", content):
            imports.append(match.group(1))

        return list(set(imports))[:50]  # 최대 50개

    def _classify_imports(self, imports: list[str]) -> tuple[list[str], list[str]]:
        """Import를 외부/내부로 분류"""

        external = []
        internal = []

        for imp in imports:
            # 상대 import는 내부
            if imp.startswith("."):
                internal.append(imp)
            # 외부 라이브러리 (일반적인 패키지명)
            elif not imp.startswith("src/") and not imp.startswith("apps/"):
                external.append(imp)
            else:
                internal.append(imp)

        return external[:30], internal[:30]  # 각각 최대 30개

    def _detect_file_framework(self, content: str) -> str | None:
        """파일의 API 프레임워크 감지"""

        for framework, patterns in self.API_PATTERNS.items():
            # Import 확인
            if "imports" in patterns and any(imp in content for imp in patterns["imports"]):
                return framework

        return None

    def _detect_file_api_patterns(self, content: str) -> list[str]:
        """파일의 API 패턴 감지"""

        patterns = []

        # Decorator 패턴
        for match in re.finditer(r"@(router|app)\.\w+", content):
            pattern = match.group(0)
            if pattern not in patterns:
                patterns.append(pattern)

        return patterns[:20]  # 최대 20개

    def _extract_endpoints(self, content: str, framework: str | None) -> list[dict]:
        """엔드포인트 정보 추출"""

        if not framework:
            return []

        endpoints = []

        # FastAPI
        if framework == "fastapi":
            # @router.post("/search")
            for match in re.finditer(
                r'@(?:router|app)\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']\)', content
            ):
                method = match.group(1).upper()
                path = match.group(2)
                endpoints.append({"method": method, "path": path})

        # Express
        elif framework == "express":
            # router.post('/search', ...)
            for match in re.finditer(
                r'(?:app|router)\.(get|post|put|delete)\(["\']([^"\']+)["\']', content
            ):
                method = match.group(1).upper()
                path = match.group(2)
                endpoints.append({"method": method, "path": path})

        # Django
        elif framework == "django":
            # url(r'^api/search/', ...)
            for match in re.finditer(r'(?:path|url)\(["\']([^"\']+)["\']', content):
                path = match.group(1)
                endpoints.append({"method": "ANY", "path": path})

        return endpoints[:50]  # 최대 50개
