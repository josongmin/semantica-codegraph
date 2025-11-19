"""저장소 프로파일링 - 프로젝트 구조 분석"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from ..core.models import RepoId, RepoProfile

logger = logging.getLogger(__name__)


class RepoProfiler:
    """
    저장소 구조 분석 및 프로파일 생성
    
    역할:
    - 프로젝트 타입/프레임워크 감지
    - 디렉토리 역할 분류
    - 엔트리포인트 찾기
    - API 패턴 감지
    """
    
    # 프레임워크 감지 패턴
    FRAMEWORK_PATTERNS = {
        "fastapi": ["from fastapi import", "import fastapi", "FastAPI("],
        "django": ["from django", "import django", "DJANGO_SETTINGS_MODULE"],
        "flask": ["from flask import", "import flask", "Flask(__name__)"],
        "express": ["express()", "require('express')", "import express"],
        "nestjs": ["@nestjs/", "NestFactory.create"],
        "spring": ["@SpringBootApplication", "import org.springframework"],
        "nextjs": ["next/", "import next"],
        "react": ["import React", "from 'react'"],
    }
    
    # API 패턴 감지
    API_PATTERNS = {
        "fastapi": ["@router.get", "@router.post", "@router.put", "@router.delete", "@app.get", "@app.post"],
        "express": ["app.get(", "app.post(", "router.get(", "router.post("],
        "django": ["def get(", "def post(", "def put(", "def delete(", "@api_view"],
        "spring": ["@GetMapping", "@PostMapping", "@PutMapping", "@DeleteMapping", "@RestController"],
    }
    
    # 디렉토리 역할 키워드
    DIR_KEYWORDS = {
        "api": ["api", "routes", "endpoints", "controllers", "handlers", "views"],
        "service": ["services", "business", "logic", "core", "domain"],
        "model": ["models", "schemas", "entities", "data", "dto"],
        "test": ["test", "tests", "__tests__", "spec"],
        "config": ["config", "configuration", "settings", "env"],
    }
    
    def profile_repo(self, repo_root: str, repo_id: RepoId) -> RepoProfile:
        """
        저장소 전체 프로파일 생성
        
        Args:
            repo_root: 저장소 루트 경로
            repo_id: 저장소 ID
        
        Returns:
            RepoProfile
        """
        logger.info(f"[RepoProfiler] 프로파일링 시작: {repo_id}")
        
        repo_path = Path(repo_root)
        profile = RepoProfile(repo_id=repo_id)
        
        # 1. 언어 분포 분석
        profile.languages, profile.primary_language = self._analyze_languages(repo_path)
        
        # 2. 의존성 파일 분석 (pyproject.toml, package.json 등)
        profile.dependencies = self._extract_dependencies(repo_path)
        
        # 3. 프레임워크 감지
        profile.framework, profile.frameworks = self._detect_frameworks(repo_path, profile.dependencies)
        
        # 4. API 패턴 감지
        profile.api_patterns = self._detect_api_patterns(repo_path, profile.framework)
        
        # 5. 디렉토리 분류
        directories = self._classify_directories(repo_path)
        profile.api_directories = directories["api"]
        profile.service_directories = directories["service"]
        profile.model_directories = directories["model"]
        profile.test_directories = directories["test"]
        profile.config_directories = directories["config"]
        
        # 6. 엔트리포인트 찾기
        profile.entry_points = self._find_entry_points(repo_path, profile.framework)
        
        # 7. 프로젝트 타입 추론
        profile.project_type = self._infer_project_type(profile)
        
        # 8. 파일 트리 생성 (간략 버전)
        profile.file_tree = self._build_simplified_tree(repo_path)
        profile.total_directories = len(list(repo_path.rglob("*/")))
        
        logger.info(f"[RepoProfiler] 완료: framework={profile.framework}, type={profile.project_type}")
        return profile
    
    def _analyze_languages(self, repo_path: Path) -> tuple[dict[str, int], str]:
        """언어 분포 분석"""
        
        lang_lines = defaultdict(int)
        
        # 확장자 → 언어 매핑
        ext_to_lang = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
        }
        
        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix
                if ext in ext_to_lang:
                    try:
                        lines = len(file_path.read_text(errors="ignore").splitlines())
                        lang_lines[ext_to_lang[ext]] += lines
                    except Exception:
                        pass
        
        # 가장 많은 언어
        primary = max(lang_lines.items(), key=lambda x: x[1])[0] if lang_lines else "unknown"
        
        return dict(lang_lines), primary
    
    def _extract_dependencies(self, repo_path: Path) -> dict[str, str]:
        """의존성 추출"""
        
        deps = {}
        
        # Python: pyproject.toml
        pyproject = repo_path / "pyproject.toml"
        if pyproject.exists():
            try:
                # Python 3.11+ tomllib, 이전 버전은 tomli
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib  # type: ignore
                
                content = tomllib.loads(pyproject.read_text())
                if "project" in content and "dependencies" in content["project"]:
                    for dep in content["project"]["dependencies"]:
                        # "fastapi>=0.104.0" → {"fastapi": "0.104.0"}
                        match = re.match(r"([a-zA-Z0-9_-]+)(?:[>=<~]+(.+))?", dep)
                        if match:
                            name, version = match.groups()
                            deps[name.lower()] = version or "*"
            except Exception as e:
                logger.warning(f"pyproject.toml 파싱 실패: {e}")
        
        # Python: requirements.txt
        requirements = repo_path / "requirements.txt"
        if requirements.exists():
            try:
                for line in requirements.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        match = re.match(r"([a-zA-Z0-9_-]+)(?:[>=<~]+(.+))?", line)
                        if match:
                            name, version = match.groups()
                            deps[name.lower()] = version or "*"
            except Exception:
                pass
        
        # Node.js: package.json
        package_json = repo_path / "package.json"
        if package_json.exists():
            try:
                content = json.loads(package_json.read_text())
                for key in ["dependencies", "devDependencies"]:
                    if key in content:
                        deps.update(content[key])
            except Exception:
                pass
        
        return deps
    
    def _detect_frameworks(self, repo_path: Path, dependencies: dict) -> tuple[str | None, list[str]]:
        """프레임워크 감지"""
        
        detected = []
        
        # 1. 의존성 기반 감지
        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            if framework in dependencies:
                detected.append(framework)
        
        # 2. 코드 패턴 기반 감지
        if not detected:
            sample_files = list(repo_path.rglob("*.py"))[:20]  # 샘플링
            sample_files += list(repo_path.rglob("*.ts"))[:20]
            sample_files += list(repo_path.rglob("*.js"))[:20]
            
            for file_path in sample_files:
                try:
                    content = file_path.read_text(errors="ignore")
                    for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                        if any(pattern in content for pattern in patterns):
                            if framework not in detected:
                                detected.append(framework)
                except Exception:
                    pass
        
        # 주 프레임워크 (우선순위)
        priority = ["fastapi", "django", "flask", "express", "spring", "nestjs"]
        primary = next((f for f in priority if f in detected), detected[0] if detected else None)
        
        return primary, detected
    
    def _detect_api_patterns(self, repo_path: Path, framework: str | None) -> list[str]:
        """API 패턴 감지"""
        
        if not framework:
            return []
        
        patterns = self.API_PATTERNS.get(framework, [])
        detected = []
        
        # 샘플 파일에서 패턴 확인
        api_dirs = [d for d in repo_path.rglob("*") if d.is_dir() and any(kw in d.name.lower() for kw in ["api", "routes", "controllers"])]
        
        sample_files = []
        for api_dir in api_dirs[:5]:
            sample_files.extend(list(api_dir.rglob("*.py"))[:10])
            sample_files.extend(list(api_dir.rglob("*.ts"))[:10])
            sample_files.extend(list(api_dir.rglob("*.js"))[:10])
        
        for file_path in sample_files[:50]:  # 최대 50개 샘플
            try:
                content = file_path.read_text(errors="ignore")
                for pattern in patterns:
                    if pattern in content and pattern not in detected:
                        detected.append(pattern)
            except Exception:
                pass
        
        return detected
    
    def _classify_directories(self, repo_path: Path) -> dict[str, list[str]]:
        """디렉토리 역할 분류"""
        
        result = {
            "api": [],
            "service": [],
            "model": [],
            "test": [],
            "config": [],
        }
        
        for dir_path in repo_path.rglob("*"):
            if not dir_path.is_dir():
                continue
            
            # 상대 경로
            rel_path = str(dir_path.relative_to(repo_path))
            dir_name_lower = dir_path.name.lower()
            
            # 숨김 폴더/node_modules 제외
            if dir_name_lower.startswith(".") or "node_modules" in rel_path or "__pycache__" in rel_path:
                continue
            
            # 키워드 매칭
            for category, keywords in self.DIR_KEYWORDS.items():
                if any(kw in dir_name_lower or kw in rel_path.lower() for kw in keywords):
                    result[category].append(rel_path + "/")
                    break
        
        return result
    
    def _find_entry_points(self, repo_path: Path, framework: str | None) -> list[str]:
        """엔트리포인트 찾기"""
        
        entry_points = []
        
        # 일반적인 엔트리포인트 파일명
        common_names = ["main.py", "app.py", "server.py", "index.ts", "index.js", "app.ts", "server.ts"]
        
        for name in common_names:
            for file_path in repo_path.rglob(name):
                # 테스트 폴더 제외
                if "test" not in str(file_path):
                    entry_points.append(str(file_path.relative_to(repo_path)))
        
        # FastAPI 특화
        if framework == "fastapi":
            for file_path in repo_path.rglob("*.py"):
                try:
                    content = file_path.read_text(errors="ignore")
                    if "FastAPI(" in content and "uvicorn.run" in content:
                        entry_points.append(str(file_path.relative_to(repo_path)))
                except Exception:
                    pass
        
        return list(set(entry_points))  # 중복 제거
    
    def _infer_project_type(self, profile: RepoProfile) -> str:
        """프로젝트 타입 추론"""
        
        # API 디렉토리가 있으면 web_api
        if profile.api_directories or profile.api_patterns:
            return "web_api"
        
        # 엔트리포인트가 CLI 형태
        if any("cli" in ep or "main" in ep for ep in profile.entry_points):
            return "cli"
        
        # 테스트만 많으면 library
        if len(profile.test_directories) > 2 and not profile.api_directories:
            return "library"
        
        return "unknown"
    
    def _build_simplified_tree(self, repo_path: Path, max_depth: int = 2) -> dict:
        """간략한 파일 트리 생성 (depth 제한)"""
        
        tree = {}
        
        for dir_path in repo_path.iterdir():
            if dir_path.is_dir() and not dir_path.name.startswith("."):
                dir_name = dir_path.name
                # 1단계만
                tree[dir_name] = {
                    "type": "directory",
                    "children": [child.name for child in dir_path.iterdir() if not child.name.startswith(".")][:10]  # 최대 10개
                }
        
        return tree

