"""청크 메타데이터 태깅"""

import logging
import re

logger = logging.getLogger(__name__)


class ChunkTagger:
    """
    청크 메타데이터 태깅
    
    역할:
    - API endpoint 청크 감지
    - HTTP method/path 추출
    - 청크 타입 세밀화
    """
    
    def tag_chunk(self, content: str, file_profile: "FileProfile | None" = None) -> dict:
        """
        청크 메타데이터 생성
        
        Args:
            content: 청크 내용
            file_profile: 파일 프로파일 (optional)
        
        Returns:
            메타데이터 dict
        """
        metadata = {
            "is_api_endpoint_chunk": False,
            "is_class_definition": False,
            "is_function_definition": False,
            "is_test_case": False,
            "is_schema_definition": False,
            "has_docstring": False,
        }
        
        # 1. 기본 태그
        metadata["has_docstring"] = self._has_docstring(content)
        metadata["is_class_definition"] = self._is_class_definition(content)
        metadata["is_function_definition"] = self._is_function_definition(content)
        metadata["is_test_case"] = self._is_test_case(content)
        metadata["is_schema_definition"] = self._is_schema_definition(content)
        
        # 2. API endpoint 감지 (파일이 API 파일인 경우)
        if file_profile and file_profile.is_api_file:
            endpoint_info = self._extract_endpoint_info(content, file_profile.api_framework)
            if endpoint_info:
                metadata["is_api_endpoint_chunk"] = True
                metadata["http_method"] = endpoint_info.get("method")
                metadata["http_path"] = endpoint_info.get("path")
                metadata["api_framework"] = file_profile.api_framework
        
        # 3. 추가 메타데이터
        metadata["line_count"] = len(content.splitlines())
        metadata["has_async"] = "async def" in content or "async function" in content
        
        return metadata
    
    def _has_docstring(self, content: str) -> bool:
        """Docstring 존재 여부"""
        return '"""' in content or "'''" in content or "/**" in content
    
    def _is_class_definition(self, content: str) -> bool:
        """클래스 정의 청크인지"""
        # Python/TS/Java class 정의
        return bool(re.search(r"^\s*(?:export\s+)?class\s+\w+", content, re.MULTILINE))
    
    def _is_function_definition(self, content: str) -> bool:
        """함수 정의 청크인지"""
        # Python/TS/JS function 정의
        patterns = [
            r"^\s*(?:async\s+)?def\s+\w+",  # Python
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+\w+",  # JS/TS
            r"^\s*(?:public|private|protected)?\s+\w+\s+\w+\s*\(",  # Java
        ]
        return any(re.search(p, content, re.MULTILINE) for p in patterns)
    
    def _is_test_case(self, content: str) -> bool:
        """테스트 케이스 청크인지"""
        patterns = [
            r"def\s+test_\w+",  # pytest
            r"@pytest\.",
            r"it\(['\"]",  # jest
            r"describe\(['\"]",  # jest
            r"@Test",  # JUnit
        ]
        return any(re.search(p, content) for p in patterns)
    
    def _is_schema_definition(self, content: str) -> bool:
        """스키마 정의 청크인지"""
        patterns = [
            r"class\s+\w+\(BaseModel\)",  # Pydantic
            r"@dataclass",  # Python dataclass
            r"interface\s+\w+\s*{",  # TypeScript interface
            r"type\s+\w+\s*=",  # TypeScript type
        ]
        return any(re.search(p, content) for p in patterns)
    
    def _extract_endpoint_info(self, content: str, framework: str | None) -> dict | None:
        """엔드포인트 정보 추출"""
        
        if not framework:
            return None
        
        # FastAPI
        if framework == "fastapi":
            # @router.post("/search")
            match = re.search(r'@(?:router|app)\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']\)', content)
            if match:
                return {
                    "method": match.group(1).upper(),
                    "path": match.group(2),
                }
        
        # Express
        elif framework == "express":
            # router.post('/search', ...)
            match = re.search(r'(?:app|router)\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']', content)
            if match:
                return {
                    "method": match.group(1).upper(),
                    "path": match.group(2),
                }
        
        # Django
        elif framework == "django":
            # @api_view(['POST'])
            method_match = re.search(r"@api_view\(\[['\"](GET|POST|PUT|DELETE|PATCH)['\"]\]\)", content)
            # path는 urls.py에서 정의되므로 함수명으로 추정
            func_match = re.search(r"def\s+(\w+)\s*\(", content)
            if method_match and func_match:
                return {
                    "method": method_match.group(1),
                    "path": f"/{func_match.group(1)}/",  # 추정
                }
        
        # Spring
        elif framework == "spring":
            # @PostMapping("/search")
            match = re.search(r'@(Get|Post|Put|Delete|Patch)Mapping\(["\']([^"\']+)["\']\)', content)
            if match:
                return {
                    "method": match.group(1).upper(),
                    "path": match.group(2),
                }
        
        return None

