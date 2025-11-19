"""검색 최적화 텍스트 생성"""

import logging
import re

from ..core.models import CodeChunk, FileProfile

logger = logging.getLogger(__name__)


class SearchTextBuilder:
    """
    Chunk에 대한 검색 최적화 메타텍스트 생성
    
    목표:
    - 자연어 질문 recall 향상 (3-5배)
    - 메타데이터를 텍스트로 명시화
    - Embedding/Lexical 검색 모두 활용
    
    Format:
        [META] File: routes/hybrid.py
        [META] Role: API
        [META] Endpoint: POST /hybrid/search
        [META] Symbol: Function hybrid_search
        [META] Contains: hybrid search logic, database access
        
        {actual code}
    """
    
    def build(
        self,
        chunk: CodeChunk,
        file_profile: FileProfile | None,
        chunk_metadata: dict,
    ) -> str:
        """
        검색용 텍스트 생성
        
        Args:
            chunk: CodeChunk
            file_profile: FileProfile (optional)
            chunk_metadata: Chunk 메타데이터 (ChunkTagger 출력)
        
        Returns:
            [META] 섹션 + 실제 코드
        """
        parts = []
        
        # 1. 파일 정보
        parts.append(f"[META] File: {chunk.file_path}")
        
        # 2. 파일 역할
        if file_profile:
            roles = self._extract_roles(file_profile)
            if roles:
                parts.append(f"[META] Role: {', '.join(roles)}")
        
        # 3. 엔드포인트 정보
        if chunk_metadata.get("is_api_endpoint_chunk"):
            method = chunk_metadata.get("http_method", "")
            path = chunk_metadata.get("http_path", "")
            if method and path:
                parts.append(f"[META] Endpoint: {method} {path}")
        
        # 4. 심볼 정보
        node_kind = chunk.attrs.get("node_kind")
        node_name = chunk.attrs.get("node_name")
        if node_kind and node_name:
            parts.append(f"[META] Symbol: {node_kind} {node_name}")
        
        # 5. 기능 키워드
        features = self._extract_features(chunk.text, chunk_metadata, file_profile)
        if features:
            parts.append(f"[META] Contains: {', '.join(features)}")
        
        # 6. 테스트 여부
        if chunk_metadata.get("is_test_case"):
            parts.append("[META] Type: Test")
        
        # 7. 스키마/모델 여부
        if chunk_metadata.get("is_schema_definition"):
            parts.append("[META] Type: Schema/Model")
        
        # 8. 실제 코드
        parts.append("")  # 빈 줄
        parts.append(chunk.text)
        
        return "\n".join(parts)
    
    def _extract_roles(self, file_profile: FileProfile) -> list[str]:
        """파일 역할 추출"""
        roles = []
        
        if file_profile.is_api_file:
            roles.append("API")
        if file_profile.is_router:
            roles.append("Router")
        if file_profile.is_service:
            roles.append("Service")
        if file_profile.is_model:
            roles.append("Model")
        if file_profile.is_schema:
            roles.append("Schema")
        if file_profile.is_config:
            roles.append("Config")
        if file_profile.is_test_file:
            roles.append("Test")
        
        return roles
    
    def _extract_features(
        self,
        code: str,
        metadata: dict,
        file_profile: FileProfile | None,
    ) -> list[str]:
        """
        코드 기능 키워드 추출 (regex 기반)
        
        목적: 자연어 질문과 매칭될 수 있는 키워드 추가
        """
        features = []
        code_lower = code.lower()
        
        # 1. DB 접근
        db_keywords = ["select ", "insert ", "update ", "delete ", ".query(", ".execute(", "cursor"]
        if any(kw in code_lower for kw in db_keywords):
            features.append("database access")
        
        # 2. 인증/보안
        auth_keywords = ["auth", "token", "jwt", "session", "permission", "credential", "password"]
        if any(kw in code_lower for kw in auth_keywords):
            features.append("authentication")
        
        # 3. 검색 로직
        if "search" in code_lower:
            features.append("search logic")
        
        # 4. 벡터/임베딩
        vector_keywords = ["embedding", "vector", "similarity", "cosine"]
        if any(kw in code_lower for kw in vector_keywords):
            features.append("vector operations")
        
        # 5. HTTP 요청/클라이언트
        http_keywords = ["requests.", "httpx.", "fetch(", "axios", "urllib"]
        if any(kw in code_lower for kw in http_keywords):
            features.append("HTTP client")
        
        # 6. 파일 I/O
        file_keywords = ["open(", "read_file", "write_file", "pathlib", "os.path"]
        if any(kw in code_lower for kw in file_keywords):
            features.append("file operations")
        
        # 7. 비동기
        if metadata.get("has_async") or "async " in code_lower or "await " in code_lower:
            features.append("async")
        
        # 8. 로깅
        log_keywords = ["logger.", "log.", "logging.", "console.log"]
        if any(kw in code_lower for kw in log_keywords):
            features.append("logging")
        
        # 9. 에러 처리
        error_keywords = ["try:", "except", "catch", "throw", "raise"]
        if any(kw in code_lower for kw in error_keywords):
            features.append("error handling")
        
        # 10. 테스트
        test_keywords = ["assert", "expect", "test_", "def test", "it(", "describe("]
        if any(kw in code_lower for kw in test_keywords):
            features.append("testing")
        
        # 11. 그래프/노드
        graph_keywords = ["graph", "node", "edge", "traverse"]
        if any(kw in code_lower for kw in graph_keywords):
            features.append("graph operations")
        
        # 12. 파싱
        parse_keywords = ["parse", "parser", "ast", "tree-sitter"]
        if any(kw in code_lower for kw in parse_keywords):
            features.append("parsing")
        
        # 13. 인덱싱
        index_keywords = ["index", "indexing", "reindex"]
        if any(kw in code_lower for kw in index_keywords):
            features.append("indexing")
        
        # 14. 캐시
        cache_keywords = ["cache", "redis", "memcache"]
        if any(kw in code_lower for kw in cache_keywords):
            features.append("caching")
        
        # 15. 스트림/배치
        stream_keywords = ["stream", "batch", "queue", "async for"]
        if any(kw in code_lower for kw in stream_keywords):
            features.append("streaming")
        
        return features[:10]  # 최대 10개로 제한

