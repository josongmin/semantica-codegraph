"""쿼리 타입 분류 (룰 기반)"""

import re
from enum import Enum


class QueryType(str, Enum):
    """쿼리 타입"""

    API_LOCATION = "api_location"  # "POST /search 어디?"
    LOG_LOCATION = "log_location"  # "로그 메시지 어디?"
    STRUCTURE = "structure"  # "전체 구조는?"
    FUNCTION_IMPL = "function_impl"  # "함수 구현 어떻게?"
    GENERAL = "general"  # 일반 쿼리


class QueryClassifier:
    """
    룰 기반 쿼리 타입 분류

    Phase 1: 간단한 regex 패턴
    Phase 2: 쿼리 로그 기반 튜닝
    Phase 3: ML 분류기 (LR or 작은 LLM)
    """

    def classify(self, query: str) -> QueryType:
        """
        쿼리 타입 분류

        Args:
            query: 쿼리 문자열

        Returns:
            QueryType
        """
        query_lower = query.lower()

        # 1. API 위치 질문
        # "/v1/search", "/api/users" 같은 경로 + 위치 키워드
        has_api_path = bool(re.search(r"/(v\d+/)?[\w/-]+", query))
        has_location_keyword = any(
            kw in query_lower for kw in ["어디", "where", "implementation", "정의", "defined"]
        )

        if has_api_path and has_location_keyword:
            return QueryType.API_LOCATION

        # 2. 로그 위치 질문
        # 따옴표 + 로그 키워드 (OR 조건이 아닌 AND 조건)
        # 따옴표 내부에 공백이 있어도 OK
        log_patterns = [
            r'"[^"]+"',  # "any text with spaces"
            r"'[^']+'",  # 'any text with spaces'
        ]
        has_quoted_string = any(re.search(p, query) for p in log_patterns)
        has_log_keyword = any(
            kw in query_lower for kw in ["로그", "log", "error", "exception", "warning", "출력"]
        )

        if has_quoted_string and has_log_keyword:
            return QueryType.LOG_LOCATION

        # 3. 구조 설명 질문
        # 명시적 bigram 또는 "how does system/project/architecture"
        structure_bigrams = [
            "전체 구조",
            "전체 아키텍처",
            "overall architecture",
            "시스템 구조",
            "프로젝트 구조",
            "architecture overview",
        ]
        has_structure_bigram = any(bigram in query_lower for bigram in structure_bigrams)

        # "how does system/project/architecture" 패턴
        has_how_does = "how does" in query_lower
        has_system_keyword = any(
            kw in query_lower for kw in ["system", "project", "architecture", "시스템", "프로젝트"]
        )

        if has_structure_bigram or (has_how_does and has_system_keyword):
            return QueryType.STRUCTURE

        # 4. 함수 구현 질문
        # "how does X work" (X가 system/project가 아닌 경우)
        impl_keywords = ["구현", "implementation", "어떻게 동작", "how to"]
        has_impl_keyword = any(kw in query_lower for kw in impl_keywords)

        if has_impl_keyword or has_how_does:
            return QueryType.FUNCTION_IMPL

        return QueryType.GENERAL
