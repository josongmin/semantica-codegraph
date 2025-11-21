"""QueryClassifier 테스트"""


from src.search.query_classifier import QueryClassifier, QueryType


class TestQueryClassifier:
    def setup_method(self):
        self.classifier = QueryClassifier()

    def test_api_location_query(self):
        """API 위치 질문 분류"""
        queries = [
            "POST /api/search 어디 정의돼?",
            "Where is /v1/users endpoint implemented?",
            "/search API가 어디있어?",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            assert result == QueryType.API_LOCATION, f"Failed for: {query}"

    def test_log_location_query(self):
        """로그 위치 질문 분류"""
        queries = [
            '"Connection failed" 로그 어디서 출력?',
            "Where is 'Authentication error' logged?",
            '"database timeout" error는 어디?',
        ]

        for query in queries:
            result = self.classifier.classify(query)
            assert result == QueryType.LOG_LOCATION, f"Failed for: {query}"

    def test_structure_query(self):
        """구조 설명 질문 분류"""
        queries = [
            "전체 구조가 어떻게 돼?",
            "Overall architecture는?",
            "How does the system work?",
            "프로젝트 구조 설명해줘",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            assert result == QueryType.STRUCTURE, f"Failed for: {query}"

    def test_function_impl_query(self):
        """함수 구현 질문 분류"""
        queries = [
            "hybrid_search 함수 구현은?",
            "How does authentication work?",
            "인증 로직 어떻게 동작해?",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            assert result == QueryType.FUNCTION_IMPL, f"Failed for: {query}"

    def test_general_query(self):
        """일반 질문 분류"""
        queries = [
            "코드베이스에서 뭐해?",
            "What is this project about?",
            "설명해줘",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            assert result == QueryType.GENERAL, f"Failed for: {query}"
