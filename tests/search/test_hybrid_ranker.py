"""HybridRanker 테스트"""

import pytest

from src.core.models import Candidate
from src.search.adapters.ranking.hybrid_ranker import HybridRanker
from src.search.query_classifier import QueryType


class TestHybridRanker:
    def setup_method(self):
        self.ranker = HybridRanker(debug_mode=True)

    def test_basic_ranking(self):
        """기본 랭킹 테스트"""
        candidates = [
            Candidate(
                repo_id="test_repo",
                chunk_id="chunk1",
                file_path="routes/api.py",
                span=(10, 0, 20, 0),
                features={
                    "lexical": 0.8,
                    "semantic_small_code": 0.7,
                    "semantic_small_node": 0.6,
                },
                metadata={"importance_score": 0.9},
            ),
            Candidate(
                repo_id="test_repo",
                chunk_id="chunk2",
                file_path="utils/helper.py",
                span=(5, 0, 15, 0),
                features={
                    "lexical": 0.5,
                    "semantic_small_code": 0.6,
                    "semantic_small_node": 0.4,
                },
                metadata={"importance_score": 0.3},
            ),
        ]

        ranked = self.ranker.rank("API 검색", candidates, max_items=2)

        assert len(ranked) == 2
        assert ranked[0].chunk_id == "chunk1"  # 더 높은 점수
        assert "final_score" in ranked[0].features
        assert "explanation" in ranked[0].metadata

    def test_api_location_weights(self):
        """API 위치 질문에 대한 가중치 테스트"""
        candidates = [
            Candidate(
                repo_id="test_repo",
                chunk_id="route_chunk",
                file_path="routes/api.py",
                span=(10, 0, 20, 0),
                features={
                    "lexical": 0.5,
                    "semantic_large_node": 0.9,  # route large (높은 가중치)
                },
                metadata={"importance_score": 0.8},
            ),
            Candidate(
                repo_id="test_repo",
                chunk_id="code_chunk",
                file_path="services/search.py",
                span=(5, 0, 15, 0),
                features={
                    "lexical": 0.9,  # lexical은 높지만 가중치 낮음
                    "semantic_small_code": 0.8,
                },
                metadata={"importance_score": 0.5},
            ),
        ]

        ranked = self.ranker.rank("POST /search 어디?", candidates, max_items=1)

        # API_LOCATION 쿼리에서는 semantic_large_node 가중치가 높아서
        # route_chunk가 우선되어야 함
        assert ranked[0].chunk_id == "route_chunk"
        assert ranked[0].metadata["explanation"]["query_type"] == "api_location"

    def test_importance_boost(self):
        """Importance boost 테스트"""
        # 동일한 base_score, 다른 importance
        candidates = [
            Candidate(
                repo_id="test_repo",
                chunk_id="important",
                file_path="api.py",
                span=(1, 0, 10, 0),
                features={"lexical": 0.5},
                metadata={"importance_score": 1.0},  # 최고 중요도
            ),
            Candidate(
                repo_id="test_repo",
                chunk_id="normal",
                file_path="util.py",
                span=(1, 0, 10, 0),
                features={"lexical": 0.5},
                metadata={"importance_score": 0.0},  # 최저 중요도
            ),
        ]

        ranked = self.ranker.rank("테스트 쿼리", candidates, max_items=2)

        # importance가 높은 것이 위로
        assert ranked[0].chunk_id == "important"
        assert ranked[0].features["importance_boost"] > ranked[1].features["importance_boost"]

    def test_llm_summary_boost(self):
        """LLM 요약 boost 테스트"""
        candidates = [
            Candidate(
                repo_id="test_repo",
                chunk_id="llm_summary",
                file_path="api.py",
                span=(1, 0, 10, 0),
                features={"lexical": 0.5},
                metadata={
                    "importance_score": 0.5,
                    "summary_method": "llm",
                },
            ),
            Candidate(
                repo_id="test_repo",
                chunk_id="template_summary",
                file_path="api.py",
                span=(11, 0, 20, 0),
                features={"lexical": 0.5},
                metadata={
                    "importance_score": 0.5,
                    "summary_method": "template",
                },
            ),
        ]

        ranked = self.ranker.rank("테스트 쿼리", candidates, max_items=2)

        # LLM 요약이 약간 높아야 함
        assert ranked[0].chunk_id == "llm_summary"
        assert ranked[0].features["summary_method_boost"] > 1.0
        assert ranked[1].features["summary_method_boost"] == 1.0

    def test_explanation_structure(self):
        """Explanation 구조 테스트"""
        candidates = [
            Candidate(
                repo_id="test_repo",
                chunk_id="test",
                file_path="test.py",
                span=(1, 0, 10, 0),
                features={
                    "lexical": 0.8,
                    "semantic_small_code": 0.7,
                    "graph": 0.5,
                },
                metadata={"importance_score": 0.6},
            ),
        ]

        ranked = self.ranker.rank("테스트", candidates, max_items=1)
        explanation = ranked[0].metadata["explanation"]

        # 필수 필드 검증
        assert "query_type" in explanation
        assert "weights" in explanation
        assert "base_score" in explanation
        assert "importance_score" in explanation
        assert "importance_boost_factor" in explanation
        assert "final_score" in explanation
        assert "top_signals" in explanation
        assert len(explanation["top_signals"]) <= 3

