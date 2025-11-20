"""하이브리드 랭커 (Query Type + Importance Boost)"""

import logging
from typing import Any

from ...query_classifier import QueryClassifier, QueryType
from ....core.models import Candidate

logger = logging.getLogger(__name__)


# Query Type별 가중치 설정
QUERY_TYPE_WEIGHTS = {
    QueryType.API_LOCATION: {
        "lexical": 0.1,
        "semantic_small_code": 0.1,
        "semantic_small_node": 0.2,  # route small
        "semantic_large_node": 0.5,  # route large (Phase 2)
        "graph": 0.1,
        "fuzzy": 0.0,
    },
    QueryType.LOG_LOCATION: {
        "lexical": 0.6,  # BM25 최우선
        "semantic_small_code": 0.2,
        "semantic_small_node": 0.1,
        "semantic_large_node": 0.0,
        "graph": 0.1,
        "fuzzy": 0.0,
    },
    QueryType.STRUCTURE: {
        "lexical": 0.1,
        "semantic_small_code": 0.1,
        "semantic_small_node": 0.2,
        "semantic_large_node": 0.4,  # doc/symbol large (Phase 2)
        "graph": 0.2,  # 관계 중요
        "fuzzy": 0.0,
    },
    QueryType.FUNCTION_IMPL: {
        "lexical": 0.2,
        "semantic_small_code": 0.3,
        "semantic_small_node": 0.3,
        "semantic_large_node": 0.1,
        "graph": 0.1,
        "fuzzy": 0.0,
    },
    QueryType.GENERAL: {
        "lexical": 0.3,
        "semantic_small_code": 0.3,
        "semantic_small_node": 0.2,
        "semantic_large_node": 0.1,
        "graph": 0.1,
        "fuzzy": 0.0,
    },
}


class HybridRanker:
    """
    하이브리드 랭커
    
    Phase 1 기능:
    - Query type별 가중치 차별화
    - Importance boost (보수적: 최대 10%)
    - Summary method boost (LLM 요약 3%)
    - Explainability (debug 모드)
    
    Phase 2 기능:
    - Learning to Rank (LightGBM)
    - 쿼리 로그 기반 weight 튜닝
    """

    # 보수적 boost 설정
    IMPORTANCE_BOOST_FACTOR = 0.1  # 최대 10% 증폭
    SUMMARY_LLM_BOOST = 1.03  # LLM 요약 3% 보너스

    def __init__(self, debug_mode: bool = False):
        """
        Args:
            debug_mode: 디버그 모드 (explanation 생성)
        """
        self.debug_mode = debug_mode
        self.classifier = QueryClassifier()

    def rank(
        self,
        query: str,
        candidates: list[Candidate],
        max_items: int = 10,
    ) -> list[Candidate]:
        """
        후보 리스트 랭킹
        
        Args:
            query: 쿼리 문자열
            candidates: Candidate 리스트
            max_items: 반환할 최대 항목 수
        
        Returns:
            랭킹된 Candidate 리스트
        """
        if not candidates:
            return []

        # 1. 쿼리 타입 분류
        query_type = self.classifier.classify(query)
        weights = QUERY_TYPE_WEIGHTS[query_type]

        logger.info(f"Query: '{query[:50]}...', Type: {query_type.value}")
        logger.debug(f"Weights: {weights}")

        # 2. 최종 점수 계산
        for candidate in candidates:
            # 2-1. 기본 가중치 합산
            base_score = self._calculate_base_score(candidate, weights)

            # 2-2. Importance boost (보수적)
            importance = candidate.metadata.get("importance_score", 0.5)
            importance_boost = 1.0 + self.IMPORTANCE_BOOST_FACTOR * importance

            # 2-3. Summary method boost (LLM 요약)
            summary_method_boost = 1.0
            if candidate.metadata.get("summary_method") == "llm":
                summary_method_boost = self.SUMMARY_LLM_BOOST

            # 2-4. 최종 점수
            final_score = base_score * importance_boost * summary_method_boost

            # 2-5. Candidate에 저장
            candidate.features["final_score"] = final_score
            candidate.features["base_score"] = base_score
            candidate.features["importance_boost"] = importance_boost
            candidate.features["summary_method_boost"] = summary_method_boost

            # 2-6. Explanation (debug 모드)
            if self.debug_mode:
                candidate.metadata["explanation"] = self._build_explanation(
                    query_type, weights, candidate, base_score, importance_boost, summary_method_boost
                )

        # 3. 정렬
        ranked = sorted(
            candidates, key=lambda c: c.features.get("final_score", 0.0), reverse=True
        )

        # 4. 상위 max_items개 반환
        result = ranked[:max_items]

        logger.info(
            f"Ranked {len(candidates)} candidates, returning top {len(result)} "
            f"(top score: {result[0].features['final_score']:.3f})"
        )

        return result

    def _calculate_base_score(self, candidate: Candidate, weights: dict[str, float]) -> float:
        """기본 가중치 합산"""
        base_score = 0.0

        for signal_name, weight in weights.items():
            if weight <= 0:
                continue

            signal_score = candidate.features.get(signal_name, 0.0)
            base_score += signal_score * weight

        return base_score

    def _build_explanation(
        self,
        query_type: QueryType,
        weights: dict[str, float],
        candidate: Candidate,
        base_score: float,
        importance_boost: float,
        summary_method_boost: float,
    ) -> dict[str, Any]:
        """Explanation 생성 (디버그용)"""
        # 상위 3개 시그널
        signals = [
            (name, candidate.features.get(name, 0.0), weights.get(name, 0.0))
            for name in weights.keys()
        ]
        top_signals = sorted(signals, key=lambda x: x[1] * x[2], reverse=True)[:3]

        return {
            "query_type": query_type.value,
            "weights": weights,
            "base_score": round(base_score, 4),
            "importance_score": candidate.metadata.get("importance_score", 0.5),
            "importance_boost_factor": round(importance_boost, 4),
            "summary_method": candidate.metadata.get("summary_method"),
            "summary_method_boost": round(summary_method_boost, 4),
            "final_score": round(candidate.features["final_score"], 4),
            "top_signals": [
                {
                    "name": name,
                    "raw_score": round(score, 4),
                    "weight": weight,
                    "contribution": round(score * weight, 4),
                }
                for name, score, weight in top_signals
            ],
        }

    def update_weights(self, query_type: QueryType, new_weights: dict[str, float]) -> None:
        """가중치 업데이트 (A/B 테스트용)"""
        QUERY_TYPE_WEIGHTS[query_type] = new_weights
        logger.info(f"Updated weights for {query_type.value}: {new_weights}")

