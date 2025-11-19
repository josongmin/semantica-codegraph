"""검색 결과 랭킹"""

import logging

from src.core.models import Candidate

logger = logging.getLogger(__name__)


class Ranker:
    """
    검색 결과 랭킹

    역할:
    - 다중 점수 기반 랭킹
    - 가중치 적용
    - 최종 정렬 및 필터링
    """

    def __init__(self, feature_weights: dict[str, float] | None = None):
        """
        Args:
            feature_weights: 특성별 가중치
                예: {
                    "lexical_score": 0.3,
                    "semantic_score": 0.5,
                    "graph_score": 0.2,
                    "recency_score": 0.0,
                    "popularity_score": 0.0
                }
        """
        if feature_weights is None:
            feature_weights = {
                "lexical_score": 0.3,
                "semantic_score": 0.5,
                "graph_score": 0.2,
                "recency_score": 0.0,  # 추후 구현
                "popularity_score": 0.0,  # 추후 구현
            }

        self.feature_weights = feature_weights

    def rank(
        self,
        candidates: list[Candidate],
        max_items: int,
    ) -> list[Candidate]:
        """
        후보 리스트 랭킹 및 필터링

        Args:
            candidates: Candidate 리스트
            max_items: 반환할 최대 항목 수

        Returns:
            랭킹된 Candidate 리스트 (상위 max_items개)
        """
        if not candidates:
            return []

        logger.debug(f"Ranking {len(candidates)} candidates")

        # 1. 각 candidate의 최종 점수 계산
        for candidate in candidates:
            final_score = self._calculate_final_score(candidate)
            candidate.features["final_score"] = final_score

        # 2. 점수 기준 정렬
        ranked = sorted(candidates, key=lambda c: c.features.get("final_score", 0.0), reverse=True)

        # 3. 상위 max_items개 반환
        result = ranked[:max_items]

        logger.debug(f"Ranked top {len(result)} candidates")
        return result

    def _calculate_final_score(self, candidate: Candidate) -> float:
        """
        최종 점수 계산

        최종 점수 = Σ (feature_score * weight)
        """
        final_score = 0.0

        for feature_name, weight in self.feature_weights.items():
            if weight <= 0:
                continue

            feature_score = candidate.features.get(feature_name, 0.0)
            final_score += feature_score * weight

        return final_score

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """가중치 업데이트"""
        self.feature_weights.update(new_weights)
        logger.info(f"Updated weights: {self.feature_weights}")
