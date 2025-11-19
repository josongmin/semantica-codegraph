"""Weighted Fusion 전략

여러 검색 백엔드의 결과를 가중치로 결합하는 로직
"""

import logging
from typing import TYPE_CHECKING

from .base import FusionStrategy

if TYPE_CHECKING:
    from src.core.models import Candidate

logger = logging.getLogger(__name__)


class WeightedFusion(FusionStrategy):
    """
    Weighted Sum Fusion

    각 검색 백엔드의 정규화된 점수에 가중치를 곱해서 합산합니다.

    공식:
        final_score = Σ (normalized_score_i × weight_i)
        = lexical_score × w_lexical
          + semantic_score × w_semantic
          + graph_score × w_graph
          + fuzzy_score × w_fuzzy

    장점:
    - 간단하고 직관적
    - 가중치로 각 백엔드의 중요도 조정 가능
    - 해석이 쉬움

    단점:
    - 가중치 튜닝 필요
    - 점수 스케일 차이에 민감 (정규화 필수)

    사용 예시:
        fusion = WeightedFusion()
        result = fusion.fuse_and_sort(
            candidates=candidate_list,
            weights={"lexical": 0.25, "semantic": 0.45, "graph": 0.15, "fuzzy": 0.15},
            k=20
        )
    """

    def fuse(
        self,
        candidates: list["Candidate"],
        weights: dict[str, float] | None = None,
    ) -> list["Candidate"]:
        """
        가중 합산으로 점수 융합

        Args:
            candidates: Candidate 리스트 (각 candidate.features에 정규화된 점수가 포함됨)
            weights: 각 검색 방식의 가중치 (None이면 기본값 사용)
                예: {"lexical": 0.25, "semantic": 0.45, "graph": 0.15, "fuzzy": 0.15}

        Returns:
            최종 점수가 계산된 Candidate 리스트 (in-place 수정)
        """
        # 기본 가중치
        if weights is None:
            weights = {"lexical": 0.25, "semantic": 0.45, "graph": 0.15, "fuzzy": 0.15}
        for candidate in candidates:
            total_score = (
                candidate.features.get("lexical_score", 0) * weights.get("lexical", 0)
                + candidate.features.get("semantic_score", 0) * weights.get("semantic", 0)
                + candidate.features.get("graph_score", 0) * weights.get("graph", 0)
                + candidate.features.get("fuzzy_score", 0) * weights.get("fuzzy", 0)
            )
            candidate.features["total_score"] = total_score
            candidate.features["final_score"] = total_score

            # 디버깅용: 개별 기여도 저장
            candidate.features["weighted_contributions"] = {
                backend: candidate.features.get(f"{backend}_score", 0) * weights.get(backend, 0)
                for backend in ["lexical", "semantic", "graph", "fuzzy"]
            }

        logger.debug(f"WeightedFusion: Fused {len(candidates)} candidates with weights {weights}")
        return candidates

    # fuse_and_sort는 base class에서 구현됨
