"""CombSum Fusion

정규화된 점수를 단순 합산하거나 가중 합산하는 방식입니다.
"""

import logging
from typing import TYPE_CHECKING

from .base import FusionStrategy

if TYPE_CHECKING:
    from src.core.models import Candidate

logger = logging.getLogger(__name__)


class CombSumFusion(FusionStrategy):
    """
    CombSum Fusion

    정규화된 점수를 단순 합산하거나 가중 합산하는 방식입니다.
    WeightedFusion과 유사하지만 더 단순한 로직으로 구현됩니다.

    공식 (가중치 있을 때):
        final_score = Σ (normalized_score_i × weight_i)

    공식 (가중치 없을 때):
        final_score = Σ normalized_score_i

    장점:
    - **매우 간단하고 빠름**: 최소한의 연산
    - **가중치 선택적**: 가중치 사용 여부 선택 가능
    - **해석 용이**: 직관적인 점수 합산

    단점:
    - 정규화에 의존적 (정규화가 잘못되면 결과도 나쁨)
    - WeightedFusion과 크게 다르지 않음

    사용 예시:
        # 가중치 없이 단순 합산
        fusion = CombSumFusion(use_weights=False)
        result = fusion.fuse_and_sort(candidates, weights=None, k=20)

        # 가중치 사용
        fusion = CombSumFusion(use_weights=True)
        result = fusion.fuse_and_sort(
            candidates,
            weights={"lexical": 1.0, "semantic": 1.5, "graph": 0.5, "fuzzy": 0.5},
            k=20
        )
    """

    def __init__(self, use_weights: bool = True):
        """
        Args:
            use_weights: 가중치 사용 여부
                - True: 가중치를 적용한 합산 (가중 합산)
                - False: 모든 점수를 동등하게 합산 (단순 합산)
        """
        self.use_weights = use_weights
        logger.debug(f"Initialized CombSumFusion with use_weights={use_weights}")

    def fuse(
        self,
        candidates: list["Candidate"],
        weights: dict[str, float] | None = None,
    ) -> list["Candidate"]:
        """
        CombSum으로 점수 융합

        Args:
            candidates: Candidate 리스트
            weights: 각 검색 방식의 가중치 (use_weights=True일 때만 사용)

        Returns:
            CombSum 점수가 계산된 Candidate 리스트
        """
        if not candidates:
            return candidates

        # 가중치 설정
        if self.use_weights and weights is None:
            # 기본 가중치: 모두 동등하게
            weights = {
                "lexical": 1.0,
                "semantic": 1.0,
                "graph": 1.0,
                "fuzzy": 1.0,
            }

        # 점수 계산
        for candidate in candidates:
            if self.use_weights and weights:
                # 가중 합산
                total_score = sum(
                    candidate.features.get(f"{backend}_score", 0.0) * weights.get(backend, 0.0)
                    for backend in ["lexical", "semantic", "graph", "fuzzy"]
                )
            else:
                # 단순 합산
                total_score = sum(
                    candidate.features.get(key, 0.0)
                    for key in ["lexical_score", "semantic_score", "graph_score", "fuzzy_score"]
                )

            candidate.features["total_score"] = total_score
            candidate.features["final_score"] = total_score
            candidate.features["combsum_weighted"] = self.use_weights

        mode = "weighted" if self.use_weights else "unweighted"
        logger.debug(f"CombSum: Fused {len(candidates)} candidates ({mode})")
        return candidates
