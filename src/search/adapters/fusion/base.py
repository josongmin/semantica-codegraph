"""Fusion Strategy 추상 클래스"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.models import Candidate


class FusionStrategy(ABC):
    """
    Fusion 전략 추상 클래스

    여러 검색 백엔드의 점수를 융합하는 전략을 정의합니다.
    각 전략은 이 인터페이스를 구현해야 합니다.

    구현해야 할 전략:
    - WeightedFusion: 가중 합산
    - ReciprocalRankFusion (RRF): 순위 기반
    - CombSumFusion: 단순 합산
    """

    @abstractmethod
    def fuse(
        self,
        candidates: list["Candidate"],
        weights: dict[str, float] | None = None,
    ) -> list["Candidate"]:
        """
        점수 융합

        Args:
            candidates: 점수가 포함된 Candidate 리스트
                각 candidate.features에는 다음 점수들이 포함됨:
                - lexical_score: Lexical 검색 점수 (정규화됨)
                - semantic_score: Semantic 검색 점수 (정규화됨)
                - graph_score: Graph 검색 점수 (정규화됨)
                - fuzzy_score: Fuzzy 검색 점수 (정규화됨)
            weights: 각 검색 방식의 가중치 (전략에 따라 선택적)
                예: {"lexical": 0.25, "semantic": 0.45, "graph": 0.15, "fuzzy": 0.15}

        Returns:
            최종 점수가 계산된 Candidate 리스트 (in-place 수정)
            각 candidate.features에 다음 필드가 추가됨:
            - total_score: 계산된 총 점수
            - final_score: 최종 점수 (total_score와 동일)
        """
        pass

    def fuse_and_sort(
        self,
        candidates: list["Candidate"],
        weights: dict[str, float] | None = None,
        k: int | None = None,
    ) -> list["Candidate"]:
        """
        점수 융합 + 정렬 + 상위 k개 반환

        Args:
            candidates: Candidate 리스트
            weights: 가중치 (전략에 따라 선택적)
            k: 반환할 최대 개수 (None이면 전체 반환)

        Returns:
            최종 점수 기준으로 정렬된 Candidate 리스트 (상위 k개)
        """
        # 점수 융합
        self.fuse(candidates, weights)

        # 정렬 (final_score 기준 내림차순)
        candidates.sort(key=lambda c: c.features.get("final_score", 0), reverse=True)

        # 상위 k개
        if k is not None:
            return candidates[:k]
        return candidates
