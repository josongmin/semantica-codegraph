"""Reciprocal Rank Fusion (RRF)

순위 기반 융합 방식으로 가중치 없이도 효과적으로 작동합니다.
"""

import logging
from typing import TYPE_CHECKING

from .base import FusionStrategy

if TYPE_CHECKING:
    from src.core.models import Candidate

logger = logging.getLogger(__name__)


class ReciprocalRankFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion (RRF)

    순위 기반 융합 방식으로 가중치 없이도 효과적으로 작동합니다.

    공식:
        RRF_score(d) = Σ 1 / (k + rank_i(d))

        여기서:
        - d: 문서(청크)
        - k: RRF 상수 (일반적으로 60)
        - rank_i(d): 백엔드 i에서 문서 d의 순위 (0-based)

    장점:
    - **가중치 튜닝 불필요**: 자동으로 균형 잡힌 융합
    - **스케일 독립적**: 점수 스케일 차이에 영향 없음 (순위만 사용)
    - **검증된 성능**: 여러 연구에서 우수한 성능 입증
    - **간단한 구현**: 복잡한 정규화 불필요

    단점:
    - 백엔드별 중요도 조정 불가 (모든 백엔드가 동등하게 취급)
    - 실제 점수 정보 손실 (순위만 사용)

    예시:
        Backend 1에서 1위(rank=0): 1/(60+0) = 0.0167
        Backend 2에서 3위(rank=2): 1/(60+2) = 0.0161
        Backend 3에서 2위(rank=1): 1/(60+1) = 0.0164
        → RRF score = 0.0167 + 0.0161 + 0.0164 = 0.0492

    참고:
    - Cormack et al. (2009) "Reciprocal Rank Fusion outperforms Condorcet"
    - https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

    사용 예시:
        fusion = ReciprocalRankFusion(k=60)
        result = fusion.fuse_and_sort(
            candidates=candidate_list,
            weights=None,  # RRF는 가중치 불필요
            k=20
        )
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: RRF 상수 (일반적으로 60, 범위: 1~100)
                - 작을수록 상위 순위의 영향력이 큼
                - 클수록 순위 간 차이가 완화됨
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.k = k
        logger.debug(f"Initialized ReciprocalRankFusion with k={k}")

    def fuse(
        self,
        candidates: list["Candidate"],
        weights: dict[str, float] | None = None,
    ) -> list["Candidate"]:
        """
        순위 기반 융합

        Args:
            candidates: Candidate 리스트
            weights: 사용되지 않음 (RRF는 가중치 불필요)

        Returns:
            RRF 점수가 계산된 Candidate 리스트
        """
        if not candidates:
            return candidates

        # 각 백엔드별로 점수 기준 순위 계산
        backend_rankings = self._compute_rankings(candidates)

        # RRF 점수 계산
        for candidate in candidates:
            rrf_score = 0.0
            rank_details = {}

            for backend_name, rankings in backend_rankings.items():
                chunk_id = candidate.chunk_id

                # 해당 백엔드에서 이 청크의 순위 (없으면 최하위)
                if chunk_id in rankings:
                    rank = rankings[chunk_id]
                    contribution = 1.0 / (self.k + rank)
                    rrf_score += contribution
                    rank_details[backend_name] = {"rank": rank, "contribution": contribution}

            candidate.features["total_score"] = rrf_score
            candidate.features["final_score"] = rrf_score
            candidate.features["rrf_score"] = rrf_score
            candidate.features["rrf_rank_details"] = rank_details  # 디버깅용

        logger.debug(
            f"RRF: Fused {len(candidates)} candidates from {len(backend_rankings)} backends (k={self.k})"
        )
        return candidates

    def _compute_rankings(self, candidates: list["Candidate"]) -> dict[str, dict[str, int]]:
        """
        각 백엔드별로 점수 기준 순위 계산

        Args:
            candidates: Candidate 리스트

        Returns:
            {backend_name: {chunk_id: rank}} 형태의 딕셔너리
            rank는 0-based (0이 최상위)
        """
        backend_scores: dict[str, dict[str, float]] = {
            "lexical": {},
            "semantic": {},
            "graph": {},
            "fuzzy": {},
        }

        # 각 backend별 점수 수집
        for candidate in candidates:
            chunk_id = candidate.chunk_id
            for backend in backend_scores:
                score_key = f"{backend}_score"
                score = candidate.features.get(score_key, 0.0)
                if score > 0:
                    backend_scores[backend][chunk_id] = score

        # 점수 기준으로 순위 매기기
        backend_rankings: dict[str, dict[str, int]] = {}

        for backend, scores in backend_scores.items():
            if not scores:
                continue

            # 점수 내림차순 정렬 (높은 점수가 상위 순위)
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # 순위 부여 (0부터 시작)
            rankings = {chunk_id: rank for rank, (chunk_id, _) in enumerate(sorted_items)}
            backend_rankings[backend] = rankings

        return backend_rankings
