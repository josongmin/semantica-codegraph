"""검색 결과 Fusion 모듈

여러 검색 백엔드의 점수를 융합하는 전략들을 제공합니다.

사용 가능한 전략:
- WeightedFusion: 가중 합산 (기본)
- ReciprocalRankFusion: 순위 기반 (RRF)
- CombSumFusion: 단순/가중 합산
"""

from .base import FusionStrategy
from .combsum import CombSumFusion
from .rrf import ReciprocalRankFusion
from .weighted_fusion import WeightedFusion

__all__ = [
    "FusionStrategy",
    "WeightedFusion",
    "ReciprocalRankFusion",
    "CombSumFusion",
]
