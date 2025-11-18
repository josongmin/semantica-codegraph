"""Ranker Port (검색 결과 랭킹)"""

from typing import Protocol

from ...core.models import Candidate


class RankerPort(Protocol):
    """검색 결과 랭킹 포트"""

    def rank(
        self,
        candidates: list[Candidate],
        max_items: int,
    ) -> list[Candidate]:
        """후보 리스트 랭킹 및 필터링"""
        ...

