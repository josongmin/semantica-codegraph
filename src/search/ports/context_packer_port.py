"""Context Packer Port (LLM 컨텍스트 패킹)"""

from typing import Protocol

from ...core.models import Candidate, LocationContext, PackedContext


class ContextPackerPort(Protocol):
    """검색 결과를 LLM 컨텍스트로 패킹"""

    def pack(
        self,
        candidates: list[Candidate],
        max_tokens: int,
        location_ctx: LocationContext,
    ) -> PackedContext:
        """후보를 토큰 제한 내에서 컨텍스트로 패킹"""
        ...

