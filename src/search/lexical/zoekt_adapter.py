from typing import Dict, List, Optional

from ...core.models import CodeChunk, ChunkResult, RepoId
from ..ports.lexical_search_port import LexicalSearchPort


class ZoektAdapter(LexicalSearchPort):
    """Zoekt를 사용한 BM25 검색 어댑터"""

    def __init__(self, zoekt_url: str, index_prefix: str = "chunks"):
        self.zoekt_url = zoekt_url
        self.index_prefix = index_prefix
        # TODO: Zoekt 클라이언트 초기화

    def index_chunks(self, chunks: List[CodeChunk]) -> None:
        """청크 인덱싱"""
        # TODO: Zoekt 인덱싱 구현
        raise NotImplementedError("ZoektAdapter.index_chunks not implemented yet")

    def search(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        filters: Optional[Dict] = None,
    ) -> List[ChunkResult]:
        """검색 실행"""
        # TODO: Zoekt 검색 구현
        raise NotImplementedError("ZoektAdapter.search not implemented yet")

    def delete_repo_index(self, repo_id: RepoId) -> None:
        """저장소 인덱스 삭제"""
        # TODO: Zoekt 인덱스 삭제 구현
        raise NotImplementedError("ZoektAdapter.delete_repo_index not implemented yet")

