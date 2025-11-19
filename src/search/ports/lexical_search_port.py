"""Lexical Search Port (BM25, 키워드 기반 검색)"""

from typing import Protocol

from ...core.models import ChunkResult, CodeChunk, RepoId


class LexicalSearchPort(Protocol):
    """키워드 기반 검색 포트 (BM25, TF-IDF 등)"""

    def index_chunks(self, chunks: list[CodeChunk]) -> None:
        """청크 인덱싱"""
        ...

    def search(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        filters: dict | None = None,
    ) -> list[ChunkResult]:
        """키워드 검색 실행"""
        ...

    def delete_repo_index(self, repo_id: RepoId) -> None:
        """저장소 인덱스 삭제"""
        ...
