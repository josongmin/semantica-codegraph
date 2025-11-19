"""Semantic Search Port (임베딩 기반 검색)"""

from typing import Protocol

from ...core.models import ChunkResult, RepoId


class SemanticSearchPort(Protocol):
    """임베딩 기반 의미론적 검색 포트"""

    def embed_text(self, text: str) -> list[float]:
        """텍스트를 벡터로 변환"""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """다수 텍스트를 벡터로 변환"""
        ...

    def index_chunks(
        self,
        repo_id: RepoId,
        chunk_ids: list[str],
        texts: list[str],
    ) -> None:
        """청크 임베딩 인덱싱"""
        ...

    def search(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        filters: dict | None = None,
    ) -> list[ChunkResult]:
        """의미론적 검색 실행"""
        ...

    def delete_repo_index(self, repo_id: RepoId) -> None:
        """저장소 인덱스 삭제"""
        ...
