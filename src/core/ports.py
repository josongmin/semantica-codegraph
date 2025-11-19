from typing import Protocol

from .models import (
    ChunkResult,
    CodeChunk,
    CodeEdge,
    CodeNode,
    IndexingStatus,
    RawRelation,
    RawSymbol,
    RepoId,
)


# Repo 스캐너
class RepoScannerPort(Protocol):
    def list_files(self, repo_id: RepoId, repo_root: str) -> list[dict]:
        """
        인덱싱 대상 파일 메타데이터 리스트 반환
        각 item 예시:
        {
          "repo_id": ...,
          "repo_root": ...,
          "path": "src/app/main.py",
          "abs_path": ".../repo/src/app/main.py",
          "language": "python",
          "is_test": False,
          ...
        }
        """
        ...


# 파서
class ParserPort(Protocol):
    def parse_file(self, file_meta: dict) -> tuple[list[RawSymbol], list[RawRelation]]:
        """
        한 파일에 대한 RawSymbol/RawRelation 리스트 반환
        file_meta는 RepoScannerPort에서 넘겨준 dict 그대로 사용
        """
        ...


# 그래프 저장소
class GraphStorePort(Protocol):
    def save_graph(self, nodes: list[CodeNode], edges: list[CodeEdge]) -> None:
        """
        노드와 엣지를 그래프 저장소에 저장

        Args:
            nodes: 저장할 CodeNode 리스트
            edges: 저장할 CodeEdge 리스트
        """
        ...

    def get_node(self, repo_id: RepoId, node_id: str) -> CodeNode | None:
        """
        노드 ID로 단일 노드 조회

        Args:
            repo_id: 저장소 ID
            node_id: 노드 ID

        Returns:
            CodeNode 또는 None
        """
        ...

    def get_node_by_location(
        self,
        repo_id: RepoId,
        file_path: str,
        line: int,
        column: int = 0,
    ) -> CodeNode | None:
        """
        파일 위치로 노드 조회

        Args:
            repo_id: 저장소 ID
            file_path: 파일 경로
            line: 라인 번호
            column: 컬럼 번호

        Returns:
            CodeNode 또는 None
        """
        ...

    def neighbors(
        self,
        repo_id: RepoId,
        node_id: str,
        edge_types: list[str] | None = None,
        k: int = 1,
    ) -> list[CodeNode]:
        """
        노드의 이웃 노드 조회

        Args:
            repo_id: 저장소 ID
            node_id: 노드 ID
            edge_types: 엣지 타입 필터 (예: ["calls", "uses"])
            k: 홉 수 (1 = 직접 이웃만)

        Returns:
            CodeNode 리스트
        """
        ...

    def neighbors_with_edges(
        self,
        repo_id: RepoId,
        node_id: str,
        edge_types: list[str] | None = None,
        k: int = 1,
    ) -> list[tuple[CodeNode, str, int]]:
        """
        노드의 이웃 노드를 edge 정보와 함께 조회

        Args:
            repo_id: 저장소 ID
            node_id: 노드 ID
            edge_types: 엣지 타입 필터 (예: ["calls", "uses"])
            k: 홉 수 (1 = 직접 이웃만)

        Returns:
            (CodeNode, edge_type, depth) 튜플 리스트
        """
        ...

    def list_nodes(
        self,
        repo_id: RepoId,
        kinds: list[str] | None = None,
    ) -> list[CodeNode]:
        """
        저장소의 모든 노드 조회

        Args:
            repo_id: 저장소 ID
            kinds: 필터할 노드 종류 (예: ["Class", "Function"])
                   None이면 모든 종류 반환

        Returns:
            CodeNode 리스트
        """
        ...

    def delete_repo(self, repo_id: RepoId) -> None:
        """
        저장소 전체 삭제

        Args:
            repo_id: 삭제할 저장소 ID
        """
        ...


# 청크 저장소
class ChunkStorePort(Protocol):
    def save_chunks(self, chunks: list[CodeChunk]) -> None:
        """
        청크를 저장소에 저장

        Args:
            chunks: 저장할 CodeChunk 리스트
        """
        ...

    def get_chunk(self, repo_id: RepoId, chunk_id: str) -> CodeChunk | None:
        """
        청크 ID로 단일 청크 조회

        Args:
            repo_id: 저장소 ID
            chunk_id: 청크 ID

        Returns:
            CodeChunk 또는 None
        """
        ...

    def get_chunks_by_node(
        self,
        repo_id: RepoId,
        node_id: str,
    ) -> list[CodeChunk]:
        """
        노드 ID로 연관된 청크 조회

        Args:
            repo_id: 저장소 ID
            node_id: 노드 ID

        Returns:
            CodeChunk 리스트
        """
        ...


# 임베딩 저장소 / 검색
class EmbeddingStorePort(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        텍스트를 임베딩 벡터로 변환

        Args:
            texts: 변환할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        ...

    def save_embeddings(
        self,
        repo_id: RepoId,
        chunk_ids: list[str],
        vectors: list[list[float]],
    ) -> None:
        """
        임베딩 벡터를 저장소에 저장

        Args:
            repo_id: 저장소 ID
            chunk_ids: 청크 ID 리스트
            vectors: 임베딩 벡터 리스트
        """
        ...

    def search_by_vector(
        self,
        repo_id: RepoId,
        vector: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[ChunkResult]:
        """
        벡터 유사도 기반 검색

        Args:
            repo_id: 저장소 ID
            vector: 쿼리 벡터
            k: 반환할 결과 수
            filters: 추가 필터 (예: {"language": "python"})

        Returns:
            ChunkResult 리스트 (유사도 순)
        """
        ...


# 인덱스 매니저
class IndexManagerPort(Protocol):
    def start_indexing(self, repo_id: RepoId, repo_root: str) -> None:
        """
        저장소 인덱싱 시작

        Args:
            repo_id: 저장소 ID
            repo_root: 저장소 루트 경로
        """
        ...

    def get_status(self, repo_id: RepoId) -> IndexingStatus | None:
        """
        인덱싱 상태 조회

        Args:
            repo_id: 저장소 ID

        Returns:
            IndexingStatus 또는 None
        """
        ...

    def cancel_indexing(self, repo_id: RepoId) -> None:
        """
        인덱싱 취소

        Args:
            repo_id: 저장소 ID
        """
        ...
