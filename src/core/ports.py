from typing import Dict, List, Optional, Protocol, Tuple

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
    def list_files(self, repo_id: RepoId, repo_root: str) -> List[Dict]:
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
    def parse_file(self, file_meta: Dict) -> Tuple[List[RawSymbol], List[RawRelation]]:
        """
        한 파일에 대한 RawSymbol/RawRelation 리스트 반환
        file_meta는 RepoScannerPort에서 넘겨준 dict 그대로 사용
        """
        ...


# 그래프 저장소
class GraphStorePort(Protocol):
    def save_graph(self, nodes: List[CodeNode], edges: List[CodeEdge]) -> None:
        ...

    def get_node(self, repo_id: RepoId, node_id: str) -> Optional[CodeNode]:
        """단일 노드 조회"""
        ...

    def get_node_by_location(
        self,
        repo_id: RepoId,
        file_path: str,
        line: int,
        column: int = 0,
    ) -> Optional[CodeNode]:
        ...

    def neighbors(
        self,
        repo_id: RepoId,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        k: int = 1,
    ) -> List[CodeNode]:
        ...

    def delete_repo(self, repo_id: RepoId) -> None:
        """저장소 전체 삭제"""
        ...


# 청크 저장소
class ChunkStorePort(Protocol):
    def save_chunks(self, chunks: List[CodeChunk]) -> None:
        ...

    def get_chunk(self, repo_id: RepoId, chunk_id: str) -> Optional[CodeChunk]:
        """단일 청크 조회"""
        ...

    def get_chunks_by_node(
        self,
        repo_id: RepoId,
        node_id: str,
    ) -> List[CodeChunk]:
        ...


# 임베딩 저장소 / 검색
class EmbeddingStorePort(Protocol):
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...

    def save_embeddings(
        self,
        repo_id: RepoId,
        chunk_ids: List[str],
        vectors: List[List[float]],
    ) -> None:
        ...

    def search_by_vector(
        self,
        repo_id: RepoId,
        vector: List[float],
        k: int,
        filters: Optional[Dict] = None,
    ) -> List[ChunkResult]:
        ...


# 인덱스 매니저
class IndexManagerPort(Protocol):
    def start_indexing(self, repo_id: RepoId, repo_root: str) -> None:
        """인덱싱 시작"""
        ...

    def get_status(self, repo_id: RepoId) -> Optional[IndexingStatus]:
        """인덱싱 상태 조회"""
        ...

    def cancel_indexing(self, repo_id: RepoId) -> None:
        """인덱싱 취소"""
        ...

