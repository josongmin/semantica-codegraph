"""Graph Search Port (코드 그래프 탐색 기반 검색)"""

from typing import Protocol

from ...core.models import CodeNode, RepoId


class GraphSearchPort(Protocol):
    """코드 그래프 탐색 기반 검색 포트"""

    def get_node(self, repo_id: RepoId, node_id: str) -> CodeNode | None:
        """노드 조회"""
        ...

    def get_node_by_location(
        self,
        repo_id: RepoId,
        file_path: str,
        line: int,
        column: int = 0,
    ) -> CodeNode | None:
        """위치로 노드 조회"""
        ...

    def expand_neighbors(
        self,
        repo_id: RepoId,
        node_id: str,
        edge_types: list[str] | None = None,
        k: int = 1,
    ) -> list[CodeNode]:
        """노드 이웃 확장 (k-hop)"""
        ...

    def expand_neighbors_with_edges(
        self,
        repo_id: RepoId,
        node_id: str,
        edge_types: list[str] | None = None,
        k: int = 1,
    ) -> list[tuple[CodeNode, str, int]]:
        """
        노드 이웃을 edge 정보와 함께 확장 (k-hop)

        Returns:
            (CodeNode, edge_type, depth) 튜플 리스트
        """
        ...
