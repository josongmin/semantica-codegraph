"""Graph Search Port (코드 그래프 탐색 기반 검색)"""

from typing import List, Optional, Protocol

from ...core.models import CodeNode, RepoId


class GraphSearchPort(Protocol):
    """코드 그래프 탐색 기반 검색 포트"""

    def get_node(self, repo_id: RepoId, node_id: str) -> Optional[CodeNode]:
        """노드 조회"""
        ...

    def get_node_by_location(
        self,
        repo_id: RepoId,
        file_path: str,
        line: int,
        column: int = 0,
    ) -> Optional[CodeNode]:
        """위치로 노드 조회"""
        ...

    def expand_neighbors(
        self,
        repo_id: RepoId,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        k: int = 1,
    ) -> List[CodeNode]:
        """노드 이웃 확장 (k-hop)"""
        ...

