"""PostgreSQL 기반 그래프 탐색 어댑터"""

import logging
from typing import List, Optional

from ...core.models import CodeNode, RepoId
from ...core.ports import GraphStorePort
from ..ports.graph_search_port import GraphSearchPort

logger = logging.getLogger(__name__)


class PostgresGraphSearch(GraphSearchPort):
    """
    PostgreSQL 기반 그래프 탐색
    
    역할:
    - 노드 조회 (ID, 위치 기반)
    - k-hop 이웃 확장
    - 관계 기반 탐색
    """

    def __init__(self, graph_store: GraphStorePort):
        """
        Args:
            graph_store: 그래프 스토어
        """
        self.graph_store = graph_store

    def get_node(self, repo_id: RepoId, node_id: str) -> Optional[CodeNode]:
        """
        노드 조회
        
        Args:
            repo_id: 저장소 ID
            node_id: 노드 ID
        
        Returns:
            CodeNode 또는 None
        """
        return self.graph_store.get_node(repo_id, node_id)

    def get_node_by_location(
        self,
        repo_id: RepoId,
        file_path: str,
        line: int,
        column: int = 0,
    ) -> Optional[CodeNode]:
        """
        위치로 노드 조회
        
        Args:
            repo_id: 저장소 ID
            file_path: 파일 경로
            line: 라인 번호 (0-based)
            column: 컬럼 번호 (0-based)
        
        Returns:
            해당 위치의 가장 작은 노드 (가장 구체적인 노드)
        """
        return self.graph_store.get_node_by_location(repo_id, file_path, line, column)

    def expand_neighbors(
        self,
        repo_id: RepoId,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        k: int = 1,
    ) -> List[CodeNode]:
        """
        노드 이웃 확장 (k-hop)
        
        Args:
            repo_id: 저장소 ID
            node_id: 시작 노드 ID
            edge_types: 관계 타입 필터 (None이면 모든 타입)
                예: ["calls", "inherits"]
            k: 확장 거리 (홉 수)
        
        Returns:
            이웃 노드 리스트
        """
        if k <= 0:
            return []

        visited = set()
        current_level = [node_id]
        neighbors = []

        for hop in range(k):
            next_level = []
            
            for current_id in current_level:
                if current_id in visited:
                    continue
                
                visited.add(current_id)
                
                # 현재 노드의 이웃 조회
                node_neighbors = self.graph_store.neighbors(
                    repo_id,
                    current_id,
                    edge_types=edge_types
                )
                
                for neighbor in node_neighbors:
                    if neighbor.id not in visited:
                        neighbors.append(neighbor)
                        next_level.append(neighbor.id)
            
            current_level = next_level
            
            if not current_level:
                break

        logger.debug(f"Expanded {len(neighbors)} neighbors for {node_id} (k={k})")
        return neighbors

