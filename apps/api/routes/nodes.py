"""코드 그래프 노드 API"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.core.models import RepoId
from src.core.bootstrap import create_bootstrap
from src.search.graph.postgres_graph_adapter import PostgresGraphSearch

router = APIRouter()
bootstrap = create_bootstrap()


class NodeResponse(BaseModel):
    """노드 응답"""
    repo_id: str
    id: str
    kind: str
    language: str
    file_path: str
    span: List[int]  # [start_line, start_col, end_line, end_col]
    name: str
    text: str
    attrs: dict


class NeighborsResponse(BaseModel):
    """이웃 노드 응답"""
    node: NodeResponse
    neighbors: List[NodeResponse]
    total: int


@router.get("/{repo_id}/{node_id}", response_model=NodeResponse)
async def get_node(repo_id: str, node_id: str):
    """노드 조회"""
    try:
        graph_search = PostgresGraphSearch(bootstrap.graph_store)
        node = graph_search.get_node(repo_id, node_id)
        
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
        
        return NodeResponse(
            repo_id=node.repo_id,
            id=node.id,
            kind=node.kind,
            language=node.language,
            file_path=node.file_path,
            span=list(node.span),
            name=node.name,
            text=node.text,
            attrs=node.attrs,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{repo_id}/by-location", response_model=NodeResponse)
async def get_node_by_location(
    repo_id: str = Query(..., description="저장소 ID"),
    file_path: str = Query(..., description="파일 경로"),
    line: int = Query(..., description="라인 번호 (0-based)"),
    column: int = Query(0, description="컬럼 번호 (0-based)"),
):
    """위치로 노드 조회"""
    try:
        graph_search = PostgresGraphSearch(bootstrap.graph_store)
        node = graph_search.get_node_by_location(
            repo_id=repo_id,
            file_path=file_path,
            line=line,
            column=column,
        )
        
        if not node:
            raise HTTPException(status_code=404, detail="Node not found at location")
        
        return NodeResponse(
            repo_id=node.repo_id,
            id=node.id,
            kind=node.kind,
            language=node.language,
            file_path=node.file_path,
            span=list(node.span),
            name=node.name,
            text=node.text,
            attrs=node.attrs,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{repo_id}/{node_id}/neighbors", response_model=NeighborsResponse)
async def get_neighbors(
    repo_id: str,
    node_id: str,
    k: int = Query(1, description="이웃 깊이 (k-hop)"),
    edge_types: Optional[List[str]] = Query(None, description="엣지 타입 필터"),
):
    """노드 이웃 확장"""
    try:
        graph_search = PostgresGraphSearch(bootstrap.graph_store)
        
        # 노드 조회
        node = graph_search.get_node(repo_id, node_id)
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # 이웃 확장
        neighbors = graph_search.expand_neighbors(
            repo_id=repo_id,
            node_id=node_id,
            edge_types=edge_types,
            k=k,
        )
        
        return NeighborsResponse(
            node=NodeResponse(
                repo_id=node.repo_id,
                id=node.id,
                kind=node.kind,
                language=node.language,
                file_path=node.file_path,
                span=list(node.span),
                name=node.name,
                text=node.text,
                attrs=node.attrs,
            ),
            neighbors=[
                NodeResponse(
                    repo_id=n.repo_id,
                    id=n.id,
                    kind=n.kind,
                    language=n.language,
                    file_path=n.file_path,
                    span=list(n.span),
                    name=n.name,
                    text=n.text,
                    attrs=n.attrs,
                )
                for n in neighbors
            ],
            total=len(neighbors),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

