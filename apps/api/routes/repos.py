"""저장소 관리 API"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.core.models import RepoId, RepoMetadata, IndexingResult
from src.core.bootstrap import create_bootstrap

router = APIRouter()
bootstrap = create_bootstrap()


class IndexRequest(BaseModel):
    """인덱싱 요청"""
    repo_path: str
    repo_id: Optional[str] = None
    name: Optional[str] = None


class IndexResponse(BaseModel):
    """인덱싱 응답"""
    repo_id: str
    status: str
    nodes_count: int
    edges_count: int
    chunks_count: int


class RepoResponse(BaseModel):
    """저장소 정보 응답"""
    repo_id: str
    name: str
    root_path: str
    git_url: Optional[str] = None
    default_branch: str
    languages: List[str]
    total_files: int
    total_nodes: int
    total_edges: int
    indexing_status: str
    indexed_at: Optional[str] = None


@router.post("/", response_model=IndexResponse)
async def index_repository(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
):
    """
    저장소 인덱싱 시작
    
    백그라운드로 실행되며 즉시 응답 반환
    """
    try:
        result = bootstrap.pipeline.index_repository(
            root_path=request.repo_path,
            repo_id=request.repo_id,
            name=request.name,
        )
        
        return IndexResponse(
            repo_id=result.repo_id,
            status="completed",
            nodes_count=result.nodes_count,
            edges_count=result.edges_count,
            chunks_count=result.chunks_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[RepoResponse])
async def list_repositories():
    """저장소 목록 조회"""
    try:
        repos = bootstrap.repo_store.list_all()
        return [
            RepoResponse(
                repo_id=repo.repo_id,
                name=repo.name,
                root_path=repo.root_path,
                git_url=repo.git_url,
                default_branch=repo.default_branch,
                languages=repo.languages or [],
                total_files=repo.total_files,
                total_nodes=repo.total_nodes,
                total_edges=repo.total_edges,
                indexing_status=repo.indexing_status,
                indexed_at=repo.indexed_at.isoformat() if repo.indexed_at else None,
            )
            for repo in repos
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{repo_id}", response_model=RepoResponse)
async def get_repository(repo_id: str):
    """저장소 정보 조회"""
    try:
        repo = bootstrap.repo_store.get(repo_id)
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        return RepoResponse(
            repo_id=repo.repo_id,
            name=repo.name,
            root_path=repo.root_path,
            git_url=repo.git_url,
            default_branch=repo.default_branch,
            languages=repo.languages or [],
            total_files=repo.total_files,
            total_nodes=repo.total_nodes,
            total_edges=repo.total_edges,
            indexing_status=repo.indexing_status,
            indexed_at=repo.indexed_at.isoformat() if repo.indexed_at else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{repo_id}")
async def delete_repository(repo_id: str):
    """저장소 삭제"""
    try:
        bootstrap.repo_store.delete(repo_id)
        bootstrap.graph_store.delete_repo(repo_id)
        return {"status": "deleted", "repo_id": repo_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{repo_id}/status")
async def get_indexing_status(repo_id: str):
    """인덱싱 상태 조회"""
    try:
        repo = bootstrap.repo_store.get(repo_id)
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        return {
            "repo_id": repo_id,
            "status": repo.indexing_status,
            "progress": repo.indexing_progress,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

