"""저장소 관리 API"""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.core.bootstrap import create_bootstrap

router = APIRouter()
bootstrap = create_bootstrap()


class IndexRequest(BaseModel):
    """인덱싱 요청"""

    repo_path: str
    repo_id: str | None = None
    name: str | None = None


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
    git_url: str | None = None
    default_branch: str
    languages: list[str]
    total_files: int
    total_nodes: int
    total_chunks: int  # RepoMetadata 실제 필드
    indexing_status: str
    indexed_at: str | None = None


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
        # repo_id 미리 생성 (즉시 응답을 위해)
        repo_id = request.repo_id or request.name or Path(request.repo_path).name

        # 백그라운드 태스크로 인덱싱 실행
        background_tasks.add_task(
            _run_indexing,
            root_path=request.repo_path,
            repo_id=repo_id,
            name=request.name,
        )

        # 즉시 응답 반환
        return IndexResponse(
            repo_id=repo_id,
            status="started",
            nodes_count=0,
            edges_count=0,
            chunks_count=0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/", response_model=list[RepoResponse])
async def list_repositories():
    """저장소 목록 조회"""
    try:
        repos = bootstrap.repo_store.list_all()
        return [
            RepoResponse(
                repo_id=repo.repo_id,
                name=repo.name,
                root_path=repo.root_path,
                git_url=repo.attrs.get("git_url") if repo.attrs else None,
                default_branch=repo.attrs.get("default_branch", "main") if repo.attrs else "main",
                languages=repo.languages or [],
                total_files=repo.total_files,
                total_nodes=repo.total_nodes,
                total_chunks=repo.total_chunks,
                indexing_status=repo.indexing_status,
                indexed_at=repo.indexed_at.isoformat() if repo.indexed_at else None,
            )
            for repo in repos
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
            git_url=repo.attrs.get("git_url") if repo.attrs else None,
            default_branch=repo.attrs.get("default_branch", "main") if repo.attrs else "main",
            languages=repo.languages or [],
            total_files=repo.total_files,
            total_nodes=repo.total_nodes,
            total_chunks=repo.total_chunks,
            indexing_status=repo.indexing_status,
            indexed_at=repo.indexed_at.isoformat() if repo.indexed_at else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{repo_id}")
async def delete_repository(repo_id: str):
    """저장소 삭제"""
    try:
        bootstrap.repo_store.delete(repo_id)
        bootstrap.graph_store.delete_repo(repo_id)
        return {"status": "deleted", "repo_id": repo_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _run_indexing(root_path: str, repo_id: str | None, name: str | None):
    """백그라운드에서 실행되는 인덱싱 작업"""
    import logging

    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Starting background indexing for {repo_id}")
        result = await bootstrap.pipeline.index_repository_async(
            root_path=root_path,
            repo_id=repo_id,
            name=name,
        )
        logger.info(
            f"Indexing completed: {result.total_nodes} nodes, "
            f"{result.total_edges} edges, {result.total_chunks} chunks"
        )
    except Exception as e:
        logger.error(f"Indexing failed for {repo_id}: {e}", exc_info=True)
