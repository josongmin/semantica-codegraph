"""코드 검색 API"""


from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.core.bootstrap import create_bootstrap
from src.core.models import LocationContext
from src.search.graph.postgres_graph_adapter import PostgresGraphSearch
from src.search.retriever.hybrid_retriever import HybridRetriever
from src.search.semantic.pgvector_adapter import PgVectorSemanticSearch

router = APIRouter()
bootstrap = create_bootstrap()


class SearchRequest(BaseModel):
    """검색 요청"""
    query: str
    repo_id: str
    k: int = 20
    file_path: str | None = None
    line: int | None = None
    column: int | None = None
    weights: dict | None = None  # {"lexical": 0.3, "semantic": 0.5, "graph": 0.2}


class SearchResult(BaseModel):
    """검색 결과"""
    chunk_id: str
    file_path: str
    span: list[int]  # [start_line, start_col, end_line, end_col]
    score: float
    features: dict


class SearchResponse(BaseModel):
    """검색 응답"""
    query: str
    repo_id: str
    results: list[SearchResult]
    total: int


@router.post("/", response_model=SearchResponse)
async def search_code(request: SearchRequest):
    """
    하이브리드 코드 검색

    Lexical + Semantic + Graph 검색을 통합하여 실행
    """
    try:
        # LocationContext 생성 (선택적)
        location_ctx = None
        if request.file_path and request.line is not None:
            location_ctx = LocationContext(
                repo_id=request.repo_id,
                file_path=request.file_path,
                line=request.line,
                column=request.column or 0,
            )

        # HybridRetriever 생성
        graph_search = PostgresGraphSearch(bootstrap.graph_store)
        semantic_search = PgVectorSemanticSearch(
            embedding_service=bootstrap.embedding_service,
            embedding_store=bootstrap.embedding_store,
            chunk_store=bootstrap.chunk_store,
        )

        retriever = HybridRetriever(
            lexical_search=bootstrap.lexical_search,
            semantic_search=semantic_search,
            graph_search=graph_search,
        )

        # 검색 실행
        candidates = retriever.retrieve(
            repo_id=request.repo_id,
            query=request.query,
            k=request.k,
            location_ctx=location_ctx,
            weights=request.weights,
        )

        # 결과 변환
        results = [
            SearchResult(
                chunk_id=c.chunk_id,
                file_path=c.file_path,
                span=list(c.span),
                score=c.features.get("total_score", 0.0),
                features=c.features,
            )
            for c in candidates
        ]

        return SearchResponse(
            query=request.query,
            repo_id=request.repo_id,
            results=results,
            total=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lexical")
async def search_lexical(
    query: str = Query(..., description="검색 쿼리"),
    repo_id: str = Query(..., description="저장소 ID"),
    k: int = Query(10, description="결과 개수"),
):
    """Lexical 검색 (키워드 기반)"""
    try:
        results = bootstrap.lexical_search.search(
            repo_id=repo_id,
            query=query,
            k=k,
        )

        return {
            "query": query,
            "repo_id": repo_id,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "file_path": r.file_path,
                    "span": list(r.span),
                    "score": r.score,
                }
                for r in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/semantic")
async def search_semantic(
    query: str = Query(..., description="검색 쿼리"),
    repo_id: str = Query(..., description="저장소 ID"),
    k: int = Query(10, description="결과 개수"),
):
    """Semantic 검색 (의미론적)"""
    try:
        semantic_search = PgVectorSemanticSearch(
            embedding_service=bootstrap.embedding_service,
            embedding_store=bootstrap.embedding_store,
            chunk_store=bootstrap.chunk_store,
        )

        results = semantic_search.search(
            repo_id=repo_id,
            query=query,
            k=k,
        )

        return {
            "query": query,
            "repo_id": repo_id,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "file_path": r.file_path,
                    "span": list(r.span),
                    "score": r.score,
                }
                for r in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

