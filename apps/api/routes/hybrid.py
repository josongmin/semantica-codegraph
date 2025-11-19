"""Hybrid Search API 엔드포인트"""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Path, Query

from src.core.bootstrap import create_bootstrap
from src.core.models import LocationContext
from src.search.adapters.graph.postgres_graph_adapter import PostgresGraphSearch
from src.search.adapters.retriever.hybrid_retriever import HybridRetriever
from src.search.adapters.semantic.pgvector_adapter import PgVectorSemanticSearch

from ..model.hybrid import (
    CandidateExplainResponse,
    CandidateExtraFeatures,
    CandidateFeatures,
    CandidateScores,
    ChunkFetchRequest,
    ChunkFetchResponse,
    ChunkInfo,
    ChunkMetadata,
    ExplainedCandidate,
    GraphExpandRequest,
    GraphExpandResponse,
    GraphNeighbor,
    HybridCandidate,
    HybridSearchRequest,
    HybridSearchResponse,
    SessionPreferenceRequest,
    SessionPreferenceResponse,
    SymbolCandidate,
    SymbolSearchRequest,
    SymbolSearchResponse,
)

router = APIRouter()
bootstrap = create_bootstrap()
logger = logging.getLogger(__name__)

# 세션별 설정 저장 (메모리 기반, 프로덕션에서는 Redis 등 사용)
_session_preferences: dict[str, dict] = {}


def _get_snippet(text: str, max_length: int = 200) -> str:
    """텍스트에서 스니펫 추출"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def _rerank_with_metadata(candidates: list, query: str, repo_profile, repo_store) -> list:
    """
    메타데이터 기반 재순위화

    Args:
        candidates: Candidate 리스트
        query: 검색 쿼리
        repo_profile: RepoProfile
        repo_store: RepoMetadataStore

    Returns:
        재순위화된 Candidate 리스트
    """
    query_lower = query.lower()

    # 쿼리 타입 분석
    query_type = _analyze_query_type(query_lower)

    # 파일 프로파일 로드 (캐싱)
    file_profile_cache = {}

    for candidate in candidates:
        file_path = candidate.file_path

        # 파일 프로파일 조회
        if file_path not in file_profile_cache:
            try:
                file_profile = repo_store.get_file_profile(candidate.repo_id, file_path)
                file_profile_cache[file_path] = file_profile
            except Exception:
                file_profile_cache[file_path] = None

        file_profile = file_profile_cache[file_path]

        # 기본 점수
        original_score = candidate.features.get("final_score", 0.0)
        boost = 1.0

        # 1. 쿼리 타입별 파일 역할 매칭
        if query_type == "api" and file_profile:
            if file_profile.is_api_file or file_profile.is_router:
                boost *= 1.5  # API 파일 50% 부스트
            # 외부 API 클라이언트는 페널티
            if "evaluator" in file_path.lower() or "client" in file_path.lower():
                boost *= 0.5

        elif query_type == "service" and file_profile:
            if file_profile.is_service:
                boost *= 1.4

        elif query_type == "model" and file_profile:
            if file_profile.is_model or file_profile.is_schema:
                boost *= 1.4

        elif query_type == "config" and file_profile:
            if file_profile.is_config:
                boost *= 1.5

        # 2. 디렉토리 패턴 매칭
        if query_type == "api":
            # API 디렉토리 부스트
            if any(api_dir in file_path for api_dir in repo_profile.api_directories):
                boost *= 1.3
            # 벤치마크/테스트 페널티
            if "benchmark" in file_path.lower() or "evaluator" in file_path.lower():
                boost *= 0.6

        # 3. 테스트 파일 페널티 (일반 쿼리)
        if file_profile and file_profile.is_test_file and query_type != "test":
            boost *= 0.7

        # 최종 점수 업데이트
        candidate.features["final_score"] = original_score * boost
        candidate.features["metadata_boost"] = boost

    # 재정렬
    candidates.sort(key=lambda c: c.features.get("final_score", 0.0), reverse=True)

    return candidates


def _analyze_query_type(query: str) -> str:
    """쿼리 타입 분석"""

    # API 관련
    if any(kw in query for kw in ["api", "endpoint", "route", "handler", "controller"]):
        return "api"

    # 서비스/비즈니스 로직
    if any(kw in query for kw in ["service", "business", "logic", "usecase"]):
        return "service"

    # 모델/스키마
    if any(kw in query for kw in ["model", "schema", "entity", "dto", "data"]):
        return "model"

    # 설정
    if any(kw in query for kw in ["config", "setting", "env", "configuration"]):
        return "config"

    # 테스트
    if any(kw in query for kw in ["test", "spec", "테스트"]):
        return "test"

    return "general"


def _candidate_to_hybrid_candidate(candidate, repo_id: str, candidate_id: str) -> HybridCandidate:
    """Candidate를 HybridCandidate로 변환"""
    features = candidate.features

    # 점수 추출
    scores = CandidateScores(
        lexical=features.get("lexical_score"),
        semantic=features.get("semantic_score"),
        graph=features.get("graph_score"),
        fuzzy=features.get("fuzzy_score"),
    )

    # 추가 특성
    extra_features = None
    if features.get("has_span") is not None or features.get("tfidf_score") is not None:
        extra_features = CandidateExtraFeatures(
            has_span=features.get("has_span"),
            tfidf_score=features.get("tfidf_score"),
        )

    # 청크에서 텍스트 가져오기
    snippet = ""
    language = "unknown"
    symbol_name = None
    node_id = None

    try:
        chunk = bootstrap.chunk_store.get_chunk(repo_id, candidate.chunk_id)
        if chunk:
            snippet = _get_snippet(chunk.text)
            language = chunk.language
            node_id = chunk.node_id

            # 노드에서 심볼명 가져오기
            if node_id:
                node = bootstrap.graph_store.get_node(repo_id, node_id)
                if node:
                    symbol_name = node.name
    except Exception as e:
        logger.debug(f"Failed to get chunk info: {e}")

    return HybridCandidate(
        candidate_id=candidate_id,
        doc_id=candidate.chunk_id,  # doc_id는 chunk_id와 동일
        chunk_id=candidate.chunk_id,
        file_path=candidate.file_path,
        span=list(candidate.span),
        snippet=snippet,
        language=language,
        symbol_name=symbol_name,
        node_id=node_id,
        final_score=features.get("total_score", 0.0),
        scores=scores,
        extra_features=extra_features,
    )


@router.post("/search", response_model=HybridSearchResponse)
async def hybrid_search(request: HybridSearchRequest):
    """
    하이브리드 검색 (Primary Retrieval API)

    쿼리 + 현재 위치를 기반으로 lexical/semantic/fuzzy/graph 백엔드를 모두 사용해
    후보를 통합 검색하고, 최종적으로 가장 유의미한 코드/텍스트 청크 후보들을 반환.
    """
    try:
        # LocationContext 생성
        location_ctx = None
        repo_id = None

        if request.location_ctx:
            repo_id = request.location_ctx.repo_id
            location_ctx = LocationContext(
                repo_id=request.location_ctx.repo_id,
                file_path=request.location_ctx.file_path,
                line=request.location_ctx.line,
                column=request.location_ctx.column,
                symbol_name=request.location_ctx.symbol_name,
            )

        # repo_id가 없으면 기본 저장소 사용 (첫 번째 저장소)
        if not repo_id:
            repos = bootstrap.repo_store.list_all()
            if not repos:
                raise HTTPException(
                    status_code=400, detail="No repositories available. repo_id is required."
                )
            repo_id = repos[0].repo_id
            logger.info(f"No repo_id provided, using default: {repo_id}")

        # 타입 체크: repo_id는 이 시점에서 반드시 str
        assert repo_id is not None, "repo_id must be set at this point"

        # 세션 설정 가져오기
        weights = None
        backend_enable = None
        if request.session_id and request.session_id in _session_preferences:
            prefs = _session_preferences[request.session_id]
            weights = prefs.get("weights")
            backend_enable = prefs.get("backend_enable")

        # 백엔드 힌트 처리
        if request.backend_hints:
            logger.debug(f"Backend hints: {request.backend_hints}")
            # backend_hints가 있으면 해당 백엔드만 활성화
            if backend_enable is None:
                backend_enable = {}
            for backend in ["lexical", "semantic", "graph", "fuzzy"]:
                if backend not in backend_enable:
                    backend_enable[backend] = backend in request.backend_hints

        # HybridRetriever 생성
        graph_search = PostgresGraphSearch(bootstrap.graph_store)
        semantic_search = PgVectorSemanticSearch(
            embedding_service=bootstrap.embedding_service,
            embedding_store=bootstrap.embedding_store,
        )

        retriever = HybridRetriever(
            lexical_search=bootstrap.lexical_search,
            semantic_search=semantic_search,
            graph_search=graph_search,
            fuzzy_search=bootstrap.fuzzy_search,
            chunk_store=bootstrap.chunk_store,
        )

        # 검색 실행
        candidates = retriever.retrieve(
            repo_id=repo_id,
            query=request.query,
            k=request.k,
            location_ctx=location_ctx,
            weights=weights,
        )

        # 메타데이터 기반 재순위화
        try:
            repo_profile = bootstrap.repo_store.get_profile(repo_id)
            if repo_profile:
                candidates = _rerank_with_metadata(
                    candidates, request.query, repo_profile, bootstrap.repo_store
                )
                logger.debug(
                    f"Reranked candidates using repo profile (framework={repo_profile.framework})"
                )
        except Exception as e:
            logger.warning(f"Metadata reranking failed (계속 진행): {e}")

        # 결과 변환
        hybrid_candidates = []
        for c in candidates:
            candidate_id = f"cand-{uuid.uuid4().hex[:8]}"
            hybrid_candidates.append(_candidate_to_hybrid_candidate(c, repo_id, candidate_id))

        return HybridSearchResponse(
            query=request.query,
            k=request.k,
            candidates=hybrid_candidates,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/chunks", response_model=ChunkFetchResponse)
async def fetch_chunks(request: ChunkFetchRequest):
    """
    청크 상세 조회 (Chunk Loader)

    /hybrid/search에서 candidate_id/chunk_id만 받은 상태에서
    청크 전체 내용을 로딩할 때 사용. snippet 길이 제한, context loss 문제를 해결하기 위해
    chunk full content를 API로 가져오는 역할.
    """
    try:
        chunks = []

        for chunk_id in request.chunk_ids:
            # repo_id 추출 (chunk_id 형식에 따라 다를 수 있음)
            # 현재는 첫 번째 청크에서 repo_id를 가져옴
            # 실제로는 chunk_id에 repo_id가 포함되어야 함
            # 임시로 모든 저장소에서 검색 (비효율적이지만 동작)
            chunk_found = None

            # 저장소 목록 가져오기
            repos = bootstrap.repo_store.list_all()
            for repo in repos:
                chunk = bootstrap.chunk_store.get_chunk(repo.repo_id, chunk_id)
                if chunk:
                    chunk_found = (chunk, repo.repo_id)
                    break

            if not chunk_found:
                continue

            chunk, repo_id = chunk_found

            # 노드 ID 가져오기
            node_ids = [chunk.node_id] if chunk.node_id else []

            # 이웃 노드 포함 여부
            if request.include_neighbors and chunk.node_id:
                try:
                    graph_search = PostgresGraphSearch(bootstrap.graph_store)
                    neighbors = graph_search.expand_neighbors(
                        repo_id=repo_id,
                        node_id=chunk.node_id,
                        k=1,
                    )
                    node_ids.extend([n.id for n in neighbors])
                except Exception as e:
                    logger.debug(f"Failed to get neighbors: {e}")

            chunks.append(
                ChunkInfo(
                    chunk_id=chunk.id,
                    doc_id=chunk.id,
                    file_path=chunk.file_path,
                    span=list(chunk.span),
                    content=chunk.text if request.include_source else "",
                    language=chunk.language,
                    node_ids=node_ids,
                    metadata=ChunkMetadata(
                        size=len(chunk.text),
                        type=chunk.attrs.get("type", "code"),
                    ),
                )
            )

        return ChunkFetchResponse(chunks=chunks)
    except Exception as e:
        logger.error(f"Chunk fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/graph", response_model=GraphExpandResponse)
async def expand_graph(request: GraphExpandRequest):
    """
    코드 그래프 기반 맥락 확장 (Context Expander)

    함수/클래스/모듈의 호출 관계, 정의 관계, import 관계 등을 따라
    연관된 주변 코드(context)를 가져옴.
    graph score(depth, edge type) 기반으로 후보 제공.
    """
    try:
        # repo_id 추출 (node_id나 chunk_id에서)
        repo_id = None

        node_id = request.node_id

        if request.chunk_id:
            # chunk에서 repo_id 가져오기
            repos = bootstrap.repo_store.list_all()
            for repo in repos:
                chunk = bootstrap.chunk_store.get_chunk(repo.repo_id, request.chunk_id)
                if chunk:
                    repo_id = repo.repo_id
                    # chunk_id가 있으면 node_id도 업데이트
                    if not node_id and chunk.node_id:
                        node_id = chunk.node_id
                    break

        if not repo_id:
            # node_id에서 repo_id 추출 시도
            if not node_id:
                raise HTTPException(status_code=400, detail="node_id or chunk_id is required")

            repos = bootstrap.repo_store.list_all()
            for repo in repos:
                node = bootstrap.graph_store.get_node(repo.repo_id, node_id)
                if node:
                    repo_id = repo.repo_id
                    break

        if not repo_id:
            raise HTTPException(status_code=400, detail="Could not determine repo_id")

        if not node_id:
            raise HTTPException(status_code=400, detail="node_id is required")

        # 그래프 확장 (edge 정보 포함)
        neighbors_with_edges = bootstrap.graph_store.neighbors_with_edges(
            repo_id=repo_id,
            node_id=node_id,
            edge_types=request.edge_types,
            k=request.max_depth,
        )

        # 결과 변환
        graph_neighbors = []
        for neighbor_node, edge_type, depth in neighbors_with_edges[: request.k]:
            # 노드에서 청크 찾기
            chunks = bootstrap.chunk_store.get_chunks_by_node(repo_id, neighbor_node.id)
            snippet = ""
            if chunks:
                snippet = _get_snippet(chunks[0].text)

            # 그래프 점수 계산 (간단한 방식: depth 기반)
            graph_score = 1.0 / (depth + 1)

            graph_neighbors.append(
                GraphNeighbor(
                    candidate_id=f"cand-{uuid.uuid4().hex[:8]}",
                    node_id=neighbor_node.id,
                    file_path=neighbor_node.file_path,
                    span=list(neighbor_node.span),
                    snippet=snippet,
                    edge_type=edge_type.upper(),  # API 스펙에 맞게 대문자 변환
                    depth=depth,
                    graph_score=graph_score,
                )
            )

        return GraphExpandResponse(neighbors=graph_neighbors)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph expand failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/symbols")
async def search_symbols_direct(
    repo_id: str = Query(..., description="저장소 ID"),
    query: str = Query(..., description="검색 쿼리 (함수/클래스 이름)"),
    kind: str | None = Query(None, description="심볼 종류 필터 (Function, Class, Method)"),
    fuzzy: bool = Query(True, description="퍼지 매칭 여부"),
    decorator: str | None = Query(None, description="Decorator 패턴 (@router.post 등)"),
    k: int = Query(20, description="결과 개수"),
):
    """
    심볼 직접 검색 (SymbolIndex)

    code_nodes 테이블을 직접 조회하여 빠른 심볼 검색.

    Examples:
        GET /hybrid/symbols?repo_id=my-repo&query=HybridRetriever&kind=Class
        GET /hybrid/symbols?repo_id=my-repo&query=search&fuzzy=true
        GET /hybrid/symbols?repo_id=my-repo&decorator=@router.post
    """
    try:
        # Decorator 검색
        if decorator:
            results = bootstrap.symbol_search.search_by_decorator(
                repo_id=repo_id,
                decorator_pattern=decorator,
                k=k,
            )
        else:
            # 이름 검색
            results = bootstrap.symbol_search.search_by_name(
                repo_id=repo_id,
                query=query,
                kind=kind,
                fuzzy=fuzzy,
                k=k,
            )

        # Response 변환
        candidates = []
        for node in results:
            # 청크에서 스니펫 가져오기
            chunks = bootstrap.chunk_store.get_chunks_by_node(repo_id, node.id)
            snippet = ""
            if chunks:
                snippet = _get_snippet(chunks[0].text)

            candidates.append(
                SymbolCandidate(
                    candidate_id=f"cand-{uuid.uuid4().hex[:8]}",
                    symbol_name=node.name,
                    match_score=1.0,  # 직접 조회이므로 스코어는 1.0
                    file_path=node.file_path,
                    span=list(node.span),
                    snippet=snippet,
                    metadata={
                        "kind": node.kind,
                        "language": node.language,
                        "decorators": node.attrs.get("decorators", []),
                        "docstring": node.attrs.get("docstring"),
                    },
                )
            )

        return SymbolSearchResponse(
            symbol_query=query if not decorator else decorator,
            candidates=candidates,
        )

    except Exception as e:
        logger.error(f"Symbol search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/symbols", response_model=SymbolSearchResponse)
async def search_symbols(request: SymbolSearchRequest):
    """
    심볼 기반 검색 (Fuzzy Search - 기존 방식)

    네이밍 기반 fuzzy + graph 인덱스로 심볼(클래스/함수/메서드/변수) 검색.
    코드레포에서 "이 함수 어디?" 같은 질문을 빠르게 만족시키는 Fast-path.

    Note: 이 엔드포인트는 fuzzy_search를 사용하는 기존 방식입니다.
          새로운 SymbolIndex 기반 검색은 GET /hybrid/symbols를 사용하세요.
    """
    try:
        if not bootstrap.fuzzy_search:
            raise HTTPException(status_code=503, detail="Fuzzy search not available")

        # 모든 저장소에서 검색
        repos = bootstrap.repo_store.list_all()
        all_candidates = []

        for repo in repos:
            matches = bootstrap.fuzzy_search.search_symbols(
                repo_id=repo.repo_id,
                query=request.symbol_query,
                k=request.k,
            )

            for match in matches:
                # 노드에서 정보 가져오기
                node = bootstrap.graph_store.get_node(repo.repo_id, match.node_id)
                if not node:
                    continue

                # 청크에서 스니펫 가져오기
                chunks = bootstrap.chunk_store.get_chunks_by_node(repo.repo_id, match.node_id)
                snippet = ""
                if chunks:
                    snippet = _get_snippet(chunks[0].text)

                all_candidates.append(
                    SymbolCandidate(
                        candidate_id=f"cand-{uuid.uuid4().hex[:8]}",
                        symbol_name=match.matched_text,  # FuzzyMatch는 matched_text 사용
                        match_score=match.score,  # 이미 0-1 스케일로 정규화됨
                        file_path=match.file_path or node.file_path,
                        span=list(node.span),
                        snippet=snippet,
                    )
                )

        # 점수 순 정렬 및 k개 제한
        all_candidates.sort(key=lambda x: x.match_score, reverse=True)
        all_candidates = all_candidates[: request.k]

        return SymbolSearchResponse(
            symbol_query=request.symbol_query,
            candidates=all_candidates,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Symbol search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/explain/{candidate_id}", response_model=CandidateExplainResponse)
async def explain_candidate(
    candidate_id: str = Path(..., description="후보 ID"),
    chunk_id: str | None = Query(
        None, description="청크 ID (candidate_id가 chunk-xxx 형식이 아닌 경우 필수)"
    ),
    repo_id: str | None = Query(None, description="저장소 ID"),
):
    """
    검색 근거(Explainability) 제공 (Scoring Inspector)

    특정 candidate가 왜 선택되었는지
    lexical/semantic/fuzzy/graph 각 backend의 score와 rank를 설명.
    """
    try:
        # chunk_id가 없으면 candidate_id에서 추출 시도 (형식: cand-xxx 또는 chunk-xxx)
        if not chunk_id:
            if candidate_id.startswith("chunk-"):
                chunk_id = candidate_id
            else:
                # candidate_id만으로는 정보를 찾을 수 없으므로 chunk_id 필요
                raise HTTPException(
                    status_code=400,
                    detail="chunk_id is required. Use chunk_id from /hybrid/search response.",
                )

        # repo_id가 없으면 모든 저장소에서 검색
        if not repo_id:
            repos = bootstrap.repo_store.list_all()
            if not repos:
                raise HTTPException(status_code=400, detail="No repositories available")
            # 첫 번째 저장소에서 시도
            repo_id = repos[0].repo_id

        # 청크 조회
        chunk = bootstrap.chunk_store.get_chunk(repo_id, chunk_id)
        if not chunk:
            # 다른 저장소에서도 시도
            for repo in repos:
                chunk = bootstrap.chunk_store.get_chunk(repo.repo_id, chunk_id)
                if chunk:
                    repo_id = repo.repo_id
                    break

        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_id}")

        # 하이브리드 검색을 다시 실행하여 점수 정보 가져오기
        # (실제로는 세션에 저장된 정보를 사용해야 하지만, 여기서는 간단히 재검색)
        # 실제 구현에서는 세션에 후보 정보를 저장하고 참조해야 함

        # 임시로 기본 features 생성
        features = CandidateFeatures(
            lexical_score=0.0,
            semantic_score=0.0,
            graph_score=0.0,
            fuzzy_score=0.0,
            has_span=1.0 if chunk.span else 0.0,
            tfidf_score=None,
        )

        explained_candidate = ExplainedCandidate(
            candidate_id=candidate_id,
            doc_id=chunk.id,
            file_path=chunk.file_path,
            span=list(chunk.span),
            final_score=0.0,  # 실제로는 세션에서 가져와야 함
            features=features,
            backend_ranks=None,  # 실제로는 세션에서 가져와야 함
            debug_info=None,  # 실제로는 세션에서 가져와야 함
        )

        return CandidateExplainResponse(candidate=explained_candidate)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explain candidate failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/session/preferences", response_model=SessionPreferenceResponse)
async def set_session_preferences(request: SessionPreferenceRequest):
    """
    세션별 검색 전략 제어 (Retrieval Strategy Configurator)

    fusion mode(weighted / rrf / hybrid), 가중치 튜닝,
    특정 backend 활성/비활성 등을 세션 단위로 설정.
    """
    try:
        _session_preferences[request.session_id] = {
            "fusion_mode": request.fusion_mode,
            "weights": request.weights,
            "backend_enable": request.backend_enable,
        }

        return SessionPreferenceResponse(session_id=request.session_id, applied=True)
    except Exception as e:
        logger.error(f"Set session preferences failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/endpoints")
async def list_endpoints(
    repo_id: str = Query(..., description="저장소 ID"),
    method: str | None = Query(None, description="HTTP 메서드 필터 (GET, POST, ...)"),
    path_pattern: str | None = Query(None, description="경로 패턴 (/search, /api 등)"),
    framework: str | None = Query(None, description="프레임워크 필터 (fastapi, django, spring)"),
    k: int = Query(100, description="결과 개수"),
):
    """
    API 엔드포인트 목록 조회 (RouteIndex)

    저장소의 모든 API 엔드포인트를 조회하거나 필터링.

    Examples:
        GET /hybrid/endpoints?repo_id=my-repo
        GET /hybrid/endpoints?repo_id=my-repo&method=POST
        GET /hybrid/endpoints?repo_id=my-repo&path_pattern=search
        GET /hybrid/endpoints?repo_id=my-repo&framework=fastapi

    Returns:
        - total: 전체 엔드포인트 개수
        - by_file: 파일별 그룹핑
        - routes: 전체 라우트 리스트
    """
    try:
        # RouteStore가 없으면 에러
        if not hasattr(bootstrap, "route_store") or not bootstrap.route_store:
            raise HTTPException(
                status_code=503,
                detail="RouteIndex not available. Run migration 004 and reindex repository.",
            )

        # Route 검색
        routes = bootstrap.route_store.search_routes(
            repo_id=repo_id,
            method=method,
            path_pattern=path_pattern,
            framework=framework,
            k=k,
        )

        # 파일별 그룹핑
        by_file = {}
        for route in routes:
            file_path = route["file_path"]
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append(route)

        return {
            "repo_id": repo_id,
            "total": len(routes),
            "filters": {
                "method": method,
                "path_pattern": path_pattern,
                "framework": framework,
            },
            "by_file": by_file,
            "routes": routes,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List endpoints failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
