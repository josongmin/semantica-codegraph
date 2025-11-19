"""Hybrid Search API 스키마"""


from pydantic import BaseModel


class LocationContextModel(BaseModel):
    """위치 컨텍스트"""

    repo_id: str
    file_path: str
    line: int
    column: int = 0
    symbol_name: str | None = None
    node_id: str | None = None


class HybridSearchRequest(BaseModel):
    """하이브리드 검색 요청"""

    query: str
    k: int = 20
    location_ctx: LocationContextModel | None = None
    intent: str | None = None  # "question_answering", "code_exploration" 등
    backend_hints: list[str] | None = None  # ["lexical", "semantic", "graph", "fuzzy"]
    session_id: str | None = None


class CandidateScores(BaseModel):
    """후보 점수"""

    lexical: float | None = None
    semantic: float | None = None
    graph: float | None = None
    fuzzy: float | None = None


class CandidateExtraFeatures(BaseModel):
    """후보 추가 특성"""

    has_span: float | None = None
    tfidf_score: float | None = None


class HybridCandidate(BaseModel):
    """하이브리드 검색 후보"""

    candidate_id: str
    doc_id: str
    chunk_id: str
    file_path: str
    span: list[int]  # [start_line, start_col, end_line, end_col]
    snippet: str
    language: str
    symbol_name: str | None = None
    node_id: str | None = None
    final_score: float
    scores: CandidateScores
    extra_features: CandidateExtraFeatures | None = None


class HybridSearchResponse(BaseModel):
    """하이브리드 검색 응답"""

    query: str
    k: int
    candidates: list[HybridCandidate]


class ChunkFetchRequest(BaseModel):
    """청크 조회 요청"""

    chunk_ids: list[str]
    include_source: bool = True
    include_neighbors: bool = False


class ChunkMetadata(BaseModel):
    """청크 메타데이터"""

    size: int | None = None
    type: str | None = None


class ChunkInfo(BaseModel):
    """청크 정보"""

    chunk_id: str
    doc_id: str
    file_path: str
    span: list[int]  # [start_line, start_col, end_line, end_col]
    content: str
    language: str
    node_ids: list[str]
    metadata: ChunkMetadata | None = None


class ChunkFetchResponse(BaseModel):
    """청크 조회 응답"""

    chunks: list[ChunkInfo]


class GraphExpandRequest(BaseModel):
    """그래프 확장 요청"""

    node_id: str
    chunk_id: str | None = None
    edge_types: list[str] | None = None  # ["CALLS", "SAME_FILE"] 등
    max_depth: int = 2
    k: int = 20
    session_id: str | None = None


class GraphNeighbor(BaseModel):
    """그래프 이웃 노드"""

    candidate_id: str
    node_id: str
    file_path: str
    span: list[int]  # [start_line, start_col, end_line, end_col]
    snippet: str
    edge_type: str
    depth: int
    graph_score: float


class GraphExpandResponse(BaseModel):
    """그래프 확장 응답"""

    neighbors: list[GraphNeighbor]


class SymbolSearchRequest(BaseModel):
    """심볼 검색 요청"""

    symbol_query: str
    k: int = 20
    session_id: str | None = None


class SymbolCandidate(BaseModel):
    """심볼 검색 후보"""

    candidate_id: str
    symbol_name: str
    match_score: float
    file_path: str
    span: list[int]  # [start_line, start_col, end_line, end_col]
    snippet: str
    metadata: dict = Field(default_factory=dict)  # kind, language, decorators, docstring 등


class SymbolSearchResponse(BaseModel):
    """심볼 검색 응답"""

    symbol_query: str
    candidates: list[SymbolCandidate]


class CandidateFeatures(BaseModel):
    """후보 특성 (Explain용)"""

    lexical_score: float | None = None
    semantic_score: float | None = None
    graph_score: float | None = None
    fuzzy_score: float | None = None
    has_span: float | None = None
    tfidf_score: float | None = None


class CandidateBackendRanks(BaseModel):
    """백엔드별 순위"""

    lexical: int | None = None
    semantic: int | None = None
    graph: int | None = None
    fuzzy: int | None = None


class CandidateDebugInfo(BaseModel):
    """디버그 정보"""

    lexical_raw: float | None = None
    semantic_raw: float | None = None
    fuzzy_raw: float | None = None


class ExplainedCandidate(BaseModel):
    """설명된 후보"""

    candidate_id: str
    doc_id: str
    file_path: str
    span: list[int]  # [start_line, start_col, end_line, end_col]
    final_score: float
    features: CandidateFeatures
    backend_ranks: CandidateBackendRanks | None = None
    debug_info: CandidateDebugInfo | None = None


class CandidateExplainResponse(BaseModel):
    """후보 설명 응답"""

    candidate: ExplainedCandidate


class SessionPreferenceRequest(BaseModel):
    """세션 설정 요청"""

    session_id: str
    fusion_mode: str | None = "weighted"  # "weighted" | "rrf" | "combsum"
    weights: dict[str, float] | None = None
    backend_enable: dict[str, bool] | None = None


class SessionPreferenceResponse(BaseModel):
    """세션 설정 응답"""

    session_id: str
    applied: bool
