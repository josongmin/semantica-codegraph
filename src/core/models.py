from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# 기본 타입 정의
RepoId = str
Span = tuple[int, int, int, int]  # (start_line, start_col, end_line, end_col)


# Semantic Node (노드 수준 요약 + 임베딩)
@dataclass
class SemanticNode:
    """노드 수준 요약 + 임베딩"""

    repo_id: RepoId
    node_id: str
    node_type: str  # "symbol" | "route" | "doc" | "issue"
    summary: str
    summary_method: str  # "template" | "llm"
    model: str  # "text-embedding-3-small" | "text-embedding-3-large"
    embedding: list[float]

    # 선택적 필드
    doc_type: str | None = None  # node_type='doc'일 때
    source_table: str | None = None
    source_id: str | None = None
    metadata: dict = field(default_factory=dict)

    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class SemanticSearchResult:
    """Semantic 검색 결과"""

    repo_id: RepoId
    node_id: str
    node_type: str
    summary: str
    model: str
    score: float  # 유사도 점수 (0-1)

    # 메타데이터
    source_table: str | None = None
    source_id: str | None = None
    metadata: dict = field(default_factory=dict)


# 1) 파서 → IR 빌더 사이에서 사용하는 Raw 모델들
@dataclass
class RawSymbol:
    repo_id: RepoId
    file_path: str
    language: str
    kind: str  # "File" | "Class" | "Function" | ...
    name: str
    span: Span
    attrs: dict = field(default_factory=dict)  # 언어별 추가 정보 (파라미터, 반환 타입 등)


@dataclass
class RawRelation:
    repo_id: RepoId
    file_path: str
    language: str
    type: str  # "defines" | "calls" | "inherits" | ...
    src_span: Span
    dst_span: Span
    attrs: dict = field(default_factory=dict)  # 예: 호출 위치 컬럼, generic 정보 등


# 2) 언어 독립 Code Graph IR
@dataclass
class CodeNode:
    repo_id: RepoId
    id: str
    kind: str  # "File" | "Class" | "Function" | "Method" | ...
    language: str
    file_path: str
    span: Span
    name: str
    text: str
    attrs: dict = field(default_factory=dict)  # docstring, modifiers, visibility 등


@dataclass
class CodeEdge:
    repo_id: RepoId
    src_id: str
    dst_id: str
    type: str  # "defines" | "belongs_to" | "calls" | ...
    attrs: dict = field(default_factory=dict)


# 3) 청킹/검색용 모델
@dataclass
class CodeChunk:
    repo_id: RepoId
    id: str
    node_id: str
    file_path: str
    span: Span
    language: str
    text: str
    attrs: dict = field(default_factory=dict)  # node_kind, importance_score 등


@dataclass
class ChunkResult:
    repo_id: RepoId
    chunk_id: str
    score: float
    source: str  # "bm25" | "embedding" | "graph" 등
    file_path: str
    span: Span


@dataclass
class Candidate:
    repo_id: RepoId
    chunk_id: str
    features: dict[str, Any]  # bm25_score, embedding_score, graph_score, ...
    file_path: str
    span: Span
    metadata: dict = field(default_factory=dict)  # 추가 메타데이터 (importance_score, summary_method 등)


@dataclass
class NeighborNode:
    """
    그래프 탐색 결과 노드 (edge 정보 포함)

    HybridRetriever의 graph scoring에서 사용
    """

    node: CodeNode
    edge_type: str  # "calls" | "defines" | "inherits" | "imports" | ...
    depth: int  # 시작 노드로부터의 거리 (hop count)


# 4) LLM 컨텍스트용 모델
@dataclass
class PackedSnippet:
    repo_id: RepoId
    file_path: str
    span: Span
    role: str  # "primary" | "callee" | "caller" | "type" | "test" | ...
    text: str
    meta: dict = field(default_factory=dict)  # node_id, chunk_id, feature 요약 등


@dataclass
class PackedContext:
    primary: PackedSnippet
    supporting: list[PackedSnippet] = field(default_factory=list)


# 5) 위치/컨텍스트 정보 (MCP/IDE → 엔진)
@dataclass
class LocationContext:
    repo_id: RepoId
    file_path: str
    line: int
    column: int = 0
    symbol_name: str | None = None
    filters: dict | None = None  # language, 디렉토리, test-only 등
    extra: dict | None = None  # IDE에서 오는 추가 메타 (열려있는 탭 등)


# 6) 저장소 메타데이터
@dataclass
class RepoMetadata:
    repo_id: RepoId
    name: str
    root_path: str
    languages: list[str] = field(default_factory=list)
    indexed_at: datetime | None = None
    total_files: int = 0
    total_nodes: int = 0
    total_chunks: int = 0
    indexing_status: str = "pending"  # "pending" | "indexing" | "completed" | "failed"
    indexing_progress: float = 0.0
    attrs: dict = field(default_factory=dict)  # git_url, branch, tags 등


@dataclass
class RepoProfile:
    """
    저장소 구조 프로파일 (검색 최적화용)

    Repo Profiling 단계에서 생성되며, 다음 용도로 사용:
    - 검색 쿼리 재작성 (Intent Router)
    - 파일 경로 기반 재순위화 (HybridRetriever)
    - LLM 컨텍스트 제공
    """

    repo_id: RepoId

    # 언어 분포
    languages: dict[str, int] = field(default_factory=dict)  # {"python": 15000, "typescript": 3000}
    primary_language: str = "unknown"

    # 프레임워크/라이브러리 감지
    framework: str | None = None  # "fastapi", "django", "express", "spring", etc.
    frameworks: list[str] = field(default_factory=list)  # 여러 프레임워크 사용 가능

    # API 패턴 (FastAPI, Express 등)
    api_patterns: list[str] = field(
        default_factory=list
    )  # ["@router.post", "@app.get", "express.Router"]

    # 주요 디렉토리 분류
    api_directories: list[str] = field(default_factory=list)  # ["apps/api/routes/", "api/"]
    service_directories: list[str] = field(default_factory=list)  # ["src/services/", "services/"]
    model_directories: list[str] = field(default_factory=list)  # ["src/models/", "models/"]
    test_directories: list[str] = field(default_factory=list)  # ["tests/", "test/"]
    config_directories: list[str] = field(default_factory=list)  # ["config/", "settings/"]

    # 엔트리포인트
    entry_points: list[str] = field(default_factory=list)  # ["apps/api/main.py", "src/index.ts"]

    # 의존성 정보
    dependencies: dict[str, str] = field(default_factory=dict)  # {"fastapi": "0.104.0"}

    # 프로젝트 타입 추론
    project_type: str = "unknown"  # "web_api", "library", "cli", "microservice"

    # 통계
    total_directories: int = 0
    file_tree: dict = field(default_factory=dict)  # 간략한 디렉토리 구조


@dataclass
class FileProfile:
    """
    파일 역할 프로파일 (검색 최적화용)

    File Profiling 단계에서 생성되며, 검색 재순위화에 사용
    """

    repo_id: RepoId
    file_path: str

    # 파일 역할 태그
    is_api_file: bool = False
    is_router: bool = False
    is_controller: bool = False
    is_service: bool = False
    is_model: bool = False
    is_schema: bool = False
    is_config: bool = False
    is_test_file: bool = False
    is_entry_point: bool = False

    # API 파일 상세 정보
    api_framework: str | None = None  # "fastapi", "express", "spring"
    api_patterns: list[str] = field(default_factory=list)  # ["@router.post", "@app.get"]
    endpoints: list[dict] = field(default_factory=list)  # [{"method": "POST", "path": "/search"}]

    # Import 정보
    imports: list[str] = field(default_factory=list)  # 주요 import 목록
    external_deps: list[str] = field(default_factory=list)  # 외부 라이브러리
    internal_deps: list[str] = field(default_factory=list)  # 내부 모듈

    # 통계
    line_count: int = 0
    function_count: int = 0
    class_count: int = 0


# 7) 인덱싱 상태 추적
@dataclass
class IndexingStatus:
    repo_id: RepoId
    status: str  # "pending" | "indexing" | "completed" | "failed"
    progress: float = 0.0  # 0.0 ~ 1.0
    current_file: str | None = None
    total_files: int = 0
    processed_files: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None


# 8) 저장소 설정
@dataclass
class RepoConfig:
    """저장소별 인덱싱 설정 (최소 필수)"""

    # 제외 패턴
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/.git/**",
            "**/*.min.js",
            "**/*.bundle.js",
            "**/dist/**",
            "**/build/**",
            "**/.venv/**",
            "**/.env/**",
        ]
    )

    # 언어 필터 (빈 리스트 = 모든 언어)
    languages: list[str] = field(default_factory=list)

    # 테스트 파일 포함 여부
    include_tests: bool = False

    # 텍스트/문서 파일 인덱싱 여부
    index_text_files: bool = True


# 9) 파일 메타데이터 (RepoScanner 출력)
@dataclass
class FileMetadata:
    """파일 메타데이터 (최소)"""

    file_path: str  # 저장소 루트 기준 상대 경로
    abs_path: str  # 절대 경로
    language: str  # "python" | "typescript" | "javascript" | ...


# 10) 인덱싱 결과
@dataclass
class IndexingResult:
    """인덱싱 파이프라인 실행 결과"""

    repo_id: RepoId
    status: str  # "completed" | "failed"
    total_files: int = 0
    processed_files: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    total_chunks: int = 0
    duration_seconds: float = 0.0
    error_message: str | None = None
