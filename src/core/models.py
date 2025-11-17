from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 기본 타입 정의
RepoId = str
Span = Tuple[int, int, int, int]  # (start_line, start_col, end_line, end_col)


# 1) 파서 → IR 빌더 사이에서 사용하는 Raw 모델들
@dataclass
class RawSymbol:
    repo_id: RepoId
    file_path: str
    language: str
    kind: str  # "File" | "Class" | "Function" | ...
    name: str
    span: Span
    attrs: Dict = field(default_factory=dict)  # 언어별 추가 정보 (파라미터, 반환 타입 등)


@dataclass
class RawRelation:
    repo_id: RepoId
    file_path: str
    language: str
    type: str  # "defines" | "calls" | "inherits" | ...
    src_span: Span
    dst_span: Span
    attrs: Dict = field(default_factory=dict)  # 예: 호출 위치 컬럼, generic 정보 등


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
    attrs: Dict = field(default_factory=dict)  # docstring, modifiers, visibility 등


@dataclass
class CodeEdge:
    repo_id: RepoId
    src_id: str
    dst_id: str
    type: str  # "defines" | "belongs_to" | "calls" | ...
    attrs: Dict = field(default_factory=dict)


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
    attrs: Dict = field(default_factory=dict)  # node_kind, importance_score 등


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
    features: Dict[str, float]  # bm25_score, embedding_score, graph_score, ...
    file_path: str
    span: Span


# 4) LLM 컨텍스트용 모델
@dataclass
class PackedSnippet:
    repo_id: RepoId
    file_path: str
    span: Span
    role: str  # "primary" | "callee" | "caller" | "type" | "test" | ...
    text: str
    meta: Dict = field(default_factory=dict)  # node_id, chunk_id, feature 요약 등


@dataclass
class PackedContext:
    primary: PackedSnippet
    supporting: List[PackedSnippet] = field(default_factory=list)


# 5) 위치/컨텍스트 정보 (MCP/IDE → 엔진)
@dataclass
class LocationContext:
    repo_id: RepoId
    file_path: str
    line: int
    column: int = 0
    symbol_name: Optional[str] = None
    filters: Optional[Dict] = None  # language, 디렉토리, test-only 등
    extra: Optional[Dict] = None  # IDE에서 오는 추가 메타 (열려있는 탭 등)


# 6) 저장소 메타데이터
@dataclass
class RepoMetadata:
    repo_id: RepoId
    name: str
    root_path: str
    languages: List[str] = field(default_factory=list)
    indexed_at: Optional[datetime] = None
    total_files: int = 0
    total_nodes: int = 0
    total_chunks: int = 0
    attrs: Dict = field(default_factory=dict)  # git_url, branch, tags 등


# 7) 인덱싱 상태 추적
@dataclass
class IndexingStatus:
    repo_id: RepoId
    status: str  # "pending" | "indexing" | "completed" | "failed"
    progress: float = 0.0  # 0.0 ~ 1.0
    current_file: Optional[str] = None
    total_files: int = 0
    processed_files: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


# 8) 저장소 설정
@dataclass
class RepoConfig:
    """저장소별 인덱싱 설정 (최소 필수)"""
    
    # 제외 패턴
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/node_modules/**",
        "**/__pycache__/**",
        "**/.git/**",
        "**/*.min.js",
        "**/*.bundle.js",
        "**/dist/**",
        "**/build/**",
        "**/.venv/**",
        "**/.env/**",
    ])
    
    # 언어 필터 (빈 리스트 = 모든 언어)
    languages: List[str] = field(default_factory=list)
    
    # 테스트 파일 포함 여부
    include_tests: bool = True


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
    error_message: Optional[str] = None

