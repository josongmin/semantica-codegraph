"""프로젝트 전역 Enum 및 상수 정의"""

from enum import Enum


class NodeKind(str, Enum):
    """코드 및 문서 노드 타입"""

    # 코드 노드
    Function = "Function"
    Class = "Class"
    Method = "Method"
    Variable = "Variable"
    Import = "Import"
    File = "File"

    # 문서 노드 (텍스트 파일 인덱싱)
    Document = "Document"

    def is_code_node(self) -> bool:
        """코드 노드 여부 (관계 추출 필요)"""
        return self in {
            NodeKind.Function,
            NodeKind.Class,
            NodeKind.Method,
            NodeKind.Variable,
            NodeKind.Import,
        }

    def is_document_node(self) -> bool:
        """문서 노드 여부"""
        return self == NodeKind.Document

    def supports_edges(self) -> bool:
        """이 노드가 엣지(관계)를 가질 수 있는지"""
        return self.is_code_node()


class EdgeType(str, Enum):
    """코드 그래프 엣지(관계) 타입"""

    CALLS = "calls"  # 함수/메서드 호출
    DEFINES = "defines"  # 정의 관계
    BELONGS_TO = "belongs_to"  # 소속 관계 (메서드 → 클래스)
    INHERITS = "inherits"  # 상속 관계
    IMPORTS = "imports"  # 임포트 관계
    USES = "uses"  # 사용 관계 (변수, 타입 등)
    OVERRIDES = "overrides"  # 오버라이드 관계
    IMPLEMENTS = "implements"  # 인터페이스 구현


class LexicalSearchBackend(str, Enum):
    """BM25 검색 백엔드 선택"""

    MEILISEARCH = "meilisearch"
    ZOEKT = "zoekt"


class EmbeddingModel(str, Enum):
    """임베딩 모델 선택"""

    # Mistral (코드 특화 최고)
    CODESTRAL_EMBED = "codestral-embed"

    # OpenAI 모델
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"

    # HuggingFace Sentence Transformers (코드 특화)
    ALL_MINI_LM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
    ALL_MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"
    CODEBERT_BASE = "microsoft/codebert-base"

    # Cohere
    COHERE_V3 = "embed-english-v3.0"
