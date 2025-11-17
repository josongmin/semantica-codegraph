"""프로젝트 전역 Enum 및 상수 정의"""

from enum import Enum


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

