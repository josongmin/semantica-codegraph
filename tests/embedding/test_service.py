"""EmbeddingService 테스트"""

import pytest

from src.core.enums import EmbeddingModel
from src.core.models import CodeChunk


@pytest.fixture
def sample_chunk():
    """샘플 CodeChunk"""
    return CodeChunk(
        repo_id="test",
        id="chunk-1",
        node_id="node-1",
        file_path="main.py",
        span=(0, 0, 5, 0),
        language="python",
        text="def hello():\n    return 'world'",
        attrs={"docstring": "Returns world"}
    )


def test_prepare_chunk_text(sample_chunk):
    """청크 텍스트 준비 테스트 (초기화 없이)"""
    from src.embedding.service import EmbeddingService

    # 초기화 없이 인스턴스만 생성
    service = EmbeddingService.__new__(EmbeddingService)

    # 메서드만 테스트
    text = service._prepare_chunk_text(sample_chunk)

    assert "def hello():" in text
    assert "Purpose: Returns world" in text


def test_get_dimension_codestral():
    """Codestral 차원 테스트"""
    from src.embedding.service import EmbeddingService

    service = EmbeddingService.__new__(EmbeddingService)
    service.model = EmbeddingModel.CODESTRAL_EMBED
    service.dimension = None

    dim = service.get_dimension()
    assert dim == 1536, "Codestral 기본 차원은 1536"


def test_get_dimension_openai_small():
    """OpenAI 3-small 차원 테스트"""
    from src.embedding.service import EmbeddingService

    service = EmbeddingService.__new__(EmbeddingService)
    service.model = EmbeddingModel.OPENAI_3_SMALL
    service.dimension = None

    dim = service.get_dimension()
    assert dim == 1536


def test_get_dimension_custom():
    """커스텀 차원 테스트"""
    from src.embedding.service import EmbeddingService

    service = EmbeddingService.__new__(EmbeddingService)
    service.model = EmbeddingModel.CODESTRAL_EMBED
    service.dimension = 256  # 커스텀

    dim = service.get_dimension()
    assert dim == 256, "커스텀 차원 사용"


def test_embed_text_codestral():
    """Codestral 임베딩 생성 테스트"""
    from src.core.config import Config
    from src.embedding.service import EmbeddingService

    config = Config.from_env()

    if not config.embedding_api_key:
        pytest.skip("EMBEDDING_API_KEY not set")

    service = EmbeddingService(
        model=EmbeddingModel.CODESTRAL_EMBED,
        api_key=config.embedding_api_key,
    )

    text = "def calculate_total(items): return sum(item.price for item in items)"
    vector = service.embed_text(text)

    # Codestral의 기본 차원은 1536
    assert len(vector) == 1536
    assert all(isinstance(v, float) for v in vector)
    assert all(-1 <= v <= 1 for v in vector), "벡터 값 범위 확인"


def test_embed_batch_codestral():
    """Codestral 배치 임베딩 테스트"""
    from src.core.config import Config
    from src.embedding.service import EmbeddingService

    config = Config.from_env()

    if not config.embedding_api_key:
        pytest.skip("EMBEDDING_API_KEY not set")

    service = EmbeddingService(
        model=EmbeddingModel.CODESTRAL_EMBED,
        api_key=config.embedding_api_key,
    )

    texts = [
        "def foo(): return 42",
        "class Bar: pass",
        "async def baz(): await something()"
    ]

    vectors = service.embed_texts(texts)

    assert len(vectors) == 3
    assert all(len(v) == 1536 for v in vectors)


def test_embed_chunk_codestral(sample_chunk):
    """Codestral 청크 임베딩 테스트"""
    from src.core.config import Config
    from src.embedding.service import EmbeddingService

    config = Config.from_env()

    if not config.embedding_api_key:
        pytest.skip("EMBEDDING_API_KEY not set")

    service = EmbeddingService(
        model=config.embedding_model,
        api_key=config.embedding_api_key,
    )

    vector = service.embed_chunk(sample_chunk)

    # 기본 차원 사용 (Codestral은 1536)
    assert len(vector) == 1536
    assert isinstance(vector, list)

