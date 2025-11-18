"""Config 테스트"""


from src.core.config import Config
from src.core.enums import EmbeddingModel, LexicalSearchBackend


def test_config_defaults():
    """Config 기본값 테스트"""
    config = Config()

    assert config.postgres_host == "localhost"
    assert config.postgres_port == 7711
    assert config.postgres_db == "semantica_codegraph"
    assert config.lexical_search_backend == LexicalSearchBackend.MEILISEARCH
    assert config.embedding_model == EmbeddingModel.CODESTRAL_EMBED


def test_config_from_env(monkeypatch):
    """환경변수에서 Config 로드 테스트"""
    monkeypatch.setenv("POSTGRES_HOST", "testhost")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "testdb")
    monkeypatch.setenv("LEXICAL_SEARCH_BACKEND", "zoekt")

    config = Config.from_env()

    assert config.postgres_host == "testhost"
    assert config.postgres_port == 5433
    assert config.postgres_db == "testdb"
    assert config.lexical_search_backend == LexicalSearchBackend.ZOEKT


def test_config_meilisearch():
    """MeiliSearch 설정 테스트"""
    config = Config()

    assert config.meilisearch_url == "http://localhost:7712"
    assert config.meilisearch_master_key is None


def test_config_zoekt():
    """Zoekt 설정 테스트"""
    config = Config()

    assert config.zoekt_url == "http://localhost:7713"
    assert config.zoekt_timeout == 30


def test_config_embedding():
    """임베딩 설정 테스트"""
    config = Config()

    assert config.embedding_model == EmbeddingModel.CODESTRAL_EMBED
    assert config.embedding_api_key is None
    assert config.embedding_dimension is None  # 모델 기본값 사용
    assert config.mistral_api_base == "https://api.mistral.ai/v1"

