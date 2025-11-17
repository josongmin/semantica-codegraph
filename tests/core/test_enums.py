"""Enums 테스트"""

from src.core.enums import EmbeddingModel, LexicalSearchBackend


def test_lexical_search_backend_enum():
    """LexicalSearchBackend Enum 테스트"""
    assert LexicalSearchBackend.MEILISEARCH.value == "meilisearch"
    assert LexicalSearchBackend.ZOEKT.value == "zoekt"
    
    # 문자열로 생성
    backend = LexicalSearchBackend("meilisearch")
    assert backend == LexicalSearchBackend.MEILISEARCH


def test_embedding_model_enum():
    """EmbeddingModel Enum 테스트"""
    assert EmbeddingModel.OPENAI_ADA_002.value == "text-embedding-ada-002"
    assert EmbeddingModel.ALL_MINI_LM_L6_V2.value == "sentence-transformers/all-MiniLM-L6-v2"
    assert EmbeddingModel.CODEBERT_BASE.value == "microsoft/codebert-base"
    
    # 문자열로 생성
    model = EmbeddingModel("text-embedding-3-small")
    assert model == EmbeddingModel.OPENAI_3_SMALL


def test_all_embedding_models():
    """모든 임베딩 모델 확인"""
    models = [
        EmbeddingModel.OPENAI_ADA_002,
        EmbeddingModel.OPENAI_3_SMALL,
        EmbeddingModel.OPENAI_3_LARGE,
        EmbeddingModel.ALL_MINI_LM_L6_V2,
        EmbeddingModel.ALL_MPNET_BASE_V2,
        EmbeddingModel.CODEBERT_BASE,
        EmbeddingModel.COHERE_V3
    ]
    
    assert len(models) == 7
    assert all(model.value for model in models)

