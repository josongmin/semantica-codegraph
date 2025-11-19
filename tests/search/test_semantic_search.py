"""SemanticSearch 테스트"""

from unittest.mock import MagicMock

from src.core.models import ChunkResult
from src.search.adapters.semantic.pgvector_adapter import PgVectorSemanticSearch


def test_semantic_search_embed_text():
    """텍스트 임베딩 테스트"""
    embedding_service = MagicMock()
    embedding_service.embed_text.return_value = [0.1, 0.2, 0.3]

    embedding_store = MagicMock()

    search = PgVectorSemanticSearch(embedding_service, embedding_store)

    vector = search.embed_text("test query")
    assert vector == [0.1, 0.2, 0.3]
    embedding_service.embed_text.assert_called_once_with("test query")


def test_semantic_search_index_chunks():
    """청크 인덱싱 테스트"""
    embedding_service = MagicMock()
    embedding_service.embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]

    embedding_store = MagicMock()

    search = PgVectorSemanticSearch(embedding_service, embedding_store)

    search.index_chunks(
        repo_id="test-repo", chunk_ids=["chunk-1", "chunk-2"], texts=["text 1", "text 2"]
    )

    embedding_service.embed_texts.assert_called_once_with(["text 1", "text 2"])
    embedding_store.save_embeddings.assert_called_once_with(
        "test-repo", ["chunk-1", "chunk-2"], [[0.1, 0.2], [0.3, 0.4]]
    )


def test_semantic_search_search():
    """의미론적 검색 테스트"""
    embedding_service = MagicMock()
    embedding_service.embed_text.return_value = [0.5, 0.6]

    embedding_store = MagicMock()
    embedding_store.search_by_vector.return_value = [
        ChunkResult(
            repo_id="test-repo",
            chunk_id="chunk-1",
            score=0.9,
            source="embedding",
            file_path="test.py",
            span=(0, 0, 10, 0),
        )
    ]

    search = PgVectorSemanticSearch(embedding_service, embedding_store)

    results = search.search(repo_id="test-repo", query="test query", k=10)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score == 0.9

    embedding_service.embed_text.assert_called_once_with("test query")
    embedding_store.search_by_vector.assert_called_once()


def test_semantic_search_delete_repo():
    """저장소 인덱스 삭제 테스트"""
    embedding_service = MagicMock()
    embedding_store = MagicMock()

    search = PgVectorSemanticSearch(embedding_service, embedding_store)

    search.delete_repo_index("test-repo")

    embedding_store.delete_repo_embeddings.assert_called_once_with("test-repo")
