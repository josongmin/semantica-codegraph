"""PgVector 기반 의미론적 검색 어댑터"""

import logging
from typing import Dict, List, Optional

from ...core.models import ChunkResult, CodeChunk, RepoId
from ...core.ports import EmbeddingStorePort
from ...embedding.service import EmbeddingService
from ..ports.semantic_search_port import SemanticSearchPort

logger = logging.getLogger(__name__)


class PgVectorSemanticSearch(SemanticSearchPort):
    """
    PgVector 기반 의미론적 검색
    
    역할:
    - 쿼리 텍스트를 벡터로 변환
    - pgvector로 유사도 검색
    - ChunkResult 반환
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        embedding_store: EmbeddingStorePort
    ):
        """
        Args:
            embedding_service: 임베딩 서비스
            embedding_store: 임베딩 스토어 (pgvector)
        """
        self.embedding_service = embedding_service
        self.embedding_store = embedding_store

    def embed_text(self, text: str) -> List[float]:
        """텍스트를 벡터로 변환"""
        return self.embedding_service.embed_text(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """다수 텍스트를 벡터로 변환"""
        return self.embedding_service.embed_texts(texts)

    def index_chunks(
        self,
        repo_id: RepoId,
        chunk_ids: List[str],
        texts: List[str],
    ) -> None:
        """
        청크 임베딩 인덱싱
        
        Args:
            repo_id: 저장소 ID
            chunk_ids: 청크 ID 리스트
            texts: 텍스트 리스트
        """
        if not texts:
            logger.warning("No texts to index")
            return

        logger.info(f"Indexing {len(texts)} chunks for {repo_id}")
        
        # 임베딩 생성
        vectors = self.embedding_service.embed_texts(texts)
        
        # 저장
        self.embedding_store.save_embeddings(repo_id, chunk_ids, vectors)
        
        logger.info(f"Indexed {len(vectors)} embeddings")

    def search(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        filters: Optional[Dict] = None,
    ) -> List[ChunkResult]:
        """
        의미론적 검색 실행
        
        Args:
            repo_id: 저장소 ID
            query: 검색 쿼리
            k: 반환할 결과 수
            filters: 필터 (language, file_path 등)
        
        Returns:
            ChunkResult 리스트 (코사인 유사도 기준 정렬)
        """
        logger.debug(f"Semantic search: {query} (k={k})")
        
        # 쿼리 벡터 생성
        query_vector = self.embedding_service.embed_text(query)
        
        # 벡터 검색
        results = self.embedding_store.search_by_vector(
            repo_id=repo_id,
            vector=query_vector,
            k=k,
            filters=filters
        )
        
        logger.debug(f"Found {len(results)} results")
        return results

    def delete_repo_index(self, repo_id: RepoId) -> None:
        """저장소 인덱스 삭제"""
        logger.info(f"Deleting embeddings for {repo_id}")
        self.embedding_store.delete_repo_embeddings(repo_id)

