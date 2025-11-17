"""pgvector 기반 임베딩 저장소"""

import logging
from typing import Dict, List, Optional

import psycopg2
from psycopg2.extras import execute_batch

from ..core.models import ChunkResult, RepoId
from ..core.ports import EmbeddingStorePort

logger = logging.getLogger(__name__)


class PgVectorStore(EmbeddingStorePort):
    """
    pgvector 기반 임베딩 저장소
    
    기능:
    - 임베딩 벡터 저장
    - 코사인 유사도 검색
    - 배치 저장
    """

    def __init__(
        self,
        connection_string: str,
        embedding_dimension: int = 384,
        model_name: str = "default"
    ):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
            embedding_dimension: 벡터 차원 (모델에 따라)
            model_name: 임베딩 모델 이름
        """
        self.connection_string = connection_string
        self.embedding_dimension = embedding_dimension
        self.model_name = model_name
        self._ensure_tables()

    def _get_connection(self):
        """DB 연결 생성"""
        return psycopg2.connect(self.connection_string)

    def _ensure_tables(self):
        """테이블 생성"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # embeddings 테이블
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        repo_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        model TEXT NOT NULL,
                        embedding vector({self.embedding_dimension}),
                        created_at TIMESTAMP DEFAULT NOW(),
                        PRIMARY KEY (repo_id, chunk_id, model)
                    )
                """)

                # HNSW 인덱스 (빠른 근사 검색)
                # PostgreSQL 16+ 필요
                try:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw 
                        ON embeddings USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64)
                    """)
                except psycopg2.errors.UndefinedFunction:
                    # HNSW 지원 안 되면 IVFFlat 사용
                    logger.warning("HNSW not supported, using IVFFlat")
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_embeddings_ivfflat 
                        ON embeddings USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                    """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_model 
                    ON embeddings(repo_id, model)
                """)

                conn.commit()

        logger.info("Embedding tables ensured")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트를 벡터로 변환
        
        Note: 실제 임베딩은 EmbeddingService에서 수행
              여기서는 placeholder
        """
        # TODO: Phase 6에서 EmbeddingService 연동
        raise NotImplementedError("embed_texts는 EmbeddingService에서 구현")

    def save_embeddings(
        self,
        repo_id: RepoId,
        chunk_ids: List[str],
        vectors: List[List[float]],
    ) -> None:
        """임베딩 벡터 저장"""
        if not chunk_ids or not vectors:
            logger.warning("No embeddings to save")
            return

        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids and vectors length mismatch")

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                embedding_data = [
                    (
                        repo_id,
                        chunk_id,
                        self.model_name,
                        vector
                    )
                    for chunk_id, vector in zip(chunk_ids, vectors)
                ]

                execute_batch(
                    cur,
                    """
                    INSERT INTO embeddings (repo_id, chunk_id, model, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (repo_id, chunk_id, model) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        created_at = NOW()
                    """,
                    embedding_data
                )

                conn.commit()

        logger.info(f"Saved {len(chunk_ids)} embeddings for {repo_id}")

    def search_by_vector(
        self,
        repo_id: RepoId,
        vector: List[float],
        k: int,
        filters: Optional[Dict] = None,
    ) -> List[ChunkResult]:
        """
        벡터 유사도 검색
        
        Args:
            repo_id: 저장소 ID
            vector: 쿼리 벡터
            k: 반환할 결과 수
            filters: 필터 (language, file_path 등)
        
        Returns:
            ChunkResult 리스트 (코사인 유사도 기준)
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 기본 쿼리
                query = """
                    SELECT 
                        e.chunk_id,
                        1 - (e.embedding <=> %s::vector) as score,
                        c.file_path,
                        c.span_start_line,
                        c.span_start_col,
                        c.span_end_line,
                        c.span_end_col
                    FROM embeddings e
                    JOIN code_chunks c ON e.repo_id = c.repo_id AND e.chunk_id = c.chunk_id
                    WHERE e.repo_id = %s AND e.model = %s
                """

                params = [vector, repo_id, self.model_name]

                # 필터 추가
                if filters:
                    if "language" in filters:
                        query += " AND c.language = %s"
                        params.append(filters["language"])
                    if "file_path" in filters:
                        query += " AND c.file_path = %s"
                        params.append(filters["file_path"])

                # 정렬 및 제한
                query += " ORDER BY e.embedding <=> %s::vector LIMIT %s"
                params.extend([vector, k])

                cur.execute(query, params)

                results = []
                for row in cur.fetchall():
                    results.append(
                        ChunkResult(
                            repo_id=repo_id,
                            chunk_id=row[0],
                            score=float(row[1]),
                            source="embedding",
                            file_path=row[2],
                            span=(row[3], row[4], row[5], row[6])
                        )
                    )

        logger.debug(f"Vector search found {len(results)} results")
        return results

    def delete_repo_embeddings(self, repo_id: RepoId) -> None:
        """저장소의 모든 임베딩 삭제"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM embeddings WHERE repo_id = %s",
                    (repo_id,)
                )
                deleted = cur.rowcount
                conn.commit()

        logger.info(f"Deleted {deleted} embeddings for repo: {repo_id}")

