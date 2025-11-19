"""pgvector 기반 임베딩 저장소"""

import logging

import psycopg2
from psycopg2 import pool
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
        model_name: str = "default",
        pool_size: int = 10,
        pool_max: int = 20,
    ):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
            embedding_dimension: 벡터 차원 (모델에 따라)
            model_name: 임베딩 모델 이름
            pool_size: 커넥션 풀 최소 크기
            pool_max: 커넥션 풀 최대 크기
        """
        self.connection_string = connection_string
        self.embedding_dimension = embedding_dimension
        self.model_name = model_name

        # 커넥션 풀 생성
        self._pool = pool.SimpleConnectionPool(pool_size, pool_max, connection_string)
        logger.info(f"Created connection pool: min={pool_size}, max={pool_max}")

        self._ensure_tables()

    def _get_connection(self):
        """DB 연결 풀에서 가져오기"""
        return self._pool.getconn()

    def _put_connection(self, conn):
        """DB 연결 풀에 반환"""
        self._pool.putconn(conn)

    def close(self):
        """커넥션 풀 종료"""
        if self._pool:
            self._pool.closeall()
            logger.info("Connection pool closed")

    def _ensure_tables(self):
        """테이블 생성"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # 기존 테이블 존재 여부 확인
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'embeddings'
                    )
                """
                )
                table_exists = cur.fetchone()[0]

                # 테이블이 존재하면 차원 확인
                if table_exists:
                    # information_schema에서 컬럼 정의 확인
                    cur.execute(
                        """
                        SELECT
                            udt_name,
                            character_maximum_length
                        FROM information_schema.columns
                        WHERE table_name = 'embeddings'
                        AND column_name = 'embedding'
                    """
                    )
                    try:
                        col_info = cur.fetchone()
                        if col_info:
                            # pgvector는 udt_name이 'vector'이고 차원은 별도로 저장되지 않음
                            # 대신 pg_attribute에서 확인
                            cur.execute(
                                """
                                SELECT
                                    format_type(atttypid, atttypmod) as type_name
                                FROM pg_attribute
                                WHERE attrelid = 'embeddings'::regclass
                                AND attname = 'embedding'
                            """
                            )
                            type_result = cur.fetchone()
                            if type_result and type_result[0]:
                                # vector(384) 형식에서 숫자 추출
                                import re

                                match = re.search(r"vector\((\d+)\)", type_result[0])
                                if match:
                                    existing_dim = int(match.group(1))
                                    if existing_dim != self.embedding_dimension:
                                        logger.info(
                                            f"Recreating embeddings table: "
                                            f"{existing_dim} -> {self.embedding_dimension} dimensions"
                                        )
                                        cur.execute("DROP TABLE IF EXISTS embeddings CASCADE")
                                        table_exists = False
                    except Exception as e:
                        # 오류 발생 시 재생성
                        logger.debug(f"Could not check dimension, recreating table: {e}")
                        cur.execute("DROP TABLE IF EXISTS embeddings CASCADE")
                        table_exists = False

                # embeddings 테이블 생성
                if not table_exists:
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS embeddings (
                            repo_id TEXT NOT NULL,
                            chunk_id TEXT NOT NULL,
                            model TEXT NOT NULL,
                            embedding vector({self.embedding_dimension}),
                            content_hash TEXT,
                            created_at TIMESTAMP DEFAULT NOW(),
                            PRIMARY KEY (repo_id, chunk_id, model)
                        )
                    """
                    )
                else:
                    # content_hash 컬럼 추가 (기존 테이블에)
                    try:
                        cur.execute(
                            """
                            ALTER TABLE embeddings
                            ADD COLUMN IF NOT EXISTS content_hash TEXT
                        """
                        )
                    except Exception as e:
                        logger.debug(f"content_hash column may already exist: {e}")

                # HNSW 인덱스 (빠른 근사 검색)
                # PostgreSQL 16+ 필요
                try:
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
                        ON embeddings USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64)
                    """
                    )
                except psycopg2.errors.UndefinedFunction:
                    # HNSW 지원 안 되면 IVFFlat 사용
                    logger.warning("HNSW not supported, using IVFFlat")
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_embeddings_ivfflat
                        ON embeddings USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                    """
                    )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_embeddings_model
                    ON embeddings(repo_id, model)
                """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_embeddings_content_hash
                    ON embeddings(content_hash)
                """
                )

                conn.commit()
        finally:
            self._put_connection(conn)

        logger.info("Embedding tables ensured")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
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
        chunk_ids: list[str],
        vectors: list[list[float]],
        content_hashes: list[str] | None = None,
    ) -> None:
        """임베딩 벡터 저장"""
        if not chunk_ids or not vectors:
            logger.warning("No embeddings to save")
            return

        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids and vectors length mismatch")

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if content_hashes and len(content_hashes) == len(chunk_ids):
                    # content_hash 포함
                    embedding_data: list[tuple[str, str, str, list[float], str]] = [
                        (repo_id, chunk_id, self.model_name, vector, content_hash)
                        for chunk_id, vector, content_hash in zip(
                            chunk_ids, vectors, content_hashes, strict=False
                        )
                    ]

                    execute_batch(
                        cur,
                        """
                        INSERT INTO embeddings (repo_id, chunk_id, model, embedding, content_hash)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (repo_id, chunk_id, model) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            content_hash = EXCLUDED.content_hash,
                            created_at = NOW()
                        """,
                        embedding_data,
                        page_size=500,
                    )
                else:
                    # 기존 방식 (content_hash 없음)
                    embedding_data_simple: list[tuple[str, str, str, list[float]]] = [
                        (repo_id, chunk_id, self.model_name, vector)
                        for chunk_id, vector in zip(chunk_ids, vectors, strict=False)
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
                        embedding_data_simple,
                        page_size=500,
                    )

                conn.commit()
        finally:
            self._put_connection(conn)

        logger.info(f"Saved {len(chunk_ids)} embeddings for {repo_id}")

    def get_embedding_by_content_hash(self, content_hash: str, model: str) -> list[float] | None:
        """content_hash로 기존 임베딩 조회"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT embedding
                    FROM embeddings
                    WHERE content_hash = %s AND model = %s
                    LIMIT 1
                    """,
                    (content_hash, model),
                )
                row = cur.fetchone()
                if row:
                    return list(row[0])
        finally:
            self._put_connection(conn)

        return None

    def get_embeddings_by_content_hashes(
        self, content_hashes: list[str], model: str
    ) -> dict[str, list[float]]:
        """여러 content_hash로 임베딩 조회 (배치)"""
        if not content_hashes:
            return {}

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content_hash, embedding
                    FROM embeddings
                    WHERE content_hash = ANY(%s) AND model = %s
                    """,
                    (content_hashes, model),
                )

                result = {}
                for row in cur.fetchall():
                    result[row[0]] = row[1]

                return result
        finally:
            self._put_connection(conn)

    def search_by_vector(
        self,
        repo_id: RepoId,
        vector: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[ChunkResult]:
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
        conn = self._get_connection()
        try:
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
                params.append(vector)
                params.append(k)

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
                            span=(row[3], row[4], row[5], row[6]),
                        )
                    )
        finally:
            self._put_connection(conn)

        logger.debug(f"Vector search found {len(results)} results")
        return results

    def delete_repo_embeddings(self, repo_id: RepoId) -> None:
        """저장소의 모든 임베딩 삭제"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM embeddings WHERE repo_id = %s", (repo_id,))
                deleted = cur.rowcount
                conn.commit()
        finally:
            self._put_connection(conn)

        logger.info(f"Deleted {deleted} embeddings for repo: {repo_id}")
