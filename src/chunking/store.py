"""PostgreSQL 기반 ChunkStore 구현"""

import json
import logging
from typing import List, Optional

import psycopg2
from psycopg2.extras import execute_batch

from ..core.models import CodeChunk, RepoId
from ..core.ports import ChunkStorePort

logger = logging.getLogger(__name__)


class PostgresChunkStore(ChunkStorePort):
    """
    PostgreSQL 기반 청크 저장소
    
    핵심 기능:
    - Chunk 저장/조회
    - 위치 기반 조회 (Zoekt 매핑에 필수!)
    - Node 기반 조회
    """

    def __init__(self, connection_string: str):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
        """
        self.connection_string = connection_string
        self._ensure_tables()

    def _get_connection(self):
        """DB 연결 생성"""
        return psycopg2.connect(self.connection_string)

    def _ensure_tables(self):
        """테이블 생성"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS code_chunks (
                        repo_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        node_id TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        span_start_line INTEGER NOT NULL,
                        span_start_col INTEGER NOT NULL,
                        span_end_line INTEGER NOT NULL,
                        span_end_col INTEGER NOT NULL,
                        language TEXT NOT NULL,
                        text TEXT NOT NULL,
                        token_count INTEGER,
                        attrs JSONB,
                        created_at TIMESTAMP DEFAULT NOW(),
                        PRIMARY KEY (repo_id, chunk_id)
                    )
                """)

                # Zoekt 매핑에 필수!
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_location 
                    ON code_chunks(repo_id, file_path, span_start_line, span_end_line)
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_node 
                    ON code_chunks(repo_id, node_id)
                """)

                conn.commit()

        logger.info("Chunk tables ensured")

    def save_chunks(self, chunks: List[CodeChunk]) -> None:
        """청크 저장"""
        if not chunks:
            logger.warning("No chunks to save")
            return

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                chunk_data = [
                    (
                        chunk.repo_id,
                        chunk.id,
                        chunk.node_id,
                        chunk.file_path,
                        chunk.span[0],
                        chunk.span[1],
                        chunk.span[2],
                        chunk.span[3],
                        chunk.language,
                        chunk.text,
                        json.dumps(chunk.attrs)
                    )
                    for chunk in chunks
                ]

                execute_batch(
                    cur,
                    """
                    INSERT INTO code_chunks (
                        repo_id, chunk_id, node_id, file_path,
                        span_start_line, span_start_col, span_end_line, span_end_col,
                        language, text, attrs
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (repo_id, chunk_id) DO UPDATE SET
                        node_id = EXCLUDED.node_id,
                        file_path = EXCLUDED.file_path,
                        span_start_line = EXCLUDED.span_start_line,
                        span_start_col = EXCLUDED.span_start_col,
                        span_end_line = EXCLUDED.span_end_line,
                        span_end_col = EXCLUDED.span_end_col,
                        language = EXCLUDED.language,
                        text = EXCLUDED.text,
                        attrs = EXCLUDED.attrs
                    """,
                    chunk_data
                )

                conn.commit()

        logger.info(f"Saved {len(chunks)} chunks")

    def get_chunk(self, repo_id: RepoId, chunk_id: str) -> Optional[CodeChunk]:
        """단일 청크 조회"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT repo_id, chunk_id, node_id, file_path,
                           span_start_line, span_start_col, span_end_line, span_end_col,
                           language, text, attrs
                    FROM code_chunks
                    WHERE repo_id = %s AND chunk_id = %s
                    """,
                    (repo_id, chunk_id)
                )

                row = cur.fetchone()
                if row:
                    return self._row_to_chunk(row)

        return None

    def get_chunks_by_node(
        self,
        repo_id: RepoId,
        node_id: str,
    ) -> List[CodeChunk]:
        """노드로 청크 조회"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT repo_id, chunk_id, node_id, file_path,
                           span_start_line, span_start_col, span_end_line, span_end_col,
                           language, text, attrs
                    FROM code_chunks
                    WHERE repo_id = %s AND node_id = %s
                    """,
                    (repo_id, node_id)
                )

                return [self._row_to_chunk(row) for row in cur.fetchall()]

    def find_by_location(
        self,
        repo_id: RepoId,
        file_path: str,
        line: int
    ) -> Optional[CodeChunk]:
        """
        위치로 청크 조회 (Zoekt 매핑에 필수!)
        
        Args:
            repo_id: 저장소 ID
            file_path: 파일 경로
            line: 라인 번호 (0-based)
        
        Returns:
            해당 위치를 포함하는 청크 (가장 작은 것)
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT repo_id, chunk_id, node_id, file_path,
                           span_start_line, span_start_col, span_end_line, span_end_col,
                           language, text, attrs
                    FROM code_chunks
                    WHERE repo_id = %s 
                      AND file_path = %s
                      AND span_start_line <= %s 
                      AND span_end_line >= %s
                    ORDER BY 
                        (span_end_line - span_start_line) ASC,
                        (span_end_col - span_start_col) ASC
                    LIMIT 1
                    """,
                    (repo_id, file_path, line, line)
                )

                row = cur.fetchone()
                if row:
                    return self._row_to_chunk(row)

        return None

    def _row_to_chunk(self, row) -> CodeChunk:
        """DB row → CodeChunk 변환"""
        return CodeChunk(
            repo_id=row[0],
            id=row[1],
            node_id=row[2],
            file_path=row[3],
            span=(row[4], row[5], row[6], row[7]),
            language=row[8],
            text=row[9],
            attrs=row[10] if row[10] else {}
        )

