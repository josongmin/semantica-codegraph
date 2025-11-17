"""저장소 메타데이터 관리"""

import json
import logging
from datetime import datetime
from typing import List, Optional

import psycopg2

from .models import RepoId, RepoMetadata

logger = logging.getLogger(__name__)


class RepoMetadataStore:
    """
    저장소 메타데이터 저장소
    
    역할:
    - 저장소 등록/조회/삭제
    - 인덱싱 상태 관리
    - 통계 정보 조회
    """

    def __init__(self, connection_string: str):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
        """
        self.connection_string = connection_string
        self._ensure_tables()

    def _get_connection(self):
        """
        DB 연결 생성
        
        Returns:
            psycopg2 connection
            
        Raises:
            psycopg2.OperationalError: DB 연결 실패 시
        """
        try:
            return psycopg2.connect(self.connection_string)
        except psycopg2.OperationalError as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _ensure_tables(self):
        """테이블 생성"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS repo_metadata (
                        repo_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        root_path TEXT NOT NULL,
                        git_url TEXT,
                        default_branch TEXT DEFAULT 'main',
                        languages TEXT[],
                        total_files INTEGER DEFAULT 0,
                        total_nodes INTEGER DEFAULT 0,
                        total_chunks INTEGER DEFAULT 0,
                        indexing_status TEXT DEFAULT 'pending',
                        indexing_progress FLOAT DEFAULT 0.0,
                        indexing_started_at TIMESTAMP,
                        indexing_completed_at TIMESTAMP,
                        indexing_error TEXT,
                        created_at TIMESTAMP DEFAULT NOW(),
                        last_indexed_at TIMESTAMP,
                        config JSONB
                    )
                """)

                conn.commit()

        logger.info("RepoMetadata table ensured")

    def save(self, metadata: RepoMetadata) -> None:
        """
        저장소 메타데이터 저장
        
        Args:
            metadata: 저장할 RepoMetadata
            
        Raises:
            psycopg2.Error: DB 작업 실패 시
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO repo_metadata (
                            repo_id, name, root_path, languages, 
                            total_files, total_nodes, total_chunks, config
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (repo_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            root_path = EXCLUDED.root_path,
                            languages = EXCLUDED.languages,
                            total_files = EXCLUDED.total_files,
                            total_nodes = EXCLUDED.total_nodes,
                            total_chunks = EXCLUDED.total_chunks,
                            config = EXCLUDED.config
                        """,
                        (
                            metadata.repo_id,
                            metadata.name,
                            metadata.root_path,
                            metadata.languages,
                            metadata.total_files,
                            metadata.total_nodes,
                            metadata.total_chunks,
                            json.dumps(metadata.attrs)
                        )
                    )

                    conn.commit()

            logger.info(f"Saved metadata for repo: {metadata.repo_id}")
        except psycopg2.Error as e:
            logger.error(f"Failed to save metadata for {metadata.repo_id}: {e}")
            raise

    def get(self, repo_id: RepoId) -> Optional[RepoMetadata]:
        """저장소 메타데이터 조회"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT repo_id, name, root_path, languages,
                           last_indexed_at, total_files, total_nodes, total_chunks,
                           indexing_status, indexing_progress, config
                    FROM repo_metadata
                    WHERE repo_id = %s
                    """,
                    (repo_id,)
                )

                row = cur.fetchone()
                if row:
                    # JSONB는 자동으로 dict로 변환될 수 있음
                    attrs_value = row[10]
                    if attrs_value is None:
                        attrs = {}
                    elif isinstance(attrs_value, dict):
                        attrs = attrs_value
                    else:
                        attrs = json.loads(attrs_value)
                    
                    return RepoMetadata(
                        repo_id=row[0],
                        name=row[1],
                        root_path=row[2],
                        languages=row[3] if row[3] else [],
                        indexed_at=row[4],
                        total_files=row[5],
                        total_nodes=row[6],
                        total_chunks=row[7],
                        indexing_status=row[8] if row[8] else "pending",
                        indexing_progress=row[9] if row[9] else 0.0,
                        attrs=attrs
                    )

        return None

    def list_all(self) -> List[RepoMetadata]:
        """모든 저장소 목록 조회"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT repo_id, name, root_path, languages,
                           last_indexed_at, total_files, total_nodes, total_chunks,
                           indexing_status, indexing_progress, config
                    FROM repo_metadata
                    ORDER BY last_indexed_at DESC NULLS LAST
                    """
                )

                results = []
                for row in cur.fetchall():
                    # JSONB는 자동으로 dict로 변환될 수 있음
                    attrs_value = row[10]
                    if attrs_value is None:
                        attrs = {}
                    elif isinstance(attrs_value, dict):
                        attrs = attrs_value
                    else:
                        attrs = json.loads(attrs_value)
                    
                    results.append(
                        RepoMetadata(
                            repo_id=row[0],
                            name=row[1],
                            root_path=row[2],
                            languages=row[3] if row[3] else [],
                            indexed_at=row[4],
                            total_files=row[5],
                            total_nodes=row[6],
                            total_chunks=row[7],
                            indexing_status=row[8] if row[8] else "pending",
                            indexing_progress=row[9] if row[9] else 0.0,
                            attrs=attrs
                        )
                    )
                
                return results

    def update_indexing_status(
        self,
        repo_id: RepoId,
        status: str,
        progress: float = 0.0,
        error: Optional[str] = None
    ) -> None:
        """인덱싱 상태 업데이트"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if status == "indexing":
                    # 시작
                    cur.execute(
                        """
                        UPDATE repo_metadata SET
                            indexing_status = %s,
                            indexing_progress = %s,
                            indexing_started_at = NOW()
                        WHERE repo_id = %s
                        """,
                        (status, progress, repo_id)
                    )
                elif status == "completed":
                    # 완료
                    cur.execute(
                        """
                        UPDATE repo_metadata SET
                            indexing_status = %s,
                            indexing_progress = 1.0,
                            indexing_completed_at = NOW(),
                            last_indexed_at = NOW(),
                            indexing_error = NULL
                        WHERE repo_id = %s
                        """,
                        (status, repo_id)
                    )
                elif status == "failed":
                    # 실패
                    cur.execute(
                        """
                        UPDATE repo_metadata SET
                            indexing_status = %s,
                            indexing_completed_at = NOW(),
                            indexing_error = %s
                        WHERE repo_id = %s
                        """,
                        (status, error, repo_id)
                    )
                else:
                    # 일반 상태 업데이트
                    cur.execute(
                        """
                        UPDATE repo_metadata SET
                            indexing_status = %s,
                            indexing_progress = %s
                        WHERE repo_id = %s
                        """,
                        (status, progress, repo_id)
                    )

                conn.commit()

        logger.debug(f"Updated indexing status: {repo_id} → {status}")

    def delete(self, repo_id: RepoId) -> None:
        """저장소 메타데이터 삭제"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM repo_metadata WHERE repo_id = %s",
                    (repo_id,)
                )
                conn.commit()

        logger.info(f"Deleted metadata for repo: {repo_id}")

