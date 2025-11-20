"""Route Index 저장소"""

import json
import logging
from typing import Any

from psycopg2 import pool

from ..core.models import RepoId
from .route_extractor import RouteInfo

logger = logging.getLogger(__name__)


class RouteStore:
    """
    PostgreSQL route_index 테이블 관리

    역할:
    - Route 저장
    - Route 검색 (method, path, framework)
    - Route 삭제
    """

    def __init__(self, connection_string: str, pool_size: int = 2, pool_max: int = 5):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
            pool_size: 커넥션 풀 최소 크기
            pool_max: 커넥션 풀 최대 크기
        """
        self.conn_pool = pool.SimpleConnectionPool(pool_size, pool_max, connection_string)
        logger.info(f"RouteStore: Created connection pool (min={pool_size}, max={pool_max})")

    def save_routes(self, routes: list[RouteInfo]) -> None:
        """
        Route 저장 (Batch Insert)

        Args:
            routes: RouteInfo 리스트
        """
        if not routes:
            return

        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Batch insert
                data = [
                    (
                        r.repo_id,
                        r.route_id,
                        r.http_method,
                        r.http_path,
                        r.handler_symbol_id,
                        r.handler_name,
                        r.file_path,
                        r.start_line,
                        r.end_line,
                        r.router_prefix,
                        r.framework,
                        json.dumps(r.metadata),
                    )
                    for r in routes
                ]

                cur.executemany(
                    """
                    INSERT INTO route_index (
                        repo_id, route_id, http_method, http_path,
                        handler_symbol_id, handler_name, file_path,
                        start_line, end_line, router_prefix, framework, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (repo_id, route_id) DO UPDATE SET
                        http_method = EXCLUDED.http_method,
                        http_path = EXCLUDED.http_path,
                        handler_name = EXCLUDED.handler_name,
                        start_line = EXCLUDED.start_line,
                        end_line = EXCLUDED.end_line
                    """,
                    data,
                )
                conn.commit()

            logger.info(f"Saved {len(routes)} routes to database")

        except Exception as e:
            logger.error(f"Failed to save routes: {e}")
            conn.rollback()
            raise
        finally:
            self.conn_pool.putconn(conn)

    def search_routes(
        self,
        repo_id: RepoId,
        method: str | None = None,
        path_pattern: str | None = None,
        framework: str | None = None,
        k: int = 100,
    ) -> list[dict]:
        """
        Route 검색

        Args:
            repo_id: 저장소 ID
            method: HTTP 메서드 필터 ("GET", "POST", ...)
            path_pattern: 경로 패턴 (LIKE %pattern%)
            framework: 프레임워크 필터 ("fastapi", "django", ...)
            k: 결과 개수

        Returns:
            Route 정보 dict 리스트
        """
        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                sql = """
                    SELECT route_id, http_method, http_path, handler_name,
                           file_path, start_line, end_line, framework,
                           router_prefix, metadata
                    FROM route_index
                    WHERE repo_id = %s
                """
                params: list[Any] = [repo_id]

                if method:
                    sql += " AND http_method = %s"
                    params.append(method.upper())

                if path_pattern:
                    sql += " AND http_path LIKE %s"
                    params.append(f"%{path_pattern}%")

                if framework:
                    sql += " AND framework = %s"
                    params.append(framework)

                sql += " ORDER BY http_path, http_method LIMIT %s"
                params.append(k)

                cur.execute(sql, params)
                rows = cur.fetchall()

                return [
                    {
                        "route_id": row[0],
                        "method": row[1],
                        "path": row[2],
                        "handler": row[3],
                        "file_path": row[4],
                        "start_line": row[5],
                        "end_line": row[6],
                        "framework": row[7],
                        "router_prefix": row[8],
                        "metadata": json.loads(row[9]) if row[9] else {},
                    }
                    for row in rows
                ]

        finally:
            self.conn_pool.putconn(conn)

    def get_routes_by_file(
        self,
        repo_id: RepoId,
        file_path: str,
    ) -> list[dict]:
        """
        파일별 Route 조회

        Args:
            repo_id: 저장소 ID
            file_path: 파일 경로

        Returns:
            Route 정보 dict 리스트
        """
        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT route_id, http_method, http_path, handler_name,
                           start_line, end_line, framework
                    FROM route_index
                    WHERE repo_id = %s AND file_path = %s
                    ORDER BY start_line
                    """,
                    [repo_id, file_path],
                )
                rows = cur.fetchall()

                return [
                    {
                        "route_id": row[0],
                        "method": row[1],
                        "path": row[2],
                        "handler": row[3],
                        "start_line": row[4],
                        "end_line": row[5],
                        "framework": row[6],
                    }
                    for row in rows
                ]

        finally:
            self.conn_pool.putconn(conn)

    def delete_routes(self, repo_id: RepoId) -> None:
        """
        저장소 Route 삭제

        Args:
            repo_id: 저장소 ID
        """
        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM route_index WHERE repo_id = %s",
                    [repo_id],
                )
                conn.commit()
            logger.info(f"Deleted routes for {repo_id}")

        finally:
            self.conn_pool.putconn(conn)

    def close(self):
        """커넥션 풀 종료"""
        if self.conn_pool:
            self.conn_pool.closeall()
            logger.info("RouteStore: Connection pool closed")
