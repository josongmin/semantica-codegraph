"""Query Log 저장소"""

import json
import logging
from dataclasses import dataclass, field

from psycopg2 import pool

from ..core.models import RepoId

logger = logging.getLogger(__name__)


@dataclass
class QueryLog:
    """쿼리 로그"""

    repo_id: RepoId
    query_text: str
    query_type: str | None = None
    query_embedding: list[float] | None = None
    weights: dict | None = None
    filters: dict | None = None
    k: int | None = None
    result_count: int = 0
    top_results: list[dict] = field(default_factory=list)
    clicked_node_ids: list[str] = field(default_factory=list)
    feedback_score: float | None = None
    latency_ms: int | None = None
    backend_latencies: dict | None = None
    client_info: dict | None = None


class QueryLogStore:
    """
    PostgreSQL query_logs 테이블 관리

    역할:
    - 검색 쿼리 로깅
    - 통계 조회 (query_type별, 날짜별)
    - 인기 노드 추출
    """

    def __init__(self, connection_string: str, pool_size: int = 2, pool_max: int = 5):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
            pool_size: 커넥션 풀 최소 크기
            pool_max: 커넥션 풀 최대 크기
        """
        self.conn_pool = pool.SimpleConnectionPool(pool_size, pool_max, connection_string)
        logger.info(f"QueryLogStore: Created connection pool (min={pool_size}, max={pool_max})")

    def log_query(self, query_log: QueryLog) -> int:
        """
        쿼리 로그 저장

        Args:
            query_log: QueryLog 객체

        Returns:
            생성된 로그 ID
        """
        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_logs (
                        repo_id, query_text, query_type, query_embedding,
                        weights, filters, k, result_count, top_results,
                        clicked_node_ids, feedback_score, latency_ms,
                        backend_latencies, client_info
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    [
                        query_log.repo_id,
                        query_log.query_text,
                        query_log.query_type,
                        query_log.query_embedding,
                        json.dumps(query_log.weights) if query_log.weights else None,
                        json.dumps(query_log.filters) if query_log.filters else None,
                        query_log.k,
                        query_log.result_count,
                        json.dumps(query_log.top_results),
                        query_log.clicked_node_ids,
                        query_log.feedback_score,
                        query_log.latency_ms,
                        json.dumps(query_log.backend_latencies)
                        if query_log.backend_latencies
                        else None,
                        json.dumps(query_log.client_info) if query_log.client_info else None,
                    ],
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError("Failed to get log_id from insert")
                log_id = int(row[0])
                conn.commit()
                logger.debug(f"Logged query: {query_log.query_text[:50]}... (id={log_id})")
                return log_id
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
            conn.rollback()
            raise
        finally:
            self.conn_pool.putconn(conn)

    def get_popular_nodes(
        self,
        repo_id: RepoId,
        node_type: str | None = None,
        days: int = 7,
        min_query_count: int = 5,
        k: int = 20,
    ) -> list[dict]:
        """
        자주 검색되는 노드 추출

        Args:
            repo_id: 저장소 ID
            node_type: 노드 타입 필터
            days: 조회 기간 (일)
            min_query_count: 최소 쿼리 횟수
            k: 반환할 노드 수

        Returns:
            [{node_id, node_type, query_count, avg_rank}, ...]
        """
        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                sql = """
                    SELECT
                        r->>'node_id' as node_id,
                        r->>'node_type' as node_type,
                        COUNT(*) as query_count,
                        AVG((r->>'rank')::int) as avg_rank
                    FROM query_logs,
                         jsonb_array_elements(top_results) as r
                    WHERE repo_id = %s
                      AND created_at > NOW() - INTERVAL '%s days'
                """
                params = [repo_id, days]

                if node_type:
                    sql += " AND r->>'node_type' = %s"
                    params.append(node_type)

                sql += """
                    GROUP BY r->>'node_id', r->>'node_type'
                    HAVING COUNT(*) >= %s
                    ORDER BY COUNT(*) DESC, AVG((r->>'rank')::int) ASC
                    LIMIT %s
                """
                params.extend([min_query_count, k])

                cur.execute(sql, params)
                rows = cur.fetchall()

                return [
                    {
                        "node_id": row[0],
                        "node_type": row[1],
                        "query_count": row[2],
                        "avg_rank": float(row[3]) if row[3] else None,
                    }
                    for row in rows
                ]
        finally:
            self.conn_pool.putconn(conn)

    def get_query_stats(
        self,
        repo_id: RepoId,
        days: int = 7,
    ) -> dict:
        """
        쿼리 통계

        Args:
            repo_id: 저장소 ID
            days: 조회 기간

        Returns:
            {
                'total_queries': int,
                'by_type': {query_type: count},
                'avg_latency_ms': float,
                'avg_result_count': float,
            }
        """
        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                # 전체 통계
                cur.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        AVG(latency_ms) as avg_latency,
                        AVG(result_count) as avg_results
                    FROM query_logs
                    WHERE repo_id = %s
                      AND created_at > NOW() - INTERVAL '%s days'
                    """,
                    [repo_id, days],
                )
                total, avg_lat, avg_res = cur.fetchone()

                # 타입별 통계
                cur.execute(
                    """
                    SELECT query_type, COUNT(*)
                    FROM query_logs
                    WHERE repo_id = %s
                      AND created_at > NOW() - INTERVAL '%s days'
                    GROUP BY query_type
                    """,
                    [repo_id, days],
                )
                by_type = dict(cur.fetchall())

                return {
                    "total_queries": int(total) if total else 0,
                    "by_type": by_type,
                    "avg_latency_ms": float(avg_lat) if avg_lat else 0.0,
                    "avg_result_count": float(avg_res) if avg_res else 0.0,
                }
        finally:
            self.conn_pool.putconn(conn)

    def close(self):
        """커넥션 풀 종료"""
        if self.conn_pool:
            self.conn_pool.closeall()
            logger.info("QueryLogStore: Connection pool closed")
