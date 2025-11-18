"""PostgreSQL 기반 GraphStore 구현"""

import json
import logging

from psycopg2 import pool
from psycopg2.extras import execute_batch

from ..core.models import CodeEdge, CodeNode, RepoId
from ..core.ports import GraphStorePort

logger = logging.getLogger(__name__)


class PostgresGraphStore(GraphStorePort):
    """
    PostgreSQL 기반 코드 그래프 저장소

    테이블 구조:
    - code_nodes: 노드 저장
    - code_edges: 엣지 저장

    인덱스:
    - (repo_id, id): 노드 조회
    - (repo_id, file_path, kind): 파일별 노드 조회
    - (repo_id, file_path, line): 위치 기반 조회
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        pool_max: int = 20
    ):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
                예: "host=localhost dbname=semantica user=semantica password=semantica"
            pool_size: 커넥션 풀 최소 크기
            pool_max: 커넥션 풀 최대 크기
        """
        self.connection_string = connection_string

        # 커넥션 풀 생성
        self._pool = pool.SimpleConnectionPool(
            pool_size,
            pool_max,
            connection_string
        )
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
        """
        테이블 생성 (없으면)

        Note: 전체 스키마는 migrations/001_init_schema.sql 참조
              여기서는 최소한만 생성
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # 스키마 파일 실행하는 게 더 나음
                # 여기서는 기본 테이블만 생성

                # repo_metadata (최소)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS repo_metadata (
                        repo_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        root_path TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # code_nodes
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS code_nodes (
                        repo_id TEXT NOT NULL,
                        id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        language TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        span_start_line INTEGER NOT NULL,
                        span_start_col INTEGER NOT NULL,
                        span_end_line INTEGER NOT NULL,
                        span_end_col INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        text TEXT NOT NULL,
                        attrs JSONB,
                        created_at TIMESTAMP DEFAULT NOW(),
                        PRIMARY KEY (repo_id, id)
                    )
                """)

                # code_edges
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS code_edges (
                        repo_id TEXT NOT NULL,
                        src_id TEXT NOT NULL,
                        dst_id TEXT NOT NULL,
                        type TEXT NOT NULL,
                        attrs JSONB,
                        created_at TIMESTAMP DEFAULT NOW(),
                        PRIMARY KEY (repo_id, src_id, dst_id, type)
                    )
                """)

                # 인덱스
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_nodes_file_path
                    ON code_nodes(repo_id, file_path)
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_nodes_location
                    ON code_nodes(repo_id, file_path, span_start_line, span_end_line)
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_nodes_name
                    ON code_nodes(repo_id, name)
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_edges_src
                    ON code_edges(repo_id, src_id)
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_edges_dst
                    ON code_edges(repo_id, dst_id)
                """)

                conn.commit()
        finally:
            self._put_connection(conn)

        logger.info("Database tables ensured")

    def save_graph(self, nodes: list[CodeNode], edges: list[CodeEdge]) -> None:
        """
        그래프 저장

        Args:
            nodes: CodeNode 리스트
            edges: CodeEdge 리스트
        """
        if not nodes:
            logger.warning("No nodes to save")
            return

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # 노드 저장 (배치)
                node_data = [
                    (
                        node.repo_id,
                        node.id,
                        node.kind,
                        node.language,
                        node.file_path,
                        node.span[0],  # start_line
                        node.span[1],  # start_col
                        node.span[2],  # end_line
                        node.span[3],  # end_col
                        node.name,
                        node.text,
                        json.dumps(node.attrs)
                    )
                    for node in nodes
                ]

                execute_batch(
                    cur,
                    """
                    INSERT INTO code_nodes (
                        repo_id, id, kind, language, file_path,
                        span_start_line, span_start_col, span_end_line, span_end_col,
                        name, text, attrs
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (repo_id, id) DO UPDATE SET
                        kind = EXCLUDED.kind,
                        language = EXCLUDED.language,
                        file_path = EXCLUDED.file_path,
                        span_start_line = EXCLUDED.span_start_line,
                        span_start_col = EXCLUDED.span_start_col,
                        span_end_line = EXCLUDED.span_end_line,
                        span_end_col = EXCLUDED.span_end_col,
                        name = EXCLUDED.name,
                        text = EXCLUDED.text,
                        attrs = EXCLUDED.attrs
                    """,
                    node_data,
                    page_size=500  # 배치 크기 최적화
                )

                logger.debug(f"Saved {len(nodes)} nodes")

                # 엣지 저장 (배치)
                if edges:
                    edge_data = [
                        (
                            edge.repo_id,
                            edge.src_id,
                            edge.dst_id,
                            edge.type,
                            json.dumps(edge.attrs)
                        )
                        for edge in edges
                    ]

                    execute_batch(
                        cur,
                        """
                        INSERT INTO code_edges (
                            repo_id, src_id, dst_id, type, attrs
                        ) VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (repo_id, src_id, dst_id, type) DO UPDATE SET
                            attrs = EXCLUDED.attrs
                        """,
                        edge_data,
                        page_size=500  # 배치 크기 최적화
                    )

                    logger.debug(f"Saved {len(edges)} edges")

                conn.commit()
        finally:
            self._put_connection(conn)

        logger.info(f"Graph saved: {len(nodes)} nodes, {len(edges)} edges")

    def get_node(self, repo_id: RepoId, node_id: str) -> CodeNode | None:
        """단일 노드 조회"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT repo_id, id, kind, language, file_path,
                           span_start_line, span_start_col, span_end_line, span_end_col,
                           name, text, attrs
                    FROM code_nodes
                    WHERE repo_id = %s AND id = %s
                    """,
                    (repo_id, node_id)
                )

                row = cur.fetchone()
                if row:
                    return self._row_to_node(row)
        finally:
            self._put_connection(conn)

        return None

    def get_node_by_location(
        self,
        repo_id: RepoId,
        file_path: str,
        line: int,
        column: int = 0,
    ) -> CodeNode | None:
        """위치로 노드 조회"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT repo_id, id, kind, language, file_path,
                           span_start_line, span_start_col, span_end_line, span_end_col,
                           name, text, attrs
                    FROM code_nodes
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
                    return self._row_to_node(row)
        finally:
            self._put_connection(conn)

        return None

    def neighbors(
        self,
        repo_id: RepoId,
        node_id: str,
        edge_types: list[str] | None = None,
        k: int = 1,
    ) -> list[CodeNode]:
        """
        이웃 노드 조회 (k-hop) - N+1 쿼리 최적화 적용

        Args:
            repo_id: 저장소 ID
            node_id: 시작 노드 ID
            edge_types: 필터할 엣지 타입 (None이면 전체)
            k: hop 수

        Returns:
            이웃 노드 리스트
        """
        if k <= 0:
            return []

        neighbors = []
        visited = {node_id}

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                current_nodes = [node_id]

                for _ in range(k):
                    if not current_nodes:
                        break

                    # N+1 쿼리 최적화: 배치로 한 번에 조회
                    if edge_types:
                        cur.execute(
                            """
                            SELECT dst_id FROM code_edges
                            WHERE repo_id = %s
                              AND src_id = ANY(%s)
                              AND type = ANY(%s)
                            """,
                            (repo_id, current_nodes, edge_types)
                        )
                    else:
                        cur.execute(
                            """
                            SELECT dst_id FROM code_edges
                            WHERE repo_id = %s AND src_id = ANY(%s)
                            """,
                            (repo_id, current_nodes)
                        )

                    next_nodes = []
                    for row in cur.fetchall():
                        dst_id = row[0]
                        if dst_id not in visited:
                            visited.add(dst_id)
                            next_nodes.append(dst_id)

                    current_nodes = next_nodes

                # 모든 이웃 노드 조회 (배치)
                if visited:
                    visited.discard(node_id)  # 시작 노드 제외
                    neighbor_ids = list(visited)

                    if neighbor_ids:
                        cur.execute(
                            """
                            SELECT repo_id, id, kind, language, file_path,
                                   span_start_line, span_start_col, span_end_line, span_end_col,
                                   name, text, attrs
                            FROM code_nodes
                            WHERE repo_id = %s AND id = ANY(%s)
                            """,
                            (repo_id, neighbor_ids)
                        )

                        neighbors = [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self._put_connection(conn)

        logger.debug(f"Found {len(neighbors)} neighbors for {node_id} ({k}-hop)")
        return neighbors

    def list_nodes(
        self,
        repo_id: RepoId,
        kinds: list[str] | None = None,
    ) -> list[CodeNode]:
        """
        저장소의 모든 노드 조회

        Args:
            repo_id: 저장소 ID
            kinds: 필터할 노드 종류 (None이면 전체)

        Returns:
            CodeNode 리스트
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if kinds:
                    cur.execute(
                        """
                        SELECT repo_id, id, kind, language, file_path,
                               span_start_line, span_start_col, span_end_line, span_end_col,
                               name, text, attrs
                        FROM code_nodes
                        WHERE repo_id = %s AND kind = ANY(%s)
                        ORDER BY file_path, span_start_line
                        """,
                        (repo_id, kinds)
                    )
                else:
                    cur.execute(
                        """
                        SELECT repo_id, id, kind, language, file_path,
                               span_start_line, span_start_col, span_end_line, span_end_col,
                               name, text, attrs
                        FROM code_nodes
                        WHERE repo_id = %s
                        ORDER BY file_path, span_start_line
                        """,
                        (repo_id,)
                    )

                rows = cur.fetchall()
                nodes = [self._row_to_node(row) for row in rows]
        finally:
            self._put_connection(conn)

        logger.debug(f"Listed {len(nodes)} nodes for repo {repo_id}")
        return nodes

    def delete_repo(self, repo_id: RepoId) -> None:
        """저장소 전체 삭제"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM code_edges WHERE repo_id = %s", (repo_id,))
                edges_deleted = cur.rowcount

                cur.execute("DELETE FROM code_nodes WHERE repo_id = %s", (repo_id,))
                nodes_deleted = cur.rowcount

                conn.commit()
        finally:
            self._put_connection(conn)

        logger.info(
            f"Deleted repo {repo_id}: {nodes_deleted} nodes, {edges_deleted} edges"
        )

    def _row_to_node(self, row) -> CodeNode:
        """DB row → CodeNode 변환"""
        return CodeNode(
            repo_id=row[0],
            id=row[1],
            kind=row[2],
            language=row[3],
            file_path=row[4],
            span=(row[5], row[6], row[7], row[8]),
            name=row[9],
            text=row[10],
            attrs=row[11] if row[11] else {}
        )

