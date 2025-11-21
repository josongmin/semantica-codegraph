"""Semantic Node 저장소 (PostgreSQL)"""

import json
import logging
from typing import Any

from psycopg2 import pool

from ..core.models import RepoId, SemanticNode, SemanticSearchResult

logger = logging.getLogger(__name__)


class SemanticNodeStore:
    """
    PostgreSQL semantic_nodes 테이블 관리

    역할:
    - Semantic node 저장 (symbol/route/doc 요약 + 임베딩)
    - 벡터 유사도 검색
    - 재인덱싱 지원 (clear_repo)
    """

    def __init__(self, connection_string: str, pool_size: int = 2, pool_max: int = 10):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
            pool_size: 커넥션 풀 최소 크기
            pool_max: 커넥션 풀 최대 크기
        """
        self.conn_pool = pool.SimpleConnectionPool(pool_size, pool_max, connection_string)
        logger.info(f"SemanticNodeStore: Created connection pool (min={pool_size}, max={pool_max})")

    def save(
        self,
        repo_id: RepoId,
        node_id: str,
        node_type: str,
        summary: str,
        summary_method: str,
        model: str,
        embedding: list[float],
        source_table: str | None = None,
        source_id: str | None = None,
        doc_type: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        단일 semantic node 저장

        Args:
            repo_id: 저장소 ID
            node_id: 노드 ID (원본 테이블 PK)
            node_type: 노드 타입 (symbol/route/doc/issue)
            summary: 요약 텍스트
            summary_method: 요약 방법 (template/llm)
            model: 임베딩 모델 풀 네임
            embedding: 임베딩 벡터
            source_table: 원본 테이블명
            source_id: 원본 PK
            doc_type: 문서 타입
            metadata: 메타데이터
        """
        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO semantic_nodes (
                        repo_id, node_id, node_type, doc_type,
                        summary, summary_method, model, embedding,
                        source_table, source_id, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (repo_id, node_id, node_type, model)
                    DO UPDATE SET
                        summary = EXCLUDED.summary,
                        summary_method = EXCLUDED.summary_method,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    [
                        repo_id,
                        node_id,
                        node_type,
                        doc_type,
                        summary,
                        summary_method,
                        model,
                        embedding,
                        source_table,
                        source_id,
                        json.dumps(metadata or {}),
                    ],
                )
                conn.commit()
                logger.debug(f"Saved semantic node: {node_type}/{node_id}")
        except Exception as e:
            logger.error(f"Failed to save semantic node: {e}")
            conn.rollback()
            raise
        finally:
            self.conn_pool.putconn(conn)

    def save_batch(
        self,
        nodes: list[dict],
        batch_size: int = 1000,
        on_conflict: str = "replace",
    ) -> int:
        """
        배치 저장

        Args:
            nodes: semantic node dict 리스트
            batch_size: 배치 크기 (너무 크면 트랜잭션 부담)
            on_conflict: 충돌 시 처리 (replace: 덮어쓰기, skip: 무시)

        Returns:
            저장된 row 수
        """
        if not nodes:
            return 0

        conn = self.conn_pool.getconn()
        total_saved = 0

        try:
            # 배치로 나눠서 처리
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i : i + batch_size]

                with conn.cursor() as cur:
                    data = [
                        (
                            n["repo_id"],
                            n["node_id"],
                            n["node_type"],
                            n.get("doc_type"),
                            n["summary"],
                            n["summary_method"],
                            n["model"],
                            n["embedding"],
                            n.get("source_table"),
                            n.get("source_id"),
                            json.dumps(n.get("metadata", {})),
                        )
                        for n in batch
                    ]

                    if on_conflict == "replace":
                        sql = """
                            INSERT INTO semantic_nodes (
                                repo_id, node_id, node_type, doc_type,
                                summary, summary_method, model, embedding,
                                source_table, source_id, metadata
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (repo_id, node_id, node_type, model)
                            DO UPDATE SET
                                summary = EXCLUDED.summary,
                                summary_method = EXCLUDED.summary_method,
                                embedding = EXCLUDED.embedding,
                                metadata = EXCLUDED.metadata,
                                updated_at = NOW()
                        """
                    else:  # skip
                        sql = """
                            INSERT INTO semantic_nodes (
                                repo_id, node_id, node_type, doc_type,
                                summary, summary_method, model, embedding,
                                source_table, source_id, metadata
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (repo_id, node_id, node_type, model) DO NOTHING
                        """

                    cur.executemany(sql, data)
                    total_saved += cur.rowcount
                    conn.commit()

                    logger.debug(
                        f"Saved batch {i//batch_size + 1}/{(len(nodes) + batch_size - 1)//batch_size} "
                        f"({cur.rowcount} rows)"
                    )

            logger.info(f"Saved {total_saved} semantic nodes (total: {len(nodes)})")
            return total_saved

        except Exception as e:
            logger.error(f"Failed to save semantic nodes batch: {e}")
            conn.rollback()
            raise
        finally:
            self.conn_pool.putconn(conn)

    def search(
        self,
        repo_id: RepoId,
        query_embedding: list[float],
        node_type: str | None = None,
        model: str | None = None,
        k: int = 10,
        threshold: float = 0.3,
    ) -> list[SemanticSearchResult]:
        """
        벡터 유사도 검색

        Args:
            repo_id: 저장소 ID
            query_embedding: 쿼리 벡터
            node_type: 노드 타입 필터
            model: 모델 필터
            k: 결과 수
            threshold: 유사도 임계값 (코사인 거리)

        Returns:
            SemanticSearchResult 리스트
        """
        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                sql = """
                    SELECT
                        node_id, node_type, summary, model,
                        source_table, source_id, metadata,
                        1 - (embedding <=> %s::vector) as score
                    FROM semantic_nodes
                    WHERE repo_id = %s
                """
                params = [query_embedding, repo_id]

                if node_type:
                    sql += " AND node_type = %s"
                    params.append(node_type)

                if model:
                    sql += " AND model = %s"
                    params.append(model)

                sql += """
                    AND (1 - (embedding <=> %s::vector)) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                params_list: list[Any] = [query_embedding, threshold, query_embedding, k]
                params.extend(params_list)

                cur.execute(sql, params)
                rows = cur.fetchall()

                return [
                    SemanticSearchResult(
                        repo_id=repo_id,
                        node_id=row[0],
                        node_type=row[1],
                        summary=row[2],
                        model=row[3],
                        score=float(row[7]),
                        source_table=row[4],
                        source_id=row[5],
                        metadata=row[6]
                        if isinstance(row[6], dict)
                        else (json.loads(row[6]) if row[6] else {}),
                    )
                    for row in rows
                ]
        finally:
            self.conn_pool.putconn(conn)

    def get_by_node_id(
        self,
        repo_id: RepoId,
        node_id: str,
        model: str | None = None,
    ) -> list[SemanticNode]:
        """
        특정 노드의 모든 representation 조회

        Args:
            repo_id: 저장소 ID
            node_id: 노드 ID
            model: 모델 필터 (None이면 전부)

        Returns:
            SemanticNode 리스트
        """
        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                sql = """
                    SELECT
                        node_id, node_type, doc_type,
                        summary, summary_method, model, embedding,
                        source_table, source_id, metadata,
                        created_at, updated_at
                    FROM semantic_nodes
                    WHERE repo_id = %s AND node_id = %s
                """
                params = [repo_id, node_id]

                if model:
                    sql += " AND model = %s"
                    params.append(model)

                cur.execute(sql, params)
                rows = cur.fetchall()

                return [
                    SemanticNode(
                        repo_id=repo_id,
                        node_id=row[0],
                        node_type=row[1],
                        doc_type=row[2],
                        summary=row[3],
                        summary_method=row[4],
                        model=row[5],
                        embedding=row[6],
                        source_table=row[7],
                        source_id=row[8],
                        metadata=json.loads(row[9]) if row[9] else {},
                        created_at=row[10],
                        updated_at=row[11],
                    )
                    for row in rows
                ]
        finally:
            self.conn_pool.putconn(conn)

    def clear_repo(
        self,
        repo_id: RepoId,
        node_types: list[str] | None = None,
    ) -> int:
        """
        재인덱싱 전 기존 데이터 삭제

        Args:
            repo_id: 저장소 ID
            node_types: 삭제할 node_type 리스트 (None이면 전부)

        Returns:
            삭제된 row 수
        """
        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                if node_types:
                    cur.execute(
                        """
                        DELETE FROM semantic_nodes
                        WHERE repo_id = %s AND node_type = ANY(%s)
                        """,
                        [repo_id, node_types],
                    )
                else:
                    cur.execute(
                        "DELETE FROM semantic_nodes WHERE repo_id = %s",
                        [repo_id],
                    )
                deleted = cur.rowcount
                conn.commit()
                logger.info(f"Cleared {deleted} semantic nodes for {repo_id}")
                return int(deleted)
        except Exception as e:
            logger.error(f"Failed to clear semantic nodes: {e}")
            conn.rollback()
            raise
        finally:
            self.conn_pool.putconn(conn)

    def close(self):
        """커넥션 풀 종료"""
        if self.conn_pool:
            self.conn_pool.closeall()
            logger.info("SemanticNodeStore: Connection pool closed")
