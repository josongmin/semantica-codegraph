"""PostgreSQL 기반 심볼 검색"""

import json
import logging
from typing import Any

from ....core.models import CodeNode, RepoId
from ....core.ports import GraphStorePort
from ...ports.symbol_search_port import SymbolSearchPort

logger = logging.getLogger(__name__)


class PostgresSymbolSearch(SymbolSearchPort):
    """
    PostgreSQL code_nodes 테이블 직접 조회
    
    역할:
    - 함수/클래스 이름으로 검색
    - Decorator 패턴 검색 (API route 찾기)
    - 파일 내 심볼 검색
    - 위치 기반 심볼 검색
    """
    
    def __init__(self, graph_store: GraphStorePort):
        """
        Args:
            graph_store: GraphStore (connection pool 재사용)
        """
        self.graph_store = graph_store
        self.conn_pool = graph_store._pool
    
    def search_by_name(
        self,
        repo_id: RepoId,
        query: str,
        kind: str | None = None,
        fuzzy: bool = True,
        k: int = 20,
    ) -> list[CodeNode]:
        """
        심볼 이름 검색
        
        구현:
        - fuzzy=True: LIKE %query% (부분 매칭)
        - fuzzy=False: 정확 매칭
        - 결과는 이름 길이 순 정렬 (짧은 것부터)
        """
        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                query_lower = query.lower()
                
                # SQL 작성
                if fuzzy:
                    # Trigram 유사도 + LIKE
                    sql = """
                        SELECT repo_id, id, kind, language, file_path,
                               span_start_line, span_start_col, 
                               span_end_line, span_end_col,
                               name, text, attrs
                        FROM code_nodes
                        WHERE repo_id = %s
                          AND (
                              LOWER(name) LIKE %s
                              OR similarity(name, %s) > 0.3
                          )
                    """
                    params: list[Any] = [repo_id, f"%{query_lower}%", query]
                else:
                    # 정확 매칭
                    sql = """
                        SELECT repo_id, id, kind, language, file_path,
                               span_start_line, span_start_col,
                               span_end_line, span_end_col,
                               name, text, attrs
                        FROM code_nodes
                        WHERE repo_id = %s
                          AND LOWER(name) = %s
                    """
                    params = [repo_id, query_lower]
                
                # Kind 필터
                if kind:
                    sql += " AND kind = %s"
                    params.append(kind)
                
                # 정렬: 이름 길이 ASC (짧은 것부터), 그 다음 유사도
                sql += """
                    ORDER BY 
                        LENGTH(name) ASC,
                        similarity(name, %s) DESC
                    LIMIT %s
                """
                params.extend([query, k])
                
                cur.execute(sql, params)
                rows = cur.fetchall()
                
                results = [self._row_to_node(row) for row in rows]
                logger.debug(f"Symbol search '{query}': {len(results)} results")
                return results
                
        finally:
            self.conn_pool.putconn(conn)
    
    def search_by_decorator(
        self,
        repo_id: RepoId,
        decorator_pattern: str,
        k: int = 50,
    ) -> list[CodeNode]:
        """
        Decorator 검색
        
        구현:
        - JSONB @> 연산자 사용
        - "@router.post" → "router.post" 정규화
        - 부분 매칭 지원
        """
        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                # @ 기호 제거
                decorator_clean = decorator_pattern.lstrip("@")
                
                # JSONB 검색 (배열 내 부분 매칭)
                # attrs->'decorators' 가 ["router.post('/search')"] 같은 형태
                sql = """
                    SELECT repo_id, id, kind, language, file_path,
                           span_start_line, span_start_col,
                           span_end_line, span_end_col,
                           name, text, attrs
                    FROM code_nodes
                    WHERE repo_id = %s
                      AND attrs->'decorators' IS NOT NULL
                      AND EXISTS (
                          SELECT 1
                          FROM jsonb_array_elements_text(attrs->'decorators') AS dec
                          WHERE dec LIKE %s
                      )
                    ORDER BY file_path, span_start_line
                    LIMIT %s
                """
                
                cur.execute(sql, [repo_id, f"%{decorator_clean}%", k])
                rows = cur.fetchall()
                
                results = [self._row_to_node(row) for row in rows]
                logger.debug(f"Decorator search '{decorator_pattern}': {len(results)} results")
                return results
                
        finally:
            self.conn_pool.putconn(conn)
    
    def search_by_file(
        self,
        repo_id: RepoId,
        file_path: str,
        kind: str | None = None,
        k: int = 100,
    ) -> list[CodeNode]:
        """
        파일 내 심볼 검색
        
        구현:
        - 파일 경로로 필터링
        - kind 옵션 지원
        - 라인 번호 순 정렬
        """
        conn = self.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                sql = """
                    SELECT repo_id, id, kind, language, file_path,
                           span_start_line, span_start_col,
                           span_end_line, span_end_col,
                           name, text, attrs
                    FROM code_nodes
                    WHERE repo_id = %s
                      AND file_path = %s
                """
                params = [repo_id, file_path]
                
                if kind:
                    sql += " AND kind = %s"
                    params.append(kind)
                
                sql += " ORDER BY span_start_line ASC LIMIT %s"
                params.append(k)
                
                cur.execute(sql, params)
                rows = cur.fetchall()
                
                results = [self._row_to_node(row) for row in rows]
                logger.debug(f"File symbols '{file_path}': {len(results)} results")
                return results
                
        finally:
            self.conn_pool.putconn(conn)
    
    def get_symbol_by_location(
        self,
        repo_id: RepoId,
        file_path: str,
        line: int,
    ) -> CodeNode | None:
        """
        위치로 심볼 찾기
        
        구현:
        - GraphStore의 기존 메서드 위임
        """
        return self.graph_store.get_node_by_location(
            repo_id, file_path, line, 0
        )
    
    def _row_to_node(self, row: tuple) -> CodeNode:
        """
        DB row → CodeNode 변환
        
        Args:
            row: (repo_id, id, kind, language, file_path,
                  span_start_line, span_start_col, span_end_line, span_end_col,
                  name, text, attrs)
        """
        return CodeNode(
            repo_id=row[0],
            id=row[1],
            kind=row[2],
            language=row[3],
            file_path=row[4],
            span=(row[5], row[6], row[7], row[8]),
            name=row[9],
            text=row[10],
            attrs=json.loads(row[11]) if row[11] else {},
        )

