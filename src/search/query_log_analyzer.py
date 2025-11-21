"""Query Log 분석 유틸리티"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class QueryLogAnalyzer:
    """
    Query Log 분석

    기능:
    - 인기 노드 추출 (LLM 요약 대상)
    - Query type별 weight 효과 분석
    - 평균 순위 분석
    """

    def __init__(self, query_log_store):
        """
        Args:
            query_log_store: QueryLogStore
        """
        self.store = query_log_store

    def get_candidates_for_llm_upgrade(
        self,
        repo_id: str,
        node_type: str = "symbol",
        days: int = 7,
        min_query_count: int = 5,
        k: int = 50,
    ) -> list[dict]:
        """
        LLM 요약 + 3-large 업그레이드 대상 추출

        자주 검색되는 중요 노드만 선별하여 비용 최적화

        Args:
            repo_id: 저장소 ID
            node_type: 노드 타입 ('symbol' | 'route')
            days: 조회 기간
            min_query_count: 최소 쿼리 횟수
            k: 반환할 노드 수

        Returns:
            [{node_id, node_type, query_count, avg_rank}, ...]
        """
        popular = self.store.get_popular_nodes(
            repo_id=repo_id,
            node_type=node_type,
            days=days,
            min_query_count=min_query_count,
            k=k,
        )

        logger.info(
            f"LLM upgrade candidates: {len(popular)} nodes "
            f"(type={node_type}, days={days}, min_queries={min_query_count})"
        )

        return list(popular)

    def analyze_weight_effectiveness(
        self,
        repo_id: str,
        query_type: str,
        days: int = 7,
    ) -> dict[str, Any]:
        """
        Query type별 weight 효과 분석

        각 시그널(lexical, semantic_small_code 등)의 기여도 분석

        Args:
            repo_id: 저장소 ID
            query_type: 쿼리 타입
            days: 조회 기간

        Returns:
            {
                'query_count': int,
                'avg_signal_scores': {signal_name: avg_score},
                'signal_contributions': {signal_name: avg_contribution},
            }
        """

        conn = self.store.conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Query type별 로그 조회
                cur.execute(
                    """
                    SELECT top_results, weights
                    FROM query_logs
                    WHERE repo_id = %s
                      AND query_type = %s
                      AND created_at > NOW() - INTERVAL '%s days'
                    """,
                    [repo_id, query_type, days],
                )

                rows = cur.fetchall()

                if not rows:
                    return {"query_count": 0}

                # 시그널별 점수 집계
                signal_scores: dict[str, list[float]] = {}

                for top_results, _weights in rows:
                    if not top_results or not isinstance(top_results, list):
                        continue

                    for result in top_results:
                        if not isinstance(result, dict) or "signals" not in result:
                            continue

                        signals = result["signals"]
                        for signal_name, score in signals.items():
                            if signal_name not in signal_scores:
                                signal_scores[signal_name] = []
                            signal_scores[signal_name].append(score)

                # 평균 계산
                avg_scores = {
                    name: sum(scores) / len(scores)
                    for name, scores in signal_scores.items()
                    if scores
                }

                return {
                    "query_count": len(rows),
                    "avg_signal_scores": avg_scores,
                }
        finally:
            self.store.conn_pool.putconn(conn)

    def print_summary(self, repo_id: str, days: int = 7):
        """
        전체 통계 요약 출력

        Args:
            repo_id: 저장소 ID
            days: 조회 기간
        """
        print("=" * 70)
        print(f"📊 Query Log 분석 (최근 {days}일)")
        print("=" * 70)

        # 전체 통계
        stats = self.store.get_query_stats(repo_id, days=days)
        print(f"\n총 쿼리: {stats['total_queries']}개")
        print(f"평균 레이턴시: {stats['avg_latency_ms']:.1f}ms")
        print(f"평균 결과 수: {stats['avg_result_count']:.1f}개")

        print("\nQuery Type별:")
        for qtype, count in stats["by_type"].items():
            print(f"  - {qtype}: {count}개")

        # 인기 노드 (symbol)
        print("\n인기 노드 (symbol, min 3회):")
        popular_symbols = self.store.get_popular_nodes(
            repo_id=repo_id,
            node_type="symbol",
            days=days,
            min_query_count=3,
            k=10,
        )
        if popular_symbols:
            for i, node in enumerate(popular_symbols, 1):
                print(f"  {i}. {node['node_id'][:40]}...")
                print(f"     쿼리: {node['query_count']}회, 평균 순위: {node['avg_rank']:.1f}")
        else:
            print("  (없음)")

        # 인기 노드 (route)
        print("\n인기 노드 (route, min 2회):")
        popular_routes = self.store.get_popular_nodes(
            repo_id=repo_id,
            node_type="route",
            days=days,
            min_query_count=2,
            k=5,
        )
        if popular_routes:
            for i, node in enumerate(popular_routes, 1):
                print(f"  {i}. {node['node_id']}")
                print(f"     쿼리: {node['query_count']}회, 평균 순위: {node['avg_rank']:.1f}")
        else:
            print("  (없음)")

        print("\n" + "=" * 70)
