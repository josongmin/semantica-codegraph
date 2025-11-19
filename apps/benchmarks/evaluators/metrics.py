"""평가 메트릭 계산 모듈"""

import time
from dataclasses import dataclass


@dataclass
class SearchResult:
    """검색 결과"""

    query: str
    results: list[str]  # 파일 경로 또는 chunk ID
    latency_ms: float
    retriever_name: str


@dataclass
class GroundTruth:
    """정답 데이터"""

    query: str
    relevant_items: set[str]  # 관련 있는 파일/chunk ID들


@dataclass
class EvaluationMetrics:
    """평가 메트릭 결과"""

    precision_at_k: float
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    avg_latency_ms: float
    total_queries: int

    def __str__(self) -> str:
        return (
            f"Precision@K: {self.precision_at_k:.3f}\n"
            f"Recall@K:    {self.recall_at_k:.3f}\n"
            f"MRR:         {self.mrr:.3f}\n"
            f"Avg Latency: {self.avg_latency_ms:.1f}ms\n"
            f"Total Queries: {self.total_queries}"
        )


class MetricsCalculator:
    """메트릭 계산기"""

    @staticmethod
    def precision_at_k(result: SearchResult, ground_truth: GroundTruth, k: int) -> float:
        """
        Precision@K 계산

        Args:
            result: 검색 결과
            ground_truth: 정답 데이터
            k: 상위 K개

        Returns:
            0.0 ~ 1.0 사이의 정확도
        """
        top_k = set(result.results[:k])
        relevant = top_k & ground_truth.relevant_items
        return len(relevant) / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(result: SearchResult, ground_truth: GroundTruth, k: int) -> float:
        """
        Recall@K 계산

        Args:
            result: 검색 결과
            ground_truth: 정답 데이터
            k: 상위 K개

        Returns:
            0.0 ~ 1.0 사이의 재현율
        """
        top_k = set(result.results[:k])
        relevant = top_k & ground_truth.relevant_items
        total_relevant = len(ground_truth.relevant_items)
        return len(relevant) / total_relevant if total_relevant > 0 else 0.0

    @staticmethod
    def reciprocal_rank(result: SearchResult, ground_truth: GroundTruth) -> float:
        """
        Reciprocal Rank 계산
        첫 번째 관련 결과의 순위의 역수

        Args:
            result: 검색 결과
            ground_truth: 정답 데이터

        Returns:
            0.0 ~ 1.0 사이의 값 (첫 결과면 1.0, 두번째면 0.5, ...)
        """
        for rank, item in enumerate(result.results, start=1):
            if item in ground_truth.relevant_items:
                return 1.0 / rank
        return 0.0

    @classmethod
    def evaluate_batch(
        cls, results: list[SearchResult], ground_truths: list[GroundTruth], k: int = 5
    ) -> EvaluationMetrics:
        """
        배치 평가

        Args:
            results: 검색 결과 리스트
            ground_truths: 정답 데이터 리스트
            k: 상위 K개

        Returns:
            집계된 메트릭
        """
        # 쿼리별 매칭
        gt_map = {gt.query: gt for gt in ground_truths}

        precisions = []
        recalls = []
        rrs = []
        latencies = []

        for result in results:
            if result.query not in gt_map:
                continue

            gt = gt_map[result.query]

            precisions.append(cls.precision_at_k(result, gt, k))
            recalls.append(cls.recall_at_k(result, gt, k))
            rrs.append(cls.reciprocal_rank(result, gt))
            latencies.append(result.latency_ms)

        n = len(precisions)
        if n == 0:
            return EvaluationMetrics(0, 0, 0, 0, 0)

        return EvaluationMetrics(
            precision_at_k=sum(precisions) / n,
            recall_at_k=sum(recalls) / n,
            mrr=sum(rrs) / n,
            avg_latency_ms=sum(latencies) / n,
            total_queries=n,
        )


def time_search(search_func, query: str, **kwargs) -> tuple[list[str], float]:
    """
    검색 실행 및 시간 측정

    Args:
        search_func: 검색 함수
        query: 검색 쿼리
        **kwargs: 추가 인자

    Returns:
        (결과 리스트, 지연시간(ms))
    """
    start = time.perf_counter()
    results = search_func(query, **kwargs)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    return results, latency_ms
