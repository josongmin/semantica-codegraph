"""Phoenix 통합 (RAG 품질 추적 및 임베딩 시각화)

Phoenix는 다음을 제공합니다:
- 검색 품질 분석 (Precision, Recall, MRR)
- Embeddings 시각화 (UMAP)
- Retrieval 성능 평가
- Hallucination 감지

Cloud 옵션:
- Arize Cloud: 관리형 Phoenix (추천)
- Self-hosted: 로컬 Phoenix UI
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class PhoenixManager:
    """Phoenix 초기화 및 관리"""

    def __init__(
        self,
        enabled: bool = True,
        port: int = 6006,
        host: str = "0.0.0.0",
        use_cloud: bool = False,
        api_key: str | None = None,
        collector_endpoint: str | None = None,
    ):
        self.enabled = enabled
        self.port = port
        self.host = host
        self.session = None
        self.use_cloud = use_cloud

        if not enabled:
            logger.info("Phoenix disabled")
            return

        try:
            import phoenix as px

            # Arize Cloud 사용
            if use_cloud:
                api_key = api_key or os.getenv("ARIZE_API_KEY")
                collector_endpoint_str: str = (
                    collector_endpoint
                    or os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "https://app.arize.com")
                    or "https://app.arize.com"
                )

                if not api_key:
                    logger.warning("Phoenix Cloud enabled but ARIZE_API_KEY not found")
                    self.enabled = False
                    return

                # Cloud에 연결
                os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = collector_endpoint_str
                os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={api_key}"

                logger.info(f"Phoenix Cloud (Arize): {collector_endpoint_str}")

            else:
                # Self-hosted Phoenix 앱 실행
                self.session = px.launch_app(port=port, host=host)
                logger.info(f"Phoenix UI: http://localhost:{port}")

            # OpenTelemetry 통합 (자동)
            try:
                from openinference.instrumentation.openai import OpenAIInstrumentor

                OpenAIInstrumentor().instrument()
                logger.info("Phoenix OpenAI instrumentation enabled")
            except ImportError:
                logger.debug("OpenAI instrumentation not available")

        except ImportError as e:
            logger.warning(f"Phoenix not installed: {e}")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Phoenix: {e}")
            self.enabled = False

    def log_retrieval(
        self,
        query: str,
        documents: list[dict[str, Any]],
        scores: list[float],
        labels: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """검색 결과 로깅

        Args:
            query: 검색 쿼리
            documents: 검색된 문서 리스트
                [{"id": "chunk1", "content": "...", "file_path": "..."}, ...]
            scores: 각 문서의 점수 리스트
            labels: 각 문서의 relevance 레이블 (1=relevant, 0=irrelevant)
            metadata: 추가 메타데이터
        """
        if not self.enabled:
            return

        try:
            import phoenix as px

            # Retrieval span 생성
            px.log_retrieval(
                query=query,
                documents=documents,
                scores=scores,
                labels=labels,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.debug(f"Phoenix log_retrieval failed: {e}")

    def log_embeddings(
        self,
        embeddings: list[list[float]],
        documents: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ):
        """임베딩 시각화를 위한 로깅

        Args:
            embeddings: 임베딩 벡터 리스트
            documents: 문서 정보 리스트
            metadata: 추가 메타데이터
        """
        if not self.enabled:
            return

        try:
            import phoenix as px

            px.log_embeddings(
                embeddings=embeddings,
                documents=documents,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.debug(f"Phoenix log_embeddings failed: {e}")

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        relevant_doc_ids: list[str],
        k: int = 10,
    ) -> dict[str, float]:
        """검색 품질 평가

        Args:
            query: 검색 쿼리
            retrieved_docs: 검색된 문서 (순서 보장)
            relevant_doc_ids: 실제 관련 문서 ID 리스트 (ground truth)
            k: 평가할 top-k

        Returns:
            평가 메트릭 딕셔너리
            {
                "precision@k": 0.8,
                "recall@k": 0.6,
                "mrr": 0.85,
                "ndcg@k": 0.82,
            }
        """
        if not self.enabled:
            return {}

        try:
            retrieved_ids = [doc["id"] for doc in retrieved_docs[:k]]
            relevant_set = set(relevant_doc_ids)

            # Precision@K
            relevant_retrieved = len([id_ for id_ in retrieved_ids if id_ in relevant_set])
            precision = relevant_retrieved / k if k > 0 else 0

            # Recall@K
            recall = (
                relevant_retrieved / len(relevant_set) if len(relevant_set) > 0 else 0
            )

            # MRR (Mean Reciprocal Rank)
            mrr = 0.0
            for i, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_set:
                    mrr = 1.0 / i
                    break

            # NDCG@K (간단 구현)
            dcg = sum(
                [
                    (1 if retrieved_ids[i] in relevant_set else 0) / (i + 2)
                    for i in range(min(k, len(retrieved_ids)))
                ]
            )
            idcg = sum([1 / (i + 2) for i in range(min(k, len(relevant_set)))])
            ndcg = dcg / idcg if idcg > 0 else 0

            metrics = {
                "precision@k": precision,
                "recall@k": recall,
                "mrr": mrr,
                "ndcg@k": ndcg,
            }

            logger.info(f"Retrieval metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.debug(f"Phoenix evaluate_retrieval failed: {e}")
            return {}

    def close(self):
        """Phoenix 세션 종료"""
        if self.session:
            try:
                self.session.close()
                logger.info("Phoenix session closed")
            except Exception as e:
                logger.debug(f"Failed to close Phoenix session: {e}")


# 전역 인스턴스
_phoenix_manager: PhoenixManager | None = None


def init_phoenix(
    enabled: bool = True,
    port: int = 6006,
    use_cloud: bool = False,
    api_key: str | None = None,
) -> PhoenixManager:
    """Phoenix 초기화

    Args:
        enabled: Phoenix 활성화 여부
        port: Phoenix UI 포트 (기본: 6006, self-hosted만)
        use_cloud: Arize Cloud 사용 여부
        api_key: Arize API 키 (또는 ARIZE_API_KEY 환경변수)

    Returns:
        PhoenixManager 인스턴스
    """
    global _phoenix_manager
    _phoenix_manager = PhoenixManager(
        enabled=enabled,
        port=port,
        use_cloud=use_cloud,
        api_key=api_key,
    )
    return _phoenix_manager


def get_phoenix() -> PhoenixManager | None:
    """Phoenix 인스턴스 가져오기"""
    return _phoenix_manager

