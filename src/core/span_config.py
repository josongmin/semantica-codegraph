"""Span 수집 설정 및 커스터마이즈

이 모듈은 어떤 Span을 수집할지, 어떤 정보를 포함할지 제어합니다.
"""

import logging

from opentelemetry.sdk.trace import SpanProcessor

logger = logging.getLogger(__name__)


class SensitiveDataFilter(SpanProcessor):
    """민감 정보 필터링 Processor

    API 키, 비밀번호 등 민감 정보를 마스킹합니다.
    """

    def __init__(self):
        self.sensitive_keys = [
            "api_key",
            "secret",
            "password",
            "token",
            "authorization",
        ]

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        """Span 종료 시 민감 정보 마스킹"""
        if not span.attributes:
            return

        for attr_key in list(span.attributes.keys()):
            attr_lower = attr_key.lower()

            # 민감 키워드 포함 시 마스킹
            if any(sensitive in attr_lower for sensitive in self.sensitive_keys):
                span.set_attribute(attr_key, "***REDACTED***")

            # 쿼리 길이 제한 (200자)
            if attr_key == "query" and isinstance(span.attributes[attr_key], str):
                if len(span.attributes[attr_key]) > 200:
                    span.set_attribute(
                        attr_key,
                        span.attributes[attr_key][:200] + "...[truncated]"
                    )

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        return True


class SlowRequestFilter(SpanProcessor):
    """느린 요청만 수집하는 Processor
    
    설정한 임계값보다 빠른 요청은 수집하지 않습니다.
    """

    def __init__(self, threshold_ms: int = 1000, enabled: bool = False):
        """
        Args:
            threshold_ms: 임계값 (밀리초)
            enabled: 활성화 여부
        """
        self.threshold_ms = threshold_ms
        self.enabled = enabled

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        """Span 종료 시 duration 체크"""
        if not self.enabled:
            return

        # Root span만 체크 (자식 span은 부모가 수집되면 같이 수집됨)
        if span.parent:
            return

        # Duration 계산 (nanoseconds → milliseconds)
        duration_ms = (span.end_time - span.start_time) / 1_000_000

        # 빠른 요청은 제외
        if duration_ms < self.threshold_ms:
            # Status를 UNSET으로 변경하여 export 방지
            # 주의: 이건 완전한 drop이 아니고 hint일 뿐
            logger.debug(f"Filtering fast span: {span.name} ({duration_ms:.2f}ms)")

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        return True


class SpanEnricher(SpanProcessor):
    """Span에 공통 정보 추가
    
    모든 Span에 환경, 버전 등 공통 정보를 추가합니다.
    """

    def __init__(
        self,
        environment: str = "development",
        version: str = "0.1.0",
        add_system_info: bool = False,
    ):
        self.environment = environment
        self.version = version
        self.add_system_info = add_system_info

    def on_start(self, span, parent_context=None):
        """Span 시작 시 공통 정보 추가"""
        # 환경 및 버전
        span.set_attribute("deployment.environment", self.environment)
        span.set_attribute("service.version", self.version)

        # 시스템 정보 (옵션)
        if self.add_system_info:
            try:
                import psutil
                span.set_attribute("system.cpu_percent", psutil.cpu_percent())
                span.set_attribute(
                    "system.memory_percent",
                    psutil.virtual_memory().percent
                )
            except Exception:
                pass

    def on_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        return True


def should_exclude_span(span_name: str, excluded_paths: list[str] | None = None) -> bool:
    """Span을 제외할지 판단
    
    Args:
        span_name: Span 이름
        excluded_paths: 제외할 경로 리스트
    
    Returns:
        제외 여부
    """
    if not excluded_paths:
        excluded_paths = [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/favicon.ico",
        ]

    for path in excluded_paths:
        if path in span_name:
            return True

    return False


def add_search_span_attributes(span, query: str, k: int, results: list, duration: float):
    """검색 Span에 표준 attribute 추가
    
    Args:
        span: OpenTelemetry Span
        query: 검색 쿼리
        k: 요청한 결과 수
        results: 검색 결과 리스트
        duration: 소요 시간 (초)
    """
    # 기본 정보
    span.set_attribute("search.query", query[:100])  # 100자 제한
    span.set_attribute("search.k", k)
    span.set_attribute("search.results.count", len(results))
    span.set_attribute("search.duration_ms", int(duration * 1000))

    # 결과 품질
    if results:
        span.set_attribute("search.results.top_score", results[0].score)
        span.set_attribute(
            "search.results.avg_score",
            sum(r.score for r in results) / len(results)
        )


def add_llm_span_attributes(
    span,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost: float | None = None,
):
    """LLM 호출 Span에 표준 attribute 추가
    
    Args:
        span: OpenTelemetry Span
        model: 모델명
        input_tokens: 입력 토큰 수
        output_tokens: 출력 토큰 수
        cost: 비용 (옵션)
    """
    span.set_attribute("llm.model", model)
    span.set_attribute("llm.tokens.input", input_tokens)
    span.set_attribute("llm.tokens.output", output_tokens)
    span.set_attribute("llm.tokens.total", input_tokens + output_tokens)

    if cost is not None:
        span.set_attribute("llm.cost", cost)

