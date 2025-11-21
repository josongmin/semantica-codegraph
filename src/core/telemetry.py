"""OpenTelemetry 초기화 및 관리

이 모듈은 OpenTelemetry의 Trace와 Metric을 설정합니다.
- 자동 계측 (FastAPI, httpx, psycopg2)
- OTLP Exporter (Jaeger, Prometheus 등)
- 샘플링 설정
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TelemetryManager:
    """OpenTelemetry 초기화 및 관리"""

    def __init__(
        self,
        service_name: str,
        service_version: str = "0.1.0",
        environment: str = "development",
        enabled: bool = True,
        otlp_endpoint: str = "http://localhost:4317",
        sample_rate: float = 1.0,
    ):
        self.enabled = enabled
        self.service_name = service_name

        if not enabled:
            logger.info("OpenTelemetry disabled")
            return

        try:
            from opentelemetry import metrics, trace
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

            # Resource 생성
            resource = Resource.create(
                {
                    SERVICE_NAME: service_name,
                    SERVICE_VERSION: service_version,
                    "deployment.environment": environment,
                }
            )

            # Trace Provider 설정
            sampler = TraceIdRatioBased(sample_rate)
            trace_provider = TracerProvider(resource=resource, sampler=sampler)

            otlp_trace_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            trace_provider.add_span_processor(BatchSpanProcessor(otlp_trace_exporter))
            trace.set_tracer_provider(trace_provider)

            # Metric Provider 설정
            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
            )
            metric_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            metrics.set_meter_provider(metric_provider)

            logger.info(
                f"OpenTelemetry initialized: service={service_name}, "
                f"endpoint={otlp_endpoint}, sample_rate={sample_rate}"
            )

        except ImportError as e:
            logger.warning(f"OpenTelemetry libraries not installed: {e}")
            self.enabled = False

    def get_tracer(self, name: str):
        """Tracer 가져오기"""
        if not self.enabled:
            try:
                from opentelemetry import trace

                return trace.get_tracer(name)
            except ImportError:
                return None

        from opentelemetry import trace

        return trace.get_tracer(name, self.service_name)

    def get_meter(self, name: str):
        """Meter 가져오기"""
        if not self.enabled:
            try:
                from opentelemetry import metrics

                return metrics.get_meter(name)
            except ImportError:
                return None

        from opentelemetry import metrics

        return metrics.get_meter(name, self.service_name)


# 전역 인스턴스 (Bootstrap에서 초기화)
_telemetry_manager: Optional[TelemetryManager] = None


def init_telemetry(service_name: str, config) -> TelemetryManager:
    """TelemetryManager 초기화

    Args:
        service_name: 서비스 이름 (예: "semantica-codegraph-api")
        config: Config 인스턴스

    Returns:
        TelemetryManager 인스턴스
    """
    global _telemetry_manager
    _telemetry_manager = TelemetryManager(
        service_name=service_name,
        enabled=config.otel_enabled,
        otlp_endpoint=config.otel_endpoint,
        sample_rate=config.otel_sample_rate,
        environment=config.environment,
    )
    return _telemetry_manager


def get_tracer(name: str):
    """편의 함수: Tracer 가져오기

    Args:
        name: 모듈 이름 (보통 __name__)

    Returns:
        Tracer 인스턴스 (OTEL 비활성화 시 NoOp tracer)
    """
    if _telemetry_manager:
        return _telemetry_manager.get_tracer(name)

    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return None


def get_meter(name: str):
    """편의 함수: Meter 가져오기

    Args:
        name: 모듈 이름 (보통 __name__)

    Returns:
        Meter 인스턴스 (OTEL 비활성화 시 NoOp meter)
    """
    if _telemetry_manager:
        return _telemetry_manager.get_meter(name)

    try:
        from opentelemetry import metrics

        return metrics.get_meter(name)
    except ImportError:
        return None


def setup_auto_instrumentation():
    """자동 계측 설정

    - httpx: HTTP 클라이언트 (Mistral API 등)
    - psycopg2: PostgreSQL
    - FastAPI는 별도로 apps/api/main.py에서 설정
    """
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

        HTTPXClientInstrumentor().instrument()
        Psycopg2Instrumentor().instrument()

        logger.info("Auto instrumentation enabled: httpx, psycopg2")
    except ImportError:
        logger.warning("Auto instrumentation libraries not installed")

