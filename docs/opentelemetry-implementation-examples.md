# OpenTelemetry 구현 예시

## 핵심 컴포넌트별 구현

### 1. Config 확장

```python
# src/core/config.py
from dataclasses import dataclass

@dataclass
class Config:
    # ... 기존 설정

    # OpenTelemetry 설정
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"
    otel_sample_rate: float = 1.0
    environment: str = "development"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            # ... 기존 설정
            otel_enabled=os.getenv("OTEL_ENABLED", "false") == "true",
            otel_endpoint=os.getenv("OTEL_ENDPOINT", "http://localhost:4317"),
            otel_sample_rate=float(os.getenv("OTEL_SAMPLE_RATE", "1.0")),
            environment=os.getenv("ENVIRONMENT", "development"),
        )
```

### 2. Telemetry 초기화

```python
# src/core/telemetry.py
import logging
from typing import Optional
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

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

        # Resource 생성
        resource = Resource.create({
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            "deployment.environment": environment,
        })

        # Trace Provider 설정
        sampler = TraceIdRatioBased(sample_rate)
        trace_provider = TracerProvider(resource=resource, sampler=sampler)

        otlp_trace_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        trace_provider.add_span_processor(
            BatchSpanProcessor(otlp_trace_exporter)
        )
        trace.set_tracer_provider(trace_provider)

        # Metric Provider 설정
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=otlp_endpoint)
        )
        metric_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(metric_provider)

        logger.info(
            f"OpenTelemetry initialized: service={service_name}, "
            f"endpoint={otlp_endpoint}, sample_rate={sample_rate}"
        )

    def get_tracer(self, name: str) -> trace.Tracer:
        """Tracer 가져오기"""
        if not self.enabled:
            return trace.get_tracer(name)
        return trace.get_tracer(name, self.service_name)

    def get_meter(self, name: str) -> metrics.Meter:
        """Meter 가져오기"""
        if not self.enabled:
            return metrics.get_meter(name)
        return metrics.get_meter(name, self.service_name)


# 전역 인스턴스 (Bootstrap에서 초기화)
_telemetry_manager: Optional[TelemetryManager] = None


def init_telemetry(
    service_name: str,
    config,
) -> TelemetryManager:
    """TelemetryManager 초기화"""
    global _telemetry_manager
    _telemetry_manager = TelemetryManager(
        service_name=service_name,
        enabled=config.otel_enabled,
        otlp_endpoint=config.otel_endpoint,
        sample_rate=config.otel_sample_rate,
        environment=config.environment,
    )
    return _telemetry_manager


def get_tracer(name: str) -> trace.Tracer:
    """편의 함수: Tracer 가져오기"""
    if _telemetry_manager:
        return _telemetry_manager.get_tracer(name)
    return trace.get_tracer(name)


def get_meter(name: str) -> metrics.Meter:
    """편의 함수: Meter 가져오기"""
    if _telemetry_manager:
        return _telemetry_manager.get_meter(name)
    return metrics.get_meter(name)
```

### 3. Bootstrap 통합

```python
# src/core/bootstrap.py
from src.core.telemetry import init_telemetry

class Bootstrap:
    def __init__(self, config: Config):
        self.config = config

        # OpenTelemetry 초기화
        self.telemetry = init_telemetry("semantica-codegraph", config)

        # 자동 계측 (FastAPI 앱이 있는 경우)
        if config.otel_enabled:
            self._setup_auto_instrumentation()

        # ... 기존 초기화

    def _setup_auto_instrumentation(self):
        """자동 계측 설정"""
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

            HTTPXClientInstrumentor().instrument()
            Psycopg2Instrumentor().instrument()
        except ImportError:
            logger.warning("Auto instrumentation libraries not installed")
```

### 4. 인덱싱 파이프라인 계측

```python
# src/indexer/pipeline.py
from src.core.telemetry import get_tracer, get_meter

tracer = get_tracer(__name__)
meter = get_meter(__name__)

# 메트릭 정의
files_indexed = meter.create_counter(
    "indexing.files.total",
    description="인덱싱된 파일 수",
)

chunks_created = meter.create_counter(
    "indexing.chunks.total",
    description="생성된 청크 수",
)

indexing_duration = meter.create_histogram(
    "indexing.duration",
    description="인덱싱 소요 시간 (초)",
    unit="s",
)


class IndexingPipeline:
    async def index_repository_async(
        self, repo_config: RepoConfig
    ) -> RepoProfile:
        start_time = time.time()

        with tracer.start_as_current_span("index_repository") as span:
            span.set_attribute("repo.id", repo_config.repo_id)
            span.set_attribute("repo.path", repo_config.repo_path)
            span.set_attribute("repo.languages", ",".join(repo_config.languages))

            try:
                # 1. 파일 스캔
                with tracer.start_as_current_span("scan_files"):
                    files = self.scanner.scan(repo_config.repo_path)
                    span.set_attribute("files.count", len(files))

                # 2. 파싱
                with tracer.start_as_current_span("parse_files") as parse_span:
                    parsed = await self._parse_files_async(files)
                    parse_span.set_attribute("files.parsed", len(parsed))
                    parse_span.set_attribute("files.failed", len(files) - len(parsed))

                # 3. IR 빌드
                with tracer.start_as_current_span("build_ir") as ir_span:
                    ir_nodes = self.ir_builder.build(parsed)
                    ir_span.set_attribute("ir_nodes.count", len(ir_nodes))

                # 4. 청킹
                with tracer.start_as_current_span("chunking") as chunk_span:
                    chunks = self.chunker.chunk(ir_nodes)
                    chunk_span.set_attribute("chunks.count", len(chunks))
                    chunks_created.add(len(chunks), {"repo_id": repo_config.repo_id})

                # 5. 임베딩
                with tracer.start_as_current_span("embedding") as emb_span:
                    await self.embedding_store.store_embeddings(chunks)
                    emb_span.set_attribute("embeddings.count", len(chunks))

                # 메트릭 기록
                duration = time.time() - start_time
                indexing_duration.record(
                    duration,
                    {"repo_id": repo_config.repo_id, "status": "success"}
                )
                files_indexed.add(len(files), {"repo_id": repo_config.repo_id})

                span.set_status(trace.Status(trace.StatusCode.OK))
                return profile

            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                indexing_duration.record(
                    time.time() - start_time,
                    {"repo_id": repo_config.repo_id, "status": "error"}
                )
                raise
```

### 5. 검색 파이프라인 계측

```python
# src/search/retriever/hybrid_retriever.py
from src.core.telemetry import get_tracer, get_meter

tracer = get_tracer(__name__)
meter = get_meter(__name__)

# 메트릭
search_requests = meter.create_counter(
    "search.requests.total",
    description="검색 요청 수",
)

search_latency = meter.create_histogram(
    "search.latency",
    description="검색 레이턴시 (초)",
    unit="s",
)


class HybridRetriever:
    async def search(
        self, query: str, retrieve_k: int = 100
    ) -> list[SearchResult]:
        start_time = time.time()

        with tracer.start_as_current_span("hybrid_search") as span:
            span.set_attribute("query", query[:100])  # 100자만
            span.set_attribute("retrieve_k", retrieve_k)

            try:
                # 병렬 검색
                with tracer.start_as_current_span("parallel_retrieval") as ret_span:
                    lexical_task = self.lexical.search(query)
                    semantic_task = self.semantic.search(query)
                    graph_task = self.graph.search(query)

                    lexical, semantic, graph = await asyncio.gather(
                        lexical_task, semantic_task, graph_task
                    )

                    ret_span.set_attribute("lexical.count", len(lexical))
                    ret_span.set_attribute("semantic.count", len(semantic))
                    ret_span.set_attribute("graph.count", len(graph))

                # 퓨전
                with tracer.start_as_current_span("fusion") as fusion_span:
                    merged = self.fusion.merge(lexical, semantic, graph)
                    fusion_span.set_attribute("merged.count", len(merged))

                # 리랭킹
                with tracer.start_as_current_span("reranking") as rerank_span:
                    results = await self.reranker.rerank(merged, query)
                    rerank_span.set_attribute("results.count", len(results))
                    rerank_span.set_attribute("reranker.type", type(self.reranker).__name__)

                # 메트릭
                duration = time.time() - start_time
                search_latency.record(duration, {"status": "success"})
                search_requests.add(1, {"status": "success"})

                span.set_status(trace.Status(trace.StatusCode.OK))
                return results

            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                search_latency.record(
                    time.time() - start_time,
                    {"status": "error"}
                )
                search_requests.add(1, {"status": "error"})
                raise
```

### 6. LLM 호출 계측 (OpenLIT)

```python
# src/embedding/mistral_embeddings.py
import openlit

# Bootstrap에서 초기화됨
# openlit.init(otlp_endpoint=config.otel_endpoint)

class MistralEmbeddings:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # OpenLIT가 자동으로 추적
        # - 토큰 수
        # - 비용
        # - 레이턴시
        # - 모델명
        response = await self.client.embeddings.create(
            model="mistral-embed",
            inputs=texts,
        )
        return [item.embedding for item in response.data]
```

### 7. FastAPI 통합

```python
# apps/api/main.py
from fastapi import FastAPI
from src.core.bootstrap import create_bootstrap
from src.core.telemetry import init_telemetry

app = FastAPI()

@app.on_event("startup")
async def startup():
    bootstrap = create_bootstrap()

    # OpenTelemetry 초기화
    if bootstrap.config.otel_enabled:
        init_telemetry("semantica-codegraph-api", bootstrap.config)

        # FastAPI 자동 계측
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)

        # OpenLIT 초기화 (LLM 추적)
        try:
            import openlit
            openlit.init(
                otlp_endpoint=bootstrap.config.otel_endpoint,
                application_name="semantica-codegraph-api",
            )
        except ImportError:
            logger.warning("OpenLIT not installed")

    app.state.bootstrap = bootstrap


@app.post("/api/search")
async def search(request: SearchRequest):
    # 자동 계측됨 (FastAPIInstrumentor)
    # 추가 span 필요시:
    from src.core.telemetry import get_tracer
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("api.search") as span:
        span.set_attribute("query", request.query)
        results = await app.state.bootstrap.hybrid_retriever.search(
            request.query
        )
        return {"results": results}
```

## 환경별 설정

### 개발 환경

```bash
# .env.development
OTEL_ENABLED=false  # 개발 시 비활성화
# 또는
OTEL_ENABLED=true
OTEL_ENDPOINT=http://localhost:4317
OTEL_SAMPLE_RATE=1.0  # 모든 요청 추적
ENVIRONMENT=development
```

### 스테이징 환경

```bash
# .env.staging
OTEL_ENABLED=true
OTEL_ENDPOINT=http://jaeger-collector:4317
OTEL_SAMPLE_RATE=0.5  # 50% 샘플링
ENVIRONMENT=staging
```

### 프로덕션 환경

```bash
# .env.production
OTEL_ENABLED=true
OTEL_ENDPOINT=http://jaeger-collector:4317
OTEL_SAMPLE_RATE=0.1  # 10% 샘플링
ENVIRONMENT=production
```

## 테스트

```python
# tests/test_telemetry.py
import pytest
from src.core.telemetry import TelemetryManager, get_tracer

def test_telemetry_disabled():
    tm = TelemetryManager("test-service", enabled=False)
    tracer = tm.get_tracer("test")

    with tracer.start_as_current_span("test_span"):
        pass  # 오버헤드 없음

def test_telemetry_enabled():
    tm = TelemetryManager(
        "test-service",
        enabled=True,
        otlp_endpoint="http://localhost:4317",
        sample_rate=1.0,
    )
    tracer = tm.get_tracer("test")

    with tracer.start_as_current_span("test_span") as span:
        span.set_attribute("test.key", "value")
        # Jaeger에서 확인 가능
```
