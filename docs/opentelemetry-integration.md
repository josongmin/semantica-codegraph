# OpenTelemetry 통합 가이드

## 아키텍처

### 역할 분담

```
개발 환경:
  - DetailedProfiler (기존)
  - 인덱싱 전체 분석
  - 파일별 세부 지표
  - JSON/텍스트 리포트

프로덕션 환경:
  - OpenTelemetry
  - 실시간 분산 트레이싱
  - LLM 호출 추적 (OtelLLM)
  - 메트릭/로그 수집
```

### 컴포넌트

```
┌─────────────────┐
│  FastAPI App    │
│  (Middleware)   │
└────────┬────────┘
         │ traces, metrics
         ↓
┌─────────────────┐
│  OTLP Exporter  │──→ Jaeger/Tempo
└─────────────────┘

┌─────────────────┐
│   OpenLIT       │──→ LLM 호출 추적
└─────────────────┘     (비용, 토큰)
```

## 1단계: 의존성 추가

### codegraph

```toml
# pyproject.toml
dependencies = [
    # ... 기존 의존성
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-instrumentation-fastapi>=0.41b0",
    "opentelemetry-instrumentation-httpx>=0.41b0",
    "opentelemetry-instrumentation-psycopg2>=0.41b0",
    "opentelemetry-exporter-otlp>=1.20.0",
]

llm = [
    "litellm>=1.0.0",
    "openlit>=1.0.0",  # LLM 전용 telemetry
]
```

### copilot

```toml
dependencies = [
    # ... 기존 의존성
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "openlit>=1.0.0",
]
```

## 2단계: 초기화 모듈

### `src/core/telemetry.py`

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased


def setup_telemetry(
    service_name: str,
    environment: str = "development",
    enabled: bool = True,
    otlp_endpoint: str = "http://localhost:4317",
    sample_rate: float = 1.0,
):
    """OpenTelemetry 초기화"""
    if not enabled:
        return None

    resource = Resource.create({
        "service.name": service_name,
        "service.version": "0.1.0",
        "deployment.environment": environment,
    })

    sampler = TraceIdRatioBased(sample_rate)
    provider = TracerProvider(resource=resource, sampler=sampler)
    
    # OTLP Exporter (Jaeger/Tempo)
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    trace.set_tracer_provider(provider)

    # 자동 계측
    FastAPIInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    Psycopg2Instrumentor().instrument()

    return trace.get_tracer(__name__)
```

## 3단계: FastAPI 통합

### `apps/api/main.py`

```python
from src.core.telemetry import setup_telemetry

# 환경변수
OTEL_ENABLED = os.getenv("OTEL_ENABLED", "false") == "true"
OTEL_ENDPOINT = os.getenv("OTEL_ENDPOINT", "http://localhost:4317")
OTEL_SAMPLE_RATE = float(os.getenv("OTEL_SAMPLE_RATE", "0.1"))

# 앱 초기화 시
if OTEL_ENABLED:
    tracer = setup_telemetry(
        service_name="semantica-codegraph-api",
        environment=os.getenv("ENVIRONMENT", "production"),
        otlp_endpoint=OTEL_ENDPOINT,
        sample_rate=OTEL_SAMPLE_RATE,
    )
```

## 4단계: LLM 호출 추적 (OpenLIT)

### OpenLIT 사용

```python
# src/embedding/service.py
import openlit

# 초기화 (앱 시작 시 한 번)
if OTEL_ENABLED:
    openlit.init(
        otlp_endpoint=OTEL_ENDPOINT,
        application_name="semantica-codegraph",
    )

# 기존 코드는 변경 없음
# LiteLLM 호출 자동 추적
response = await litellm.aembedding(...)
```

**추적 정보**:
- 토큰 수 (input/output)
- 레이턴시
- 모델명
- 비용 (자동 계산)
- 에러율

## 5단계: 커스텀 Span 추가

### 인덱싱 파이프라인

```python
# src/indexer/pipeline.py
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

class IndexingPipeline:
    async def index_repository_async(self, repo_config: RepoConfig):
        with tracer.start_as_current_span("index_repository") as span:
            span.set_attribute("repo.id", repo_config.repo_id)
            span.set_attribute("repo.file_count", len(files))
            
            # 파싱
            with tracer.start_as_current_span("parse_files"):
                parsed = await self._parse_files(files)
            
            # IR 빌드
            with tracer.start_as_current_span("build_ir"):
                ir_nodes = self.ir_builder.build(parsed)
            
            # 청킹
            with tracer.start_as_current_span("chunking"):
                chunks = self.chunker.chunk(ir_nodes)
            
            # 임베딩
            with tracer.start_as_current_span("embedding") as emb_span:
                emb_span.set_attribute("chunk.count", len(chunks))
                await self.embedding_store.store_embeddings(chunks)
```

### 검색 파이프라인

```python
# src/search/retriever/hybrid_retriever.py
with tracer.start_as_current_span("hybrid_search") as span:
    span.set_attribute("query", query)
    span.set_attribute("retrieve_k", retrieve_k)
    
    # 병렬 검색
    with tracer.start_as_current_span("parallel_retrieval"):
        lexical = await self.lexical.search(query)
        semantic = await self.semantic.search(query)
        graph = await self.graph.search(query)
    
    # 퓨전
    with tracer.start_as_current_span("fusion"):
        merged = self.fusion.merge(lexical, semantic, graph)
    
    # 리랭킹
    with tracer.start_as_current_span("reranking") as rerank_span:
        rerank_span.set_attribute("reranker.type", "hybrid")
        results = await self.reranker.rerank(merged, query)
```

## 6단계: 메트릭 수집

```python
from opentelemetry import metrics

meter = metrics.get_meter(__name__)

# 카운터
embedding_api_calls = meter.create_counter(
    "embedding.api.calls",
    description="임베딩 API 호출 횟수",
)

# 히스토그램
embedding_latency = meter.create_histogram(
    "embedding.latency",
    description="임베딩 API 레이턴시 (초)",
    unit="s",
)

# 사용
embedding_api_calls.add(1, {"model": "codestral-embed"})
embedding_latency.record(duration, {"model": "codestral-embed"})
```

## 7단계: 백엔드 설정

### Jaeger (Docker Compose)

```yaml
# docker-compose.yml
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
```

## 환경변수

```bash
# .env
OTEL_ENABLED=true
OTEL_ENDPOINT=http://localhost:4317
OTEL_SAMPLE_RATE=0.1  # 프로덕션: 1-10%
ENVIRONMENT=production
```

## 대시보드 예시

### LLM 비용 추적

```
┌─────────────────────────────────┐
│ 임베딩 API 비용 (일일)           │
│ $12.34                          │
│ ↑ 15% vs 어제                   │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ 모델별 토큰 사용량               │
│ codestral-embed:  1.2M tokens   │
│ gpt-4o-mini:      340K tokens   │
└─────────────────────────────────┘
```

### 검색 성능

```
┌─────────────────────────────────┐
│ P95 레이턴시                     │
│ 전체: 450ms                      │
│ - Semantic: 180ms                │
│ - Reranking: 120ms               │
│ - Lexical: 90ms                  │
└─────────────────────────────────┘
```

## 실행 순서

```bash
# 1. 백엔드 시작
docker-compose up -d jaeger

# 2. API 서버 시작 (Otel 활성화)
OTEL_ENABLED=true uvicorn apps.api.main:app

# 3. Jaeger UI 확인
open http://localhost:16686

# 4. 요청 실행
curl -X POST http://localhost:8000/api/search \
  -d '{"query": "authentication"}'

# 5. 트레이스 확인
# Jaeger UI에서 "semantica-codegraph-api" 서비스 선택
```

## 기존 프로파일러와 공존

```python
# 개발: 기존 프로파일러 사용
./index_profiler.py ../repo

# 프로덕션: OpenTelemetry
OTEL_ENABLED=true uvicorn apps.api.main:app

# 둘 다 사용 가능 (오버헤드 고려)
if config.profiler_enabled:
    pipeline.profiler = DetailedProfiler(...)
if config.otel_enabled:
    with tracer.start_span("index"):
        ...
```

