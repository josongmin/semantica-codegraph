# Span 수집 설정 가이드

## 현재 수집되는 Span

### 자동 계측 (Instrumentation)

```python
# src/core/telemetry.py에서 자동 설정됨
- FastAPI: 모든 HTTP 엔드포인트
- httpx: 외부 API 호출 (Mistral, OpenAI 등)
- psycopg2: PostgreSQL 쿼리
- OpenLIT: LLM API 호출
```

### 수동 Span

```python
# src/search/adapters/retriever/hybrid_retriever.py
- hybrid_search: 전체 검색
```

## 1. 샘플링 설정 (가장 중요!)

### 환경변수로 제어

```bash
# .env
OTEL_SAMPLE_RATE=1.0   # 100% 수집 (개발)
OTEL_SAMPLE_RATE=0.1   # 10% 수집 (프로덕션)
OTEL_SAMPLE_RATE=0.01  # 1% 수집 (대용량)
```

### 동적 샘플링

```python
# src/core/telemetry.py
from opentelemetry.sdk.trace.sampling import (
    TraceIdRatioBased,
    ParentBased,
    ALWAYS_ON,
    ALWAYS_OFF,
)

class CustomSampler:
    """조건부 샘플링"""
    
    def should_sample(self, context, trace_id, name, attributes):
        # 느린 요청은 항상 수집
        if attributes.get("http.target") == "/api/search":
            return ALWAYS_ON
        
        # 헬스체크는 무시
        if attributes.get("http.target") == "/health":
            return ALWAYS_OFF
        
        # 나머지는 비율 적용
        return TraceIdRatioBased(0.1)
```

## 2. Span 필터링

### 특정 엔드포인트만 수집

```python
# src/core/telemetry.py 수정
def setup_auto_instrumentation():
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    
    # 제외할 경로 설정
    FastAPIInstrumentor().instrument(
        excluded_urls="/health,/metrics,/docs,/openapi.json"
    )
```

### PostgreSQL 쿼리 필터

```python
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

Psycopg2Instrumentor().instrument(
    # 느린 쿼리만 수집
    enable_commenter=True,
    commenter_options={
        "db_driver": True,
        "dbapi_level": True,
        "dbapi_threadsafety": True,
    },
)
```

## 3. Span Attribute 커스터마이즈

### 검색 Span에 더 많은 정보 추가

```python
# src/search/adapters/retriever/hybrid_retriever.py
with tracer.start_as_current_span("hybrid_search") as span:
    # 기본 정보
    span.set_attribute("query", query[:100])
    span.set_attribute("k", k)
    
    # 추가 정보
    span.set_attribute("repo_id", repo_id)
    span.set_attribute("parallel", parallel)
    span.set_attribute("weights.lexical", weights.get("lexical", 0))
    span.set_attribute("weights.semantic", weights.get("semantic", 0))
    
    # 결과 정보
    span.set_attribute("results.count", len(results))
    span.set_attribute("results.top_score", results[0].score if results else 0)
    
    # 성능 정보
    span.set_attribute("duration_ms", int(duration * 1000))
```

### LLM 호출 Span

```python
# src/embedding/mistral_embeddings.py
with tracer.start_as_current_span("mistral_embedding") as span:
    span.set_attribute("model", "codestral-embed")
    span.set_attribute("input.count", len(texts))
    span.set_attribute("input.total_length", sum(len(t) for t in texts))
    
    response = await client.embeddings.create(...)
    
    # 응답 정보
    span.set_attribute("output.dimension", len(response.data[0].embedding))
    span.set_attribute("tokens", response.usage.total_tokens)
    span.set_attribute("cost", calculate_cost(response.usage))
```

## 4. 세분화된 Span 추가

### 인덱싱 파이프라인

```python
# src/indexer/pipeline.py
async def index_repository_async(self, repo_config: RepoConfig):
    with tracer.start_as_current_span("index_repository") as span:
        span.set_attribute("repo.id", repo_config.repo_id)
        span.set_attribute("repo.path", repo_config.repo_path)
        
        # 1. 파일 스캔
        with tracer.start_as_current_span("scan_files") as scan_span:
            files = self.scanner.scan(repo_config.repo_path)
            scan_span.set_attribute("files.count", len(files))
            scan_span.set_attribute("files.total_size", sum(f.size for f in files))
        
        # 2. 파싱
        with tracer.start_as_current_span("parse_files") as parse_span:
            parsed = await self._parse_files_async(files)
            parse_span.set_attribute("files.parsed", len(parsed))
            parse_span.set_attribute("files.failed", len(files) - len(parsed))
            
            # 파서별 통계
            for parser_type, count in parser_stats.items():
                parse_span.set_attribute(f"parser.{parser_type}", count)
        
        # 3. 임베딩 (세부 추적)
        with tracer.start_as_current_span("embedding") as emb_span:
            # 배치별 추적
            for i, batch in enumerate(chunks_batches):
                with tracer.start_as_current_span(f"embedding_batch_{i}"):
                    await self.embed_batch(batch)
```

### 검색 파이프라인 세부화

```python
# src/search/adapters/retriever/hybrid_retriever.py
with tracer.start_as_current_span("parallel_retrieval") as ret_span:
    # 각 백엔드별 Span
    with tracer.start_as_current_span("lexical_search") as lex_span:
        lexical = await self.lexical.search(query)
        lex_span.set_attribute("results.count", len(lexical))
        lex_span.set_attribute("results.top_score", lexical[0].score if lexical else 0)
    
    with tracer.start_as_current_span("semantic_search") as sem_span:
        semantic = await self.semantic.search(query)
        sem_span.set_attribute("results.count", len(semantic))
        sem_span.set_attribute("embedding.cache_hit", cache_hit)
    
    with tracer.start_as_current_span("graph_search") as graph_span:
        graph = await self.graph.search(query)
        graph_span.set_attribute("results.count", len(graph))
        graph_span.set_attribute("depth", 2)
```

## 5. Span Processor 커스터마이즈

### 민감 정보 필터링

```python
# src/core/telemetry.py
from opentelemetry.sdk.trace import SpanProcessor

class SensitiveDataFilter(SpanProcessor):
    """민감 정보 제거"""
    
    def on_end(self, span):
        # API 키 마스킹
        for attr in span.attributes:
            if "api_key" in attr.lower():
                span.set_attribute(attr, "***REDACTED***")
            
            # 쿼리 길이 제한
            if attr == "query" and len(span.attributes[attr]) > 200:
                span.set_attribute(attr, span.attributes[attr][:200] + "...")

# Bootstrap에서 추가
trace_provider.add_span_processor(SensitiveDataFilter())
```

### 성능 임계값 필터

```python
class SlowSpanFilter(SpanProcessor):
    """느린 Span만 수집"""
    
    def __init__(self, threshold_ms=1000):
        self.threshold_ms = threshold_ms
    
    def on_end(self, span):
        duration_ms = (span.end_time - span.start_time) / 1_000_000
        
        if duration_ms < self.threshold_ms:
            # 빠른 요청은 drop
            span.set_status(Status(StatusCode.UNSET))
```

## 6. 실용적인 설정 예시

### 개발 환경

```bash
# .env.development
OTEL_ENABLED=true
OTEL_SAMPLE_RATE=1.0  # 모든 요청 추적
OTEL_SPAN_DETAIL=high  # 상세 정보 포함
```

```python
# 모든 Span 수집
# 상세한 attribute
# DB 쿼리 포함
```

### 스테이징 환경

```bash
# .env.staging
OTEL_ENABLED=true
OTEL_SAMPLE_RATE=0.5  # 50% 샘플링
OTEL_SPAN_DETAIL=medium
```

```python
# 주요 엔드포인트 우선
# LLM 호출은 항상 추적
# DB 쿼리는 느린 것만
```

### 프로덕션 환경

```bash
# .env.production
OTEL_ENABLED=true
OTEL_SAMPLE_RATE=0.1  # 10% 샘플링
OTEL_SPAN_DETAIL=low
```

```python
# 에러 요청 항상 추적
# 정상 요청 10% 샘플링
# 헬스체크 제외
# 민감 정보 마스킹
```

## 7. 추천 설정

### 최소 설정 (비용 절감)

```python
# 수집할 Span
✓ API 엔드포인트 (10% 샘플링)
✓ LLM API 호출 (항상)
✓ 에러 발생 (항상)
✗ DB 쿼리 (제외)
✗ 내부 함수 (제외)
```

### 균형 설정 (추천)

```python
# 수집할 Span
✓ API 엔드포인트 (50% 샘플링)
✓ 검색 파이프라인 (항상)
✓ LLM API 호출 (항상)
✓ 느린 DB 쿼리 (>100ms만)
✗ 헬스체크 (제외)
```

### 최대 설정 (디버깅)

```python
# 수집할 Span
✓ 모든 요청 (100%)
✓ 모든 함수 (상세)
✓ DB 쿼리 (모두)
✓ 캐시 히트/미스
✓ 내부 로직
```

## 8. 구현 예시

### Config에 설정 추가

```python
# src/core/config.py
@dataclass
class Config:
    # ... 기존 설정
    
    # Span 수집 설정
    otel_span_detail: str = "medium"  # low, medium, high
    otel_collect_db_queries: bool = False
    otel_slow_query_threshold_ms: int = 100
    otel_exclude_paths: list[str] = None
```

### 조건부 Span 생성

```python
# src/search/adapters/retriever/hybrid_retriever.py
def retrieve(self, repo_id, query, k):
    # 상세 모드일 때만 세부 Span 생성
    if self.config.otel_span_detail == "high":
        with tracer.start_as_current_span("hybrid_search_detailed"):
            return self._retrieve_with_detailed_spans(...)
    else:
        # 간단한 Span만
        with tracer.start_as_current_span("hybrid_search"):
            return self._retrieve_simple(...)
```

## 9. 모니터링

### Span 통계 확인

Jaeger UI에서:
- **Service**: `semantica-codegraph` 선택
- **Operations**: 어떤 Span이 가장 많은지 확인
- **Duration**: 어떤 Span이 느린지 확인

### 비용 관리

```python
# 예상 비용 계산
spans_per_day = 10000 (요청) × 5 (span/request) × sample_rate
storage_cost = spans_per_day × 30 (days) × $0.0001
```

## 참고

- OpenTelemetry 샘플링: https://opentelemetry.io/docs/specs/otel/trace/sdk/#sampling
- Span Processor: https://opentelemetry.io/docs/instrumentation/python/sdk/#span-processors

