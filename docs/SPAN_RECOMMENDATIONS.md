# Span 수집 추천 설정

## 현재 프로젝트 상황 분석

### 주요 엔드포인트

```python
1. POST /api/search          # 가장 중요 ⭐⭐⭐
2. POST /api/repositories/index  # 중요 ⭐⭐
3. GET /api/repositories     # 부차적 ⭐
4. GET /health              # 제외
```

### LLM API 호출

```python
1. Mistral embedding (검색마다 1회)     # 항상 추적 ⭐⭐⭐
2. OpenAI embedding (인덱싱 시)         # 항상 추적 ⭐⭐⭐
3. Reranking LLM (설정 시)             # 항상 추적 ⭐⭐⭐
```

## 추천 설정

### 개발 환경 (현재)

```bash
# .env
OTEL_ENABLED=true
OTEL_SAMPLE_RATE=1.0          # 100% (모든 요청)
OTEL_SERVICE_NAME=semantica-codegraph
ENVIRONMENT=development

PHOENIX_ENABLED=true
PHOENIX_USE_CLOUD=true
```

**수집 Span**:
- ✅ 모든 API 요청
- ✅ 모든 검색 단계
- ✅ 모든 LLM 호출
- ✅ DB 쿼리
- ✅ 상세 attribute

**장점**: 버그 빠르게 찾기, 성능 최적화

### 프로덕션 환경 (추천)

```bash
# .env.production
OTEL_ENABLED=true
OTEL_SAMPLE_RATE=0.1          # 10% (비용 절감)
OTEL_SERVICE_NAME=semantica-codegraph
ENVIRONMENT=production

# 중요 요청은 항상 수집
OTEL_ALWAYS_SAMPLE_PATHS=/api/search,/api/repositories/index

# 제외할 경로
OTEL_EXCLUDE_PATHS=/health,/metrics,/docs

# Span 상세 레벨
OTEL_SPAN_DETAIL=medium

PHOENIX_ENABLED=true
PHOENIX_USE_CLOUD=true
```

**수집 Span**:
- ✅ 검색 API (항상 100%)
- ✅ 인덱싱 API (항상 100%)
- ⚠️  기타 API (10% 샘플링)
- ✅ LLM 호출 (항상 100%)
- ❌ 헬스체크 (제외)
- ⚠️  DB 쿼리 (느린 것만)

**장점**: 비용 90% 절감, 중요 정보 유지

## 실제 적용 코드

### Config 확장

```python
# src/core/config.py
@dataclass
class Config:
    # ... 기존 설정
    
    # Span 수집 상세 설정
    otel_span_detail: str = "medium"  # low, medium, high
    otel_always_sample_paths: list[str] = None  # 항상 수집할 경로
    otel_exclude_paths: list[str] = None  # 제외할 경로
    otel_collect_db_queries: bool = True  # DB 쿼리 수집
    otel_slow_query_threshold_ms: int = 100  # 느린 쿼리 임계값
    
    @classmethod
    def from_env(cls):
        # ... 기존 코드
        
        # Span 설정
        always_sample = os.getenv("OTEL_ALWAYS_SAMPLE_PATHS", "")
        always_sample_paths = [p.strip() for p in always_sample.split(",") if p.strip()]
        
        exclude = os.getenv("OTEL_EXCLUDE_PATHS", "/health,/metrics")
        exclude_paths = [p.strip() for p in exclude.split(",") if p.strip()]
        
        return cls(
            # ... 기존 설정
            otel_span_detail=os.getenv("OTEL_SPAN_DETAIL", "medium"),
            otel_always_sample_paths=always_sample_paths,
            otel_exclude_paths=exclude_paths,
            otel_collect_db_queries=os.getenv("OTEL_COLLECT_DB_QUERIES", "true") == "true",
            otel_slow_query_threshold_ms=int(os.getenv("OTEL_SLOW_QUERY_THRESHOLD_MS", "100")),
        )
```

### 조건부 샘플링 구현

```python
# src/core/telemetry.py
from opentelemetry.sdk.trace.sampling import (
    Sampler,
    SamplingResult,
    Decision,
    TraceIdRatioBased,
)

class SmartSampler(Sampler):
    """조건부 샘플링
    
    - 중요 경로: 항상 수집
    - 제외 경로: 수집 안 함
    - 나머지: 비율 적용
    """
    
    def __init__(
        self,
        sample_rate: float,
        always_sample_paths: list[str],
        exclude_paths: list[str],
    ):
        self.ratio_sampler = TraceIdRatioBased(sample_rate)
        self.always_sample_paths = always_sample_paths
        self.exclude_paths = exclude_paths
    
    def should_sample(self, parent_context, trace_id, name, kind, attributes, links):
        # HTTP 경로 추출
        http_target = attributes.get("http.target", "")
        
        # 제외 경로
        for path in self.exclude_paths:
            if path in http_target:
                return SamplingResult(Decision.DROP)
        
        # 항상 수집 경로
        for path in self.always_sample_paths:
            if path in http_target:
                return SamplingResult(Decision.RECORD_AND_SAMPLE)
        
        # 나머지는 비율 적용
        return self.ratio_sampler.should_sample(
            parent_context, trace_id, name, kind, attributes, links
        )
    
    def get_description(self):
        return f"SmartSampler(rate={self.ratio_sampler._rate})"


# TelemetryManager에서 사용
def setup_telemetry(config):
    # ...
    
    sampler = SmartSampler(
        sample_rate=config.otel_sample_rate,
        always_sample_paths=config.otel_always_sample_paths or ["/api/search"],
        exclude_paths=config.otel_exclude_paths or ["/health"],
    )
    
    provider = TracerProvider(resource=resource, sampler=sampler)
```

## 실전 예시

### 시나리오 1: 비용 최소화

```bash
# .env.production
OTEL_SAMPLE_RATE=0.01                    # 1%만 수집
OTEL_ALWAYS_SAMPLE_PATHS=/api/search     # 검색은 100%
OTEL_EXCLUDE_PATHS=/health,/metrics,/docs
OTEL_SPAN_DETAIL=low                     # 최소 정보
```

**결과**:
- 검색 API: 100% 추적 (중요!)
- 기타: 1% 샘플링
- Span 비용: 99% 절감

### 시나리오 2: 균형

```bash
# .env.staging
OTEL_SAMPLE_RATE=0.5                     # 50% 수집
OTEL_ALWAYS_SAMPLE_PATHS=/api/search,/api/repositories/index
OTEL_EXCLUDE_PATHS=/health
OTEL_SPAN_DETAIL=medium
```

**결과**:
- 검색/인덱싱: 100%
- 기타: 50%
- Span 비용: 50% 절감

### 시나리오 3: 디버깅 (에러 발생 시)

```bash
# .env.debug
OTEL_SAMPLE_RATE=1.0                     # 100% 수집
OTEL_SPAN_DETAIL=high                    # 모든 정보
OTEL_COLLECT_DB_QUERIES=true             # DB 쿼리 포함
```

**결과**:
- 모든 요청 추적
- 상세 정보
- DB 쿼리 포함

## 현재 프로젝트 추천

### 개발 중 (지금)

```bash
# .env (현재 설정 유지)
OTEL_ENABLED=true
OTEL_SAMPLE_RATE=1.0
PHOENIX_ENABLED=true
```

**이유**:
- 모든 요청 추적 (버그 찾기)
- 성능 최적화 데이터 수집
- 비용 걱정 없음 (로컬)

### 프로덕션 준비 시

```bash
# .env.production
OTEL_ENABLED=true
OTEL_SAMPLE_RATE=0.1                     # 10%
OTEL_ALWAYS_SAMPLE_PATHS=/api/search     # 검색 100%
OTEL_EXCLUDE_PATHS=/health,/metrics
PHOENIX_ENABLED=true
```

**이유**:
- 검색은 항상 추적 (핵심 기능)
- 기타는 10% (통계적으로 충분)
- 90% 비용 절감

## Span 데이터 예상량

### 개발 (SAMPLE_RATE=1.0)

```
하루 요청: 100회
Span/요청: 5개 (API → search → lexical/semantic/graph/fuzzy)
─────────────────
총 Span: 500개/일

저장 용량: ~50KB/일 (무시 가능)
```

### 프로덕션 (SAMPLE_RATE=0.1)

```
하루 요청: 10,000회
Span/요청: 5개
샘플링: 10%
─────────────────
총 Span: 5,000개/일

저장 용량: ~500KB/일
월간: ~15MB (무시 가능)
```

## 요약

**현재 설정 (개발)**: 완벽 ✅
- 모든 요청 추적
- 상세 정보
- 비용 없음

**변경 필요 없음!** 

프로덕션 배포 시에만 샘플링 조정하면 됩니다.

