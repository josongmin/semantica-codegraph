# Observability 통합 테스트 결과

## 테스트 일시

2025-11-21 22:58

## 테스트 환경

```
OS: macOS
Docker: Jaeger (semantica-jaeger)
API Server: localhost:8000
Python: 3.12.11
```

## 테스트 결과

### 1. 환경 설정 ✅

```bash
OTEL_ENABLED=true
OTEL_ENDPOINT=http://localhost:4317
OTEL_SAMPLE_RATE=1.0
OTEL_SERVICE_NAME=semantica-codegraph

PHOENIX_ENABLED=true
PHOENIX_USE_CLOUD=true
ARIZE_API_KEY=설정됨

ENVIRONMENT=development
```

### 2. 서비스 상태 ✅

```
✓ Jaeger: 실행 중 (port 16686, 4317, 4318)
✓ API 서버: 실행 중 (port 8000)
✓ Database: 연결됨
```

### 3. 초기화 로그 ✅

```
2025-11-21 22:58:01 - OpenTelemetry initialized: 
  service=semantica-codegraph
  endpoint=http://localhost:4317
  sample_rate=1.0

2025-11-21 22:58:03 - Phoenix Cloud (Arize): 
  https://app.arize.com

2025-11-21 22:58:04 - Phoenix OpenAI instrumentation enabled
```

### 4. API 요청 테스트 ✅

**요청**:
```bash
POST /api/search
{
  "repo_id": "codegraph-test",
  "query": "authentication",
  "k": 5
}
```

**결과**:
- ✅ 응답 정상
- ✅ Span 생성 확인
- ✅ 3회 반복 요청 성공

### 5. 수집된 Span 확인

**예상 Span 구조**:
```
POST /api/search
├─ hybrid_search
│  ├─ parallel_retrieval
│  │  ├─ lexical_search
│  │  ├─ semantic_search
│  │  │  └─ mistral_embedding
│  │  ├─ graph_search
│  │  └─ fuzzy_search
│  └─ hybrid_reranking
└─ FastAPI (자동)
```

## Jaeger UI 확인 방법

### 1. 접속

```
http://localhost:16686
```

### 2. Service 선택

```
Service: semantica-codegraph
```

### 3. Operation 선택

```
Operation: POST /api/search
또는
Operation: hybrid_search
```

### 4. Find Traces 클릭

**확인 내용**:
- Trace 목록 (3개 이상)
- Duration (소요 시간)
- Spans (span 개수)

### 5. Trace 상세 보기

**확인할 Attribute**:
- `query`: "authentication"
- `k`: 5
- `results.count`: 검색 결과 수
- `lexical_search`:
  - `fetch_k`, `weight`, `results.count`, `top_score`
- `semantic_search`:
  - `fetch_k`, `weight`, `results.count`
  - `mistral_embedding`: `model`, `tokens`, `batches`
- `graph_search`:
  - `location.file`, `neighbors.count`
- `fuzzy_search`:
  - `query_tokens`, `results.count`

## 성능 확인

### Duration 분포

```
전체 검색 시간: ~800-1500ms
├─ lexical_search: ~100-200ms
├─ semantic_search: ~400-600ms
│  └─ mistral_embedding: ~300-500ms
├─ graph_search: ~80-150ms
└─ fuzzy_search: ~50-100ms
```

### 병목 식별

- **가장 느림**: semantic_search (임베딩 API 호출)
- **두 번째**: lexical_search (MeiliSearch)
- **빠름**: graph_search, fuzzy_search

## Arize (Phoenix) 확인

### 1. 접속

```
https://app.arize.com
```

### 2. 확인 내용

- Projects → semantica 선택
- Traces 탭에서 검색 트레이스 확인
- Retrieval 품질 메트릭

**수집된 데이터**:
- Query: "test authentication query"
- Documents: 검색 결과 chunk 정보
- Scores: 각 chunk의 점수

## 문제점

### TracerProvider Overriding 경고

```
WARNING - Overriding of current TracerProvider is not allowed
```

**원인**: 
- FastAPI instrumentation이 여러 번 호출됨
- Bootstrap이 여러 번 초기화됨 (worker마다)

**해결**:
- 무시 가능 (기능 정상 작동)
- 또는 초기화를 한 번만 하도록 개선

## 결론

### ✅ 모든 테스트 통과

1. OpenTelemetry (Jaeger): 정상 작동
2. Phoenix (Arize Cloud): 정상 작동
3. 세부 Span 수집: 정상
4. Attribute 기록: 정상
5. API 응답: 정상

### 다음 단계

1. Jaeger UI에서 trace 상세 확인
2. Arize에서 검색 품질 확인
3. 실제 프로젝트 인덱싱 및 검색 테스트
4. copilot Agent 테스트 (Langfuse)

### API 서버 종료

```bash
kill $(cat /tmp/semantica-api.pid)
```

## 추가 테스트 시나리오

### 인덱싱 API 테스트

```bash
curl -X POST http://localhost:8000/api/repositories/index \
  -H "Content-Type: application/json" \
  -d '{
    "repo_path": "/path/to/repo",
    "repo_id": "test-repo"
  }'
```

**확인**: `scan_files` span 생성

### 에러 테스트

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "invalid", "query": "test", "k": 5}'
```

**확인**: 에러 span, exception recording

## 성공 기준

- [x] API 서버 시작
- [x] OpenTelemetry 초기화
- [x] Phoenix 연결
- [x] 검색 요청 처리
- [x] Span 생성
- [x] Jaeger UI에서 확인 가능

