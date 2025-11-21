# Observability 완전 통합 완료

## 통합된 도구

### 1. OpenTelemetry + Jaeger (범용 트레이싱)

**대상**: codegraph, copilot 양쪽

**추적 내용**:
- API 요청/응답
- 데이터베이스 쿼리
- 외부 API 호출
- 서비스 간 통신

**UI**: http://localhost:16686

### 2. Arize/Phoenix (검색 품질)

**대상**: codegraph

**추적 내용**:
- 검색 품질 메트릭 (Precision, Recall, MRR)
- Embeddings 시각화
- Retrieval 성능 분석

**UI**: https://app.arize.com

### 3. Langfuse (Agent 세션)

**대상**: copilot

**추적 내용**:
- Agent 실행 세션
- LLM 호출 (토큰, 비용)
- Prompt 관리
- 평가 및 피드백

**UI**: https://cloud.langfuse.com

### 4. OpenLIT (LLM 자동 추적)

**대상**: codegraph, copilot 양쪽

**추적 내용**:
- LLM API 호출 자동 추적
- 토큰 사용량
- 비용 자동 계산
- Jaeger에 통합

## 추가된 Span

### codegraph

```
POST /api/search
├─ hybrid_search (전체)
│  ├─ parallel_retrieval
│  │  ├─ lexical_search
│  │  │  └─ attributes: fetch_k, weight, results.count, top_score
│  │  ├─ semantic_search
│  │  │  ├─ mistral_embedding
│  │  │  │  └─ attributes: model, texts.count, tokens, batches
│  │  │  └─ attributes: fetch_k, weight, results.count
│  │  ├─ graph_search
│  │  │  └─ attributes: weight, location, neighbors.count
│  │  └─ fuzzy_search
│  │     └─ attributes: weight, query_tokens, results.count
│  └─ hybrid_reranking
│     └─ attributes: query_type, candidates.count, top_score
└─ FastAPI (자동)
```

**인덱싱**:
```
POST /api/repositories/index
├─ scan_files
│  └─ attributes: files.count, total_size
├─ parse_files
├─ build_ir
├─ chunking
└─ embedding
   └─ mistral_embedding (배치별)
```

### copilot

```
Agent 실행
├─ intent_router (Langfuse)
│  ├─ input: query, location_ctx
│  └─ output: execution_plan
├─ plan_executor (Langfuse)
│  ├─ input: plan_steps
│  ├─ codegraph API 호출 (OpenTelemetry)
│  └─ output: chunks_collected
└─ answer_generator (Langfuse)
   ├─ input: query, chunks_count
   ├─ LLM 호출 (OpenLIT)
   └─ output: answer_length
```

## 수집되는 정보

### 검색 파이프라인 (codegraph)

```json
{
  "span": "hybrid_search",
  "query": "authentication",
  "k": 5,
  "parallel": true,
  "repo_id": "test-repo",
  "results": {
    "count": 5,
    "top_score": 0.89
  },
  "children": [
    {
      "span": "lexical_search",
      "fetch_k": 50,
      "weight": 0.25,
      "results": {"count": 50, "top_score": 8.5},
      "duration_ms": 120
    },
    {
      "span": "semantic_search",
      "fetch_k": 80,
      "weight": 0.45,
      "children": [{
        "span": "mistral_embedding",
        "model": "codestral-embed",
        "texts": 1,
        "tokens": 50,
        "batches": 1,
        "cost": 0.0001,
        "duration_ms": 380
      }],
      "results": {"count": 80, "top_score": 0.92},
      "duration_ms": 450
    },
    {
      "span": "graph_search",
      "weight": 0.15,
      "location": {"file": "src/auth.py", "line": 45},
      "current_node": {"id": "node123", "name": "login"},
      "neighbors": {"count": 25},
      "duration_ms": 90
    },
    {
      "span": "fuzzy_search",
      "weight": 0.15,
      "query_tokens": ["auth", "login"],
      "results": {"count": 30},
      "duration_ms": 80
    }
  ],
  "total_duration_ms": 850
}
```

### Agent 실행 (copilot)

```json
{
  "trace_id": "session-abc123",
  "name": "code_understanding",
  "spans": [
    {
      "name": "intent_router",
      "input": {"query": "authentication code"},
      "output": {
        "intent": "code_question",
        "plan_steps": 2
      },
      "duration_ms": 1200
    },
    {
      "name": "plan_executor",
      "input": {"plan_steps": 2, "intent": "code_question"},
      "output": {"chunks_collected": 5, "steps_executed": 2},
      "duration_ms": 850
    },
    {
      "name": "answer_generator",
      "input": {"query": "...", "chunks_count": 5},
      "output": {"answer_length": 450, "llm_used": true},
      "generations": [{
        "model": "gpt-4o-mini",
        "input_tokens": 800,
        "output_tokens": 200,
        "cost": 0.006
      }],
      "duration_ms": 1500
    }
  ],
  "total_duration_ms": 3550,
  "total_cost": 0.0061
}
```

## 확인 방법

### 1. Jaeger (OpenTelemetry)

```bash
open http://localhost:16686
```

**확인 내용**:
- Service: `semantica-codegraph` 또는 `semantica-copilot`
- Operation 선택
- 트레이스 상세 보기
  - 전체 소요 시간
  - 각 span별 시간
  - Attribute 상세 정보

### 2. Arize (Phoenix)

```bash
open https://app.arize.com
```

**확인 내용**:
- Projects → semantica 선택
- Traces 탭: 검색 트레이스
- Embeddings 탭: UMAP 시각화
- Evaluations 탭: 검색 품질

### 3. Langfuse

```bash
open https://cloud.langfuse.com
```

**확인 내용**:
- Traces: Agent 실행 이력
- Sessions: 사용자별 대화
- Generations: LLM 호출 상세
- Dashboard: 비용 및 토큰 사용량

## 실전 테스트

### codegraph API

```bash
# API 서버 시작
cd /Users/songmin/Documents/code-jo/semantica/semantica-codegraph
uvicorn apps.api.main:app --reload

# 검색 요청
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "codegraph-test",
    "query": "authentication",
    "k": 5
  }'

# Jaeger에서 확인
# → hybrid_search span
# → lexical_search, semantic_search, graph_search, fuzzy_search
# → 각 span의 attribute 확인
```

### copilot Agent

```bash
# copilot 실행
cd /Users/songmin/Documents/code-jo/semantica/semantica-copilot
semantica-copilot

# 질문 입력
> authentication 코드가 어디 있어?

# Langfuse에서 확인
# → intent_router span
# → plan_executor span
# → answer_generator span
# → LLM 호출 상세
```

## 비용 및 성능

### 오버헤드

| 도구 | 오버헤드 | 완화 |
|------|----------|------|
| OpenTelemetry | <5% | 샘플링 10% |
| Phoenix | <1% | 비동기 |
| Langfuse | <3% | 비동기 |
| OpenLIT | <2% | 배치 |

**총 오버헤드**: <10% (개발), <2% (프로덕션 샘플링 시)

### 데이터 보존

| 도구 | 보존 기간 |
|------|----------|
| Jaeger (로컬) | 메모리 (재시작 시 삭제) |
| Arize | 30일 (무료) / 무제한 (Pro) |
| Langfuse | 14일 (무료) / 무제한 (Pro) |

## 환경변수 요약

### semantica-codegraph/.env

```bash
# OpenTelemetry + Phoenix
OTEL_ENABLED=true
OTEL_ENDPOINT=http://localhost:4317
OTEL_SAMPLE_RATE=1.0
OTEL_SERVICE_NAME=semantica-codegraph

PHOENIX_ENABLED=true
PHOENIX_USE_CLOUD=true
ARIZE_API_KEY=ak-xxx

ENVIRONMENT=development
```

### semantica-copilot/.env

```bash
# OpenTelemetry
OTEL_ENABLED=true
OTEL_ENDPOINT=http://localhost:4317
OTEL_SAMPLE_RATE=1.0
OTEL_SERVICE_NAME=semantica-copilot

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com

ENVIRONMENT=development
```

## 다음 단계

### 프로덕션 준비

```bash
# .env.production
OTEL_SAMPLE_RATE=0.1  # 10% 샘플링
ENVIRONMENT=production
```

### 대시보드 구축

- Grafana 연결
- 알림 설정
- SLO 정의

### 평가 시스템

- Langfuse Evaluations
- Phoenix Experiments
- A/B 테스팅

## 참고 문서

- `docs/OTEL_QUICKSTART.md`: 빠른 시작
- `docs/OBSERVABILITY_STACK.md`: 전체 아키텍처
- `docs/SPAN_CONFIGURATION.md`: Span 설정
- `docs/SPAN_RECOMMENDATIONS.md`: 추천 설정

