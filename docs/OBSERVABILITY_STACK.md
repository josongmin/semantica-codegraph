# Observability 스택 통합 가이드

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Semantica 프로젝트                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  semantica-codegraph (RAG 엔진)                                 │
│  ├─ OpenTelemetry → Jaeger (범용 트레이싱)                     │
│  ├─ OpenLIT → Jaeger (LLM 호출 추적)                           │
│  └─ Phoenix → UI (검색 품질 분석)                               │
│                                                                 │
│  semantica-copilot (Agent)                                      │
│  ├─ OpenTelemetry → Jaeger (범용 트레이싱)                     │
│  ├─ OpenLIT → Jaeger (LLM 호출 추적)                           │
│  └─ Langfuse → Cloud/Self-hosted (Agent 세션 추적)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 도구별 역할

### 1. OpenTelemetry (범용 인프라 추적)

**대상**: 두 프로젝트 모두

**추적 내용**:
- HTTP 요청/응답
- 데이터베이스 쿼리 (PostgreSQL)
- 외부 API 호출 (Qdrant, MeiliSearch)
- 마이크로서비스 간 통신

**확인**: http://localhost:16686 (Jaeger UI)

### 2. Phoenix (RAG 품질 추적)

**대상**: semantica-codegraph

**추적 내용**:
- 검색 품질 메트릭 (Precision@K, Recall, MRR)
- Embeddings 시각화 (UMAP)
- Retrieval 성능 분석
- Irrelevant chunks 식별

**확인**: http://localhost:6006 (Phoenix UI)

**주요 기능**:
```python
# 검색 결과 로깅
phoenix.log_retrieval(
    query="authentication",
    documents=[...],
    scores=[0.89, 0.85, ...],
)

# 임베딩 시각화
phoenix.log_embeddings(
    embeddings=chunk_embeddings,
    documents=chunk_metadata,
)
```

### 3. Langfuse (LLM 앱 추적)

**대상**: semantica-copilot

**추적 내용**:
- Agent 대화 세션
- LLM 호출 (토큰, 비용)
- Prompt 버전 관리
- 평가 및 피드백

**확인**: https://cloud.langfuse.com (또는 self-hosted)

**주요 기능**:
```python
# Agent 세션 추적
with langfuse.trace("code_understanding", user_id="user123"):
    # 각 단계 추적
    with langfuse.span("intent_router"):
        intent = router.route(query)
    
    # LLM 호출 추적
    langfuse.generation(
        model="gpt-4o-mini",
        input=prompt,
        output=response,
        usage={"input_tokens": 500, "output_tokens": 200},
    )
```

### 4. OpenLIT (LLM 호출 자동 추적)

**대상**: 두 프로젝트 모두

**추적 내용**:
- LLM API 호출 (자동)
- 토큰 사용량
- 비용 자동 계산
- 레이턴시

**확인**: Jaeger UI에 통합

## 설치

### semantica-codegraph

```bash
# 의존성 설치
pip install -e ".[otel]"

# 환경변수 설정
cat >> .env << EOF
# OpenTelemetry
OTEL_ENABLED=true
OTEL_ENDPOINT=http://localhost:4317
OTEL_SAMPLE_RATE=1.0
OTEL_SERVICE_NAME=semantica-codegraph

# Phoenix
PHOENIX_ENABLED=true
PHOENIX_PORT=6006

ENVIRONMENT=development
EOF
```

### semantica-copilot

```bash
# 의존성 설치
pip install -e ".[otel]"

# 환경변수 설정
cat >> .env << EOF
# OpenTelemetry
OTEL_ENABLED=true
OTEL_ENDPOINT=http://localhost:4317
OTEL_SAMPLE_RATE=1.0
OTEL_SERVICE_NAME=semantica-copilot

# Langfuse (선택)
LANGFUSE_PUBLIC_KEY=pk-xxx
LANGFUSE_SECRET_KEY=sk-xxx
LANGFUSE_HOST=https://cloud.langfuse.com

ENVIRONMENT=development
EOF
```

## 백엔드 실행

### Jaeger (공통)

```bash
cd semantica-codegraph
docker-compose -f docker-compose.otel.yml up -d jaeger

# 확인
open http://localhost:16686
```

### Phoenix (codegraph)

```bash
# API 서버 시작 시 자동 실행 (PHOENIX_ENABLED=true)
uvicorn apps.api.main:app --reload

# 확인
open http://localhost:6006
```

### Langfuse (copilot)

**SaaS 사용 (추천)**:
1. https://cloud.langfuse.com 회원가입
2. 프로젝트 생성
3. API 키 복사
4. `.env`에 설정

**Self-hosted**:
```bash
docker run -p 3000:3000 langfuse/langfuse
open http://localhost:3000
```

## 사용 예시

### 1. 전체 플로우 추적

**사용자 요청**:
```
"authentication 코드가 어디 있어?"
```

**추적 흐름**:
```
[Langfuse] Copilot Agent Session
├─ [Span] intent_router → "code_search"
├─ [Span] search_plan
│  └─ [LLM] gpt-4o-mini (토큰: 500)
├─ [Span] codegraph_search
│  ├─ [OTEL] HTTP POST /api/search
│  └─ [Jaeger] semantica-codegraph
│     ├─ [Span] hybrid_search
│     │  ├─ [Span] lexical_search
│     │  ├─ [Span] semantic_search
│     │  │  └─ [OpenLIT] Mistral API (토큰: 50)
│     │  └─ [Span] graph_search
│     └─ [Phoenix] Retrieval Quality
│        - Precision@5: 0.8
│        - MRR: 0.85
└─ [Span] generate_answer
   └─ [LLM] gpt-4o-mini (토큰: 1200)
```

### 2. 대시보드에서 확인

**Jaeger** (http://localhost:16686):
```
Service: semantica-copilot
├─ Trace: code_understanding (2.3s)
   └─ POST /api/search → semantica-codegraph (0.8s)
      └─ hybrid_search (0.7s)
```

**Phoenix** (http://localhost:6006):
```
Query: "authentication"
├─ Retrieved: 20 chunks
├─ Relevant: 15 (75%)
├─ MRR: 0.85
└─ Top irrelevant:
   - "UI components" (score: 0.72)
   - "database schema" (score: 0.68)
```

**Langfuse** (cloud):
```
Session: user123
├─ Total tokens: 2,250
├─ Total cost: $0.08
└─ Traces:
   - code_understanding (2.3s)
   - code_explain (1.5s)
```

## 비용

| 도구 | 비용 | 비고 |
|------|------|------|
| OpenTelemetry | 무료 | Jaeger self-hosted |
| Phoenix | 무료 | 완전 오픈소스 |
| OpenLIT | 무료 | 오픈소스 |
| Langfuse | 무료 (제한) / $99/월 | Cloud 또는 self-hosted |

**총 비용**: $0 (Langfuse 무료 티어 사용 시)

## 성능 오버헤드

| 도구 | 오버헤드 | 완화 방법 |
|------|----------|----------|
| OpenTelemetry | <5% | 샘플링 (10%) |
| Phoenix | <1% | 비동기 로깅 |
| OpenLIT | <2% | 자동 배치 |
| Langfuse | <3% | 비동기 전송 |

**개발 환경**: 샘플링 100% (모든 요청 추적)
**프로덕션**: 샘플링 10% (비용 절감)

## 문제 해결

### Jaeger UI에 트레이스가 안 보임

```bash
# 1. Jaeger 실행 확인
docker ps | grep jaeger

# 2. 환경변수 확인
echo $OTEL_ENABLED
echo $OTEL_ENDPOINT

# 3. 로그 확인
# "OpenTelemetry initialized" 메시지 확인
```

### Phoenix UI가 안 열림

```bash
# 환경변수 확인
echo $PHOENIX_ENABLED  # true

# API 서버 로그 확인
# "Phoenix UI: http://localhost:6006" 메시지 확인
```

### Langfuse 연결 실패

```bash
# API 키 확인
echo $LANGFUSE_PUBLIC_KEY
echo $LANGFUSE_SECRET_KEY

# 로그 확인
# "Langfuse initialized" 메시지 확인
```

## 다음 단계

### Phase 1 (완료) ✅
- [x] OpenTelemetry 기본 설정
- [x] Phoenix 통합
- [x] Langfuse 통합
- [x] 검색 파이프라인 추적

### Phase 2 (선택)
- [ ] Grafana 대시보드
- [ ] 커스텀 메트릭
- [ ] 프로덕션 최적화
- [ ] 알림 설정

## 참고 문서

- `docs/OTEL_QUICKSTART.md`: OpenTelemetry 빠른 시작
- `docs/opentelemetry-integration.md`: OpenTelemetry 상세
- Phoenix: https://docs.arize.com/phoenix
- Langfuse: https://langfuse.com/docs

