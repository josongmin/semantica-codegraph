# Observability 빠른 시작 (3분)

## 1. 의존성 설치

### semantica-codegraph

```bash
cd /Users/songmin/Documents/code-jo/semantica/semantica-codegraph
pip install -e ".[otel]"
```

### semantica-copilot

```bash
cd /Users/songmin/Documents/code-jo/semantica/semantica-copilot
pip install -e ".[otel]"
```

## 2. 백엔드 실행

### 옵션 A: Jaeger만 (최소 구성)

```bash
cd /Users/songmin/Documents/code-jo/semantica/semantica-codegraph
docker-compose -f docker-compose.otel.yml up -d jaeger
```

### 옵션 B: 전체 스택 (Jaeger + Langfuse Self-hosted)

```bash
cd /Users/songmin/Documents/code-jo/semantica/semantica-codegraph
docker-compose -f docker-compose.otel.yml up -d
```

**실행되는 서비스**:
- Jaeger: http://localhost:16686
- Langfuse: http://localhost:3000
- Phoenix: http://localhost:6006 (API 서버 시작 시)

## 3. 환경변수 설정

### semantica-codegraph/.env

```bash
# 기존 .env 파일에 추가
cat >> .env << EOF

# === Observability ===
OTEL_ENABLED=true
OTEL_ENDPOINT=http://localhost:4317
OTEL_SAMPLE_RATE=1.0
OTEL_SERVICE_NAME=semantica-codegraph

PHOENIX_ENABLED=true
PHOENIX_PORT=6006

ENVIRONMENT=development
EOF
```

### semantica-copilot/.env

```bash
# 기존 .env 파일에 추가
cat >> .env << EOF

# === Observability ===
OTEL_ENABLED=true
OTEL_ENDPOINT=http://localhost:4317
OTEL_SAMPLE_RATE=1.0
OTEL_SERVICE_NAME=semantica-copilot

# Langfuse (선택)
# Self-hosted 사용 시:
# LANGFUSE_PUBLIC_KEY=pk-lf-xxx  # UI에서 생성
# LANGFUSE_SECRET_KEY=sk-lf-xxx  # UI에서 생성
# LANGFUSE_HOST=http://localhost:3000
#
# Cloud 사용 시:
# LANGFUSE_PUBLIC_KEY=pk-xxx
# LANGFUSE_SECRET_KEY=sk-xxx
# LANGFUSE_HOST=https://cloud.langfuse.com

ENVIRONMENT=development
EOF
```

## 4. 서버 시작

### codegraph API

```bash
cd /Users/songmin/Documents/code-jo/semantica/semantica-codegraph
uvicorn apps.api.main:app --reload
```

**로그 확인**:
```
INFO - OpenTelemetry initialized: service=semantica-codegraph
INFO - FastAPI instrumentation enabled
INFO - OpenLIT initialized
INFO - Phoenix UI: http://localhost:6006
```

### copilot (별도 터미널)

```bash
cd /Users/songmin/Documents/code-jo/semantica/semantica-copilot
semantica-copilot
```

## 5. 테스트

### 검색 요청

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "codegraph-test",
    "query": "authentication",
    "k": 5
  }'
```

## 6. 확인

### Jaeger UI

```bash
open http://localhost:16686
```

**확인 내용**:
1. Service 드롭다운에서 `semantica-codegraph` 선택
2. Operation: `POST /api/search` 선택
3. Find Traces 클릭
4. 트레이스 상세 보기
   - 전체 소요 시간
   - 각 단계별 시간 (lexical, semantic, graph)
   - LLM API 호출 (OpenLIT)

### Phoenix UI

```bash
open http://localhost:6006
```

**확인 내용**:
1. Traces 탭: 검색 트레이스
2. Embeddings 탭: 임베딩 시각화 (UMAP)
3. Evaluations 탭: 검색 품질 메트릭

## 확인 가능한 정보

### OpenTelemetry (Jaeger)

```
POST /api/search (850ms)
├─ hybrid_search (780ms)
│  ├─ lexical_search (120ms)
│  ├─ semantic_search (450ms)
│  │  └─ Mistral API (380ms, 50 tokens)
│  ├─ graph_search (90ms)
│  └─ fuzzy_search (80ms)
└─ reranking (50ms)
```

### Phoenix

```
Query: "authentication"
├─ Retrieved: 20 chunks
├─ Precision@5: 0.8
├─ Recall@5: 0.6
├─ MRR: 0.85
└─ Top scores: [0.89, 0.85, 0.82, 0.78, 0.75]
```

### OpenLIT (Jaeger 통합)

```
LLM API 호출:
├─ Mistral codestral-embed
│  - Tokens: 50
│  - Latency: 380ms
│  - Cost: $0.0001
└─ OpenAI gpt-4o-mini (리랭킹)
   - Tokens: 1200
   - Latency: 850ms
   - Cost: $0.006
```

## 비활성화

### 임시 비활성화

```bash
# 환경변수만 변경
export OTEL_ENABLED=false
export PHOENIX_ENABLED=false
```

### 완전 비활성화

```bash
# .env 파일에서 제거 또는 false로 설정
OTEL_ENABLED=false
PHOENIX_ENABLED=false
```

## 문제 해결

### "OpenTelemetry initialized" 안 보임

```bash
# 의존성 확인
pip list | grep opentelemetry

# 미설치 시
pip install -e ".[otel]"
```

### Jaeger UI에 트레이스 없음

```bash
# Jaeger 실행 확인
docker ps | grep jaeger

# 환경변수 확인
env | grep OTEL
```

### Phoenix UI 안 열림

```bash
# 환경변수 확인
echo $PHOENIX_ENABLED  # true

# 포트 충돌 확인
lsof -i :6006
```

## 다음 단계

### Langfuse 설정 (copilot)

#### 옵션 A: Self-hosted (추천 - 데이터 로컬)

```bash
# 1. Langfuse 실행 (이미 실행했으면 스킵)
docker-compose -f docker-compose.otel.yml up -d langfuse-server

# 2. UI 접속
open http://localhost:3000

# 3. 계정 생성 및 로그인

# 4. Settings → API Keys에서 키 생성

# 5. .env에 추가
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=http://localhost:3000
```

**상세 가이드**: `docs/LANGFUSE_SELFHOSTED.md`

#### 옵션 B: Cloud (쉬움 - 관리 편리)

```bash
# 1. https://cloud.langfuse.com 회원가입
# 2. 프로젝트 생성
# 3. API 키 복사
# 4. .env에 추가
LANGFUSE_PUBLIC_KEY=pk-xxx
LANGFUSE_SECRET_KEY=sk-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 커스텀 추적 추가

```python
# Phoenix: 검색 품질 평가
from src.core.phoenix_integration import get_phoenix

phoenix = get_phoenix()
phoenix.log_retrieval(
    query="...",
    documents=[...],
    scores=[...],
)

# Langfuse: Agent 세션 추적
from semantica_copilot.observability import get_langfuse

langfuse = get_langfuse()
trace = langfuse.trace("code_understanding")
```

## 참고

- 상세 가이드: `docs/OBSERVABILITY_STACK.md`
- OpenTelemetry: `docs/OTEL_QUICKSTART.md`
- Phoenix: https://docs.arize.com/phoenix
- Langfuse: https://langfuse.com/docs

