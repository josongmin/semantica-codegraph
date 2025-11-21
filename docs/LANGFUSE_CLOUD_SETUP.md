# Langfuse Cloud 설정 가이드

## 프로젝트 정보

```
Project Name: my-lang
Project ID: cmi8whyzh0423ad0700xkqjtb
Cloud Region: EU
URL: https://cloud.langfuse.com
```

## 1. API 키 발급

### 단계

1. **로그인**
   ```
   https://cloud.langfuse.com
   ```

2. **프로젝트 선택**
   - 좌측 사이드바에서 "my-lang" 선택

3. **Settings 이동**
   - 좌측 메뉴 → Settings

4. **API Keys 탭**
   - Settings → API Keys

5. **새 키 생성**
   - "Create new secret key" 버튼 클릭
   - 이름: `semantica-copilot` (선택사항)
   - Create 클릭

6. **키 복사**
   - Public Key: `pk-lf-...`
   - Secret Key: `sk-lf-...`
   - ⚠️ Secret Key는 한 번만 표시됩니다!

## 2. 환경변수 설정

### semantica-copilot/.env

```bash
# Langfuse Cloud (EU Region)
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com

# OpenTelemetry
OTEL_ENABLED=true
OTEL_ENDPOINT=http://localhost:4317
OTEL_SAMPLE_RATE=1.0
OTEL_SERVICE_NAME=semantica-copilot
ENVIRONMENT=development
```

### 확인

```bash
# .env 파일 확인
cat /Users/songmin/Documents/code-jo/semantica/semantica-copilot/.env | grep LANGFUSE
```

## 3. 테스트

### copilot에서 Agent 실행

```python
from semantica_copilot.observability import init_langfuse

# 초기화 (환경변수에서 자동 로드)
langfuse = init_langfuse(enabled=True)

# Agent 실행 추적
trace = langfuse.trace(
    name="code_understanding",
    user_id="user123",
    session_id="session456",
    metadata={
        "repo": "semantica-codegraph",
        "query": "authentication code",
    },
)

# LLM 호출 추적
trace.generation(
    model="gpt-4o-mini",
    input="Explain authentication code",
    output="Authentication is...",
    usage={
        "input_tokens": 50,
        "output_tokens": 200,
        "total_tokens": 250,
    },
)

# 트레이스 종료 (자동 전송)
langfuse.flush()
```

### Langfuse UI에서 확인

```bash
open https://cloud.langfuse.com
```

**확인 내용**:
1. **Traces** 탭
   - `code_understanding` 트레이스 확인
   - 소요 시간, 메타데이터

2. **Generations** 탭
   - LLM 호출 상세
   - 토큰 사용량
   - 비용 (자동 계산)

3. **Sessions** 탭
   - 사용자별 세션
   - 대화 흐름

4. **Dashboard**
   - 일일 토큰 사용량
   - 비용 추이
   - 에러율

## 4. Agent 통합

### LangGraph Agent에서 사용

```python
# src/semantica_copilot/agent/graph.py
from semantica_copilot.observability import get_langfuse

class CodeUnderstandingAgent:
    def __init__(self):
        self.langfuse = get_langfuse()
        # ...
    
    async def run(self, query: str, user_id: str):
        # Agent 실행 추적
        trace = self.langfuse.trace(
            name="agent_run",
            user_id=user_id,
            metadata={"query": query},
        )
        
        try:
            # Intent Router
            with trace.span("intent_router") as span:
                intent = await self.router.route(query)
                span.end(output={"intent": intent})
            
            # Search Planning
            with trace.span("search_planning") as span:
                plan = await self.planner.plan(query)
                span.end(output={"plan": plan})
            
            # CodeGraph Search
            with trace.span("codegraph_search") as span:
                results = await self.codegraph.search(query)
                span.end(output={"results_count": len(results)})
            
            # LLM Generation
            response = await self.llm.generate(prompt)
            trace.generation(
                model="gpt-4o-mini",
                input=prompt,
                output=response,
                usage=self.llm.last_usage,
            )
            
            return response
            
        finally:
            self.langfuse.flush()
```

## 5. 비용 추적

### Langfuse UI - Dashboard

```
Daily Usage:
├─ Tokens: 45,230 (input: 12,450 / output: 32,780)
├─ Cost: $0.89
└─ Requests: 234

By Model:
├─ gpt-4o-mini: 40,120 tokens ($0.82)
└─ gpt-4o: 5,110 tokens ($0.07)

By User:
├─ user123: $0.45
└─ user456: $0.44
```

### 알림 설정

1. Settings → Notifications
2. Daily budget alert: $10
3. Email: your-email@example.com

## 6. 프롬프트 관리

### Prompt Template 저장

```python
# Langfuse UI에서 Prompt 생성
# Prompts → Create New Prompt

# 코드에서 사용
prompt_template = langfuse.get_prompt("code_explanation_v1")
prompt = prompt_template.compile(
    code=code_snippet,
    language="python",
)

response = llm.generate(prompt)
```

### 버전 관리

```
code_explanation_v1 (production)
├─ Version 1: "Explain this {language} code..."
├─ Version 2: "Analyze this {language} code..."  ← 현재
└─ Version 3: "Deep dive into {language}..."  (draft)
```

## 7. 평가 및 피드백

### 사용자 피드백 수집

```python
# 사용자가 답변에 피드백
trace.score(
    name="user_feedback",
    value=1.0,  # 1.0 = 좋음, 0.0 = 나쁨
    comment="Very helpful answer!",
)
```

### Langfuse UI에서 확인

```
Evaluations:
├─ user_feedback: 4.2/5 (avg)
├─ accuracy: 0.85
└─ relevance: 0.92
```

## 8. 팀 협업

### 멤버 추가

1. Settings → Team Members
2. Invite Member
3. Email 입력 → Send Invitation

### 역할

- **Owner**: 모든 권한
- **Admin**: 설정 제외 모든 권한
- **Member**: 읽기 + 트레이스 생성
- **Viewer**: 읽기 전용

## 문제 해결

### API 키 오류

```bash
# 환경변수 확인
env | grep LANGFUSE

# 키 재생성
# Langfuse UI → Settings → API Keys → Revoke & Create New
```

### 트레이스가 안 보임

```bash
# flush 확인
langfuse.flush()  # 버퍼 전송

# 로그 확인
# "Langfuse initialized" 메시지 확인
```

### 비용 계산 안 됨

```
Settings → Cost Tracking에서 모델별 가격 확인
자동 계산 안 되는 모델: 수동 설정 필요
```

## 데이터 보존

- **무료 플랜**: 14일
- **Pro 플랜**: 무제한

## 요금제

| 플랜 | 비용 | 트레이스/월 | 데이터 보존 |
|------|------|-------------|-------------|
| Free | $0 | 50,000 | 14일 |
| Pro | $99 | 무제한 | 무제한 |
| Enterprise | 문의 | 무제한 | 무제한 |

## 참고

- Langfuse Docs: https://langfuse.com/docs
- API Reference: https://langfuse.com/docs/api
- SDK: https://langfuse.com/docs/sdk/python

