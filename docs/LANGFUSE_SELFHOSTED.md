# Langfuse Self-hosted 가이드

## 빠른 시작

### 1. Langfuse 실행

```bash
cd /Users/songmin/Documents/code-jo/semantica/semantica-codegraph
docker-compose -f docker-compose.otel.yml up -d langfuse-server langfuse-db
```

### 2. UI 접속

```bash
open http://localhost:3000
```

### 3. 초기 설정

**첫 실행 시**:
1. http://localhost:3000 접속
2. "Sign up" 클릭
3. 계정 생성:
   - Email: admin@example.com
   - Password: (원하는 비밀번호)
4. 로그인 후 프로젝트 생성

### 4. API 키 발급

1. 로그인 → Settings → API Keys
2. "Create new secret key" 클릭
3. Public Key & Secret Key 복사
4. `.env` 파일에 추가

### 5. 환경변수 설정

**semantica-copilot/.env**:

```bash
# Langfuse (Self-hosted)
LANGFUSE_PUBLIC_KEY=pk-lf-xxx  # UI에서 복사
LANGFUSE_SECRET_KEY=sk-lf-xxx  # UI에서 복사
LANGFUSE_HOST=http://localhost:3000
```

## 보안 설정 (중요!)

### NEXTAUTH_SECRET & SALT 변경

**프로덕션 배포 전 필수!**

```bash
# 랜덤 문자열 생성
openssl rand -base64 32  # NEXTAUTH_SECRET용
openssl rand -base64 32  # SALT용
```

`docker-compose.otel.yml` 수정:

```yaml
langfuse-server:
  environment:
    - NEXTAUTH_SECRET=<생성된_문자열_1>
    - SALT=<생성된_문자열_2>
```

## 전체 스택 실행

### Jaeger + Phoenix + Langfuse 모두 실행

```bash
docker-compose -f docker-compose.otel.yml up -d
```

**실행되는 서비스**:
- Jaeger: http://localhost:16686
- Langfuse: http://localhost:3000
- langfuse-db (내부 PostgreSQL)

**Phoenix**는 API 서버 시작 시 자동 실행:
- Phoenix: http://localhost:6006

## 테스트

### copilot에서 Langfuse 사용

```python
# semantica-copilot에서
from semantica_copilot.observability import init_langfuse

# 초기화
langfuse = init_langfuse(
    enabled=True,
    # .env에서 자동 로드됨
)

# Agent 실행 추적
trace = langfuse.trace(
    name="code_understanding",
    user_id="user123",
    metadata={"repo": "semantica-codegraph"},
)

# LLM 호출 추적
with trace.span("llm_generation"):
    response = llm.generate(prompt)
    trace.generation(
        model="gpt-4o-mini",
        input=prompt,
        output=response,
        usage={"input_tokens": 500, "output_tokens": 200},
    )
```

### Langfuse UI에서 확인

1. http://localhost:3000 접속
2. Traces 탭:
   - 모든 Agent 실행 이력
   - 각 실행의 소요 시간
   - LLM 호출 상세
3. Sessions 탭:
   - 사용자별 세션
   - 대화 흐름
4. Generations 탭:
   - 모든 LLM 호출
   - 토큰 사용량
   - 비용 (자동 계산)

## Cloud vs Self-hosted 비교

| 항목 | Cloud | Self-hosted |
|------|-------|-------------|
| 설정 | 즉시 사용 | Docker 필요 |
| 비용 | 무료/유료 | 무료 (인프라 제외) |
| 데이터 | Langfuse 서버 | 로컬 서버 |
| 유지보수 | 자동 | 직접 관리 |
| 버전 업데이트 | 자동 | 수동 (이미지 업데이트) |
| 팀 협업 | 쉬움 | 네트워크 설정 필요 |

**권장**:
- 개발: Self-hosted (데이터 로컬 보관)
- 프로덕션: Cloud (관리 편의성)

## 데이터 백업

```bash
# PostgreSQL 백업
docker exec semantica-langfuse-db pg_dump \
  -U langfuse langfuse > langfuse_backup.sql

# 복원
docker exec -i semantica-langfuse-db psql \
  -U langfuse langfuse < langfuse_backup.sql
```

## 업데이트

```bash
# 최신 이미지 다운로드
docker-compose -f docker-compose.otel.yml pull langfuse-server

# 재시작
docker-compose -f docker-compose.otel.yml up -d langfuse-server
```

## 문제 해결

### UI가 안 열림

```bash
# 컨테이너 상태 확인
docker ps | grep langfuse

# 로그 확인
docker logs semantica-langfuse
docker logs semantica-langfuse-db

# 재시작
docker-compose -f docker-compose.otel.yml restart langfuse-server
```

### 데이터베이스 연결 실패

```bash
# DB 헬스체크
docker exec semantica-langfuse-db pg_isready -U langfuse

# DB 초기화 (데이터 삭제!)
docker-compose -f docker-compose.otel.yml down -v
docker-compose -f docker-compose.otel.yml up -d langfuse-server langfuse-db
```

### API 키가 작동 안 함

1. Langfuse UI에서 키 재생성
2. `.env` 파일 업데이트
3. copilot 재시작

## 포트 변경 (선택)

Grafana와 포트 충돌 시:

```yaml
# docker-compose.otel.yml
langfuse-server:
  ports:
    - "3001:3000"  # 3001로 변경

# .env
LANGFUSE_HOST=http://localhost:3001
```

## 프로덕션 배포

### 권장 설정

```yaml
langfuse-server:
  environment:
    - NODE_ENV=production
    - DATABASE_URL=postgresql://user:pass@prod-db:5432/langfuse
    - NEXTAUTH_URL=https://langfuse.yourdomain.com
    - NEXTAUTH_SECRET=<strong-random-secret>
    - SALT=<strong-random-salt>
    - TELEMETRY_ENABLED=false
    - LANGFUSE_CSP_ENFORCE_HTTPS=true
```

### HTTPS 설정 (Nginx)

```nginx
server {
    listen 443 ssl;
    server_name langfuse.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 참고

- 공식 문서: https://langfuse.com/docs/deployment/self-host
- GitHub: https://github.com/langfuse/langfuse
- Docker Hub: https://hub.docker.com/r/langfuse/langfuse

