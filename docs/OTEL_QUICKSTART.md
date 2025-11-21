# OpenTelemetry 빠른 시작 가이드

## 설치

```bash
# OpenTelemetry 의존성 설치
pip install -e ".[otel]"

# 또는 uv 사용
uv pip install -e ".[otel]"
```

## 1. Jaeger 실행 (백엔드)

```bash
# Docker Compose로 Jaeger 시작
docker-compose -f docker-compose.otel.yml up -d jaeger

# Jaeger UI 확인
open http://localhost:16686
```

## 2. 환경변수 설정

`.env` 파일에 추가:

```bash
# OpenTelemetry 활성화
OTEL_ENABLED=true

# Jaeger endpoint
OTEL_ENDPOINT=http://localhost:4317

# 샘플링 비율 (개발: 1.0, 프로덕션: 0.1)
OTEL_SAMPLE_RATE=1.0

# 서비스 이름
OTEL_SERVICE_NAME=semantica-codegraph

# 환경
ENVIRONMENT=development
```

## 3. API 서버 시작

```bash
# API 서버 실행
uvicorn apps.api.main:app --reload

# 또는
semantica-api
```

## 4. 테스트

### 검색 API 호출

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "codegraph-test",
    "query": "authentication",
    "k": 5
  }'
```

### Jaeger UI에서 트레이스 확인

1. http://localhost:16686 접속
2. Service: `semantica-codegraph` 선택
3. Operation: `POST /api/search` 선택
4. Find Traces 클릭
5. 트레이스 상세 보기
   - 전체 검색 시간
   - 각 단계별 시간 (lexical, semantic, graph, fuzzy)
   - LLM API 호출 (OpenLIT)

## 확인 가능한 정보

### 1. 검색 파이프라인

```
POST /api/search
├─ hybrid_search (전체)
│  ├─ lexical_search
│  ├─ semantic_search  
│  ├─ graph_search
│  └─ fuzzy_search
└─ reranking (선택)
```

### 2. LLM 호출 추적 (OpenLIT)

- 임베딩 API 호출
  - 모델: codestral-embed
  - 토큰 수
  - 레이턴시
  - 비용 (자동 계산)
  
- 리랭킹 LLM 호출 (설정 시)
  - 모델: gpt-4o-mini 등
  - 토큰 수
  - 비용

### 3. 메트릭

Jaeger UI에서 확인 가능:
- P50, P95, P99 레이턴시
- 에러율
- 호출 빈도

## 비활성화

```bash
# .env 파일에서
OTEL_ENABLED=false

# 또는 환경변수 제거
unset OTEL_ENABLED
```

## 문제 해결

### 트레이스가 보이지 않음

```bash
# 1. Jaeger 실행 확인
docker ps | grep jaeger

# 2. API 서버 로그 확인
# "OpenTelemetry initialized" 메시지 확인

# 3. 엔드포인트 확인
echo $OTEL_ENDPOINT
# → http://localhost:4317
```

### OpenLIT 에러

```bash
# OpenLIT 설치 확인
pip list | grep openlit

# 미설치 시
pip install openlit
```

## 다음 단계

- `docs/opentelemetry-integration.md`: 전체 아키텍처
- `docs/opentelemetry-implementation-examples.md`: 코드 예시
- `docs/opentelemetry-deployment.md`: 프로덕션 배포
- `docs/opentelemetry-recommendation.md`: 도입 가이드

## 샘플 대시보드 쿼리

### 검색 레이턴시 P95

```
Service: semantica-codegraph
Operation: hybrid_search
Duration: P95
```

### LLM 비용 (일일)

OpenLIT 대시보드에서 확인:
- http://localhost:16686 → OpenLIT Metrics
- 토큰 사용량
- 예상 비용

## FAQ

**Q: 개발 중에도 켜놓을까요?**
A: 샘플링 1.0으로 설정하면 오버헤드가 거의 없습니다. 
   병목 확인에 유용하므로 켜놓는 것을 추천합니다.

**Q: 프로덕션에서는?**
A: 샘플링을 0.1 (10%)로 낮추고, 
   Prometheus + Grafana 추가를 권장합니다.

**Q: 비용이 많이 들까요?**
A: 개발 환경에서는 무료입니다 (Jaeger local).
   프로덕션에서도 self-hosted 시 서버 비용만 발생합니다.

