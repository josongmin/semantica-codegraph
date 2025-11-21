# OpenTelemetry 배포 가이드

## Docker Compose 설정

### 전체 스택

```yaml
# docker-compose.otel.yml
version: '3.8'

services:
  # Jaeger (All-in-One)
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: jaeger
    ports:
      - "16686:16686"  # Jaeger UI
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP
    environment:
      - COLLECTOR_OTLP_ENABLED=true
      - SPAN_STORAGE_TYPE=badger
      - BADGER_EPHEMERAL=false
      - BADGER_DIRECTORY_VALUE=/badger/data
      - BADGER_DIRECTORY_KEY=/badger/key
    volumes:
      - jaeger-data:/badger
    networks:
      - otel-network

  # Prometheus (메트릭 저장)
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - otel-network

  # Grafana (대시보드)
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - otel-network

  # OpenTelemetry Collector (선택사항)
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    container_name: otel-collector
    command: ["--config=/etc/otel-collector-config.yml"]
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
      - "8889:8889"   # Prometheus metrics
    volumes:
      - ./otel-collector-config.yml:/etc/otel-collector-config.yml
    networks:
      - otel-network

volumes:
  jaeger-data:
  prometheus-data:
  grafana-data:

networks:
  otel-network:
    driver: bridge
```

### Prometheus 설정

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'otel-collector'
    static_configs:
      - targets: ['otel-collector:8889']

  - job_name: 'semantica-api'
    static_configs:
      - targets: ['host.docker.internal:8000']
```

### OTEL Collector 설정

```yaml
# otel-collector-config.yml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024

  memory_limiter:
    check_interval: 1s
    limit_mib: 512

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true

  prometheus:
    endpoint: "0.0.0.0:8889"

  logging:
    loglevel: info

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [jaeger, logging]

    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus, logging]
```

## Grafana 대시보드

### 데이터소스 설정

```yaml
# grafana/datasources/datasources.yml
apiVersion: 1

datasources:
  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    isDefault: false

  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

### 대시보드 JSON (예시)

```json
{
  "dashboard": {
    "title": "Semantica Codegraph - Overview",
    "panels": [
      {
        "title": "검색 요청 수 (시간당)",
        "targets": [
          {
            "expr": "rate(search_requests_total[1h])"
          }
        ]
      },
      {
        "title": "검색 P95 레이턴시",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, search_latency_bucket)"
          }
        ]
      },
      {
        "title": "임베딩 API 호출 수",
        "targets": [
          {
            "expr": "sum(embedding_api_calls_total) by (model)"
          }
        ]
      }
    ]
  }
}
```

## 실행 순서

### 1. 백엔드 시작

```bash
# Jaeger만
docker-compose -f docker-compose.otel.yml up -d jaeger

# 전체 스택
docker-compose -f docker-compose.otel.yml up -d
```

### 2. 설정 확인

```bash
# Jaeger UI
open http://localhost:16686

# Prometheus
open http://localhost:9090

# Grafana
open http://localhost:3000
# 로그인: admin / admin
```

### 3. API 서버 시작

```bash
# 환경변수 설정
export OTEL_ENABLED=true
export OTEL_ENDPOINT=http://localhost:4317
export OTEL_SAMPLE_RATE=1.0

# API 실행
uvicorn apps.api.main:app --reload
```

### 4. 트래픽 생성

```bash
# 검색 요청
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication", "k": 5}'

# 인덱싱 요청
curl -X POST http://localhost:8000/api/repositories/index \
  -H "Content-Type: application/json" \
  -d '{
    "repo_path": "/path/to/repo",
    "repo_id": "test-repo"
  }'
```

### 5. 트레이스 확인

```
1. Jaeger UI (http://localhost:16686)
2. Service 선택: "semantica-codegraph-api"
3. Operation 선택: "POST /api/search"
4. Find Traces 클릭
5. 트레이스 상세 보기
```

## Kubernetes 배포

### Deployment

```yaml
# k8s/api-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: semantica-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: semantica-api
  template:
    metadata:
      labels:
        app: semantica-api
    spec:
      containers:
      - name: api
        image: semantica-codegraph:latest
        env:
        - name: OTEL_ENABLED
          value: "true"
        - name: OTEL_ENDPOINT
          value: "http://otel-collector:4317"
        - name: OTEL_SAMPLE_RATE
          value: "0.1"
        - name: ENVIRONMENT
          value: "production"
        ports:
        - containerPort: 8000
```

### Service

```yaml
# k8s/otel-collector-service.yml
apiVersion: v1
kind: Service
metadata:
  name: otel-collector
spec:
  selector:
    app: otel-collector
  ports:
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
  - name: otlp-http
    port: 4318
    targetPort: 4318
```

## 프로덕션 체크리스트

### 성능

- [ ] 샘플링 설정 (1-10%)
- [ ] Batch 크기 최적화
- [ ] 메모리 제한 설정
- [ ] 불필요한 attribute 제거

### 보안

- [ ] OTLP endpoint TLS 활성화
- [ ] 민감 정보 필터링 (query 일부만)
- [ ] 접근 제어 (Jaeger UI)

### 모니터링

- [ ] Collector 메트릭 모니터링
- [ ] Drop rate 확인
- [ ] Export 실패 알림

### 비용

- [ ] 스토리지 크기 제한
- [ ] 데이터 보존 기간 설정
- [ ] 샘플링으로 비용 절감

## 문제 해결

### Trace가 보이지 않음

```bash
# 1. Collector 로그 확인
docker logs otel-collector

# 2. Exporter 테스트
curl http://localhost:4317

# 3. 앱 로그 확인
# "OpenTelemetry initialized" 메시지 확인
```

### 메트릭이 수집되지 않음

```bash
# Prometheus target 확인
open http://localhost:9090/targets

# OTEL Collector 메트릭 확인
curl http://localhost:8889/metrics
```

### 높은 오버헤드

```bash
# 샘플링 낮추기
export OTEL_SAMPLE_RATE=0.01  # 1%

# Batch 크기 증가
# otel-collector-config.yml에서
# processors.batch.send_batch_size 증가
```

## 단계별 롤아웃

### Phase 1: 개발 환경

```
- Jaeger local 실행
- 샘플링 100%
- 개발자 피드백 수집
```

### Phase 2: 스테이징

```
- Collector 도입
- 샘플링 50%
- 대시보드 구축
```

### Phase 3: 프로덕션 (Canary)

```
- 10% 트래픽만 계측
- 샘플링 10%
- 알림 설정
```

### Phase 4: 전체 배포

```
- 100% 트래픽 계측
- 샘플링 최적화 (1-10%)
- 자동화된 모니터링
```
