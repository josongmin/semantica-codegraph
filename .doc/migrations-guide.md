# Database Migrations

## 스키마 초기화

```bash
# Docker로 PostgreSQL 시작
docker-compose up -d postgres

# 스키마 적용
psql -h localhost -U semantica -d semantica_codegraph < migrations/001_init_schema.sql
```

## 환경변수

`.env`:
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=semantica
POSTGRES_PASSWORD=semantica
POSTGRES_DB=semantica_codegraph
```

## 테이블 구조

### 1. repo_metadata
- 저장소 메타데이터
- 인덱싱 상태 추적
- 통계 정보

### 2. code_nodes
- 코드 그래프 노드
- 심볼 정보 (Class, Function, Method 등)

### 3. code_edges
- 코드 그래프 엣지
- 관계 정보 (defines, calls, inherits 등)

### 4. code_chunks
- RAG/검색용 청크
- Zoekt 매핑용 위치 인덱스

### 5. embeddings
- pgvector 임베딩
- 의미론적 검색용

### 6. code_files (선택)
- 파일 메타데이터
- 증분 인덱싱용

## 연결 문자열 형식

```python
connection_string = "host=localhost port=5432 dbname=semantica_codegraph user=semantica password=semantica"
```

또는

```python
from src.core.config import Config

config = Config.from_env()
connection_string = (
    f"host={config.postgres_host} "
    f"port={config.postgres_port} "
    f"dbname={config.postgres_db} "
    f"user={config.postgres_user} "
    f"password={config.postgres_password}"
)
```

