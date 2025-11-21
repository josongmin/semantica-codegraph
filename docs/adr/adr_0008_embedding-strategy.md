# ADR 0008 – Embedding Dimension 전략

## Status
Accepted

## Context

다양한 embedding 모델 지원 필요성:

- OpenAI text-embedding-3-small: 1536 차원
- OpenAI text-embedding-3-large: 3072 차원
- 향후 다른 모델 추가 가능성

단일 테이블에 다양한 차원을 섞으면:
- 인덱스 효율 저하
- 비교 불가능 (차원이 다른 벡터는 유사도 계산 불가)

## Decision

다음 전략을 적용한다:

### 1. 기본 테이블: 1536 차원 고정

```sql
CREATE TABLE semantic_nodes (
    id TEXT PRIMARY KEY,
    symbol_id TEXT,
    content TEXT,
    embedding vector(1536),  -- fixed dimension
    level INTEGER,
    chunk_index INTEGER
);

CREATE INDEX idx_semantic_nodes_embedding
ON semantic_nodes
USING ivfflat (embedding vector_cosine_ops);
```

### 2. 추가 차원 필요 시: 별도 테이블

```sql
CREATE TABLE semantic_nodes_3072 (
    id TEXT PRIMARY KEY,
    symbol_id TEXT,
    content TEXT,
    embedding vector(3072),
    level INTEGER,
    chunk_index INTEGER
);
```

### 3. 검색 시: Union Merge

```python
def hybrid_search(query: str, model: str = "1536"):
    if model == "1536":
        results = search_semantic_nodes(query)
    elif model == "3072":
        results = search_semantic_nodes_3072(query)

    return merge_and_rank(results)
```

## Alternatives Considered

### 하나의 테이블에 다양한 차원 섞기
- 인덱스 생성 불가 (pgvector는 고정 차원 필요)
- 검색 효율 저하

### Dynamic dimension per row
- PostgreSQL vector 타입이 지원하지 않음
- 성능 문제

### 모든 벡터를 최대 차원으로 패딩
- 저장 공간 낭비
- 검색 성능 저하

## Consequences

### Positive
- 확장성과 일관성 유지
- 각 모델별 최적 인덱스 구성 가능
- 모델 전환 시 기존 데이터 유지

### Negative
- 테이블 관리 복잡도 증가
- 여러 모델 동시 사용 시 저장 공간 증가

## Notes

현재는 text-embedding-3-small (1536) 만 사용하며, 필요 시 확장 가능한 구조로 설계되었다.
