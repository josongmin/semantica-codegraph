# ADR 0006 – Hybrid Retrieval (BM25 + Vector + Graph) 채택

## Status
Accepted

## Context

코드 검색은 단일 기법만으로는 품질이 부족하다:

- **BM25**: 키워드 매칭에 강하지만 의미 이해 부족
- **Vector**: 의미 유사도에 강하지만 정확한 키워드 매칭 약함
- **Graph**: 관계 기반 탐색에 강하지만 초기 진입점 필요

Sourcegraph, Cody, Serena, RepoFusion 사례에서도 hybrid가 사실상 필수라는 결론.

## Decision

다음 3개를 weighted-fusion 방식으로 결합한다:

### 1. BM25 (Keyword Search)
- FTS5 기반 전문 검색
- 정확한 키워드 매칭

### 2. Vector Similarity
- pgvector 기반 semantic search
- 의미적 유사도 계산

### 3. CallGraph (RouteIndex) 기반 Rerank
- 관련 심볼 확장
- 호출 관계 기반 우선순위 조정

### Fusion 전략

```python
final_score = (
    w_bm25 * bm25_score +
    w_vector * vector_score +
    w_graph * graph_score
)

# Default weights
w_bm25 = 0.3
w_vector = 0.5
w_graph = 0.2
```

## Alternatives Considered

### BM25 단독
- 의미 검색 불가
- 동의어 처리 약함

### Vector 단독
- 정확한 키워드 미스
- 검색 품질 불안정

### Graph 단독
- 초기 진입점 찾기 어려움
- 단독 사용 불가

## Consequences

### Positive
- LLM에게 가장 적합한 context를 안정적으로 공급
- 검색 품질 최대화
- 다양한 쿼리 타입 지원

### Negative
- 구현 복잡도 증가
- 가중치 튜닝 필요
- 검색 시간 증가 (캐싱으로 완화)

## Notes

Hybrid retrieval은 현대 코드 검색 엔진의 표준이며, Codegraph의 핵심 경쟁력이다.


