# ADR 0009 – Search API 구조 분리

## Status
Accepted

## Context

에이전트가 검색을 목적별로 호출해야 하는 이유:

- 하나의 통합 API는 플래너가 계획 세우기 어려움
- 용도별 최적화 불가
- 재현성 확보 어려움

예시 시나리오:
- "이 함수 정의 찾기" → 심볼 검색
- "이 함수를 사용하는 코드" → 그래프 탐색
- "인증 관련 코드" → 하이브리드 검색

## Decision

Search API를 목적별로 분리한다.

### API 구조

```
GET /api/repos/{owner}/{repo}/hybrid/search
  - 일반 하이브리드 검색 (BM25 + Vector + Graph)
  - 쿼리: query, top_k, filters

GET /api/repos/{owner}/{repo}/hybrid/chunks
  - 청크 레벨 검색
  - 특정 레벨(module/symbol/impl) 지정 가능

GET /api/repos/{owner}/{repo}/hybrid/graph
  - 그래프 기반 탐색
  - 시작점 symbol_id로부터 관계 확장

GET /api/repos/{owner}/{repo}/symbols
  - 심볼 메타데이터 검색
  - 이름, kind, 파일 등으로 필터링
```

### API 예시

```python
# 1. 일반 검색
GET /api/repos/myorg/myrepo/hybrid/search?query=authentication&top_k=10

# 2. 심볼 검색
GET /api/repos/myorg/myrepo/symbols?name=AuthService&kind=class

# 3. 그래프 탐색
GET /api/repos/myorg/myrepo/hybrid/graph?symbol_id=func_123&depth=2

# 4. 청크 검색
GET /api/repos/myorg/myrepo/hybrid/chunks?query=login&level=symbol
```

## Alternatives Considered

### 단일 통합 API
- 모든 파라미터를 하나의 엔드포인트에 몰아넣기
- 플래너 제어 어려움
- 복잡도 증가

### RPC 스타일 API
- 목적 불명확
- RESTful 원칙 위반

## Consequences

### Positive
- Agent Planner가 조합 호출 쉽게 가능
- 목적별 최적화 가능
- 재현성 확보
- API 문서화 명확

### Negative
- 엔드포인트 수 증가
- API 버전 관리 필요
- 클라이언트 구현 복잡도 증가

## Notes

각 API는 독립적으로 사용 가능하며, Agent Planner는 이들을 조합하여 복잡한 검색 시나리오를 구현할 수 있다.
