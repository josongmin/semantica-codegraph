# ADR 0004 – RouteIndex (CallGraph) 구축

## Status
Accepted

## Context

LLM이 다음과 같은 질문에 답하려면 호출 그래프가 필요하다:

- "이 함수는 어디서 호출되나?"
- "이 라우트는 어떤 핸들러와 연결됐나?"
- "이 모듈을 변경하면 어디에 영향을 주나?"

텍스트 기반 검색만으로는 이러한 관계를 정확히 파악할 수 없다.

## Decision

파일 간 의존 관계와 호출 관계를 나타내는 `RouteIndex` (CallGraph)를 구축한다.

### 저장 정보

- `source_symbol` → `target_symbol` 관계
- import graph
- module-level dependency
- function call relationships

### 데이터 모델

```python
@dataclass
class Edge:
    source: str  # symbol_id
    target: str  # symbol_id
    edge_type: EdgeType  # CALL, IMPORT, INHERIT, etc.
    location: Span
```

## Alternatives Considered

### 텍스트 기반 경로 추적
- 정밀도 낮음
- false positive 많음

### 단순 AST 수평 탐색
- 크로스 파일 추적 불가
- 간접 호출 파악 어려움

### 실시간 호출 분석
- 성능 오버헤드 큼
- 대규모 코드베이스에서 불가능

## Consequences

### Positive
- 검색 품질 비약적 향상
- LLM이 계획 기반 수정 가능
- 영향 범위 분석 가능

### Negative
- 인덱싱 시간 증가
- 동적 호출 추적 한계 (reflection, dynamic import)

## Notes

RouteIndex는 정적 분석 기반이므로 동적 호출은 일부 누락될 수 있다. 하지만 대부분의 코드베이스에서 정적 분석만으로도 충분한 품질을 제공한다.
