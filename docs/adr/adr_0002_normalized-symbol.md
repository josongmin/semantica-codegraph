# ADR 0002 – NormalizedSymbol 도입

## Status
Accepted

## Context

언어마다 클래스·함수·변수 정의 방식이 달라 검색·분석이 통합되지 않는다:

- Python: `def`, `class`, `async def`
- TypeScript: `function`, `class`, `interface`, `type`
- 통합 심볼 메타모델이 없으면 다언어 레포지토리 확장 시 유지보수 비용 증가

## Decision

언어별 AST 파서를 기반으로 공통 구조의 `NormalizedSymbol`을 도입한다.

### 핵심 필드

```python
@dataclass
class NormalizedSymbol:
    name: str
    kind: SymbolKind  # function, class, method, variable, etc.
    span: Span        # start_line, end_line, start_col, end_col
    file: str
    parameters: List[Parameter]
    return_type: Optional[str]
    parent: Optional[str]  # enclosing class/module
```

## Alternatives Considered

### 파일 단위 인덱싱
- 심볼 레벨 검색 불가
- 정밀도 저하

### 언어별 심볼 구조를 그대로 사용
- 크로스 랭귀지 검색 불가
- 확장성 부족

## Consequences

### Positive
- SymbolIndex / RouteIndex 구조화를 위한 기반 마련
- 다언어 통합 검색 가능
- 심볼 기반 코드 편집 지원

### Negative
- 언어별 매핑 로직 구현 필요
- 언어 특화 정보 손실 가능성

## Notes

NormalizedSymbol은 Codegraph의 핵심 데이터 모델이며, 모든 인덱싱과 검색의 기반이 된다.
