# ADR 0005 – SymbolIndex 구축

## Status
Accepted

## Context

심볼 레벨에서 함수, 클래스 정의를 추적해야 에이전트가 정확한 위치를 찾을 수 있다:

- 파일 단위 검색은 너무 광범위함
- 라인 단위 검색은 컨텍스트 부족
- 심볼 단위가 코드 편집의 최적 단위

## Decision

`SymbolIndex`에 모든 심볼의 메타데이터를 저장한다.

### 저장 항목

```sql
CREATE TABLE symbols (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,  -- function, class, method, variable
    file_path TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    start_col INTEGER,
    end_col INTEGER,
    parameters TEXT,  -- JSON
    return_type TEXT,
    parent_symbol_id TEXT,  -- enclosing class/module
    docstring TEXT,
    FOREIGN KEY (parent_symbol_id) REFERENCES symbols(id)
);

CREATE INDEX idx_symbols_name ON symbols(name);
CREATE INDEX idx_symbols_file ON symbols(file_path);
CREATE INDEX idx_symbols_kind ON symbols(kind);
```

## Alternatives Considered

### 파일 단위 색인
- 심볼 기반 검색 불가
- 정밀 편집 어려움

### Embedding-only 색인
- 정확한 위치 정보 부족
- 메타데이터 검색 불가

### 라인 단위 색인
- 컨텍스트 경계 불명확
- 심볼 범위 파악 어려움

## Consequences

### Positive
- LLM이 심볼 단위 코드 편집 수행 가능
- 정확한 위치 정보 제공
- 메타데이터 기반 필터링 가능

### Negative
- 인덱스 크기 증가
- 코드 변경 시 업데이트 필요

## Notes

SymbolIndex는 RouteIndex와 함께 Codegraph의 핵심 인덱스이다. SymbolIndex는 "무엇"을, RouteIndex는 "관계"를 담당한다.


