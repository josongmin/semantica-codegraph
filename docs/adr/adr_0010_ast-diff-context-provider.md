# ADR 0010 – AST 기반 Diff Context Provider 구축

## Status
Accepted

## Context

Copilot diff generator가 부분 diff patch를 생성하기 위해 필요한 것:

- 정확한 코드 블록 위치 (start_line, end_line)
- 심볼 정의 위치
- 호출되는 라인
- surrounding context (import, 상위 클래스 등)

LLM이 이러한 정보를 추측하면:
- 잘못된 위치 수정
- 중복 코드 생성
- 컨텍스트 누락으로 문법 오류

## Decision

Codegraph는 다음 정보를 제공한다:

### 1. AST 기반 코드 블록 Span

```python
@dataclass
class CodeSpan:
    file_path: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    symbol_id: str
    kind: SymbolKind
```

### 2. 심볼 정의 위치

```python
GET /api/repos/{owner}/{repo}/symbols/{symbol_id}/definition
→ {
    "file_path": "src/auth.py",
    "start_line": 45,
    "end_line": 67,
    "content": "...",
    "imports": [...],
    "parent_context": {...}
}
```

### 3. 호출되는 라인

```python
GET /api/repos/{owner}/{repo}/symbols/{symbol_id}/references
→ [
    {"file": "src/main.py", "line": 123, "context": "..."},
    {"file": "tests/test_auth.py", "line": 45, "context": "..."}
]
```

### 4. Surrounding Context

```python
GET /api/repos/{owner}/{repo}/symbols/{symbol_id}/context
→ {
    "imports": [...],
    "parent_class": {...},
    "sibling_methods": [...],
    "dependencies": [...]
}
```

## Alternatives Considered

### 에이전트가 파일 전체 파싱
- 비효율적
- 파싱 로직 중복
- 정확도 저하

### LLM에게 위치 추측 요청
- 오류 증가
- hallucination 발생
- 재현성 부족

### diff 도구 의존 (git diff 등)
- 기존 파일 기준 필요
- 신규 심볼 추가 시 부적합

## Consequences

### Positive
- 안전하고 일관적인 diff 생성 가능
- 대규모 리팩터링의 토대 마련
- LLM의 정확도 향상
- 재현 가능한 코드 수정

### Negative
- API 복잡도 증가
- 컨텍스트 추출 로직 구현 필요
- API 호출 횟수 증가 가능

## Notes

AST 기반 Diff Context Provider는 Codegraph와 Copilot 간의 핵심 인터페이스이다. 이를 통해 Copilot은 구조적으로 올바른 코드 수정을 수행할 수 있다.
