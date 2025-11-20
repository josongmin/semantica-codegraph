# ADR 0003 – 다언어 파싱 아키텍처 설계

## Status
Accepted

## Context

Python, TypeScript, JavaScript, Go 등 서로 다른 언어를 동일한 인덱스 시스템에서 다뤄야 한다:

- 각 언어의 AST 구조가 상이함
- 새로운 언어 추가 시 확장 가능한 구조 필요

## Decision

tree-sitter 기반 언어 파서를 채택하고, 각 언어 파싱 결과를 `NormalizedSymbol`로 매핑하는 adapter layer를 생성한다.

### 아키텍처

```
LanguageParser (Abstract)
    ├── PythonParser (tree-sitter-python)
    ├── TypeScriptParser (tree-sitter-typescript)
    └── [Future] GoParser, RustParser, etc.

각 Parser는 NormalizedSymbol[]을 반환
```

### 파싱 파이프라인

1. 파일 읽기
2. tree-sitter로 AST 생성
3. 언어별 visitor로 심볼 추출
4. NormalizedSymbol로 변환
5. SymbolIndex에 저장

## Alternatives Considered

### Custom parser 직접 개발
- 개발 비용 과다
- 언어 업데이트 추적 어려움

### ctags 기반 인덱싱
- 정확도 낮음
- AST 레벨 정보 부족

### LSP 의존
- 실시간 파싱 오버헤드
- 모든 언어에 LSP 서버 필요

## Consequences

### Positive
- 언어 추가가 쉬워짐 (새 adapter만 구현)
- Codegraph는 multi-language indexer가 됨
- tree-sitter의 안정성과 성능 활용

### Negative
- tree-sitter 의존성
- 언어별 adapter 유지보수 필요

## Notes

tree-sitter는 GitHub, Atom, Neovim 등에서 사용되는 검증된 파서 생성기이다.


