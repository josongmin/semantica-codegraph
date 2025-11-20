# ADR 0007 – RAPTOR 기반 계층형 Chunking

## Status
Accepted

## Context

코드 context를 LLM에 전달할 때 문제점:

- 파일 단위 청킹: 너무 크고 관련 없는 정보 포함
- 고정 크기 슬라이스(512/1024 토큰): 의미 단위가 무너지고 중요 정보 사라짐
- 단순 청킹: AST 구조 무시

RAPTOR(Recursive Abstractive Processing for Tree-Organized Retrieval)는 AST 기반 계층(chunk-tree)을 형성해 LLM retrieval을 강화한다.

## Decision

RAPTOR-style 계층 구조를 채택한다.

### 계층 구조

```
Level 1: Module Summary
  - 파일 전체의 요약
  - 주요 export, public API

Level 2: Class/Function Block Summary
  - 각 클래스, 함수의 요약
  - 시그니처, docstring, 핵심 로직

Level 3: Leaf-level AST Chunk
  - 실제 코드 블록
  - 메서드, 함수 본문
```

### Chunking 전략

```python
def create_semantic_nodes(symbol: NormalizedSymbol) -> List[SemanticNode]:
    # Level 1: Module-level
    module_node = create_module_summary(symbol.file)
    
    # Level 2: Symbol-level
    symbol_node = create_symbol_summary(symbol)
    
    # Level 3: Implementation-level
    impl_chunks = chunk_implementation(symbol.body)
    
    return [module_node, symbol_node] + impl_chunks
```

## Alternatives Considered

### Naive 512~1024 토큰 슬라이스
- 의미 경계 무시
- 컨텍스트 손실

### 단일 embedding chunk
- 파일 크기 제한
- 검색 정밀도 저하

### 함수 단위만 청킹
- 모듈/클래스 레벨 컨텍스트 손실
- 계층 정보 부족

## Consequences

### Positive
- LLM이 필요한 레벨의 context만 획득 가능
- 검색 품질 안정
- 토큰 사용 효율화

### Negative
- 인덱싱 복잡도 증가
- 저장 공간 증가 (각 레벨별 중복)
- 요약 품질 의존성

## Notes

RAPTOR는 원래 논문 검색을 위해 제안되었으나, 코드의 계층적 구조와 잘 맞아떨어진다. AST 기반 청킹과 결합하면 최적의 코드 retrieval을 제공한다.


