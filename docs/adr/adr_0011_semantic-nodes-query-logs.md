# ADR 0011 – Semantic Nodes + Query Logs 전략

## Status
Accepted (2025-11-20)

## Context

### 문제
코드 청크 임베딩만으로는 특정 질문 유형의 recall이 낮음:
- "POST /search API 어디?" → Route 정보 필요
- "IndexingPipeline 클래스는?" → Symbol 요약 필요
- "전체 구조는?" → 문서/설계 요약 필요

### 기존 구조의 한계
```
embeddings 테이블 → code_chunks만 연결
→ Symbol/Route/Doc 요약은 별도 테이블 필요
```

## Decision

### 1. Semantic Nodes 테이블 (통합 노드 요약)

**설계 원칙:**
- 모든 고급 노드(symbol/route/doc/issue)를 단일 테이블로 통합
- 템플릿 요약 / LLM 요약 구분
- 다중 모델 지원 (3-small, 3-large)
- 비용 효율 (선택적 업그레이드)

**스키마:**
```sql
CREATE TABLE semantic_nodes (
    repo_id TEXT NOT NULL,
    node_id TEXT NOT NULL,              -- 원본 테이블 PK (prefix 없이)
    node_type TEXT NOT NULL,            -- 'symbol' | 'route' | 'doc' | 'issue'
    doc_type TEXT,                      -- 'readme' | 'adr' | 'design'

    summary TEXT NOT NULL,              -- 검색용 텍스트
    summary_method TEXT NOT NULL,       -- 'template' | 'llm'
    model TEXT NOT NULL,                -- 'text-embedding-3-small' (풀 네임)
    embedding vector(1536),

    source_table TEXT,
    source_id TEXT,
    metadata JSONB DEFAULT '{}',        -- importance_score, query_count 등

    PRIMARY KEY (repo_id, node_id, node_type, model)
);
```

### 2. 2단계 임베딩 전략

**Phase 1: 템플릿 + 3-small (저비용)**
```
모든 퍼블릭 심볼 → 템플릿 summary + 3-small
모든 Route → 템플릿 summary + 3-small
비용: ~$0.02/repo (10k symbols)
```

**Phase 2: 선택적 LLM + 3-large (고품질)**
```
자주 검색되는 상위 50개만 → LLM summary + 3-large
나머지 → 템플릿 + 3-small (그대로)
비용: ~$0.03/repo (LLM 50개만)
99.5% LLM 비용 절감
```

### 3. Importance 기반 필터링

**Importance 계산 (0~1, 보수적):**
```python
Public: 0.3
API handler: 0.4
Docstring: 0.1
In-degree > 10: 0.2
API/Router 파일: 0.1
```

**랭킹 boost (과도 방지):**
```python
importance_boost = 1.0 + 0.1 * importance  # 최대 10%
llm_summary_boost = 1.03                   # LLM 요약 3%
```

### 4. Query Type별 Hybrid Ranking

**Query Classifier (룰 기반 → ML):**
```python
API_LOCATION: "/api/search 어디?"
LOG_LOCATION: '"error" 로그 어디?'
STRUCTURE: "전체 구조는?"
FUNCTION_IMPL: "함수 구현은?"
GENERAL: 일반 질문
```

**Query Type별 가중치:**
```python
API_LOCATION: {
    lexical: 0.1,
    semantic_large_node: 0.5,  # route large (Phase 2)
    semantic_small_node: 0.2,
}

LOG_LOCATION: {
    lexical: 0.6,              # BM25 우선
    semantic_small_code: 0.2,
}

STRUCTURE: {
    semantic_large_node: 0.4,  # doc/symbol large
    graph: 0.2,                # 관계 중요
}
```

### 5. Query Logs (데이터 기반 개선)

**query_logs 테이블:**
```sql
CREATE TABLE query_logs (
    id BIGSERIAL PRIMARY KEY,
    repo_id TEXT,
    query_text TEXT,
    query_type TEXT,
    weights JSONB,
    top_results JSONB,              -- [{node_id, score, signals}, ...]
    latency_ms INTEGER,
    created_at TIMESTAMP
);
```

**활용:**
- 인기 노드 추출 (LLM 업그레이드 대상)
- Weight 효과 분석
- A/B 테스트

**node_popularity 테이블:**
```sql
CREATE TABLE node_popularity (
    repo_id TEXT,
    node_id TEXT,
    query_count_7d INTEGER,
    avg_rank FLOAT,
    PRIMARY KEY (repo_id, node_id)
);
```

## Implementation

### Phase 1 (완료)
- semantic_nodes 테이블 + Migration 005
- SemanticNodeStore (PostgreSQL)
- SymbolSummaryBuilder (템플릿 기반)
- Bootstrap 다중 모델 (small/large)
- Pipeline 통합 (route/symbol semantic)
- QueryClassifier + HybridRanker
- 테스트: 71개 통과

**결과:**
- 649개 semantic nodes 생성 (643 symbols + 6 routes)
- 템플릿 summary + 3-small
- Avg importance: 0.412 (symbols), 0.8 (routes)

### Phase 2 (완료)
- query_logs + node_popularity 테이블
- QueryLogStore (로깅 + 통계)
- HybridRetriever 자동 로깅
- API/MCP QueryClassifier 통합
- QueryLogAnalyzer (인기 노드 추출)

**결과:**
- 검색 시 자동 query_type 로깅
- 인기 노드 추출 로직 구현
- Weight 효과 분석 준비

## Alternatives Considered

### 모든 심볼에 LLM 요약 + 3-large
- 비용: 10k symbols × $0.001 = $10/repo
- 불필요한 노드에도 비용 지출
- **거부:** 선택적 업그레이드로 99% 절감

### 별도 테이블 (symbol_summaries, route_embeddings)
- 확장성 낮음 (doc, issue 추가 시 또 테이블 증가)
- 검색 로직 중복
- **거부:** semantic_nodes로 통합

### Query type을 LLM으로 분류
- 비용 증가 (검색마다 LLM 호출)
- 레이턴시 증가
- **거부:** 룰 기반 시작 → 로그 기반 튜닝 → ML 분류기

### Query logs를 파일로 저장
- 분석 어려움 (SQL 불가)
- 통계 느림
- **거부:** PostgreSQL에 저장

## Consequences

### Positive
- **확장성:** symbol/route/doc/issue 단일 구조
- **비용 효율:** 템플릿 기본 + 선택적 LLM
- **데이터 기반:** query_logs로 실사용 패턴 반영
- **Explainability:** Debug 모드로 결과 추적

### Negative
- **초기 설정 복잡도:** 2단계 임베딩 관리
- **로그 용량:** 장기 운영 시 query_logs 증가

### Mitigation
- **로그 정리:** 90일 이상 자동 삭제
- **샘플링:** 프로덕션에서 10-20% 샘플링

## Performance

### 인덱싱 성능
```
106개 파일 → 15.3초
649개 semantic nodes 생성
- Routes: 6개
- Symbols: 643개
비용: ~$0.02/repo (3-small)
```

### 검색 성능
```
Semantic search: 30-80% 유사도
Query logging: +50-100ms 오버헤드
전체 레이턴시: ~400ms
```

### Recall 향상 (예상)
```
"API 어디?" 질문: 2-3배 recall 향상
"Symbol 찾기": 3-5배 recall 향상
```

## Future Work

### Phase 3 (예정)
- 일주일간 query_logs 수집
- 인기 노드 50개 추출
- LLM 요약 생성 + 3-large
- Weight 미세 조정
- 문서 인덱싱 (README, ADR)

### Phase 4 (장기)
- Learning to Rank (LightGBM)
- ML 기반 query classifier
- 이슈/PR 인덱싱
- Voyage/Cohere A/B 테스트

## References
- ADR 0005: Symbol Index
- ADR 0008: Embedding Strategy
- Phase 1 구현: migrations/005_semantic_nodes.sql
- Phase 2 구현: migrations/006_query_logs.sql

## Notes

**구현 완료 (2025-11-20):**
- Phase 1: Semantic Nodes (symbol/route)
- Phase 2: Query Logs (자동 수집)
- 테스트: 71개 통과
- 실제 데이터: 649 nodes, 9 query logs

**다음 단계:**
- 1-2주 실사용 후 query_logs 분석
- 인기 노드 LLM 업그레이드
- Weight 튜닝
