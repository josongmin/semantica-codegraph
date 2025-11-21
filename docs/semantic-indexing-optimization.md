# Semantic Indexing 최적화 가이드

## 현재 문제점

### 프로파일링 결과 (126파일, 767개 symbol)
```
총 시간: 16963ms
- Summary 생성: 24ms (0.1%)
- 임베딩 생성: 3737ms (22.0%)
- DB 저장: 13202ms (77.8%) ← 병목

메모리:
- Bootstrap: +500MB
- Peak: 622MB
```

### 근본 원인

**1. 메모리 기반 일괄 처리**
```python
# 현재: 741개를 전부 메모리에
semantic_nodes = []  # 741개 전체
for node in all_nodes:
    semantic_nodes.append({...})

# 임베딩 생성
embeddings = embed_texts(all_summaries)  # 741개 한 번에

# DB 저장
save_batch(semantic_nodes, batch_size=1000)  # 741개 한 번에
```

**문제:**
- 피크 메모리 과다
- 큰 리포에서 스케일 불가 (10배 → 5GB+)
- 중간 실패 시 전체 재시작

**2. DB 저장 비효율**
```python
# executemany 사용
cur.executemany(sql, data)  # 각 row 개별 실행
conn.commit()  # 매 배치마다
```

**문제:**
- executemany는 느림 (execute_batch보다 5-10배)
- 배치마다 commit (트랜잭션 오버헤드)
- ON CONFLICT DO UPDATE (인덱스 조회 비용)

**3. Bootstrap 비대화**
```
Bootstrap 초기화: 3.70초, +500MB
```

**문제:**
- 사용하지 않는 backend도 모두 초기화
- 모든 언어 파서 로딩
- Search backend (index 시에는 불필요)

## 최적화 방안

### Phase 1: DB 저장 최적화 (즉시 적용 가능)

#### 1-1. executemany → execute_batch
```python
from psycopg2.extras import execute_batch

# 변경 전
cur.executemany(sql, data)

# 변경 후
execute_batch(cur, sql, data, page_size=500)
```

**효과:**
- 5-10배 빠름
- 예상: 13202ms → 1500-2500ms

#### 1-2. 트랜잭션 최적화
```python
# 변경 전: 매 배치마다 commit
for i in range(0, len(nodes), batch_size):
    batch = nodes[i : i + batch_size]
    execute_batch(cur, sql, batch)
    conn.commit()  # 매번

# 변경 후: 마지막에 한 번만
for i in range(0, len(nodes), batch_size):
    batch = nodes[i : i + batch_size]
    execute_batch(cur, sql, batch, page_size=500)
    # commit 없음

conn.commit()  # 마지막에만
```

**효과:**
- 2-3배 빠름
- 예상: 2000ms → 700-1000ms

#### 1-3. 재인덱싱 시 DELETE 후 INSERT
```python
# 변경 전: ON CONFLICT DO UPDATE
INSERT ... ON CONFLICT ... DO UPDATE SET ...

# 변경 후: 재인덱싱 시
DELETE FROM semantic_nodes WHERE repo_id = %s AND node_type = 'symbol';
INSERT ... (ON CONFLICT 없음)
```

**효과:**
- 인덱스 조회 비용 제거
- 2배 빠름
- 예상: 700ms → 350ms

**총 예상 개선: 13202ms → 350ms (37배)**

### Phase 2: 스트리밍 처리 (중요)

#### 2-1. Symbol 배치 단위 처리
```python
SYMBOL_BATCH_SIZE = 100  # 100개씩 처리

for i in range(0, len(indexable_nodes), SYMBOL_BATCH_SIZE):
    batch_nodes = indexable_nodes[i:i+SYMBOL_BATCH_SIZE]
    
    # 1. Summary 생성 (배치)
    summaries = [builder.build(node) for node in batch_nodes]
    
    # 2. 임베딩 생성 (배치)
    embeddings = embedding_service.embed_texts(summaries)
    
    # 3. DB 저장 (배치)
    semantic_nodes = [...]
    semantic_node_store.save_batch(semantic_nodes, batch_size=100)
    
    # 4. 메모리 정리
    del semantic_nodes, embeddings, summaries
    
    # 5. 진행률 업데이트
    progress = 0.7 + 0.3 * (i + SYMBOL_BATCH_SIZE) / len(indexable_nodes)
    repo_store.update_indexing_status(repo_id, "indexing", progress=progress)
```

**효과:**
- 피크 메모리: 622MB → ~200MB
- 중간 저장으로 신뢰성 향상
- 진행률 업데이트 세밀화

#### 2-2. ORM 세션 관리 (SQLAlchemy 사용 시)
```python
for batch in batches:
    # 처리
    session.bulk_save_objects(semantic_nodes)
    session.flush()
    
    # 세션 정리
    session.expunge_all()
    
    # GC (큰 배치 후에만)
    if batch_index % 10 == 0:
        import gc
        gc.collect()
```

**효과:**
- 메모리 누수 방지
- 장시간 실행 안정성

### Phase 3: Bootstrap 최적화

#### 3-1. Lazy Loading
```python
class Bootstrap:
    @property
    def parser(self):
        if self._parser is None:
            # 필요한 언어만 로드
            languages = self.config.languages  # ['python']만
            self._parser = create_parser(languages)
        return self._parser
```

**효과:**
- Bootstrap: 3.70초 → ~1초
- 메모리: +500MB → ~150MB

#### 3-2. 모드별 Bootstrap Profile
```python
# Index 모드
bootstrap_index = Bootstrap.for_indexing(
    languages=['python'],
    enable_search=False,  # search backend 제외
)

# Search 모드
bootstrap_search = Bootstrap.for_search(
    enable_indexing=False,  # parser 제외
)
```

**효과:**
- 불필요한 초기화 제거
- 50% 시간/메모리 절약

### Phase 4: Phase 세분화

#### 4-1. DB 저장 Phase 분리
```python
# 현재
profiler.start_sub_phase("indexing_core")
# ... 전체 처리 ...
profiler.end_sub_phase()

# 개선
profiler.start_sub_phase("graph_nodes_save")
graph_store.save_nodes(nodes)
profiler.end_sub_phase()

profiler.start_sub_phase("chunks_save")
chunk_store.save_chunks(chunks)
profiler.end_sub_phase()

profiler.start_sub_phase("edges_save")
graph_store.save_edges(edges)
profiler.end_sub_phase()

profiler.start_sub_phase("semantic_symbols_save")
semantic_node_store.save_batch(symbols)
profiler.end_sub_phase()
```

**효과:**
- 병목 구간 정밀 식별
- DB vs 로직 비율 분리
- 최적화 우선순위 명확

### Phase 5: 대규모 리포 대비

#### 5-1. 스케일 예측 (10배)
```
현재 (126파일, 885노드):
- 시간: 127초
- 메모리: 622MB

예상 (1260파일, 8850노드):
- 시간: 1270초 (~21분)
- 메모리: 6GB+
- 임베딩 API: 90회 → 비용 증가
```

#### 5-2. 스트리밍 아키텍처
```python
def index_repository_streaming(repo_id, root_path):
    """스트리밍 방식 인덱싱"""
    
    # 파일 단위 스트리밍
    for file_batch in scan_files_batched(root_path, batch_size=50):
        # 1. 파싱
        nodes, edges = parse_files(file_batch)
        
        # 2. 즉시 저장
        graph_store.save_nodes(nodes)
        graph_store.save_edges(edges)
        
        # 3. 메모리 정리
        del nodes, edges
        gc.collect()
    
    # Symbol semantic 스트리밍
    for symbol_batch in get_symbols_batched(repo_id, batch_size=100):
        summaries = build_summaries(symbol_batch)
        embeddings = embed_texts(summaries)
        save_semantic_nodes(symbol_batch, embeddings)
        
        del summaries, embeddings
        gc.collect()
```

**효과:**
- 메모리: O(n) → O(1) (상수)
- 대규모 리포 처리 가능
- 중단/재개 가능

#### 5-3. 체크포인트 시스템
```python
# 진행 상태 저장
checkpoint = {
    "last_processed_file": "src/indexer/pipeline.py",
    "last_processed_node": 542,
    "phase": "semantic_symbols",
}

# 재개
if checkpoint_exists:
    resume_from(checkpoint)
else:
    start_from_scratch()
```

**효과:**
- 실패 시 재시작 불필요
- 부분 재인덱싱 가능

### Phase 6: 인덱스 관리

#### 6-1. 대량 삽입 시 인덱스 비활성화
```python
# 재인덱싱 시 (clean_before=True)
if clean_before and large_repo:
    # 인덱스 삭제
    cur.execute("DROP INDEX IF EXISTS idx_semantic_nodes_embedding")
    
    # 대량 삽입
    for batch in batches:
        execute_batch(cur, sql, batch, page_size=1000)
    
    # 인덱스 재생성
    cur.execute("""
        CREATE INDEX idx_semantic_nodes_embedding 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)
```

**효과:**
- 10-20배 빠름 (대량 삽입 시)
- 예상: 13202ms → 500-1000ms

#### 6-2. COPY 명령 (최고 성능)
```python
from io import StringIO
import csv

# CSV 생성
csv_buffer = StringIO()
writer = csv.writer(csv_buffer, delimiter='\t')

for node in semantic_nodes:
    writer.writerow([
        node['repo_id'],
        node['node_id'],
        node['summary'],
        json.dumps(node['embedding']),
        ...
    ])

# COPY
csv_buffer.seek(0)
cur.copy_expert("""
    COPY semantic_nodes (repo_id, node_id, summary, embedding, ...)
    FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t')
""", csv_buffer)
```

**효과:**
- 50-100배 빠름
- 예상: 13202ms → 130-260ms

## 구현 우선순위

### 즉시 (Critical)
1. execute_batch 적용 (5-10배 개선)
2. 트랜잭션 최적화 (2-3배 개선)
3. 재인덱싱 시 DELETE+INSERT (2배 개선)

**예상 총 개선: 13202ms → ~500ms (26배)**

### 단기 (High Priority)
4. 배치 단위 처리 (100개씩)
5. 메모리 정리 (GC, session clear)
6. Phase 세분화 (DB 저장 분리)

**예상 개선: 메모리 622MB → ~200MB (3배)**

### 중기 (Medium Priority)
7. Bootstrap 최적화 (lazy loading)
8. 모드별 Bootstrap Profile
9. 인덱스 관리 전략

**예상 개선: Bootstrap 3.70초 → ~1초, 메모리 -350MB**

### 장기 (Future)
10. 스트리밍 아키텍처
11. 체크포인트 시스템
12. COPY 명령 활용

**예상 개선: 대규모 리포 처리 가능, O(n) → O(1) 메모리**

## 스케일 시나리오

### 현재 (126파일)
```
시간: 127초
메모리: 622MB
비용: $0.000166
```

### 10배 (1260파일)
```
현재 구조:
- 시간: 1270초 (~21분)
- 메모리: 6GB+
- 비용: $0.00166

Phase 1 최적화 후:
- 시간: 400초 (~7분)
- 메모리: 6GB+
- 비용: $0.00166

Phase 2 최적화 후:
- 시간: 400초
- 메모리: 500MB (O(1))
- 비용: $0.00166
```

### 100배 (12600파일, 대규모 모노레포)
```
현재 구조:
- 불가능 (메모리 초과)

Phase 1+2 최적화 후:
- 시간: 4000초 (~67분)
- 메모리: 500MB
- 비용: $0.0166

Phase 1+2+3 최적화 후:
- 시간: 2500초 (~42분)
- 메모리: 300MB
- 비용: $0.0166
```

## 구현 계획

### Step 1: execute_batch 적용 (1시간)
```python
# src/indexer/semantic_node_store.py
from psycopg2.extras import execute_batch

def save_batch(self, nodes, batch_size=1000):
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i+batch_size]
        execute_batch(cur, sql, data, page_size=500)
    
    conn.commit()  # 마지막에만
```

### Step 2: 배치 단위 처리 (2시간)
```python
# src/indexer/pipeline.py
SYMBOL_BATCH_SIZE = 100

for i in range(0, len(indexable), SYMBOL_BATCH_SIZE):
    batch = indexable[i:i+SYMBOL_BATCH_SIZE]
    
    # Summary + Embed + Save
    summaries = [builder.build(n) for n in batch]
    embeddings = embedding_service.embed_texts(summaries)
    save_batch(make_semantic_nodes(batch, embeddings))
    
    # 메모리 정리
    del summaries, embeddings
    if i % 1000 == 0:
        gc.collect()
```

### Step 3: Phase 세분화 (30분)
```python
profiler.start_sub_phase("nodes_db_save")
graph_store.save_nodes(nodes)
profiler.end_sub_phase()

profiler.start_sub_phase("chunks_db_save")
chunk_store.save_chunks(chunks)
profiler.end_sub_phase()
```

### Step 4: Bootstrap 최적화 (1시간)
```python
# src/core/bootstrap.py
def for_indexing(config):
    """인덱싱 전용 Bootstrap (search backend 제외)"""
    return Bootstrap(
        enable_lexical_search=False,
        enable_semantic_search=False,
        ...
    )
```

### Step 5: 인덱스 관리 (1시간)
```python
def save_batch_with_index_management(nodes, clean_before=False):
    if clean_before and len(nodes) > 1000:
        # 인덱스 삭제
        drop_indexes()
        
        # 대량 삽입
        copy_from_csv(nodes)
        
        # 인덱스 재생성
        create_indexes()
    else:
        # 일반 삽입
        execute_batch(...)
```

## 벤치마크 목표

### Phase 1 완료 후
```
Symbol Semantics:
- Summary: 24ms
- 임베딩: 3737ms
- DB 저장: 500ms (26배 개선)
- 합계: 4261ms

전체: 127초 → ~30초
```

### Phase 2 완료 후
```
메모리: 622MB → 200MB (3배 개선)
대규모 리포: 처리 가능
```

### Phase 3 완료 후
```
Bootstrap: 3.70초 → 1초
전체: 30초 → 27초
```

## 마이그레이션 전략

### 호환성 유지
```python
# 기존 인터페이스 유지
def save_batch(nodes, batch_size=1000, streaming=False):
    if streaming:
        return _save_batch_streaming(nodes, batch_size)
    else:
        return _save_batch_legacy(nodes, batch_size)
```

### 점진적 적용
1. execute_batch 적용 (하위 호환)
2. 새 플래그로 스트리밍 모드 추가
3. 기본값 변경 (충분한 테스트 후)

## 참고

- 현재 구현: `src/indexer/semantic_node_store.py`
- 프로파일링: `.profiler/reports/`
- 관련 이슈: DB 저장 병목 (77.8%)

