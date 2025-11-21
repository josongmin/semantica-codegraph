# 인덱싱 과정 전체 정리

## 개요

Semantica Codegraph의 인덱싱 파이프라인은 소스 코드를 파싱, 분석, 변환하여 검색 가능한 형태로 저장하는 전체 과정입니다.

**핵심 정책 (옵션 B 구조)**:
- `code_chunks.text` = 원본 코드 (raw_code) - 코드 뷰어용
- `code_chunks.attrs["search_text"]` = 검색 최적화 텍스트 ([META] + raw_code) - 검색 인덱싱용
- Meilisearch/Qdrant는 `attrs["search_text"]` 우선 사용, 없으면 `text` 사용

## 인덱스 레이어 구조

파이프라인은 4가지 검색 인덱스를 생성합니다:

### 1. Lexical Index (BM25 검색)
- **단위**: Chunk
- **백엔드**: Meilisearch
- **소스**: `code_chunks.attrs["search_text"]` (fallback: `text`)
- **생성**: Step 10
- **용도**: 키워드 기반 정확 매칭

### 2. Chunk Semantic Index (벡터 검색)
- **단위**: Chunk
- **백엔드**: Qdrant
- **소스**: `embedding(code_chunks.attrs["search_text"])`
- **생성**: Step 11
- **용도**: 자연어 질문 의미 기반 검색

### 3. Route Semantic Index (API 검색)
- **단위**: Route (API 엔드포인트)
- **백엔드**: `semantic_nodes` (PostgreSQL + pgvector)
- **소스**: Route 템플릿 기반 summary
- **생성**: Step 13
- **용도**: API 엔드포인트 의미 기반 검색

### 4. Symbol Semantic Index (심볼 검색)
- **단위**: Symbol (Function/Class)
- **백엔드**: `semantic_nodes` (PostgreSQL + pgvector)
- **소스**: Symbol 템플릿 기반 summary
- **생성**: Step 14
- **용도**: 함수/클래스 의미 기반 검색

**저장소 분리 정책**:
- Chunk 임베딩: Qdrant (빠른 벡터 검색, 고차원 최적화)
- Route/Symbol 임베딩: Postgres pgvector (관계형 데이터와 결합, 트랜잭션 보장)

## 전체 플로우

```
저장소 루트 경로
  ↓
[Step 1] 파일 스캔 (RepoScanner)
  ↓
[Step 2] Repo Profiling (프로젝트 구조 분석)
  ↓
[Step 3] 파일별 파싱 (Parser) + IR 변환 (IRBuilder)
  ↓
[Step 4] 그래프 저장 (GraphStore)
  ↓
[Step 5] File Profiling (파일 역할 태깅)
  ↓
[Step 6] Graph Ranking (노드 중요도 계산)
  ↓
[Step 7] 청킹 (Chunker)
  ↓
[Step 8] 청크 저장 (ChunkStore)
  ↓
[Step 9] Chunk Tagging + search_text 생성
  ↓
[Step 10] Lexical 인덱싱 (Meilisearch)
  ↓
[Step 11] Chunk Semantic 인덱싱 (Qdrant)
  ↓
[Step 12] Route 추출 및 저장 (API 엔드포인트)
  ↓
[Step 13] Route Semantic 인덱싱 (semantic_nodes/pgvector)
  ↓
[Step 14] Symbol Semantic 인덱싱 (semantic_nodes/pgvector)
  ↓
[Step 15] 메타데이터 업데이트 및 완료
```

## 단계별 상세 설명

### Step 1: 파일 스캔 (RepoScanner)

**목적**: 저장소 내 인덱싱 대상 파일 목록 수집

**처리 내용**:
- 디렉토리 재귀 탐색 (.git, node_modules 등 제외)
- 언어 감지 (확장자 기반: .py → python, .ts → typescript 등)
- 제외 패턴 적용 (exclude_patterns 설정)
- 언어 필터링 (languages 설정)
- 텍스트 파일 필터링 (index_text_files 설정)
- 테스트 파일 필터링 (include_tests 설정)

**출력**: FileMetadata 리스트
- file_path: 상대 경로
- abs_path: 절대 경로
- language: 언어 (python, typescript 등)

**성능**: 1000개 파일 약 0.5초

### Step 2: Repo Profiling (프로젝트 구조 분석)

**목적**: 저장소 전체 구조 및 특성 파악

**처리 내용**:
- 언어 분포 분석 (파일 수, 라인 수 기준)
- 의존성 추출 (pyproject.toml, package.json, requirements.txt)
- 프레임워크 감지 (FastAPI, Django, Flask, Express 등)
- API 패턴 감지 (@router.get, @app.post 등)
- 디렉토리 역할 분류 (API, Service, Model, Test, Config)
- 엔트리포인트 찾기 (main.py, app.py 등)
- 프로젝트 타입 추론 (web_api, cli, library 등)

**출력**: RepoProfile
- framework: 주 프레임워크
- frameworks: 감지된 모든 프레임워크
- project_type: web_api, cli, library 등
- api_directories: API 디렉토리 목록
- service_directories: Service 디렉토리 목록
- model_directories: Model 디렉토리 목록
- test_directories: Test 디렉토리 목록
- entry_points: 엔트리포인트 파일 목록
- dependencies: 의존성 패키지
- languages: 언어별 라인 수
- primary_language: 주 언어

**저장**: repo_profiles 테이블

### Step 3: 파일별 파싱 + IR 변환

**목적**: 소스 코드를 AST로 파싱하고 코드 그래프 IR로 변환

**처리 내용**:

#### 3-1. 파싱 (Parser)
- Tree-sitter 기반 파서 사용 (언어별)
- 파싱 결과 캐시 확인 (ParseCache)
- 캐시 미스 시 파싱 실행
- RawSymbol, RawRelation 추출

**파싱 결과**:
- RawSymbol: 심볼 이름, kind, 파일 경로, 라인 범위, 부모, docstring
- RawRelation: 소스/대상 심볼, 관계 종류, 파일 경로

#### 3-2. IR 변환 (IRBuilder)
- RawSymbol → CodeNode 변환
- RawRelation → CodeEdge 변환
- 소스 코드 텍스트 첨부
- 노드/엣지 메타데이터 생성

**IR 결과**:
- CodeNode: id, kind, name, file_path, span, text, language, attrs
- CodeEdge: src_id, dst_id, kind, attrs

**병렬 처리**:
- 파일 5개 이상 시 병렬 파싱 (ProcessPoolExecutor)
- 워커 수: CPU 코어 수 (기본값)
- 진행률 업데이트: 5% 단위

**성능**: 1000개 파일 약 10초 (병렬), 30초 (순차)

### Step 4: 그래프 저장 (GraphStore)

**목적**: 코드 그래프를 PostgreSQL에 영구 저장

**입력**: 메모리의 `all_nodes`, `all_edges` (Step 3에서 누적)

**처리 내용**:
```python
# 배치 저장
self.graph_store.save_nodes(all_nodes)
self.graph_store.save_edges(all_edges)
```

**DB 스키마**:
- code_nodes: repo_id, node_id, kind, name, file_path, span, language, text (원본 코드), attrs
- code_edges: repo_id, src_id, dst_id, kind, attrs

**주의**: 
- all_nodes는 메모리에 계속 유지됨
- Step 7 청킹 시 메모리의 all_nodes를 바로 사용
- GraphStore에서 다시 조회하지 않음 (성능 최적화)

**성능**: 1000개 노드 약 0.5초

### Step 5: File Profiling (파일 역할 태깅)

**목적**: 각 파일의 역할 및 특성 분석

**처리 내용**:
- 경로 기반 역할 태깅 (API, Service, Model, Test 등)
- 내용 기반 역할 태깅 (코드 패턴 스캔)
- Import 분석 (외부/내부 의존성 분류)
- API 프레임워크 감지 (FastAPI, Flask 등)
- API 엔드포인트 추출 (@router.get("/path") 등)

**출력**: FileProfile 리스트
- is_api_file: API 파일 여부
- is_router: 라우터 파일 여부
- is_service_file: 서비스 파일 여부
- is_model_file: 모델 파일 여부
- is_test_file: 테스트 파일 여부
- roles: 역할 목록
- api_framework: API 프레임워크
- api_patterns: API 패턴 목록
- endpoints: 엔드포인트 정보
- imports: Import 목록
- external_deps: 외부 의존성
- internal_deps: 내부 의존성
- line_count: 라인 수

**저장**: file_profiles 테이블 (배치 저장)

**성능**: 1000개 파일 약 3초

### Step 6: Graph Ranking (노드 중요도 계산)

**목적**: 코드 그래프 기반 노드 중요도 점수 계산

**처리 내용**:
- PageRank 스타일 알고리즘
- In-degree 계산 (얼마나 많이 호출/참조되는가)
- Out-degree 계산 (얼마나 많이 호출/참조하는가)
- 중요도 점수 계산: (in_degree * 0.7 + out_degree * 0.3) / max(total_nodes * 0.1, 1)
- 0~1 범위로 정규화
- 배치 처리 (100개씩)

**업데이트**: code_nodes.attrs["importance_score"]

**성능**: 1000개 노드 약 1초

### Step 7: 청킹 (Chunker)

**목적**: CodeNode를 검색 가능한 CodeChunk로 변환

**입력**: 메모리에 있는 `all_nodes` 리스트 (Step 3-5에서 누적된 노드)
- GraphStore에서 다시 가져오지 않음
- 메모리 기반 처리로 빠른 속도

**핵심 전략**: Symbol 기반 청킹 (1 Node = 1 Chunk) + 조건부 파일 요약 청크

**처리 흐름**:

#### 7-1. 파일별 노드 그룹화
```python
# 파일별로 노드 분류
nodes_by_file: dict[str, list[CodeNode]] = {}
file_nodes: dict[str, CodeNode] = {}

for node in all_nodes:
    if node.kind == "File":
        file_nodes[file_path] = node
    else:
        nodes_by_file[file_path].append(node)
```

#### 7-2. Symbol 노드 → 청크 변환

**원칙: 1 Node = 1 Chunk**

```python
for node in symbol_nodes:
    # 토큰 수 체크
    token_count = count_tokens(node.text)  # tiktoken
    
    if token_count > 15000:
        # 분할 (라인 기반 + 5줄 오버랩)
        chunks.extend(split_node_by_tokens(node))
    else:
        # 그대로 청크 생성
        chunk = CodeChunk(
            text=node.text,  # raw_code
            attrs={
                "node_kind": node.kind,
                "node_name": node.name,
            }
        )
        chunks.append(chunk)
```

**토큰 제한:**
- 최대: 15,000 토큰 (Mistral Codestral Embed 16K 제한 - 안전 마진 1K)
- 초과 시: 자동 분할 (라인 기반, 5줄 오버랩)
- 실제 사례: IndexingPipeline (7050 토큰), HybridRetriever (7052 토큰)

#### 7-3. 조건부 파일 요약 청크 생성

**생성 조건:**

1. **설정 파일**: 무조건 생성
   ```python
   file_ext in ['.yaml', '.yml', '.toml', '.json', '.env', '.ini', '.conf']
   ```

2. **코드 파일**: 심볼 5개 이상
   ```python
   file_ext in ['.py', '.ts', '.js'] and len(symbol_nodes) >= 5
   ```

3. **API 파일**: 무조건 생성
   ```python
   'api' in file_path or 'route' in file_path
   ```

**파일 요약 청크 구조:**
```
# File: src/indexer/pipeline.py
# Language: python

# File Description:
인덱싱 파이프라인
(파일 상단 docstring 자동 추출)

# Contains 14 symbols:

## Classes (2 total):
  - IndexingPipeline
  - ParallelConfig

## Functions (12 total):
  - index_repository
  - _parse_file
  - _build_graph
  ... and 9 more

# API Endpoints:
  POST /hybrid/search -> hybrid_search
  GET /status -> get_status
```

**특징:**
- 원본 코드 포함 안 함 (중복 방지)
- 메타데이터만 포함 (평균 500-1000자)
- 파일 레벨 질문 recall 대폭 향상

#### 7-4. 청킹 결과

**Codegraph 프로젝트 (2025-11-21):**
- Symbol 청크: 726개 (Function/Class/Method 노드)
- 파일 요약 청크: 90개 (조건 만족 파일, +12%)
- 총 청크: 816개

**주요 파일 예시:**
```
src/core/ports.py:       31개 청크 (30 Symbol + 1 File Summary)
src/core/bootstrap.py:   30개 청크 (29 Symbol + 1 File Summary)
src/indexer/pipeline.py: 15개 청크 (14 Symbol + 1 File Summary)
```

**출력**: CodeChunk 리스트
- `id`: 청크 ID
- `node_id`: 원본 노드 ID
- `text`: 원본 코드 (raw_code) - 코드 뷰어용
- `attrs`: 메타데이터
  - `node_kind`, `node_name`: 노드 정보
  - `is_file_summary`: 파일 요약 청크 여부
  - `symbol_count`: 파일 내 심볼 개수 (파일 요약인 경우)
  - `search_text`: Step 9에서 생성

**참고**: CodeChunk.text는 항상 raw_code, 검색용 텍스트는 attrs["search_text"]에 별도 생성

**성능**: 1000개 노드 약 0.5초

### Step 8: 청크 저장 (ChunkStore)

**목적**: CodeChunk를 PostgreSQL에 영구 저장

**처리 흐름**:

```python
def save_chunks(chunks: list[CodeChunk]):
    # 배치 INSERT
    execute_batch(cur, """
        INSERT INTO code_chunks 
        (repo_id, chunk_id, node_id, file_path, span_start_line, span_end_line,
         language, text, attrs)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (repo_id, chunk_id) 
        DO UPDATE SET 
            text = EXCLUDED.text,
            attrs = EXCLUDED.attrs
    """, [
        (chunk.repo_id, chunk.id, chunk.node_id, chunk.file_path,
         chunk.span[0], chunk.span[2], chunk.language,
         chunk.text,  # raw_code 저장
         json.dumps(chunk.attrs))  # search_text는 attrs 안에 (Step 9에서 추가)
        for chunk in chunks
    ])
```

**DB 스키마**:
```sql
CREATE TABLE code_chunks (
    text TEXT NOT NULL,     -- 원본 코드 (raw_code) - 코드 뷰어용
    attrs JSONB,            -- search_text, metadata 등
    ...
)
```

**attrs JSONB 구조** (Step 9에서 확장됨):
```json
{
  "node_kind": "Function",
  "node_name": "verify_token",
  "is_file_summary": false,
  "search_text": "[META]...\n\ncode...",  // Step 9에서 추가
  "metadata": {...}                       // Step 9에서 추가
}
```

**인덱스**:
- `idx_chunks_location`: (repo_id, file_path, span_start_line, span_end_line)
- `idx_chunks_node`: (repo_id, node_id)

**성능**: 1000개 청크 약 0.5초

### Step 9: Chunk Tagging + search_text 생성

**목적**: 청크에 메타데이터 태깅 및 검색 최적화 텍스트 생성

**입력**: 
- CodeChunk 리스트 (DB에 저장된 상태, text = raw_code)
- FileProfile 리스트 (Step 5에서 생성)

**처리 흐름**:

#### 9-1. Chunk Tagging (ChunkTagger)

**역할**: 코드 내용 분석하여 의미 태그 추가

**처리**:
1. 정규식 기반 코드 스캔
   ```python
   metadata = {
       "has_docstring": '"""' in chunk.text or "'''" in chunk.text,
       "is_class_definition": re.search(r'^\s*class\s+\w+', chunk.text, MULTILINE),
       "is_function_definition": re.search(r'^\s*def\s+\w+', chunk.text, MULTILINE),
       "is_async": 'async def' in chunk.text,
   }
   ```

2. API endpoint 패턴 매칭 (file_profile.is_api_file인 경우만)
   ```python
   # FastAPI
   if '@app.post("/search")' in chunk.text:
       metadata["is_api_endpoint_chunk"] = True
       metadata["http_method"] = "POST"
       metadata["http_path"] = "/search"
       metadata["api_framework"] = file_profile.api_framework
   ```

3. 타입 분류
   - `is_test_case`: test_ 접두사, pytest/unittest 패턴
   - `is_schema_definition`: Schema, Model 클래스 감지

**출력 (chunk.attrs["metadata"])**:
```json
{
  "is_function_definition": true,
  "is_api_endpoint_chunk": true,
  "http_method": "POST",
  "http_path": "/hybrid/search",
  "api_framework": "fastapi",
  "has_async": true,
  "has_docstring": true,
  "line_count": 45
}
```

#### 9-2. SearchTextBuilder

**역할**: 검색 최적화 텍스트 생성 ([META] + raw_code)

**처리**:
1. [META] 헤더 생성
   ```python
   parts = []
   parts.append(f"[META] File: {chunk.file_path}")
   
   # FileProfile.roles 사용 (repo_profiler/file_profiler가 선행 분석)
   if file_profile:
       parts.append(f"[META] Role: {', '.join(file_profile.roles)}")
   
   # API 엔드포인트 정보
   if metadata.get("is_api_endpoint_chunk"):
       parts.append(f"[META] Endpoint: {metadata['http_method']} {metadata['http_path']}")
   
   parts.append(f"[META] Symbol: {chunk.attrs['node_kind']} {chunk.attrs['node_name']}")
   ```

2. 기능 키워드 추출 (15개 패턴 스캔)
   ```python
   features = []
   if 'auth' in chunk.text or 'token' in chunk.text:
       features.append('authentication')
   if 'select' in chunk.text or 'query(' in chunk.text:
       features.append('database access')
   if 'embedding' in chunk.text or 'vector' in chunk.text:
       features.append('vector operations')
   # ... 12가지 더
   
   parts.append(f"[META] Contains: {', '.join(features[:5])}")  # 최대 5개
   ```

3. [META] 길이 제한
   - 전체 토큰의 20% 이하로 제한
   - 초과 시 키워드 요약

4. 원본 코드와 결합
   ```python
   parts.append("")
   parts.append(chunk.text)  # raw_code
   search_text = "\n".join(parts)
   ```

**출력 (chunk.attrs["search_text"])**:
```
[META] File: auth/jwt_handler.py
[META] Role: Authentication, Service
[META] Symbol: Function verify_token
[META] Contains: authentication, database access

def verify_token(token: str) -> User:
    payload = jwt.decode(token, SECRET_KEY)
    user_id = payload.get("user_id")
    return User.query.get(user_id)
```

**주의**: chunk.text는 그대로 (raw_code), search_text는 attrs에 추가

**DB 업데이트**:
```python
# 배치 UPDATE
execute_batch(cur, """
    UPDATE code_chunks 
    SET attrs = jsonb_set(attrs, '{search_text}', %s::jsonb) ||
                jsonb_set(attrs, '{metadata}', %s::jsonb)
    WHERE repo_id = %s AND chunk_id = %s
""", [
    (json.dumps(search_text), json.dumps(metadata), repo_id, chunk_id)
    for chunk, metadata, search_text in zip(chunks, metadatas, search_texts)
])
```

**성능**: 1000개 청크 약 2초

### Step 10: Lexical 인덱싱 (Meilisearch/Zoekt)

**목적**: 텍스트 기반 검색 인덱스 구축 (BM25)

**입력**: CodeChunk 리스트 (attrs["search_text"] 포함)

**처리**:
```python
# Meilisearch 문서 생성
documents = []
for chunk in chunks:
    doc = {
        "id": f"{chunk.repo_id}:{chunk.id}",
        "text": chunk.attrs.get("search_text", chunk.text),  # search_text 우선
        "file_path": chunk.file_path,
        "language": chunk.language,
        "node_kind": chunk.attrs.get("node_kind"),
        **chunk.attrs,  # 모든 attrs 포함
    }
    documents.append(doc)

# 배치 인덱싱
index.add_documents(documents, batch_size=1000)
```

**검색 대상**: `attrs["search_text"]` ([META] + raw_code)
- [META] 섹션의 키워드로 BM25 점수 향상
- 원본 코드도 포함되어 코드 검색 가능

**성능**: 1000개 청크 약 1초

### Step 11: Chunk Semantic 인덱싱 (Qdrant)

**목적**: 청크 단위 시맨틱 인덱스 생성 (벡터 임베딩 생성 및 Qdrant 저장)

**처리 내용**:

#### 11-1. 임베딩 생성 (EmbeddingService)

**입력 텍스트 결정**:
```python
def get_embedding_input(chunk: CodeChunk) -> str:
    # 1. search_text 우선 (SearchTextBuilder 출력)
    search_text = chunk.attrs.get("search_text")
    if search_text:
        return str(search_text)
    
    # 2. Fallback: raw_code + docstring
    parts = [chunk.text]  # raw_code
    if chunk.attrs.get("docstring"):
        parts.append(f"# Purpose: {chunk.attrs['docstring']}")
    return "\n".join(parts)
```

**임베딩 정책**:
- 현재: `attrs["search_text"]` 전체 임베딩 ([META] + raw_code)
- 장점: 자연어 질문과 코드 모두에 반응
- [META] 섹션이 시맨틱 검색 컨텍스트 제공

**API 호출**:
- 모델: Mistral Codestral Embed (기본), OpenAI text-embedding-3-small/large
- 배치 처리:
  - Mistral: 350개 청크/배치 (토큰 제한 고려)
  - OpenAI: 200개 청크/배치
- 비동기 병렬: 최대 20개 배치 동시 실행

**토큰 제한 체크**:
```python
if count_tokens(embedding_input) > 15000:
    logger.warning(f"Skipped large chunk {chunk.id}")
    vectors.append(None)  # 스킵
```

**토큰 제한 정책**:
- 청킹 단계(Step 7)에서 이미 chunk.text 기준 15,000 토큰 이하로 분할됨
- [META] 섹션 추가로 search_text가 약간 증가하나, [META]는 전체의 20% 이하로 제한
- 따라서 search_text가 15,000 토큰을 넘는 경우는 드물며, 넘을 경우 해당 청크는 Chunk Semantic Index에서 제외됨
- 제외된 청크도 Lexical Index와 Graph 기반 검색은 여전히 가능

**중복 제거**: content_hash 기반 캐시 (선택)

#### 11-2. 임베딩 저장 (EmbeddingStore)

**Qdrant 업로드**:
```python
points = []
for chunk, vector in zip(chunks, vectors):
    if vector is None:
        continue  # 토큰 초과 청크 스킵
    
    point = PointStruct(
        id=f"{chunk.repo_id}:{chunk.id}",
        vector=vector,  # 1536 dim (Codestral)
        payload={
            "chunk_id": chunk.id,
            "file_path": chunk.file_path,
            "node_kind": chunk.attrs.get("node_kind"),
            "span_start_line": chunk.span[0],
        }
    )
    points.append(point)

client.upsert(collection_name=repo_id, points=points)
```

**Collection 설정**:
- Collection: {repo_id}
- Vector dim: 1536 (Codestral Embed)
- Distance: Cosine
- Payload: 필터링용 메타데이터

**향후 확장 (embedding_mode)**:
- `full`: search_text 전체 (현재)
- `code_only`: raw_code만
- `meta_plus_snippet`: [META] + 코드 일부
- `adaptive`: 청크 타입별 다른 전략

**성능**: 
- 1000개 청크 약 30초 (Mistral, 병렬)
- 캐시 히트율에 따라 단축 가능

### Step 12: Route 추출 및 저장 (API 엔드포인트)

**목적**: API 엔드포인트 정보 추출 및 인덱싱

**처리 내용**:
- API 파일만 처리 (FileProfile.is_api_file == True)
- RouteExtractor로 엔드포인트 추출:
  - FastAPI: @router.get("/path"), @app.post("/path")
  - Flask: @app.route("/path", methods=["GET"])
  - Express: app.get("/path", ...)
- RouteInfo 생성:
  - route_id: 고유 ID
  - repo_id: 저장소 ID
  - file_path: 파일 경로
  - handler_name: 핸들러 함수명
  - http_method: GET, POST 등
  - http_path: /api/search 등
  - framework: fastapi, flask 등
  - line_number: 라인 번호

**저장**: route_index 테이블

**성능**: 100개 API 파일 약 1초

### Step 13: Route Semantic 인덱싱 (semantic_nodes/pgvector)

**목적**: API 엔드포인트 단위 시맨틱 인덱스 생성 (Route 템플릿 summary + 임베딩)

**처리 내용**:
- Route 템플릿 summary 생성:
  ```
  POST /hybrid/search: hybrid_search in apps/api/routes/hybrid.py
  ```
- OpenAI 3-small 임베딩 생성
- SemanticNodeStore에 저장:
  - repo_id, node_id (route_id), node_type ("route")
  - summary: 템플릿 summary
  - summary_method: "template"
  - model: "openai-3-small"
  - embedding: 벡터
  - source_table: "route_index"
  - metadata: http_method, http_path 등

**저장**: semantic_nodes 테이블

**성능**: 100개 route 약 5초

### Step 14: Symbol Semantic 인덱싱 (semantic_nodes/pgvector)

**목적**: 함수/클래스 단위 시맨틱 인덱스 생성 (Symbol 템플릿 summary + 임베딩)

**처리 내용**:
- 인덱싱 대상 필터링:
  - Function/Class/Method만
  - Private 제외 (_로 시작, __init__ 제외)
  - 테스트 파일 제외 (선택적)
  - Migration 파일 제외
  - 상한선: 20,000개 (비용 제어)
- **Step 6에서 계산된 importance_score를 기준으로, 상위 N개 symbol만 Semantic 인덱싱 대상에 포함함** (Importance 순으로 정렬 후 상위만 선택)
- SymbolSummaryBuilder로 템플릿 summary 생성:
  ```
  Function verify_token in auth/jwt_handler.py
  JWT 토큰 검증 및 사용자 조회
  ```
- OpenAI 3-small 임베딩 생성
- SemanticNodeStore에 저장

**저장**: semantic_nodes 테이블

**성능**: 5000개 symbol 약 60초

### Step 15: 메타데이터 업데이트 및 완료

**목적**: 인덱싱 결과 요약 및 상태 업데이트

**처리 내용**:
- RepoMetadata 업데이트:
  - total_files: 파일 수
  - total_nodes: 노드 수
  - total_chunks: 청크 수
  - languages: 언어 목록
- 인덱싱 상태: completed, progress: 1.0
- indexed_at 타임스탬프 업데이트

**출력**: IndexingResult
- repo_id, status, total_files, processed_files
- total_nodes, total_edges, total_chunks
- duration_seconds, error_message

## 병렬 처리 전략

### 파일 파싱
- 임계값: 파일 5개 이상
- 방식: ProcessPoolExecutor (CPU 집약적)
- 워커 수: CPU 코어 수 (기본값)

### 임베딩 생성
- 방식: 비동기 병렬 (asyncio.gather)
- 동시 실행 수: 최대 20개 배치
- 배치 크기: 모델별 최적값 (Mistral: 350, OpenAI: 200)

## 캐싱 전략

### 파싱 캐시 (ParseCache)
- 위치: {cache_root}/parse_cache/{repo_id}/{file_hash}.json
- 키: (repo_id, file_abs_path)
- 값: (RawSymbol[], RawRelation[])
- 효과: 재인덱싱 시 파싱 단계만 캐시 사용

### 임베딩 저장소 및 캐시

**Chunk 임베딩**:
- 저장: Qdrant collection `{repo_id}`
- 캐시 키: content_hash (MD5)
- 캐시 위치: Qdrant 내부 (동일 content_hash 재사용)
- 특징: 고차원 벡터 검색 최적화

**Route/Symbol 임베딩**:
- 저장: `semantic_nodes` 테이블 (PostgreSQL + pgvector)
- 캐시: 현재 없음 (필요 시 추가 가능)
- 특징: 관계형 데이터와 결합, 트랜잭션 보장

**저장소 분리 이유**:
- Chunk: 대량 벡터 검색 (Qdrant 최적화)
- Route/Symbol: 소량 + 메타데이터 결합 쿼리 (Postgres 적합)

## 진행률 추적

인덱싱 진행률은 0.0 ~ 1.0 범위로 업데이트:

- 0.0: 시작
- 0.0 ~ 0.3: Step 1~4 (파일 스캔, Repo Profiling, 파싱, 그래프 저장) - 5% 단위 업데이트
- 0.3 ~ 0.5: Step 5~7 (File Profiling, Graph Ranking, Chunker) - 청킹 완료
- 0.5 ~ 0.7: Step 8~10 (청크 저장, Tagging/search_text, Lexical 인덱싱)
- 0.7: Lexical 인덱싱 완료
- 0.7 ~ 1.0: Step 11~14 (시맨틱 인덱싱 3종) - 5% 단위 업데이트
  - Step 11: Chunk Semantic 인덱싱 (Qdrant)
  - Step 13: Route Semantic 인덱싱 (semantic_nodes)
  - Step 14: Symbol Semantic 인덱싱 (semantic_nodes)
- 1.0: 완료

## 성능 특성

### 전체 인덱싱 시간 (예시: 1000개 파일, 5000개 노드)

| 단계 | 시간 | 비고 |
|------|------|------|
| 파일 스캔 | 0.5초 | |
| Repo Profiling | 2초 | |
| 파싱 + IR 변환 | 10초 | 병렬 처리 |
| 그래프 저장 | 0.5초 | |
| File Profiling | 3초 | |
| Graph Ranking | 1초 | |
| 청킹 | 0.5초 | |
| 청크 저장 | 0.5초 | |
| Chunk Tagging | 2초 | |
| Lexical 인덱싱 | 1초 | |
| 임베딩 생성 | 30초 | 병렬 처리, 캐시 없음 |
| Route 추출 | 1초 | |
| Route Semantic | 5초 | |
| Symbol Semantic | 60초 | |
| 합계 | ~118초 | 약 2분 |

### 메모리 사용량

- 1000개 노드: ~15MB
- 10000개 노드: ~150MB

## 에러 처리

- 파일 파싱 실패: 로깅 후 계속 진행
- 임베딩 생성 실패: 로깅 후 계속 진행
- 전체 실패: 상태를 failed로 업데이트, 에러 메시지 저장

## 재인덱싱

**기본 전략**:
- 파싱 결과만 캐시 사용 (파싱 단계만 스킵)
- 그래프 저장, 청킹, 임베딩은 매번 재수행
- ON CONFLICT → UPDATE로 기존 데이터 덮어쓰기

**삭제된 파일/노드/청크 정리 (Conservative 정책)**:
- 현재 버전에서는 삭제된 파일/노드/청크를 자동 제거하지 않는 conservative 전략을 사용함
- 운영 중 데이터 손실 리스크를 피하기 위해, GC는 별도의 배치 또는 관리 도구에서 수행하는 것을 전제로 함
- 필요 시 수동 정리:
  ```sql
  -- 특정 repo 전체 삭제
  DELETE FROM code_nodes WHERE repo_id = 'target-repo';
  DELETE FROM code_chunks WHERE repo_id = 'target-repo';
  DELETE FROM semantic_nodes WHERE repo_id = 'target-repo';
  ```

**향후 개선 옵션**:
- Aggressive 모드: 재인덱싱 시작 시 기존 데이터 truncate 후 재삽입
- Incremental 모드: 변경된 파일만 선택적 재인덱싱 + 삭제 파일 자동 제거

## 스토리지 최적화 포인트

**code_nodes.text와 code_chunks.text 중복**:
- 현재는 code_nodes.text와 code_chunks.text에 동일한 raw_code를 저장함
- 저장소가 커지면 텍스트 중복 비용이 클 수 있음
- 추후 스토리지 비용 최적화 시 code_nodes.text를 요약/부분 저장으로 줄일 수 있음
