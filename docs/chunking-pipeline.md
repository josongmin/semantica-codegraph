# Codegraph 청킹 파이프라인

## 개요

이 문서는 소스 코드를 검색 가능한 청크로 변환하는 전체 파이프라인을 설명합니다.

**핵심 설계:**
- `CodeChunk.text` = 원본 코드 (raw_code)
- `attrs["search_text"]` = 검색 최적화 텍스트 ([META] + raw_code)
- Meilisearch/Qdrant는 `search_text` 우선 사용

## 1. High-level Overview

```
소스 코드 파일 (Python, TypeScript)
  ↓
Parser (tree-sitter)
  ↓
CodeNode 리스트 (AST 노드)
  ↓
[Stage 1] Chunker: Node → Chunk 변환
  입력: CodeNode 리스트 (Function, Class, Method, File 등)
  처리:
    1. 파일별로 노드 그룹화
    2. Symbol 노드 → 1개 청크 생성 (토큰 체크 후 필요시 분할)
    3. File 노드 → 조건 확인 후 파일 요약 청크 생성
  출력: CodeChunk 리스트 (text, span, attrs 포함)
  ↓
[Stage 2] ChunkTagger: 메타데이터 태깅
  입력: CodeChunk + FileProfile
  처리:
    1. 청크 내용 정규식 스캔
    2. API endpoint 패턴 매칭 (@app.post, @router.get 등)
    3. 타입 분류 (class, function, test, schema 등)
  출력: 메타데이터 dict (attrs에 추가)
  ↓
[Stage 3] SearchTextBuilder: 검색용 텍스트 생성
  입력: CodeChunk + FileProfile + 메타데이터
  처리:
    1. [META] 헤더 생성 (파일 경로, 역할, 심볼 정보)
    2. 기능 키워드 추출 (코드 스캔)
    3. API 엔드포인트 정보 추가
    4. 원본 코드와 결합
  출력: search_text (chunk.attrs에 저장)
  ↓
[Stage 4] ChunkStore: PostgreSQL 저장
  입력: CodeChunk 리스트
  처리:
    1. 배치 INSERT (execute_batch)
    2. ON CONFLICT → UPDATE
    3. 인덱스 자동 업데이트
  출력: DB에 영구 저장
  ↓
검색 인덱스 (Lexical, Semantic)
  - Meilisearch: search_text 인덱싱
  - Qdrant: embedding 벡터 저장
```

## 2. 파이프라인 Stage별 상세

### 데이터 필드 정의

**CodeChunk 필드:**
```python
CodeChunk(
    text: str           # 원본 코드 (raw_code) - 코드 뷰어용
    attrs: dict         # 메타데이터
      - "search_text"   # 검색 최적화 텍스트 ([META] + raw_code)
      - "metadata"      # ChunkTagger 출력 (is_api_endpoint_chunk 등)
      - "node_kind"     # Function, Class, Method 등
      - "node_name"     # 심볼 이름
)
```

**검색 인덱스 입력:**
- Meilisearch: `attrs["search_text"]` 우선, 없으면 `text`
- Qdrant: `embedding(attrs["search_text"])` 우선, 없으면 `embedding(text)`

### 입력 데이터 흐름

**1. 파서 출력 (CodeNode)**
```python
# 예시: auth/jwt_handler.py 파일 파싱 결과
nodes = [
    CodeNode(id="file-1", kind="File", name="auth/jwt_handler.py", text="전체파일내용..."),
    CodeNode(id="class-1", kind="Class", name="JWTHandler", text="class JWTHandler:\n..."),
    CodeNode(id="func-1", kind="Function", name="verify_token", text="def verify_token():\n..."),
    CodeNode(id="func-2", kind="Function", name="create_token", text="def create_token():\n..."),
]
```

**2. Stage 1 → Chunker 처리**
```python
# 파일별로 그룹화
files = {
    "auth/jwt_handler.py": {
        "file_node": CodeNode(kind="File", ...),
        "symbol_nodes": [
            CodeNode(kind="Class", ...),
            CodeNode(kind="Function", ...),
            CodeNode(kind="Function", ...),
        ]
    }
}

# Symbol 노드 → 청크 변환
for node in symbol_nodes:
    if count_tokens(node.text) > 15000:
        chunks.extend(split_node(node))  # 분할
    else:
        chunks.append(node_to_chunk(node))  # 그대로

# File 노드 → 조건부 요약 청크
if should_create_summary(file_node, symbol_nodes):
    summary_chunk = build_file_summary(file_node, symbol_nodes)
    chunks.append(summary_chunk)
```

**3. Stage 2 → ChunkTagger 처리**

입력: CodeChunk + FileProfile

FileProfile은 repo_profiler/file_profiler가 선행 분석:
- `is_api_file`: API 라우터 파일 여부
- `roles`: ["API", "Service", "Model"] 등
- `api_framework`: "fastapi", "flask" 등

```python
for chunk in chunks:
    # file_profile 조회
    file_profile = file_profile_map.get(chunk.file_path)

    # 코드 스캔
    metadata = {
        "has_docstring": '"""' in chunk.text,
        "is_class_definition": re.search(r'class\s+\w+', chunk.text),
        "is_async": 'async def' in chunk.text,
    }

    # API endpoint 감지 (file_profile.is_api_file == True인 경우만)
    if file_profile and file_profile.is_api_file:
        if '@app.post("/search")' in chunk.text:
            metadata["is_api_endpoint_chunk"] = True
            metadata["http_method"] = "POST"
            metadata["http_path"] = "/search"

    chunk.attrs["metadata"] = metadata
```

**4. Stage 3 → SearchTextBuilder 처리**

입력: CodeChunk + FileProfile + 메타데이터

**기능 키워드 추출 정책:**
- 패턴 기반: 15개 카테고리 (authentication, database access, HTTP client 등)
- 도메인 특화: 낮은 빈도 키워드 우선 포함 (예: reranker, qdrant)
- 목표: 자연어 질문 recall 3-5배 향상

**[META] 길이 제한:**
- META 섹션은 전체 토큰의 20% 이하로 제한
- 초과 시 키워드 요약

```python
def build_search_text(chunk, file_profile, metadata):
    parts = []

    # [META] 섹션
    parts.append(f"[META] File: {chunk.file_path}")

    # FileProfile.roles 사용
    if file_profile:
        roles = file_profile.roles  # ["API", "Service"]
        parts.append(f"[META] Role: {', '.join(roles)}")

    if metadata.get("is_api_endpoint_chunk"):
        parts.append(f"[META] Endpoint: {metadata['http_method']} {metadata['http_path']}")

    parts.append(f"[META] Symbol: {chunk.attrs['node_kind']} {chunk.attrs['node_name']}")

    # 기능 키워드 추출 (15개 패턴 스캔)
    features = extract_features(chunk.text, file_profile)
    # 예: ["authentication", "database access", "error handling"]
    parts.append(f"[META] Contains: {', '.join(features[:5])}")  # 최대 5개

    # 원본 코드
    parts.append("")
    parts.append(chunk.text)  # raw_code 그대로

    search_text = "\n".join(parts)

    # 길이 제한 체크 (META는 전체의 20% 이하)
    meta_tokens = count_tokens(parts[:-2])  # META 섹션만
    total_tokens = count_tokens(search_text)
    if meta_tokens > total_tokens * 0.2:
        # META 압축 (키워드 요약)
        parts = compress_meta(parts)

    return "\n".join(parts)

# 결과: chunk.text는 그대로, attrs["search_text"]에 저장
chunk.attrs["search_text"] = """
[META] File: auth/jwt_handler.py
[META] Role: Authentication, Service
[META] Symbol: Function verify_token
[META] Contains: authentication, database access

def verify_token(token: str) -> User:
    ...
"""
```

**5. Stage 4 → ChunkStore 저장**

**DB 스키마:**
```sql
CREATE TABLE code_chunks (
    text TEXT NOT NULL,     -- 원본 코드 (raw_code)
    attrs JSONB,            -- search_text, metadata 등
    ...
)
```

**저장 로직:**
```python
def save_chunks(chunks):
    conn = get_connection()
    with conn.cursor() as cur:
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
             json.dumps(chunk.attrs))  # search_text는 attrs 안에
            for chunk in chunks
        ])
    conn.commit()
```

**검색 인덱스:**
```python
# Meilisearch
documents = [{
    "text": chunk.attrs.get("search_text", chunk.text),  # search_text 우선
    ...
}]

# Qdrant (Embedding)
embedding_input = chunk.attrs.get("search_text")  # search_text 우선
if not embedding_input:
    embedding_input = chunk.text  # fallback: raw_code
```

### 데이터 변환 전체 과정

```python
# 입력: CodeNode
node = CodeNode(
    kind="Function",
    name="verify_token",
    text="def verify_token(token: str):\n    return jwt.decode(token)"
)

# Stage 1: Chunker
chunk = CodeChunk(
    node_id=node.id,
    text=node.text,  # raw_code 유지
    attrs={
        "node_kind": "Function",
        "node_name": "verify_token"
    }
)

# Stage 2: ChunkTagger
chunk.attrs["metadata"] = {
    "is_function_definition": True,
    "has_docstring": False,
    "is_async": False,
}

# Stage 3: SearchTextBuilder
chunk.attrs["search_text"] = """
[META] File: auth/jwt_handler.py
[META] Symbol: Function verify_token
[META] Contains: authentication

def verify_token(token: str):
    return jwt.decode(token)
"""
# 주의: chunk.text는 그대로 (raw_code)

# Stage 4: ChunkStore → PostgreSQL
INSERT INTO code_chunks (text, attrs) VALUES (
    chunk.text,  # raw_code
    json.dumps(chunk.attrs)  # search_text는 attrs 안에
)

# 최종: 검색 인덱스
# Meilisearch: attrs["search_text"] 인덱싱
# Qdrant: embedding(attrs["search_text"]) 저장
```

**임베딩 입력 정책:**
- 현재: `attrs["search_text"]` 전체 ([META] + raw_code)
- 향후 확장 옵션 (embedding_mode):
  - `code_only`: raw_code만 임베딩
  - `meta_plus_snippet`: [META] + 코드 일부
  - `adaptive`: 청크 타입별 다른 전략

### Stage 1: Chunker (Node → Chunk)

### 입력: CodeNode

```python
CodeNode(
    id="node-123",
    kind="Function",
    name="verify_jwt_token",
    file_path="auth/jwt_handler.py",
    span=(10, 0, 25, 4),  # (start_line, start_col, end_line, end_col)
    text="""def verify_jwt_token(token: str) -> User:
    \"\"\"JWT 토큰 검증\"\"\"
    try:
        payload = jwt.decode(token, SECRET_KEY)
        user_id = payload.get("user_id")
        return User.query.get(user_id)
    except JWTError:
        raise AuthenticationError("Invalid token")
""",
    language="python",
    parent_id="node-122",  # Class or File
    children=[],
)
```

### 청킹 전략

#### 1. Node-based (기본)
```python
# Symbol 노드: 1 Node = 1 Chunk
# File 노드: 조건부 파일 요약 청크 생성
# 가장 단순하고 빠름
# 대부분의 경우 최적

strategy = "node_based"
max_tokens = 15000  # Codestral Embed 16K 제한 (안전 마진 1K)
enable_file_summary = True  # 파일 요약 청크 생성
min_symbols_for_summary = 5  # 최소 심볼 개수
```

**동작:**
1. Symbol 노드 (Function/Class/Method): 1개 청크로 변환
2. File 노드: 조건부 파일 요약 청크 생성
   - 설정 파일 (.yaml, .toml 등) → 무조건 생성
   - 코드 파일 → 심볼 5개 이상 시 생성
   - API 파일 → 무조건 생성
3. 토큰 수 체크 (max_tokens 초과 시 분할)

```python
def _node_to_chunk(node: CodeNode) -> CodeChunk:
    return CodeChunk(
        id=f"chunk-{node.id}",
        node_id=node.id,
        file_path=node.file_path,
        span=node.span,
        text=node.text,
        language=node.language,
        token_count=count_tokens(node.text),  # tiktoken
        attrs={
            "node_kind": node.kind,
            "node_name": node.name,
            "parent_id": node.parent_id,
        }
    )
```

#### 2. Size-based (크기 기반 분할)
```python
# 큰 노드를 max_lines 기준으로 분할
strategy = "size_based"
max_lines = 100
overlap_lines = 5
```

**예시: 200줄 함수**
```python
# 원본 함수 (200줄)
def large_function():
    # 0-100줄
    ...
    # 100-200줄
    ...

# → 2개 청크로 분할
Chunk 1: 줄 0-105 (오버랩 5줄)
Chunk 2: 줄 95-200 (오버랩 5줄)
```

**오버랩 이유:**
- 청크 경계의 컨텍스트 보존
- 검색 누락 방지

#### 3. Hierarchical (계층적)
```python
# Class → 1개 청크 + 각 Method → 개별 청크
strategy = "hierarchical"
```

**예시: Class**
```python
# 원본
class UserService:
    def create_user(self): ...
    def delete_user(self): ...
    def update_user(self): ...

# → 4개 청크
Chunk 1: 전체 클래스 (개요)
Chunk 2: create_user (상세)
Chunk 3: delete_user (상세)
Chunk 4: update_user (상세)
```

### 파일 요약 청크 생성 (NEW)

**목적:**
- "이 파일이 뭐 하는 파일인지" 질문에 답변
- API 엔드포인트 전체 목록 제공
- 설정 파일 검색 가능

**생성 조건:**

```python
# 1. 설정 파일: 무조건 생성
file_ext in ['.yaml', '.yml', '.toml', '.json', '.env', '.ini', '.conf']

# 2. 코드 파일: 심볼 N개 이상
file_ext in ['.py', '.ts', '.js'] and symbol_count >= 5

# 3. API 파일: 무조건 생성
'api' in file_path or 'route' in file_path
```

**파일 요약 청크 내용:**

```
# File: src/indexer/pipeline.py
# Language: python

# File Description:
인덱싱 파이프라인

# Contains 14 symbols:

## Classes (2 total):
  - IndexingPipeline
  - ParallelConfig

## Functions (12 total):
  - index_repository
  - _parse_file
  - _build_graph
  ...

# API Endpoints:
  POST /hybrid/search -> hybrid_search
  GET /status -> get_status
```

**효과:**
- 청크 수: +12% (코드그래프: 726→816개)
- 파일 레벨 질문 recall 대폭 향상
- Symbol 청크와 중복 최소화

### 토큰 카운팅 (tiktoken)

```python
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(encoder.encode(text))

# 예시
code = "def hello(): print('world')"
tokens = count_tokens(code)  # ~10 tokens
```

**토큰 제한:**
- Mistral Codestral Embed: 16,384 토큰
- 설정: 15,000 토큰 (안전 마진 1K)
- 초과 시 자동 분할 (7050 토큰 큰 클래스도 처리 가능)

### 병렬 처리

```python
# 100개 이상 노드 → 병렬 처리
with ThreadPoolExecutor(max_workers=4) as executor:
    chunks = executor.map(process_node, nodes)

# 속도: 1000 노드 → 2초 (순차) vs 0.5초 (병렬)
```

### 출력: CodeChunk

```python
CodeChunk(
    id="chunk-node-123",
    node_id="node-123",
    file_path="auth/jwt_handler.py",
    span=(10, 0, 25, 4),
    text="def verify_jwt_token(token: str) -> User:\n...",  # raw_code
    language="python",
    attrs={
        "node_kind": "Function",
        "node_name": "verify_jwt_token",
        "parent_id": "node-122",
        # "search_text"는 Stage 3에서 추가됨
    }
)
```

### Stage 2: ChunkTagger (메타데이터 태깅)

### 역할

- API endpoint 감지
- 청크 타입 분류
- HTTP method/path 추출

### 태그 종류

```python
metadata = {
    # 기본 태그
    "is_class_definition": True/False,
    "is_function_definition": True/False,
    "has_docstring": True/False,

    # 특수 태그
    "is_api_endpoint_chunk": True/False,
    "is_test_case": True/False,
    "is_schema_definition": True/False,

    # API 정보 (endpoint인 경우)
    "http_method": "POST",
    "http_path": "/api/search",

    # 비동기
    "has_async": True/False,
}
```

### API Endpoint 감지 예시

```python
# FastAPI 예시
@app.post("/hybrid/search")
async def hybrid_search(request: HybridSearchRequest):
    ...

# → 태깅
{
    "is_api_endpoint_chunk": True,
    "http_method": "POST",
    "http_path": "/hybrid/search",
    "has_async": True,
}
```

**지원 프레임워크:**
- FastAPI: `@app.get`, `@router.post`
- Flask: `@app.route`
- Express: `app.get(`, `router.post(`

### Stage 3: SearchTextBuilder (검색 최적화)

### 목표

자연어 질문의 recall을 **3-5배** 향상

### 변환 예시

**원본 코드:**
```python
async def hybrid_search(request: HybridSearchRequest):
    retriever = HybridRetriever(...)
    candidates = retriever.retrieve(
        query=request.query,
        k=request.k
    )
    return {"results": candidates}
```

**검색용 텍스트:**
```
[META] File: apps/api/routes/hybrid.py
[META] Role: API, Router
[META] Endpoint: POST /hybrid/search
[META] Symbol: Function hybrid_search
[META] Contains: search logic, database access, async

async def hybrid_search(request: HybridSearchRequest):
    retriever = HybridRetriever(...)
    candidates = retriever.retrieve(
        query=request.query,
        k=request.k
    )
    return {"results": candidates}
```

### [META] 섹션 구성

| 필드 | 설명 | 예시 |
|------|------|------|
| File | 파일 경로 | `routes/hybrid.py` |
| Role | 파일 역할 | `API, Router, Service` |
| Endpoint | HTTP endpoint | `POST /hybrid/search` |
| Symbol | 심볼 정보 | `Function hybrid_search` |
| Contains | 기능 키워드 | `search logic, database access` |
| Type | 특수 타입 | `Test`, `Schema/Model` |

### 기능 키워드 추출

코드에서 자동으로 키워드를 추출:

```python
# 1. DB 접근
"select", "insert", "query(" → "database access"

# 2. 인증/보안
"auth", "token", "jwt" → "authentication"

# 3. 검색
"search", "retrieve" → "search logic"

# 4. 벡터
"embedding", "vector" → "vector operations"

# 5. HTTP 클라이언트
"requests.", "httpx." → "HTTP client"

# 6. 비동기
"async", "await" → "async"

# 7. 에러 처리
"try:", "except" → "error handling"

# ... 15가지 패턴
```

### 검색 개선 효과

**질문: "사용자 인증 로직이 어디 있어?"**

❌ **[META] 없을 때:**
```
def verify_jwt_token(token: str):
    payload = jwt.decode(token, SECRET_KEY)
    ...
```
- "인증" 단어 없음 → 낮은 BM25 점수
- Semantic만 의존

✅ **[META] 있을 때:**
```
[META] Contains: authentication, database access
[META] Symbol: Function verify_jwt_token

def verify_jwt_token(token: str):
    payload = jwt.decode(token, SECRET_KEY)
    ...
```
- "authentication" 키워드 명시 → 높은 BM25 점수
- Lexical + Semantic 모두 매칭

**결과: Recall 3-5배 향상**

### Stage 4: ChunkStore (PostgreSQL 저장)

#### 테이블 스키마

```sql
CREATE TABLE code_chunks (
    repo_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    span_start_line INTEGER NOT NULL,
    span_start_col INTEGER NOT NULL,
    span_end_line INTEGER NOT NULL,
    span_end_col INTEGER NOT NULL,
    language TEXT NOT NULL,
    text TEXT NOT NULL,  -- 원본 코드 (raw_code) - 코드 뷰어용
    token_count INTEGER,
    attrs JSONB,         -- search_text, metadata 등
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (repo_id, chunk_id)
);
```

**attrs JSONB 구조:**
```json
{
  "search_text": "[META] File: ...\n\ndef func()...",
  "metadata": {
    "is_api_endpoint_chunk": true,
    "http_method": "POST",
    "http_path": "/search"
  },
  "node_kind": "Function",
  "node_name": "search_handler"
}
```

### 인덱스

```sql
-- 1. 위치 기반 조회 (Zoekt 매핑에 필수)
CREATE INDEX idx_chunks_location
ON code_chunks(repo_id, file_path, span_start_line, span_end_line);

-- 2. Node 기반 조회
CREATE INDEX idx_chunks_node
ON code_chunks(repo_id, node_id);
```

**사용 예시:**
```python
# Zoekt가 file:line 반환 → Chunk 조회
chunk = chunk_store.get_chunk_by_location(
    repo_id="my-repo",
    file_path="auth/jwt.py",
    line=15
)
```

### 저장 프로세스

```python
def save_chunks(chunks: list[CodeChunk]):
    # 배치 저장 (execute_batch)
    conn = get_connection()
    with conn.cursor() as cur:
        execute_batch(
            cur,
            """
            INSERT INTO code_chunks
            (repo_id, chunk_id, node_id, ..., text)
            VALUES (%s, %s, %s, ..., %s)
            ON CONFLICT (repo_id, chunk_id)
            DO UPDATE SET text = EXCLUDED.text
            """,
            [(chunk.repo_id, chunk.id, ...) for chunk in chunks]
        )
    conn.commit()

# 속도: 1000 청크 → ~0.5초
```

## 3. 성능 특성 및 튜닝

### 전체 성능

### 인덱싱 속도

| 단계 | 1000 노드 | 10000 노드 |
|------|----------|-----------|
| Parsing | 1s | 10s |
| **Chunking** | **0.5s** | **5s** |
| Tagging | 0.3s | 3s |
| Search Text | 0.2s | 2s |
| DB 저장 | 0.5s | 5s |
| **합계** | **2.5s** | **25s** |

### 메모리 사용

```
1000 노드 (평균 50줄/노드):
- CodeNode: ~5MB
- CodeChunk: ~10MB (검색 텍스트 포함)
- 최대: ~15MB

10000 노드:
- 최대: ~150MB
```

### 청크 통계 (실제 프로젝트)

```
Codegraph 프로젝트 (2025-11-21 기준):
- 파일: 121개
- CodeNode: 852개
- CodeChunk: 816개
  - Symbol 청크: 726개 (1 Node = 1 Chunk)
  - 파일 요약 청크: 90개 (조건부 생성)
- 평균 청크 크기: 42줄
- 평균 토큰 수: 185 토큰
- 최대 청크: 7050 토큰 (큰 클래스, 분할됨)

주요 파일:
- src/core/ports.py: 31개 청크 (30 Symbol + 1 File Summary)
- src/core/bootstrap.py: 30개 청크 (29 Symbol + 1 File Summary)
- src/indexer/pipeline.py: 15개 청크 (14 Symbol + 1 File Summary)
```

### 설정 튜닝

### 청킹 전략 선택

```python
# 추천: node_based (기본)
strategy = "node_based"
max_tokens = 7000

# 큰 함수가 많은 경우
strategy = "size_based"
max_lines = 100
overlap_lines = 5

# 클래스 중심 프로젝트
strategy = "hierarchical"
```

### 토큰 제한

```python
# Mistral Codestral Embed (16K)
max_tokens = 15000  # 안전 마진 1K

# OpenAI text-embedding-3-* (8K)
max_tokens = 7000

# 로컬 모델 (512)
max_tokens = 400
```

### 파일 요약 청크

```python
# 활성화 (권장)
enable_file_summary = True
min_symbols_for_summary = 5

# 비활성화
enable_file_summary = False

# 환경 변수
export CHUNKER_ENABLE_FILE_SUMMARY=true
export CHUNKER_MIN_SYMBOLS_FOR_SUMMARY=5
```

### 메타데이터 활성화

```python
# 검색 품질 향상 (추천)
enable_search_text = True
enable_chunk_tagging = True

# 속도 우선 (비추천)
enable_search_text = False
```

### 청크 품질 체크

### 좋은 청크 예시

✅ **적절한 크기 (20-100줄)**
```python
def authenticate_user(username: str, password: str) -> User:
    """사용자 인증"""
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        return user
    raise AuthenticationError()
```

✅ **명확한 경계 (함수/클래스 단위)**

✅ **충분한 컨텍스트 ([META] 포함)**

### 나쁜 청크 예시

❌ **너무 큰 청크 (1000줄+)**
- 임베딩 품질 저하
- 검색 정확도 감소

❌ **불완전한 청크 (중간에 잘림)**
```python
# 함수 시작부만
def large_function():
    # 처음 50줄만...
```

❌ **컨텍스트 없음 (메타데이터 누락)**

## 4. 참고

### 소스 코드
- Chunker: `src/chunking/chunker.py`
- FileSummaryBuilder: `src/chunking/file_summary_builder.py` (NEW)
- ChunkTagger: `src/chunking/chunk_tagger.py`
- SearchTextBuilder: `src/chunking/search_text_builder.py`
- ChunkStore: `src/chunking/store.py`

### 문서
- ADR: `docs/adr/013-file-summary-chunks.md`
- RAPTOR Chunking: `docs/adr/adr_0007_raptor-chunking.md`
