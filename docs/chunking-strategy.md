# 코드 청킹 전략

## 개요

코드를 검색 가능한 청크로 분할하는 전략을 정의합니다.

**핵심 원칙:**
- 의미 단위 보존: 함수/클래스 경계 유지
- 검색 최적화: Symbol 청크 + 파일 요약 청크
- 크기 제어: 토큰 제한 준수 (15K 토큰)
- 컨텍스트 유지: 메타데이터 풍부화

## 1. 기본 전략: Symbol 기반 청킹

### 원칙

**1 Node = 1 Chunk**
- CodeNode(Function/Class/Method) → CodeChunk 1:1 매핑
- 파서가 추출한 AST 노드를 그대로 청크로 변환
- 가장 단순하고 빠름
- 의미 경계 자동 보장

### 노드 타입별 처리

```
File 노드: 조건부 파일 요약 청크 생성 (후술)
Class 노드: 1개 청크 (전체 클래스)
Function 노드: 1개 청크
Method 노드: 1개 청크
```

### 장점

1. **의미 완전성**
   - 함수/클래스가 중간에 잘리지 않음
   - 코드 단위가 명확

2. **파싱 일관성**
   - 파서 출력을 그대로 사용
   - 언어별 차이를 파서가 처리

3. **디버깅 용이**
   - Node ID와 Chunk ID 1:1 대응
   - 추적 간단

4. **속도**
   - 1000 노드 → 0.5초
   - 추가 분할/병합 로직 없음

### 단점 및 대응

**문제: 큰 클래스/함수**
- 클래스가 1000줄 넘어가면 토큰 제한 초과
- 대응: 토큰 수 체크 후 자동 분할

**문제: 작은 함수 여러 개**
- 2-3줄 함수가 개별 청크가 되면 컨텍스트 부족
- 대응: 파일 요약 청크로 보완 (후술)

## 2. 토큰 제한 및 분할

### 제한 정책

```python
MAX_TOKEN_LIMIT = 15000  # Mistral Codestral Embed 16K 제한
SAFE_CHAR_LIMIT = 45000  # 15000 * 3 (토큰 추정용)
```

**체크 순서:**
1. 문자 수 체크: `len(text) > 45000`인 경우만 정밀 토큰 카운팅
2. 토큰 카운팅: `tiktoken.encode(text)` 사용
3. 분할 여부 결정: `token_count > 15000`

### 분할 전략

**라인 기반 분할 + 오버랩**

```python
# 200줄 함수 예시
def large_function():
    # 0-100줄
    ...
    # 100-200줄
    ...

# 분할 결과
Chunk 1: 줄 0-105 (100 + 오버랩 5)
Chunk 2: 줄 95-200 (오버랩 5 + 100)
```

**오버랩 이유:**
- 청크 경계의 컨텍스트 보존
- 검색 누락 방지
- 기본 5줄 오버랩

**분할 알고리즘:**

```python
def split_node_by_tokens(node):
    chunks = []
    lines = node.text.split('\n')

    current_pos = 0
    while current_pos < len(lines):
        # 토큰 제한 내에서 최대한 많은 라인 포함
        end_pos = current_pos + 1
        while end_pos < len(lines):
            chunk_text = '\n'.join(lines[current_pos:end_pos+1])
            if count_tokens(chunk_text) > 15000:
                break
            end_pos += 1

        # 청크 생성
        chunk = create_chunk(lines[current_pos:end_pos])
        chunks.append(chunk)

        # 오버랩 적용
        current_pos = end_pos - 5  # 5줄 오버랩

    return chunks
```

### 실제 사례

**Codegraph 프로젝트:**
- 총 852개 노드
- 816개 청크 생성
- 분할된 청크: 2개 (큰 클래스 2개)
  - IndexingPipeline: 7050 토큰 → 2개 청크
  - HybridRetriever: 7052 토큰 → 2개 청크

## 3. 조건부 파일 요약 청크

### 목적

Symbol 청크만으로는 부족한 경우:
- "이 파일이 뭐 하는 파일인지"
- "API 엔드포인트 전체 목록"
- "설정 파일 내용"

### 생성 조건

**1. 설정 파일: 무조건 생성**
```python
file_ext in ['.yaml', '.yml', '.toml', '.json', '.env', '.ini', '.conf']
```
- Symbol이 없는 파일도 검색 가능
- 설정 값 전체를 하나의 청크로

**2. 코드 파일: 심볼 N개 이상**
```python
file_ext in ['.py', '.ts', '.tsx', '.js', '.jsx'] and symbol_count >= 5
```
- 심볼이 많은 "허브 파일"
- 파일 레벨 개요 제공

**3. API 파일: 무조건 생성**
```python
'api' in file_path or 'route' in file_path
```
- 엔드포인트 목록 제공
- API 문서 역할

### 파일 요약 청크 구조

```
# File: src/indexer/pipeline.py
# Language: python

# File Description:
인덱싱 파이프라인
(파일 상단 docstring)

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
- 메타데이터만 포함
- 평균 500-1000자
- 검색 품질 향상, 비용 효율적

### 효과 측정

**Codegraph 프로젝트:**
- Symbol 청크: 726개
- 파일 요약 청크: 90개 (+12%)
- 총 청크: 816개

**검색 개선:**
- 파일 레벨 질문 recall 대폭 향상
- "인덱싱 파이프라인 어디 있어?" → 파일 요약 청크 직접 히트
- API 엔드포인트 목록 조회 가능

## 4. 청크 크기 최적화

### 권장 크기

**라인 수:**
```
최소: 10줄 (너무 작으면 컨텍스트 부족)
최적: 20-100줄 (대부분의 함수/클래스)
최대: 15000 토큰 (분할 필요)
```

**토큰 수:**
```
평균: 185 토큰 (Codegraph 실측)
최소: 50 토큰
최대: 15000 토큰
```

### 언어별 특성

**Python:**
- 평균 함수: 20-50줄
- 평균 클래스: 100-300줄
- 큰 클래스: 500+ 줄 (분할 필요)

**TypeScript:**
- 평균 함수: 10-30줄
- React 컴포넌트: 50-150줄
- 큰 컴포넌트: 300+ 줄

### 크기 분포 (Codegraph)

```
816개 청크:
  < 50줄:    324개 (40%)  - 작은 함수
  50-100줄:  312개 (38%)  - 표준 함수/메서드
  100-300줄: 156개 (19%)  - 클래스
  > 300줄:    24개 (3%)   - 큰 클래스 (일부 분할됨)
```

## 5. 컨텍스트 유지 전략

### 메타데이터 풍부화

**Stage 1: Chunker - 구조 정보**
```python
attrs = {
    "node_kind": "Function",
    "node_name": "verify_token",
    "parent_id": "class-123",  # 소속 클래스
}
```

**Stage 2: ChunkTagger - 의미 태그**
```python
attrs["metadata"] = {
    "is_function_definition": True,
    "is_api_endpoint_chunk": True,
    "http_method": "POST",
    "http_path": "/auth/verify",
    "has_async": True,
}
```

**Stage 3: SearchTextBuilder - 검색 텍스트**
```python
attrs["search_text"] = """
[META] File: auth/handler.py
[META] Role: Authentication, API
[META] Endpoint: POST /auth/verify
[META] Symbol: Function verify_token
[META] Contains: authentication, database access

def verify_token(token: str):
    ...
"""
```

### [META] 섹션 설계

**목적:**
- 자연어 질문 recall 3-5배 향상
- BM25 검색 품질 개선
- 시맨틱 검색 컨텍스트 제공

**길이 제한:**
- 전체 토큰의 20% 이하
- 초과 시 키워드 요약

**예시:**

질문: "사용자 인증 로직 어디 있어?"

[META] 없을 때:
```python
def verify_token(token: str):
    payload = jwt.decode(token)
    ...
```
- "인증" 단어 없음 → 낮은 BM25 점수

[META] 있을 때:
```
[META] Contains: authentication, database access
def verify_token(token: str):
    ...
```
- "authentication" 명시 → 높은 BM25 점수

## 6. 특수 케이스 처리

### 빈 파일 / Symbol 없는 파일

**설정 파일:**
```
config.yaml (50줄)
→ 파일 요약 청크 1개 생성
→ 전체 내용 포함
```

**__init__.py:**
```python
# 비어있거나 import만
→ Symbol 청크 0개
→ 파일 요약 청크 생성 안 함 (의도적)
```

### 테스트 파일

```python
test_auth.py:
  - test_verify_token (Function 청크)
  - test_create_token (Function 청크)
  - test_invalid_token (Function 청크)
  → 각각 개별 청크
  → 파일 요약 청크도 생성 (심볼 5개 이상)
```

### API 라우터 파일

```python
routes/auth.py:
  - login (Function 청크 + API 태그)
  - logout (Function 청크 + API 태그)
  - refresh (Function 청크 + API 태그)
  → 각 엔드포인트 개별 청크
  → 파일 요약 청크 (엔드포인트 목록)
```

## 7. 검색 인덱스 전략

### Lexical 검색 (Meilisearch)

**입력:** `attrs["search_text"]`
- [META] 섹션으로 키워드 매칭 강화
- 원본 코드도 포함 (코드 검색)

**예시:**
```
질문: "POST /search API"
→ [META] Endpoint: POST /search 매칭
→ 높은 BM25 점수
```

### Semantic 검색 (Qdrant)

**입력:** `embedding(attrs["search_text"])`
- [META] + 코드 혼합 임베딩
- 자연어 질문에 강함

**예시:**
```
질문: "사용자 인증하는 함수"
→ [META] Contains: authentication
→ 높은 유사도
```

### Hybrid 검색

```
BM25(search_text) * 0.3 +
Semantic(embedding) * 0.7
→ 최종 점수
```

## 8. 성능 고려사항

### 청킹 속도

```
1000 노드:
- 파일별 그룹화: 0.1초
- Symbol 청크 생성: 0.3초 (병렬)
- 파일 요약 청크: 0.1초
- 합계: 0.5초
```

### 메모리 사용

```
1000 노드:
- CodeNode: 5MB
- CodeChunk (text + attrs): 10MB
- search_text 추가: +3MB
- 합계: 18MB
```

### 인덱싱 비용

```
816개 청크 (Codegraph):
- Meilisearch: ~2MB 인덱스
- Qdrant: ~10MB (1536 dim 임베딩)
- 임베딩 API 호출: 816회
```

## 9. 품질 체크리스트

### 좋은 청크

1. **적절한 크기**
   - 20-100줄
   - 100-500 토큰

2. **명확한 경계**
   - 함수/클래스 단위
   - 의미 완전성

3. **풍부한 메타데이터**
   - [META] 섹션
   - 태그 정보

4. **검색 가능성**
   - 자연어 키워드
   - 코드 패턴

### 나쁜 청크

1. **너무 큼**
   - 1000줄+
   - 10000 토큰+
   - 해결: 분할

2. **너무 작음**
   - 2-3줄
   - 컨텍스트 부족
   - 해결: 파일 요약 청크로 보완

3. **불완전**
   - 함수 중간에 잘림
   - 해결: Symbol 기반 청킹으로 방지

4. **메타데이터 부족**
   - [META] 없음
   - 해결: SearchTextBuilder 활성화

## 10. 향후 개선 방향

### 계층적 청킹 (미래)

```python
class UserService:
    def create_user(): ...
    def delete_user(): ...

→ 현재:
  - Class 청크 1개 (전체)
  - Method 청크 2개 (각각)

→ 미래 옵션:
  - Class 청크 1개 (개요용)
  - Method 청크 2개 (상세용)
```

### 적응적 분할 (미래)

```python
# 청크 타입별 다른 크기 제한
Function: max 10000 토큰
Class: max 15000 토큰
Test: max 5000 토큰 (짧게)
```

### 임베딩 최적화 (미래)

```python
# 현재: search_text 전체 임베딩
embedding(attrs["search_text"])

# 옵션 1: 코드만
embedding(chunk.text)

# 옵션 2: META + 일부
embedding([META] + chunk.text[:1000])

# 옵션 3: 적응적
if chunk.attrs["node_kind"] == "Class":
    embedding([META] + summary)
else:
    embedding(full_text)
```

## 11. 설정

### 환경 변수

```bash
# 기본 전략
CHUNKER_STRATEGY=node_based
CHUNKER_MAX_TOKENS=15000
CHUNKER_OVERLAP_LINES=5

# 파일 요약 청크
CHUNKER_ENABLE_FILE_SUMMARY=true
CHUNKER_MIN_SYMBOLS_FOR_SUMMARY=5
```

### 코드 설정

```python
from src.chunking.chunker import Chunker

chunker = Chunker(
    strategy="node_based",
    max_tokens=15000,
    overlap_lines=5,
    enable_file_summary=True,
    min_symbols_for_summary=5,
)
```

## 참고

- 구현: `src/chunking/chunker.py`
- 파일 요약: `src/chunking/file_summary_builder.py`
- 파이프라인: `docs/chunking-pipeline.md`
- ADR: `docs/adr/013-file-summary-chunks.md`
