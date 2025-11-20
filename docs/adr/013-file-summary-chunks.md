# ADR 013: 조건부 파일 요약 청크 생성

## 상태
승인됨 (2025-11-21)

## 컨텍스트

### 문제
기존 청킹 전략은 Function/Class/Method 같은 Symbol 노드만 청크로 변환하고, File 노드는 무조건 스킵했습니다.

**발견된 이슈:**
- "이 파일이 뭐 하는 파일인지" 같은 파일 레벨 질문에 답변 불가
- API 엔드포인트 전체 목록 조회 어려움
- 설정 파일(yaml, toml) 등은 Symbol이 없어 검색 불가

**기존 설계:**
```python
# File 노드는 스킵
if node.kind == "File":
    return []
```

### 검토한 대안

**1. 모든 File 노드를 청크로 변환**
- 장점: 모든 파일 검색 가능
- 단점: 
  - Symbol 청크와 내용 중복
  - 큰 파일은 토큰 제한 초과
  - 비용 증가 (임베딩/저장)

**2. File 노드는 계속 스킵**
- 장점: 비용 효율적, 중복 없음
- 단점: 파일 레벨 질문에 답변 불가

**3. 조건부 파일 요약 청크 생성 (선택)**
- 핵심/API/설정 파일만 선택적으로 요약 청크 생성
- 전체 코드가 아닌 메타데이터(파일 경로, docstring, 심볼 목록, API 엔드포인트)만 포함
- 비용 효율적이면서 검색 품질 향상

## 결정

**조건부 파일 요약 청크 생성 전략을 채택합니다.**

### 생성 조건

**1. 설정 파일: 무조건 생성**
```
.yaml, .yml, .toml, .json, .env, .ini, .conf
```

**2. 코드 파일: 심볼 N개 이상 (기본 5개)**
```
.py, .ts, .tsx, .js, .jsx, .java, .go, .rs
```
- 심볼이 많은 "허브 파일"은 파일 레벨 개요가 유용

**3. API 파일: 무조건 생성**
```
파일 경로에 'api' 또는 'route' 포함
```
- API 엔드포인트 목록 제공

### 요약 청크 내용

```
# File: src/indexer/pipeline.py
# Language: python

# File Description:
(파일 상단 docstring 또는 주석)

# Contains 14 symbols:

## Classes (2 total):
  - IndexingPipeline
  - ParallelConfig

## Functions (12 total):
  - index_repository
  - _parse_file
  ...

# API Endpoints:
  POST /hybrid/search -> hybrid_search
  GET /status -> get_status
```

### Config 파라미터

```python
chunker_enable_file_summary: bool = True
chunker_min_symbols_for_summary: int = 5
```

환경 변수:
```bash
CHUNKER_ENABLE_FILE_SUMMARY=true|false
CHUNKER_MIN_SYMBOLS_FOR_SUMMARY=5
```

## 구현

### 새 파일
- `src/chunking/file_summary_builder.py`: FileSummaryBuilder 클래스

### 수정 파일
- `src/chunking/chunker.py`: 조건부 요약 청크 생성 로직
- `src/core/config.py`: 새 파라미터 추가
- `src/core/bootstrap.py`: Chunker 초기화 시 파라미터 전달
- `src/indexer/pipeline.py`: source_files 수집 및 전달

### 추가 수정
- `src/indexer/pipeline.py` (L1173): 토큰 제한 7000 → 15000
  - Mistral Codestral Embed는 16K 토큰 지원
  - 큰 클래스(7050 토큰) 임베딩 스킵 이슈 해결

## 결과

### 청크 수 변화
- Before: 726개 청크
- After: 816개 청크
- 증가: +90개 파일 요약 청크 (~12%)

### 주요 파일 예시
```
src/indexer/pipeline.py: 14개 Symbol 청크 + 1개 요약 청크
apps/api/routes/hybrid.py: 12개 Symbol 청크 + 1개 요약 청크
src/core/ports.py: 30개 Symbol 청크 + 1개 요약 청크
```

### 검색 개선
- "인덱싱 파이프라인이 뭐 하는 파일인지" → 파일 요약 청크 검색 ✓
- "하이브리드 검색 API가 어디 있는지" → API 엔드포인트 정보 검색 ✓
- "config.yaml에 뭐 설정되어 있는지" → 설정 파일 청크 검색 ✓

## 영향

### 비용
- 임베딩: +90개 청크 (~12% 증가)
- 저장소: 요약 청크는 평균 500-1000자로 작아서 영향 미미

### 성능
- 청킹 시간: +5% 미만 (파일 docstring 추출 오버헤드)
- 검색 시간: 영향 없음 (벡터 검색은 청크 수에 로그 스케일)

### 검색 품질
- 파일 레벨 질문 recall 대폭 향상
- Symbol 레벨 질문은 기존과 동일

## 참고

### 관련 이슈
- "많은 파일이 노드는 있는데 청크가 0개" 문제 분석
- 7000 토큰 제한으로 인한 임베딩 스킵 이슈

### 관련 파일
- `src/chunking/chunker.py`
- `src/chunking/file_summary_builder.py`
- `docs/chunking-pipeline.md`

