# 리트리버 벤치마크

Semantica와 Cody(Sourcegraph) 리트리버 성능 비교 도구

## 빠른 시작 ⚡

### 1줄 실행

```bash
./benchmark
```

또는

```bash
python apps/benchmarks/run.py
```

그냥 실행하면 대화형으로 안내합니다!

### 사전 체크

```bash
python apps/benchmarks/check.py
```

환경이 제대로 설정되었는지 확인합니다.

## 간단한 사용법

```bash
# 기본 (대화형 - 가장 쉬움)
./benchmark

# 빠른 테스트 (5개 쿼리)
./benchmark quick

# 정확도 평가 (정답 데이터 사용)
./benchmark eval

# Cody와 비교
./benchmark cody

# 전체 벤치마크
./benchmark full
```

## 구조

```
apps/benchmarks/
├── run.py               # 🎯 간단한 실행 스크립트 (시작점)
├── check.py             # ✅ 사전 환경 체크
├── compare.py           # 고급 비교 스크립트
├── evaluators/
│   ├── metrics.py       # 평가 메트릭 (Precision, Recall, MRR)
│   ├── semantica.py     # Semantica 평가기
│   └── cody.py          # Cody 평가기
└── datasets/
    ├── semantica_queries.txt         # 이 프로젝트용 쿼리
    └── semantica_ground_truth.json   # 정답 데이터
```

## 상세 사용법

### 방법 1: 가장 쉬운 방법 (추천)

대화형 인터페이스:

```bash
./benchmark
```

또는

```bash
python apps/benchmarks/run.py
```

프롬프트가 안내합니다:
```
📁 저장소 설정
  저장소 ID (기본: semantica-codegraph): [엔터]

📝 테스트 쿼리 선택
  1. 기본 쿼리 5개 (빠름)
  2. 전체 쿼리 10개 (상세)
  3. 직접 입력
  선택: 1

[1/5] '설정 파일'
  ⏱️  145.3ms
    1. src/core/config.py
    2. src/core/bootstrap.py
    3. apps/api/main.py
```

### 방법 2: 명령어로 바로 실행

```bash
# 빠른 테스트
./benchmark quick

# 정확도 평가
./benchmark eval
```

### 방법 3: 고급 - 배치 모드

쿼리 파일과 정답 데이터로 자동 평가:

```bash
python -m apps.benchmarks.compare \
  --repo-id my-repo \
  --cody-repo github.com/owner/repo \
  --queries apps/benchmarks/datasets/example_queries.txt \
  --ground-truth apps/benchmarks/datasets/example_ground_truth.json \
  --k 5
```

출력 예시:
```
================================================================================
리트리버 성능 비교
================================================================================
저장소: my-repo
쿼리 수: 10
K: 5

[Semantica 결과]
Precision@K: 0.720
Recall@K:    0.650
MRR:         0.810
Avg Latency: 145.3ms
Total Queries: 10

[Cody 결과]
Precision@K: 0.680
Recall@K:    0.620
MRR:         0.750
Avg Latency: 234.5ms
Total Queries: 10

================================================================================
비교
================================================================================
메트릭               Semantica       Cody            차이
--------------------------------------------------------------------------------
Precision@K         0.720           0.680           +0.040
Recall@K            0.650           0.620           +0.030
MRR                 0.810           0.750           +0.060
Avg Latency (ms)    145.3           234.5           -89.2
```

## 환경 설정

### Semantica 설정

`.env` 파일에 설정:
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=7711
POSTGRES_USER=semantica
POSTGRES_PASSWORD=semantica
POSTGRES_DB=semantica_codegraph

MEILISEARCH_URL=http://localhost:7712
EMBEDDING_API_KEY=your-api-key
```

### Cody (Sourcegraph) 설정

Sourcegraph API 토큰 필요:

1. https://sourcegraph.com 로그인
2. Settings > Access tokens > Generate new token
3. 환경변수 설정:

```bash
export SOURCEGRAPH_TOKEN=sgp_xxxxxxxxxxxxx
```

또는 `.env` 파일에:
```bash
SOURCEGRAPH_TOKEN=sgp_xxxxxxxxxxxxx
```

## 정답 데이터 작성

`ground_truth.json` 형식:

```json
[
  {
    "query": "검색 쿼리",
    "relevant_items": [
      "관련있는/파일/경로1.py",
      "관련있는/파일/경로2.py"
    ]
  }
]
```

작성 팁:
1. 각 쿼리당 2-5개 정도의 관련 파일 선정
2. 실제 개발자가 찾으려는 코드를 기준으로
3. 너무 많으면 Recall이 낮아짐

## 평가 메트릭

### Precision@K
상위 K개 결과 중 관련있는 결과의 비율
- 1.0 = 모든 결과가 관련있음
- 0.5 = 절반만 관련있음

### Recall@K
전체 관련 결과 중 상위 K개에 포함된 비율
- 1.0 = 모든 관련 결과를 찾음
- 0.5 = 절반만 찾음

### MRR (Mean Reciprocal Rank)
첫 번째 관련 결과의 순위의 역수
- 1.0 = 첫 결과가 관련있음
- 0.5 = 두 번째 결과가 첫 관련 결과
- 0.333 = 세 번째 결과가 첫 관련 결과

### Latency
검색 응답 시간 (밀리초)

## 확장하기

### 커스텀 평가기 추가

`apps/benchmarks/evaluators/my_retriever.py`:

```python
from .metrics import SearchResult, time_search

class MyRetrieverEvaluator:
    def search(self, query: str, k: int = 5) -> SearchResult:
        results, latency_ms = time_search(my_search_func, query, k=k)
        return SearchResult(
            query=query,
            results=results,
            latency_ms=latency_ms,
            retriever_name="my-retriever"
        )
```

### 새 메트릭 추가

`apps/benchmarks/evaluators/metrics.py`의 `MetricsCalculator`에 메서드 추가:

```python
@staticmethod
def my_metric(result: SearchResult, ground_truth: GroundTruth) -> float:
    # 계산 로직
    return score
```

## 다음 단계

1. CodeSearchNet 데이터셋 통합
2. 자동 정답 생성 (LLM 활용)
3. 시각화 (그래프, 차트)
4. CI/CD 통합 (성능 회귀 감지)
5. A/B 테스트 프레임워크
