# 벤치마크 사용법 - 완전 가이드

## 🎯 목표

Semantica 리트리버의 성능을 측정하고 Cody와 비교

## ⚡ 가장 빠른 시작 (3단계)

### 1단계: 환경 확인
```bash
python benchmarks/check.py
```

모든 항목이 ✅ 면 OK!

### 2단계: 벤치마크 실행
```bash
./benchmark
```

또는

```bash
python benchmarks/run.py
```

### 3단계: 결과 확인

자동으로 출력됩니다!

## 📋 명령어 치트시트

```bash
# 환경 체크
python benchmarks/check.py

# 기본 실행 (대화형)
./benchmark

# 빠른 테스트 (5개 쿼리)
./benchmark quick

# 정확도 평가 (10개 쿼리 + 정답 데이터)
./benchmark eval

# Cody와 비교
./benchmark cody

# 전체 벤치마크
./benchmark full

# 도움말
./benchmark help
```

## 📝 실행 시나리오

### 시나리오 A: 첫 실행 (초보자)

```bash
# 1. 환경 체크
python benchmarks/check.py
# → 문제 있으면 안내에 따라 해결

# 2. 실행
./benchmark
# → 프롬프트에서 엔터만 연속으로 치면 됨

# 3. 결과 확인
# → 평균 응답 속도, 검색 결과 확인
```

**예상 시간**: 1분

### 시나리오 B: 정확도 평가 (중급자)

```bash
# 1. 정확도 평가 실행
./benchmark eval

# 2. 메트릭 확인
# - Precision@5: 0.780
# - Recall@5: 0.720
# - MRR: 0.850
# - Latency: 156ms

# 3. 개선점 파악
# → 어떤 쿼리에서 낮은 점수?
```

**예상 시간**: 2분

### 시나리오 C: Cody 비교 (고급자)

```bash
# 1. Sourcegraph 토큰 설정
export SOURCEGRAPH_TOKEN=sgp_xxxxx

# 2. 비교 실행
./benchmark cody
# 또는 전체 비교
./benchmark full

# 3. 결과 비교
# Semantica vs Cody
# → 어느 쪽이 더 나은가?
```

**예상 시간**: 5분 (API 호출 포함)

### 시나리오 D: 커스텀 평가 (전문가)

```bash
# 1. 쿼리 파일 작성
cat > my_queries.txt << EOF
사용자 인증 로직
데이터베이스 연결
API 엔드포인트
EOF

# 2. 정답 데이터 작성 (JSON)
# ... (예제 참조)

# 3. 실행
python -m benchmarks.compare \
  --repo-id my-project \
  --queries my_queries.txt \
  --ground-truth my_ground_truth.json \
  --k 5

# 4. 결과 분석
# → 정량적 메트릭 확인
```

**예상 시간**: 30분 (데이터 작성 포함)

## 🔧 환경 설정

### 필수 사항

1. **PostgreSQL** (데이터베이스)
```bash
docker-compose up -d postgres
```

2. **MeiliSearch** (키워드 검색)
```bash
docker-compose up -d meilisearch
```

3. **인덱싱된 저장소**
```bash
semantica index /path/to/repo
# 또는
python -m apps.cli.main index /path/to/repo
```

### 선택 사항

4. **Sourcegraph 토큰** (Cody 비교 시)
```bash
export SOURCEGRAPH_TOKEN=sgp_xxxxx
```

## 📊 결과 해석

### 출력 예시

```
[1/5] '설정 파일'
  ⏱️  145.3ms
    1. src/core/config.py        ← 1순위
    2. src/core/bootstrap.py     ← 2순위
    3. apps/api/main.py          ← 3순위

📊 통계
총 쿼리:      5개
평균 응답:    143.2ms          ← 빠를수록 좋음
가장 빠름:    132.1ms
가장 느림:    156.7ms
총 결과:      15개

✅ 응답 속도: 빠름 (200ms 미만)
```

### 메트릭 기준

| 메트릭 | 우수 | 양호 | 보통 | 개선 필요 |
|--------|------|------|------|-----------|
| **Precision@5** | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |
| **Recall@5** | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |
| **MRR** | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |
| **Latency** | < 100ms | 100-200ms | 200-500ms | > 500ms |

### 개선 방법

**Precision이 낮으면**:
- 가중치 조정 (`src/core/config.py`)
- 퍼지 매칭 임계값 조정

**Recall이 낮으면**:
- K 값 증가 (더 많은 결과 반환)
- 검색 범위 확대

**MRR이 낮으면**:
- 랭킹 알고리즘 개선
- 가중치 재조정

**Latency가 높으면**:
- 인덱스 최적화
- 캐시 활용
- 병렬 처리

## 🐛 문제 해결

### 문제 1: "초기화 실패"

**원인**: PostgreSQL/MeiliSearch 미실행

**해결**:
```bash
docker-compose up -d
python benchmarks/check.py  # 확인
```

### 문제 2: "결과 없음"

**원인**: 저장소 미인덱싱

**해결**:
```bash
# 저장소 인덱싱
semantica index /path/to/repo

# 확인
semantica list-repos
```

### 문제 3: "명령어를 찾을 수 없음"

**원인**: 실행 권한 없음

**해결**:
```bash
chmod +x benchmark
# 또는
python benchmarks/run.py
```

### 문제 4: "Cody API 에러"

**원인**: Sourcegraph 토큰 미설정

**해결**:
```bash
# 토큰 발급: https://sourcegraph.com/user/settings/tokens
export SOURCEGRAPH_TOKEN=sgp_xxxxx
```

### 문제 5: "import 에러"

**원인**: 의존성 미설치

**해결**:
```bash
pip install -e .
# 또는
pip install -r requirements.txt
```

## 📁 파일 위치

```
프로젝트 루트/
├── benchmark              ← 실행 스크립트 (여기서 시작)
│
└── benchmarks/
    ├── run.py             ← 메인 Python 스크립트
    ├── check.py           ← 환경 체크
    ├── quickstart.py      ← 빠른 시작
    ├── compare.py         ← 고급 비교
    │
    ├── datasets/
    │   ├── semantica_queries.txt        ← 기본 쿼리
    │   └── semantica_ground_truth.json  ← 정답 데이터
    │
    └── evaluators/
        ├── metrics.py     ← 메트릭 계산
        ├── semantica.py   ← Semantica 평가기
        └── cody.py        ← Cody 평가기
```

## 📚 추가 문서

- `benchmarks/README.md` - 전체 가이드
- `.temp/benchmark-quick-guide.md` - 빠른 시작
- `.temp/benchmark-simple-summary.md` - 초간단 요약
- `.temp/BENCHMARK_COMPLETE.md` - 완성 요약
- `.temp/cody-comparison-summary.md` - Cody 비교

## 🎯 실전 체크리스트

### 첫 실행 체크리스트

- [ ] Docker 실행 (`docker ps` 확인)
- [ ] 환경 체크 (`python benchmarks/check.py`)
- [ ] 벤치마크 실행 (`./benchmark`)
- [ ] 결과 확인 (평균 응답 속도)
- [ ] 개선점 파악

### 정확도 평가 체크리스트

- [ ] 정답 데이터 확인
- [ ] 정확도 평가 실행 (`./benchmark eval`)
- [ ] Precision/Recall/MRR 확인
- [ ] 낮은 점수 쿼리 분석
- [ ] 개선 계획 수립

### Cody 비교 체크리스트

- [ ] Sourcegraph 계정 생성
- [ ] API 토큰 발급
- [ ] 환경변수 설정
- [ ] 비교 실행 (`./benchmark cody`)
- [ ] 결과 분석 (어느 쪽이 더 나은가?)

## 💡 팁과 요령

### 팁 1: 빠른 반복
```bash
# 수정 → 테스트 → 확인 반복
vim src/core/config.py  # 가중치 조정
./benchmark quick       # 빠른 테스트
./benchmark eval        # 정확도 확인
```

### 팁 2: 커스텀 쿼리로 취약점 찾기
```bash
# 실패한 쿼리만 모아서 재평가
./benchmark
# → 선택: 3 (직접 입력)
# → 문제 있던 쿼리 입력
```

### 팁 3: 로그 저장
```bash
./benchmark > benchmark_results.txt 2>&1
# → 나중에 분석 가능
```

### 팁 4: 여러 버전 비교
```bash
# 버전 1
./benchmark eval > v1_results.txt

# 설정 변경
vim src/core/config.py

# 버전 2
./benchmark eval > v2_results.txt

# 비교
diff v1_results.txt v2_results.txt
```

## 🚀 다음 단계

### 단계 1: 기본 (오늘)
1. `python benchmarks/check.py` 실행
2. `./benchmark` 실행
3. 결과 확인

### 단계 2: 평가 (이번 주)
1. `./benchmark eval` 실행
2. 메트릭 분석
3. 약점 파악

### 단계 3: 개선 (다음 주)
1. 가중치 조정
2. 재평가
3. 성능 향상 확인

### 단계 4: 비교 (선택)
1. Sourcegraph 계정
2. `./benchmark cody` 실행
3. Cody와 비교

## 요약

**가장 간단한 방법**:
```bash
./benchmark
```

**환경 체크**:
```bash
python benchmarks/check.py
```

**정확도 평가**:
```bash
./benchmark eval
```

**Cody 비교**:
```bash
export SOURCEGRAPH_TOKEN=sgp_xxxxx
./benchmark cody
```

**끝!** 🎉
