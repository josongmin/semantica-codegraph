# 벤치마크 빠른 시작

Semantica 리트리버 성능 측정 및 Cody 비교 도구

## 실행 (1줄)

```bash
./benchmark
```

## 사전 체크

```bash
python apps/benchmarks/check.py
```

## 명령어

```bash
./benchmark          # 대화형 (기본)
./benchmark quick    # 빠른 테스트
./benchmark eval     # 정확도 평가
./benchmark cody     # Cody 비교
```

## 문서

- `.doc/benchmarks-guide.md` - 전체 가이드
- `.doc/HOW_TO_BENCHMARK.md` - 상세 사용법
- `.doc/benchmark-quick-guide.md` - 빠른 가이드

## 문제 해결

### 환경 체크 실패

```bash
# Docker 서비스 시작
docker-compose up -d

# 저장소 인덱싱
semantica index /path/to/repo
```

### 명령어 없음

```bash
# 권한 추가
chmod +x benchmark

# 또는 Python으로 직접
python apps/benchmarks/run.py
```

## 구조

```
apps/benchmarks/
├── run.py              # 메인 실행기
├── check.py            # 환경 체크
├── compare.py          # 고급 비교
├── evaluators/         # 평가기들
└── datasets/           # 쿼리/정답 데이터
```

## 평가 메트릭

- **Precision@K**: 상위 K개 중 관련 결과 비율
- **Recall@K**: 전체 관련 결과 중 찾은 비율
- **MRR**: 첫 관련 결과의 순위
- **Latency**: 검색 응답 시간

## 즉시 시작

1. `python apps/benchmarks/check.py` - 환경 체크
2. `./benchmark` - 벤치마크 실행
3. 결과 확인

끝!
