#!/usr/bin/env python3
"""
벤치마크 실행 - 간단한 래퍼

사용법:
    # 가장 간단 (Semantica만)
    python benchmarks/run.py
    
    # Cody와 비교
    python benchmarks/run.py --with-cody
    
    # 커스텀 쿼리
    python benchmarks/run.py --queries my_queries.txt
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.evaluators.semantica import SemanticaEvaluator
from benchmarks.evaluators.metrics import GroundTruth, MetricsCalculator


def simple_benchmark():
    """가장 간단한 벤치마크 - 대화형"""
    print("=" * 80)
    print("🚀 Semantica 벤치마크")
    print("=" * 80)
    print()
    
    # 저장소 ID
    print("📁 저장소 설정")
    repo_id = input("  저장소 ID (기본: semantica-codegraph): ").strip()
    if not repo_id:
        repo_id = "semantica-codegraph"
    print(f"  ✓ 저장소: {repo_id}")
    print()
    
    # 테스트 쿼리 선택
    print("📝 테스트 쿼리 선택")
    print("  1. 기본 쿼리 5개 (빠름)")
    print("  2. 전체 쿼리 10개 (상세)")
    print("  3. 직접 입력")
    choice = input("  선택 (1-3, 기본: 1): ").strip() or "1"
    print()
    
    if choice == "1":
        queries = [
            "설정 파일",
            "데이터베이스 연결",
            "검색 구현",
            "파서",
            "테스트"
        ]
        print(f"  ✓ 기본 쿼리 {len(queries)}개 사용")
    elif choice == "2":
        queries_file = project_root / "benchmarks/datasets/semantica_queries.txt"
        if queries_file.exists():
            with open(queries_file) as f:
                queries = [line.strip() for line in f if line.strip()]
            print(f"  ✓ 전체 쿼리 {len(queries)}개 사용")
        else:
            print("  ✗ 쿼리 파일 없음, 기본 쿼리 사용")
            queries = ["설정", "검색", "파서"]
    else:
        print("  쿼리를 입력하세요 (빈 줄로 종료):")
        queries = []
        while True:
            q = input("    > ").strip()
            if not q:
                break
            queries.append(q)
        print(f"  ✓ 쿼리 {len(queries)}개 입력됨")
    
    if not queries:
        print("  ✗ 쿼리가 없습니다.")
        return
    
    print()
    print("🔧 Semantica 초기화 중...")
    try:
        evaluator = SemanticaEvaluator()
        print("  ✓ 초기화 완료")
    except Exception as e:
        print(f"  ✗ 초기화 실패: {e}")
        print()
        print("💡 문제 해결:")
        print("  - PostgreSQL이 실행 중인가요? (docker-compose up -d)")
        print("  - MeiliSearch가 실행 중인가요?")
        print("  - .env 파일이 설정되어 있나요?")
        return
    
    print()
    print("=" * 80)
    print("🔍 검색 실행 중...")
    print("=" * 80)
    print()
    
    k = 3
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] '{query}'")
        try:
            result = evaluator.search(repo_id, query, k)
            results.append(result)
            
            print(f"  ⏱️  {result.latency_ms:.1f}ms")
            if result.results:
                for j, path in enumerate(result.results[:k], 1):
                    print(f"    {j}. {path}")
            else:
                print("    (결과 없음)")
        except Exception as e:
            print(f"  ✗ 에러: {e}")
        print()
    
    # 간단한 통계
    if results:
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        print("=" * 80)
        print("📊 통계")
        print("=" * 80)
        print(f"총 쿼리:      {len(results)}개")
        print(f"평균 응답:    {avg_latency:.1f}ms")
        print(f"가장 빠름:    {min(r.latency_ms for r in results):.1f}ms")
        print(f"가장 느림:    {max(r.latency_ms for r in results):.1f}ms")
        
        # 결과 품질 간단 체크
        total_results = sum(len(r.results) for r in results)
        print(f"총 결과:      {total_results}개")
        print()
        
        # 성능 평가
        if avg_latency < 200:
            print("✅ 응답 속도: 빠름 (200ms 미만)")
        elif avg_latency < 500:
            print("⚠️  응답 속도: 보통 (200-500ms)")
        else:
            print("❌ 응답 속도: 느림 (500ms 이상)")
    
    print()
    print("=" * 80)
    print()
    print("💡 다음 단계:")
    print("  - 정답 데이터로 정확도 평가: python benchmarks/run.py --evaluate")
    print("  - Cody와 비교: python benchmarks/run.py --with-cody")
    print("  - 커스텀 쿼리: python benchmarks/run.py --queries my_queries.txt")
    print()


def full_evaluation(repo_id: str):
    """정답 데이터로 완전한 평가"""
    print("=" * 80)
    print("📊 정확도 평가")
    print("=" * 80)
    print()
    
    # 정답 데이터 로드
    gt_file = Path(__file__).parent / "datasets/semantica_ground_truth.json"
    if not gt_file.exists():
        print(f"✗ 정답 데이터 없음: {gt_file}")
        print()
        print("💡 정답 데이터 생성:")
        print("  benchmarks/datasets/semantica_ground_truth.json 파일을 작성하세요.")
        return
    
    import json
    with open(gt_file) as f:
        gt_data = json.load(f)
    
    ground_truths = [
        GroundTruth(item["query"], set(item["relevant_items"]))
        for item in gt_data
    ]
    
    queries = [gt.query for gt in ground_truths]
    
    print(f"📝 쿼리: {len(queries)}개")
    print(f"📁 저장소: {repo_id}")
    print()
    
    # 평가 실행
    print("🔧 초기화 중...")
    evaluator = SemanticaEvaluator()
    
    print("🔍 검색 중...")
    results = evaluator.batch_search(repo_id, queries, k=5)
    
    print("📊 평가 중...")
    metrics = MetricsCalculator.evaluate_batch(results, ground_truths, k=5)
    
    print()
    print("=" * 80)
    print("결과")
    print("=" * 80)
    print(metrics)
    print()
    
    # 평가
    score = 0
    if metrics.precision_at_k > 0.6:
        print("✅ Precision: 좋음")
        score += 1
    else:
        print("❌ Precision: 개선 필요")
    
    if metrics.recall_at_k > 0.5:
        print("✅ Recall: 좋음")
        score += 1
    else:
        print("❌ Recall: 개선 필요")
    
    if metrics.mrr > 0.7:
        print("✅ MRR: 좋음")
        score += 1
    else:
        print("❌ MRR: 개선 필요")
    
    if metrics.avg_latency_ms < 200:
        print("✅ Latency: 빠름")
        score += 1
    else:
        print("⚠️  Latency: 보통")
    
    print()
    print(f"종합 점수: {score}/4")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="간단한 벤치마크 실행")
    parser.add_argument("--evaluate", action="store_true", help="정답 데이터로 정확도 평가")
    parser.add_argument("--with-cody", action="store_true", help="Cody와 비교")
    parser.add_argument("--repo-id", help="저장소 ID")
    parser.add_argument("--queries", help="쿼리 파일 경로")
    
    args = parser.parse_args()
    
    if args.evaluate:
        repo_id = args.repo_id or "semantica-codegraph"
        full_evaluation(repo_id)
    elif args.with_cody:
        print("Cody 비교는 아직 준비 중입니다.")
        print("임시로 compare.py를 사용하세요:")
        print("  python -m benchmarks.compare --interactive")
    else:
        simple_benchmark()


if __name__ == "__main__":
    main()

