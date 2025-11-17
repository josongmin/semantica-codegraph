"""리트리버 비교 스크립트

Usage:
    python -m benchmarks.compare \
        --repo-id my-repo \
        --queries queries.txt \
        --ground-truth ground_truth.json \
        --k 5

또는 인터랙티브 모드:
    python -m benchmarks.compare --interactive
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.evaluators.semantica import SemanticaEvaluator
from benchmarks.evaluators.cody import CodyEvaluator
from benchmarks.evaluators.metrics import GroundTruth, MetricsCalculator


def load_queries(file_path: str) -> List[str]:
    """쿼리 파일 로드 (한 줄에 하나씩)"""
    with open(file_path) as f:
        return [line.strip() for line in f if line.strip()]


def load_ground_truth(file_path: str) -> List[GroundTruth]:
    """
    정답 데이터 로드
    
    JSON 형식:
    [
      {
        "query": "User authentication logic",
        "relevant_items": ["src/auth/login.py", "src/auth/jwt.py"]
      },
      ...
    ]
    """
    with open(file_path) as f:
        data = json.load(f)
    
    return [
        GroundTruth(
            query=item["query"],
            relevant_items=set(item["relevant_items"])
        )
        for item in data
    ]


def compare_retrievers(
    semantica_eval: SemanticaEvaluator,
    cody_eval: CodyEvaluator | None,
    repo_id: str,
    cody_repo: str,
    queries: List[str],
    ground_truths: List[GroundTruth],
    k: int = 5
):
    """리트리버 비교 실행"""
    
    print("=" * 80)
    print("리트리버 성능 비교")
    print("=" * 80)
    print(f"저장소: {repo_id}")
    print(f"쿼리 수: {len(queries)}")
    print(f"K: {k}")
    print()
    
    # 1. Semantica 평가
    print("[1/2] Semantica 리트리버 평가 중...")
    semantica_results = semantica_eval.batch_search(repo_id, queries, k)
    semantica_metrics = MetricsCalculator.evaluate_batch(
        semantica_results, ground_truths, k
    )
    
    print("\n[Semantica 결과]")
    print(semantica_metrics)
    print()
    
    # 2. Cody 평가 (옵션)
    if cody_eval:
        print("[2/2] Cody 리트리버 평가 중...")
        try:
            cody_results = cody_eval.batch_search(queries, cody_repo, k)
            cody_metrics = MetricsCalculator.evaluate_batch(
                cody_results, ground_truths, k
            )
            
            print("\n[Cody 결과]")
            print(cody_metrics)
            print()
            
            # 비교 표
            print("=" * 80)
            print("비교")
            print("=" * 80)
            print(f"{'메트릭':<20} {'Semantica':<15} {'Cody':<15} {'차이':<15}")
            print("-" * 80)
            
            metrics_list = [
                ("Precision@K", semantica_metrics.precision_at_k, cody_metrics.precision_at_k),
                ("Recall@K", semantica_metrics.recall_at_k, cody_metrics.recall_at_k),
                ("MRR", semantica_metrics.mrr, cody_metrics.mrr),
                ("Avg Latency (ms)", semantica_metrics.avg_latency_ms, cody_metrics.avg_latency_ms),
            ]
            
            for name, sem_val, cody_val in metrics_list:
                diff = sem_val - cody_val
                diff_str = f"{diff:+.3f}" if name != "Avg Latency (ms)" else f"{diff:+.1f}"
                print(f"{name:<20} {sem_val:<15.3f} {cody_val:<15.3f} {diff_str:<15}")
            
        except Exception as e:
            print(f"Cody 평가 실패: {e}")
            print("Sourcegraph API 토큰이 설정되어 있는지 확인하세요.")
    else:
        print("[2/2] Cody 평가 생략 (API 토큰 없음)")
    
    print()
    print("=" * 80)


def interactive_mode():
    """인터랙티브 모드 (수동 비교)"""
    print("=" * 80)
    print("인터랙티브 비교 모드")
    print("=" * 80)
    print()
    
    # 설정 입력
    repo_id = input("저장소 ID: ").strip()
    k = int(input("K (결과 수, 기본 5): ").strip() or "5")
    
    # Semantica 초기화
    print("\nSemantica 리트리버 초기화 중...")
    semantica_eval = SemanticaEvaluator()
    
    # Cody 초기화 (옵션)
    cody_eval = None
    use_cody = input("Cody와 비교하시겠습니까? (y/N): ").strip().lower() == "y"
    if use_cody:
        cody_repo = input("Cody 저장소 이름 (예: github.com/owner/repo): ").strip()
        try:
            cody_eval = CodyEvaluator()
            print("Cody API 연결 성공")
        except ValueError as e:
            print(f"Cody 초기화 실패: {e}")
    
    print()
    print("=" * 80)
    print("쿼리 입력 (종료: 빈 줄)")
    print("=" * 80)
    
    # 쿼리 입력 루프
    while True:
        query = input("\n쿼리: ").strip()
        if not query:
            break
        
        print(f"\n[검색 중: '{query}']")
        
        # Semantica 검색
        sem_result = semantica_eval.search(repo_id, query, k)
        print(f"\n[Semantica] ({sem_result.latency_ms:.1f}ms)")
        for i, path in enumerate(sem_result.results, 1):
            print(f"  {i}. {path}")
        
        # Cody 검색
        if cody_eval:
            cody_result = cody_eval.search(query, cody_repo, k)
            print(f"\n[Cody] ({cody_result.latency_ms:.1f}ms)")
            for i, path in enumerate(cody_result.results, 1):
                print(f"  {i}. {path}")
    
    print("\n종료합니다.")


def main():
    parser = argparse.ArgumentParser(description="리트리버 성능 비교")
    parser.add_argument("--interactive", action="store_true", help="인터랙티브 모드")
    parser.add_argument("--repo-id", help="Semantica 저장소 ID")
    parser.add_argument("--cody-repo", help="Cody 저장소 이름 (예: github.com/owner/repo)")
    parser.add_argument("--queries", help="쿼리 파일 경로 (한 줄에 하나)")
    parser.add_argument("--ground-truth", help="정답 데이터 JSON 파일")
    parser.add_argument("--k", type=int, default=5, help="반환할 결과 수 (기본: 5)")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
        return
    
    # 배치 모드
    if not args.repo_id or not args.queries or not args.ground_truth:
        parser.error("배치 모드는 --repo-id, --queries, --ground-truth 필수")
    
    # 데이터 로드
    queries = load_queries(args.queries)
    ground_truths = load_ground_truth(args.ground_truth)
    
    # 평가기 초기화
    semantica_eval = SemanticaEvaluator()
    
    cody_eval = None
    if args.cody_repo:
        try:
            cody_eval = CodyEvaluator()
        except ValueError as e:
            print(f"Cody 초기화 실패: {e}")
    
    # 비교 실행
    compare_retrievers(
        semantica_eval,
        cody_eval,
        args.repo_id,
        args.cody_repo or "",
        queries,
        ground_truths,
        args.k
    )


if __name__ == "__main__":
    main()

