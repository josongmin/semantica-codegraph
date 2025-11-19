#!/usr/bin/env python3
"""
벤치마크 실행 - 간단한 래퍼

사용법:
    # 가장 간단 (Semantica만)
    python apps/benchmarks/run.py

    # Cody와 비교
    python apps/benchmarks/run.py --with-cody

    # 커스텀 쿼리
    python apps/benchmarks/run.py --queries my_queries.txt
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from apps.benchmarks.evaluators.metrics import GroundTruth, MetricsCalculator
from apps.benchmarks.evaluators.semantica import SemanticaEvaluator

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


@dataclass
class QueryAnalysis:
    """쿼리별 상세 분석"""

    query: str
    precision: float
    recall: float
    reciprocal_rank: float
    expected_count: int
    found_count: int
    missing_items: list[str]
    unexpected_items: list[str]


def analyze_query(result, ground_truth: GroundTruth, k: int) -> QueryAnalysis:
    """쿼리별 상세 분석 생성"""
    top_k = set(result.results[:k])
    relevant = top_k & ground_truth.relevant_items
    missing = ground_truth.relevant_items - top_k
    unexpected = top_k - ground_truth.relevant_items

    precision = MetricsCalculator.precision_at_k(result, ground_truth, k)
    recall = MetricsCalculator.recall_at_k(result, ground_truth, k)
    rr = MetricsCalculator.reciprocal_rank(result, ground_truth)

    return QueryAnalysis(
        query=result.query,
        precision=precision,
        recall=recall,
        reciprocal_rank=rr,
        expected_count=len(ground_truth.relevant_items),
        found_count=len(relevant),
        missing_items=sorted(missing),
        unexpected_items=sorted(unexpected),
    )


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
        queries = ["설정 파일", "데이터베이스 연결", "검색 구현", "파서", "테스트"]
        print(f"  ✓ 기본 쿼리 {len(queries)}개 사용")
    elif choice == "2":
        queries_file = project_root / "apps/benchmarks/datasets/semantica_queries.txt"
        if queries_file.exists():
            with queries_file.open() as f:
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
    print("  - 정답 데이터로 정확도 평가: python apps/benchmarks/run.py --evaluate")
    print("  - Cody와 비교: python apps/benchmarks/run.py --with-cody")
    print("  - 커스텀 쿼리: python apps/benchmarks/run.py --queries my_queries.txt")
    print()


def full_evaluation(repo_id: str, save_results: bool = True):
    """정답 데이터로 완전한 평가"""
    if HAS_RICH:
        console = Console()
        console.print("[bold cyan]📊 정확도 평가[/bold cyan]")
        console.print()
    else:
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
        print("  apps/benchmarks/datasets/semantica_ground_truth.json 파일을 작성하세요.")
        return

    with gt_file.open() as f:
        gt_data = json.load(f)

    ground_truths = [GroundTruth(item["query"], set(item["relevant_items"])) for item in gt_data]

    k = 5

    if HAS_RICH:
        console.print(f"📝 쿼리: {len(ground_truths)}개")
        console.print(f"📁 저장소: {repo_id}")
        console.print(f"🔢 K: {k}")
        console.print()
    else:
        print(f"📝 쿼리: {len(ground_truths)}개")
        print(f"📁 저장소: {repo_id}")
        print(f"🔢 K: {k}")
        print()

    # 평가 실행
    print("🔧 초기화 중...")
    evaluator = SemanticaEvaluator()

    # 검색 실행 (프로그레스 바 포함)
    results = []
    query_analyses = []

    if HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("검색 실행 중...", total=len(ground_truths))

            for gt in ground_truths:
                result = evaluator.search(repo_id, gt.query, k)
                results.append(result)

                # 쿼리별 분석
                analysis = analyze_query(result, gt, k)
                query_analyses.append(analysis)

                progress.update(task, advance=1)
    else:
        print("🔍 검색 중...")
        for i, gt in enumerate(ground_truths, 1):
            result = evaluator.search(repo_id, gt.query, k)
            results.append(result)

            analysis = analyze_query(result, gt, k)
            query_analyses.append(analysis)

            print(f"  [{i}/{len(ground_truths)}] {gt.query[:40]}...")

    # 전체 메트릭 계산
    metrics = MetricsCalculator.evaluate_batch(results, ground_truths, k)

    print()

    # 결과 출력
    if HAS_RICH:
        _print_results_rich(console, metrics, query_analyses)
    else:
        _print_results_plain(metrics, query_analyses)

    # 결과 저장
    if save_results:
        output_dir = project_root / ".temp/benchmark_results"
        _save_results(metrics, query_analyses, repo_id, output_dir)

    # 종합 평가
    score = 0
    if metrics.precision_at_k > 0.6:
        score += 1
    if metrics.recall_at_k > 0.5:
        score += 1
    if metrics.mrr > 0.7:
        score += 1
    if metrics.avg_latency_ms < 200:
        score += 1

    return score, metrics, query_analyses


def _print_results_rich(console: Console, metrics, analyses):
    """Rich 라이브러리를 사용한 결과 출력"""
    # 쿼리별 결과 테이블
    table = Table(title="쿼리별 평가 결과", show_lines=True)
    table.add_column("쿼리", style="cyan", width=35)
    table.add_column("Prec", justify="right", style="green")
    table.add_column("Recall", justify="right", style="yellow")
    table.add_column("RR", justify="right", style="magenta")
    table.add_column("매칭", justify="center")
    table.add_column("상태", justify="center")

    for a in analyses:
        query_display = a.query[:32] + "..." if len(a.query) > 35 else a.query

        if a.recall >= 0.7:
            status, style = "✅", "green"
        elif a.recall >= 0.5:
            status, style = "⚠️", "yellow"
        else:
            status, style = "❌", "red"

        matching = f"{a.found_count}/{a.expected_count}"

        table.add_row(
            query_display,
            f"{a.precision:.2f}",
            f"{a.recall:.2f}",
            f"{a.reciprocal_rank:.2f}",
            matching,
            status,
            style=style,
        )

    console.print(table)
    console.print()

    # 전체 메트릭 테이블
    metrics_table = Table(title="전체 메트릭", show_header=False)
    metrics_table.add_column("메트릭", style="cyan bold", width=20)
    metrics_table.add_column("값", style="white", width=15)
    metrics_table.add_column("기준", style="dim", width=20)
    metrics_table.add_column("평가", justify="center")

    metrics_data = [
        (
            "Precision@K",
            f"{metrics.precision_at_k:.3f}",
            "> 0.6 (Good)",
            "✅" if metrics.precision_at_k > 0.6 else "❌",
        ),
        (
            "Recall@K",
            f"{metrics.recall_at_k:.3f}",
            "> 0.5 (Good)",
            "✅" if metrics.recall_at_k > 0.5 else "❌",
        ),
        ("MRR", f"{metrics.mrr:.3f}", "> 0.7 (Good)", "✅" if metrics.mrr > 0.7 else "❌"),
        (
            "Avg Latency",
            f"{metrics.avg_latency_ms:.1f}ms",
            "< 200ms (Fast)",
            "✅" if metrics.avg_latency_ms < 200 else "⚠️",
        ),
    ]

    for name, value, threshold, status in metrics_data:
        metrics_table.add_row(name, value, threshold, status)

    console.print(metrics_table)

    # 실패 케이스
    failed = [a for a in analyses if a.recall < 0.5]
    if failed:
        console.print()
        console.print("[bold red]실패 케이스 분석[/bold red] (Recall < 0.5)\n")

        for i, a in enumerate(failed, 1):
            console.print(f"[bold]{i}. {a.query}[/bold]")
            console.print(f"   Recall: {a.recall:.2f}")

            if a.missing_items:
                console.print("   [red]누락:[/red]")
                for item in a.missing_items[:2]:
                    console.print(f"     - {item}")
                if len(a.missing_items) > 2:
                    console.print(f"     ... 외 {len(a.missing_items) - 2}개")
            console.print()


def _print_results_plain(metrics, analyses):
    """Plain 텍스트 결과 출력"""
    print("=" * 80)
    print("결과")
    print("=" * 80)
    print(metrics)
    print()

    # 평가
    if metrics.precision_at_k > 0.6:
        print("✅ Precision: 좋음")
    else:
        print("❌ Precision: 개선 필요")

    if metrics.recall_at_k > 0.5:
        print("✅ Recall: 좋음")
    else:
        print("❌ Recall: 개선 필요")

    if metrics.mrr > 0.7:
        print("✅ MRR: 좋음")
    else:
        print("❌ MRR: 개선 필요")

    if metrics.avg_latency_ms < 200:
        print("✅ Latency: 빠름")
    else:
        print("⚠️  Latency: 보통")

    print()

    # 실패 케이스
    failed = [a for a in analyses if a.recall < 0.5]
    if failed:
        print(f"실패 케이스: {len(failed)}개")
        for a in failed:
            print(f"  - {a.query} (Recall: {a.recall:.2f})")
        print()


def _save_results(metrics, analyses, repo_id, output_dir):
    """결과를 JSON으로 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now()
    run_data = {
        "timestamp": timestamp.isoformat(),
        "repo_id": repo_id,
        "summary": {
            "precision_at_k": metrics.precision_at_k,
            "recall_at_k": metrics.recall_at_k,
            "mrr": metrics.mrr,
            "avg_latency_ms": metrics.avg_latency_ms,
            "total_queries": metrics.total_queries,
        },
        "queries": [
            {
                "query": a.query,
                "precision": a.precision,
                "recall": a.recall,
                "reciprocal_rank": a.reciprocal_rank,
                "found": a.found_count,
                "expected": a.expected_count,
                "missing": a.missing_items,
                "unexpected": a.unexpected_items,
            }
            for a in analyses
        ],
    }

    filename = f"benchmark_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    output_file = output_dir / filename

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2, ensure_ascii=False)

    if HAS_RICH:
        console = Console()
        console.print(f"[green]✅ 결과 저장:[/green] {output_file}")
    else:
        print(f"✅ 결과 저장: {output_file}")


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
        result = full_evaluation(repo_id)
        if result:
            score, metrics, analyses = result
            print()
            print(f"종합 점수: {score}/4")
            if score >= 3:
                print("✅ 전반적으로 좋은 성능!")
            else:
                print("⚠️  개선이 필요합니다.")
            print()
    elif args.with_cody:
        print("Cody 비교는 아직 준비 중입니다.")
        print("임시로 compare.py를 사용하세요:")
        print("  python -m apps.benchmarks.compare --interactive")
    else:
        simple_benchmark()


if __name__ == "__main__":
    main()
