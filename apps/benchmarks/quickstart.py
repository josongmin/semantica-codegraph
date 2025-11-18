"""빠른 시작 - Semantica vs Cody 비교

간단한 쿼리로 빠르게 비교해보기
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from apps.benchmarks.evaluators.cody import CodyEvaluator
from apps.benchmarks.evaluators.semantica import SemanticaEvaluator


def main():
    print("=" * 80)
    print("Semantica vs Cody 빠른 비교")
    print("=" * 80)
    print()

    # 설정 입력
    repo_id = input("Semantica 저장소 ID (예: my-repo): ").strip()
    if not repo_id:
        print("저장소 ID를 입력해주세요.")
        return

    # Cody 사용 여부
    use_cody = input("Cody와 비교? (y/N): ").strip().lower() == "y"
    cody_eval = None
    cody_repo = None

    if use_cody:
        cody_repo = input("Cody 저장소 (예: github.com/owner/repo): ").strip()
        if not cody_repo:
            print("저장소 이름을 입력해주세요.")
            return

        try:
            cody_eval = CodyEvaluator()
            print("✓ Cody API 연결 성공")
        except ValueError as e:
            print(f"✗ Cody 초기화 실패: {e}")
            print("  SOURCEGRAPH_TOKEN 환경변수를 설정하세요.")
            return

    # Semantica 초기화
    print("\nSemantica 리트리버 초기화 중...")
    try:
        semantica_eval = SemanticaEvaluator()
        print("✓ Semantica 초기화 성공")
    except Exception as e:
        print(f"✗ Semantica 초기화 실패: {e}")
        return

    print()
    print("=" * 80)
    print("기본 테스트 쿼리 5개")
    print("=" * 80)

    # 기본 테스트 쿼리
    test_queries = [
        "설정 파일 로드",
        "데이터베이스 연결",
        "검색 리트리버",
        "파서 구현",
        "테스트 코드"
    ]

    k = 3  # 상위 3개만 표시

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/5] '{query}'")
        print("-" * 80)

        # Semantica 검색
        try:
            sem_result = semantica_eval.search(repo_id, query, k)
            print(f"\n  Semantica ({sem_result.latency_ms:.1f}ms):")
            for j, path in enumerate(sem_result.results[:k], 1):
                print(f"    {j}. {path}")
        except Exception as e:
            print(f"  Semantica 에러: {e}")

        # Cody 검색
        if cody_eval and cody_repo:
            try:
                cody_result = cody_eval.search(query, cody_repo, k)
                print(f"\n  Cody ({cody_result.latency_ms:.1f}ms):")
                for j, path in enumerate(cody_result.results[:k], 1):
                    print(f"    {j}. {path}")
            except Exception as e:
                print(f"  Cody 에러: {e}")

    print()
    print("=" * 80)
    print("완료!")
    print()
    print("더 자세한 평가를 원하시면:")
    print("  python -m apps.benchmarks.compare --interactive")
    print()


if __name__ == "__main__":
    main()

