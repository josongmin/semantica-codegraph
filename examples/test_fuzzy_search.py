"""퍼지 검색 기능 테스트 예제"""

from src.core.bootstrap import create_bootstrap


def main():
    """퍼지 검색 기능 테스트"""
    # Bootstrap 생성
    bootstrap = create_bootstrap()
    
    # 저장소 ID (이미 인덱싱된 저장소)
    repo_id = "test-repo"
    
    print("=== 퍼지 검색 기능 테스트 ===\n")
    
    # 1. 정확한 매칭
    print("1. 정확한 매칭: 'UserService'")
    results = bootstrap.fuzzy_search.search_symbols(
        repo_id=repo_id,
        query="UserService",
        threshold=0.9,
        k=5,
    )
    print_results(results)
    
    # 2. 오타 허용
    print("\n2. 오타 허용: 'UserServce' (i 누락)")
    results = bootstrap.fuzzy_search.search_symbols(
        repo_id=repo_id,
        query="UserServce",
        threshold=0.8,
        k=5,
    )
    print_results(results)
    
    # 3. 대소문자 무시
    print("\n3. 대소문자 무시: 'userservice'")
    results = bootstrap.fuzzy_search.search_symbols(
        repo_id=repo_id,
        query="userservice",
        threshold=0.8,
        k=5,
    )
    print_results(results)
    
    # 4. 축약형
    print("\n4. 축약형: 'UsrSvc'")
    results = bootstrap.fuzzy_search.search_symbols(
        repo_id=repo_id,
        query="UsrSvc",
        threshold=0.7,
        k=5,
    )
    print_results(results)
    
    # 5. 하이브리드 검색 (퍼지 포함)
    print("\n5. 하이브리드 검색 (퍼지 포함): 'UserServce login'")
    candidates = bootstrap.hybrid_retriever.retrieve(
        repo_id=repo_id,
        query="UserServce login",
        k=10,
        weights={
            "lexical": 0.25,
            "semantic": 0.45,
            "graph": 0.15,
            "fuzzy": 0.15,
        }
    )
    print(f"Retrieved {len(candidates)} candidates:")
    for i, candidate in enumerate(candidates[:5], 1):
        print(f"  {i}. {candidate.file_path}")
        print(f"     Scores: {candidate.features}")
    
    # 6. Threshold 비교
    print("\n6. Threshold 비교:")
    for threshold in [0.95, 0.85, 0.75]:
        results = bootstrap.fuzzy_search.search_symbols(
            repo_id=repo_id,
            query="loging",  # 오타
            threshold=threshold,
            k=5,
        )
        print(f"   Threshold {threshold}: {len(results)} results")


def print_results(results):
    """검색 결과 출력"""
    if not results:
        print("  (결과 없음)")
        return
    
    for i, match in enumerate(results, 1):
        print(f"  {i}. {match.matched_text} (score: {match.score:.3f})")
        print(f"     File: {match.file_path}")
        print(f"     Kind: {match.kind}")


if __name__ == "__main__":
    main()

