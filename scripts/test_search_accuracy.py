#!/usr/bin/env python
"""
검색 정확도 테스트 스크립트

fixture 코드를 기반으로 의도한 검색 결과가 나오는지 검증
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.bootstrap import create_bootstrap


def setup_test_repos(bootstrap):
    """테스트 레포지토리 인덱싱"""
    print("=== 테스트 레포 인덱싱 ===\n")

    # Python 프로젝트
    python_path = project_root / "tests/fixtures/python_project"
    if python_path.exists():
        print("Python 프로젝트 인덱싱...")
        result = bootstrap.pipeline.index_repository(
            root_path=str(python_path),
            repo_id="search-test-python",
            name="Search Test Python"
        )
        print(f"  ✓ {result.total_files}파일, {result.total_nodes}노드, {result.total_chunks}청크\n")

    # TypeScript 프로젝트
    ts_path = project_root / "tests/fixtures/typescript_project"
    if ts_path.exists():
        print("TypeScript 프로젝트 인덱싱...")
        result = bootstrap.pipeline.index_repository(
            root_path=str(ts_path),
            repo_id="search-test-typescript",
            name="Search Test TypeScript"
        )
        print(f"  ✓ {result.total_files}파일, {result.total_nodes}노드, {result.total_chunks}청크\n")


def test_search_case(retriever, chunk_store, repo_id, query, expected_keywords, test_name):
    """단일 검색 테스트 케이스"""
    print(f"\n[{test_name}]")
    print(f"쿼리: \"{query}\"")
    print(f"기대: {', '.join(expected_keywords)}")

    results = retriever.retrieve(
        repo_id=repo_id,
        query=query,
        k=5
    )

    print(f"결과: {len(results)}개")

    # 결과 출력 (상위 5개 확인)
    found_keywords = []
    for i, candidate in enumerate(results[:5], 1):
        # Candidate에서 chunk_id로 실제 청크 조회
        chunk = chunk_store.get_chunk(repo_id, candidate.chunk_id)
        if not chunk:
            continue

        # 키워드 매칭 확인
        matches = [kw for kw in expected_keywords if kw.lower() in chunk.text.lower()]
        found_keywords.extend(matches)

        # 점수는 features에서 가져오기
        score = candidate.features.get("final_score", 0.0)
        is_md = '.md' in chunk.file_path.lower()
        marker = '[MD]' if is_md else ''
        print(f"  {i}. {marker} {chunk.file_path} (점수: {score:.3f})")
        if matches:
            print(f"     매칭: {', '.join(matches)}")

    # 검증
    found_keywords = list(set(found_keywords))
    coverage = len(found_keywords) / len(expected_keywords) * 100 if expected_keywords else 0

    if coverage >= 50:
        print(f"✓ 통과 ({coverage:.0f}% 키워드 매칭)")
        return True
    else:
        print(f"✗ 실패 ({coverage:.0f}% 키워드 매칭)")
        return False


def run_markdown_tests(bootstrap, retriever, chunk_store):
    """마크다운 문서 검색 테스트"""
    print("\n" + "="*60)
    print("마크다운 문서 검색 테스트")
    print("="*60)

    test_cases_python = [
        {
            "query": "middleware pattern request response chain",
            "expected": ["Middleware", "Pattern", "요청", "응답"],
            "name": "미들웨어 문서"
        },
        {
            "query": "repository pattern data access abstraction",
            "expected": ["Repository", "Pattern", "데이터 접근"],
            "name": "레포지토리 문서"
        },
        {
            "query": "calculate total order product",
            "expected": ["calculate_total", "Order", "총액"],
            "name": "사용 예시"
        },
    ]

    test_cases_ts = [
        {
            "query": "async await pattern clean asynchronous",
            "expected": ["Async", "Await", "Pattern", "asynchronous"],
            "name": "비동기 패턴 문서"
        },
        {
            "query": "generic programming cache typescript",
            "expected": ["Generic", "Cache", "TypeScript"],
            "name": "제네릭 문서"
        },
        {
            "query": "validation service centralized logic",
            "expected": ["Validation", "Service", "validation logic"],
            "name": "유효성 검사 문서"
        },
    ]

    passed = 0
    total = 0

    # Python README 테스트
    for test_case in test_cases_python:
        total += 1
        if test_search_case(
            retriever, chunk_store, "search-test-python",
            test_case["query"],
            test_case["expected"],
            f"[Python] {test_case['name']}"
        ):
            passed += 1

    # TypeScript README 테스트
    for test_case in test_cases_ts:
        total += 1
        if test_search_case(
            retriever, chunk_store, "search-test-typescript",
            test_case["query"],
            test_case["expected"],
            f"[TypeScript] {test_case['name']}"
        ):
            passed += 1

    print(f"\n결과: {passed}/{total} 통과")
    return passed, total


def run_python_tests(bootstrap, retriever, chunk_store):
    """Python 프로젝트 검색 테스트"""
    print("\n" + "="*60)
    print("Python 코드 검색 테스트")
    print("="*60)

    repo_id = "search-test-python"

    test_cases = [
        # 기본 기능
        {
            "query": "calculate total price",
            "expected": ["calculate_total", "Order", "price"],
            "name": "총액 계산"
        },
        {
            "query": "user authentication login",
            "expected": ["AuthService", "authenticate", "user"],
            "name": "사용자 인증"
        },
        {
            "query": "email validation check",
            "expected": ["validate_email", "email", "@"],
            "name": "이메일 검증"
        },
        {
            "query": "admin permission management",
            "expected": ["Admin", "permission", "grant", "revoke"],
            "name": "권한 관리"
        },
        {
            "query": "product discount apply",
            "expected": ["Product", "apply_discount", "price"],
            "name": "할인 적용"
        },
        {
            "query": "cache store retrieve",
            "expected": ["Cache", "get", "set"],
            "name": "캐시 구현"
        },
        # 고급 패턴
        {
            "query": "decorator require authentication",
            "expected": ["require_auth", "decorator", "current_user"],
            "name": "인증 데코레이터"
        },
        {
            "query": "middleware chain request response",
            "expected": ["Middleware", "process_request", "process_response"],
            "name": "미들웨어 체인"
        },
        {
            "query": "router path matching handler",
            "expected": ["Router", "Route", "matches", "handler"],
            "name": "라우터 패턴"
        },
        {
            "query": "dependency injection container service",
            "expected": ["ServiceContainer", "register", "inject"],
            "name": "의존성 주입"
        },
        {
            "query": "event bus publish subscribe",
            "expected": ["EventBus", "publish", "subscribe"],
            "name": "이벤트 버스"
        },
        {
            "query": "command pattern execute undo",
            "expected": ["Command", "execute", "undo"],
            "name": "커맨드 패턴"
        },
        {
            "query": "repository find save delete",
            "expected": ["Repository", "find_by_id", "save"],
            "name": "레포지토리 패턴"
        },
        {
            "query": "query builder select where",
            "expected": ["QueryBuilder", "select", "where", "build"],
            "name": "쿼리 빌더"
        },
        {
            "query": "specification pattern filter",
            "expected": ["Specification", "is_satisfied_by"],
            "name": "스펙 패턴"
        },
    ]

    passed = 0
    for test_case in test_cases:
        if test_search_case(
            retriever, chunk_store, repo_id,
            test_case["query"],
            test_case["expected"],
            test_case["name"]
        ):
            passed += 1

    print(f"\n결과: {passed}/{len(test_cases)} 통과")
    return passed, len(test_cases)


def run_typescript_tests(bootstrap, retriever, chunk_store):
    """TypeScript 프로젝트 검색 테스트"""
    print("\n" + "="*60)
    print("TypeScript 코드 검색 테스트")
    print("="*60)

    repo_id = "search-test-typescript"

    test_cases = [
        # 기본 기능
        {
            "query": "calculate order total",
            "expected": ["calculateTotal", "Order", "reduce"],
            "name": "주문 총액"
        },
        {
            "query": "user greeting message",
            "expected": ["greet", "UserModel", "Hello"],
            "name": "인사 메시지"
        },
        {
            "query": "admin permission check",
            "expected": ["AdminUser", "checkPermission", "permissions"],
            "name": "권한 확인"
        },
        {
            "query": "product availability stock",
            "expected": ["isAvailable", "Product", "stock"],
            "name": "재고 확인"
        },
        # 고급 패턴
        {
            "query": "user repository save find",
            "expected": ["InMemoryUserRepository", "save", "findById"],
            "name": "사용자 레포지토리"
        },
        {
            "query": "order service create calculate",
            "expected": ["OrderService", "createOrder", "calculateOrderTotal"],
            "name": "주문 서비스"
        },
        {
            "query": "email validation service",
            "expected": ["ValidationService", "validateUser", "email"],
            "name": "유효성 검사"
        },
        {
            "query": "debounce throttle function",
            "expected": ["debounce", "throttle", "setTimeout"],
            "name": "함수 제어"
        },
        {
            "query": "array chunk unique flatten",
            "expected": ["chunk", "unique", "flatten"],
            "name": "배열 유틸"
        },
        {
            "query": "cache generic get set",
            "expected": ["Cache", "get", "set", "Map"],
            "name": "제네릭 캐시"
        },
        {
            "query": "async retry error handling",
            "expected": ["retry", "async", "Promise"],
            "name": "비동기 재시도"
        },
        {
            "query": "format currency date internationalization",
            "expected": ["formatCurrency", "formatDate", "Intl"],
            "name": "포맷 함수"
        },
    ]

    passed = 0
    for test_case in test_cases:
        if test_search_case(
            retriever, chunk_store, repo_id,
            test_case["query"],
            test_case["expected"],
            test_case["name"]
        ):
            passed += 1

    print(f"\n결과: {passed}/{len(test_cases)} 통과")
    return passed, len(test_cases)


def main():
    """메인 실행"""
    print("검색 정확도 테스트\n")

    # Bootstrap 초기화
    bootstrap = create_bootstrap()

    # 레포 인덱싱
    setup_test_repos(bootstrap)

    # HybridRetriever 초기화
    from src.search.retriever.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(
        lexical_search=bootstrap.lexical_search,
        semantic_search=bootstrap.semantic_search,
        graph_search=bootstrap.graph_search,
        fuzzy_search=bootstrap.fuzzy_search,
        chunk_store=bootstrap.chunk_store,
        config=bootstrap.config
    )

    # 테스트 실행
    python_passed, python_total = run_python_tests(bootstrap, retriever, bootstrap.chunk_store)
    ts_passed, ts_total = run_typescript_tests(bootstrap, retriever, bootstrap.chunk_store)
    md_passed, md_total = run_markdown_tests(bootstrap, retriever, bootstrap.chunk_store)

    # 전체 결과
    total_passed = python_passed + ts_passed + md_passed
    total_tests = python_total + ts_total + md_total

    print("\n" + "="*60)
    print("전체 결과")
    print("="*60)
    print(f"통과: {total_passed}/{total_tests}")
    print(f"정확도: {total_passed/total_tests*100:.1f}%")

    if total_passed == total_tests:
        print("\n✓ 모든 검색 테스트 통과")
        return 0
    else:
        print(f"\n✗ {total_tests - total_passed}개 테스트 실패")
        return 1


if __name__ == "__main__":
    sys.exit(main())

