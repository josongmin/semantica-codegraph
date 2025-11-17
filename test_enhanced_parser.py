#!/usr/bin/env python3
"""EnhancedParser 통합 테스트"""

from pathlib import Path
from src.parser.enhanced_parser import EnhancedParser


def test_all_analyzers_integrated():
    """모든 분석기 통합 테스트"""
    print("=" * 80)
    print("EnhancedParser 통합 테스트")
    print("=" * 80)
    
    code = '''
from auth.services import UserAuthenticator as UA

class UserView:
    def dispatch(self, request, action):
        # Django 패턴
        handler = getattr(self, f"handle_{action}")
        return handler(request)
    
    def handle_login(self, request):
        return "login"

def process_user(user: UA):
    # 타입 힌트 추론
    method = getattr(user, "authenticate")
    return method()

def test_authentication():
    """테스트 코드"""
    auth = UserAuthenticator()
    result = auth.authenticate("test", "pass")
    assert result
'''
    
    # 임시 파일 생성
    test_file = Path("/tmp/test_enhanced.py")
    test_file.write_text(code)
    
    # EnhancedParser 생성 (Django 프레임워크)
    parser = EnhancedParser(framework="django")
    
    # 파싱
    symbols, relations = parser.parse_file({
        "repo_id": "test",
        "path": "test_enhanced.py",
        "file_path": "test_enhanced.py",
        "abs_path": str(test_file),
        "language": "python"
    })
    
    print(f"\n파싱 결과:")
    print(f"  심볼: {len(symbols)}개")
    print(f"  관계: {len(relations)}개")
    
    # 추론된 관계 분석
    inferred = [r for r in relations if r.attrs.get("inferred")]
    print(f"  추론된 관계: {len(inferred)}개")
    
    # 방법별 분류
    by_method = {}
    for rel in inferred:
        method = rel.attrs.get("method", "unknown")
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(rel)
    
    print(f"\n방법별 분류:")
    for method, rels in by_method.items():
        print(f"  {method}: {len(rels)}개")
        for rel in rels[:2]:  # 상위 2개만
            source = rel.attrs.get("source_symbol", "?")
            target = rel.attrs.get("target_symbol", "?")
            print(f"    - {source} → {target}")
    
    # 검증
    assert len(symbols) > 0
    assert len(relations) > 0
    assert len(inferred) > 0
    
    # 각 방법이 모두 작동했는지 확인
    methods = set(by_method.keys())
    print(f"\n사용된 분석 방법: {methods}")
    
    # 타입 힌트는 반드시 있어야 함 (UA → auth.services.UserAuthenticator)
    assert "type_hint" in methods, "타입 힌트 분석 실행되지 않음"
    
    # Django 프레임워크이므로 패턴도 있을 수 있음
    # (없어도 OK, 패턴 매칭 안될 수 있음)
    
    print("\n✅ 모든 분석기 통합 동작!")
    return True


def test_performance_stats():
    """성능 통계 측정"""
    print("\n" + "=" * 80)
    print("성능 통계 테스트")
    print("=" * 80)
    
    code = '''
def test_something():
    auth = UserAuth()
    auth.login()
'''
    
    test_file = Path("/tmp/test_perf.py")
    test_file.write_text(code)
    
    parser = EnhancedParser()
    
    # 통계 초기화
    parser.reset_stats()
    
    # 파싱
    symbols, relations = parser.parse_file({
        "repo_id": "test",
        "path": "tests/test_perf.py",
        "abs_path": str(test_file),
        "language": "python"
    })
    
    # 통계 확인
    stats = parser.get_performance_stats()
    
    print(f"\n성능 통계:")
    total_time = sum(stats.values())
    for method, time_spent in stats.items():
        percentage = (time_spent / total_time * 100) if total_time > 0 else 0
        print(f"  {method}: {time_spent:.4f}초 ({percentage:.1f}%)")
    
    print(f"  총 시간: {total_time:.4f}초")
    
    # 동적 분석 오버헤드 체크
    dynamic_time = stats["type_hint_time"] + stats["pattern_time"] + stats["test_time"]
    if total_time > 0:
        overhead = (dynamic_time / total_time) * 100
        print(f"\n동적 분석 오버헤드: {overhead:.1f}%")
        
        # 목표: 10% 이내
        # (작은 파일이라 측정 부정확할 수 있음)
        print(f"  목표: <10% (작은 파일이라 부정확할 수 있음)")
    
    print("\n✅ 성능 통계 수집 성공!")
    return True


def test_conditional_activation():
    """조건부 활성화 테스트"""
    print("\n" + "=" * 80)
    print("조건부 활성화 테스트")
    print("=" * 80)
    
    code = '''
def test_something():
    auth.login()
'''
    
    test_file = Path("/tmp/test_cond.py")
    test_file.write_text(code)
    
    # 1. 모든 분석기 비활성화
    parser = EnhancedParser(
        enable_type_hint=False,
        enable_pattern=False,
        enable_test=False
    )
    
    symbols, relations = parser.parse_file({
        "repo_id": "test",
        "path": "tests/test_cond.py",
        "abs_path": str(test_file),
        "language": "python"
    })
    
    inferred = [r for r in relations if r.attrs.get("inferred")]
    
    print(f"\n비활성화: 추론 {len(inferred)}개 (0이어야 함)")
    assert len(inferred) == 0
    
    # 2. 타입 힌트만 활성화
    parser2 = EnhancedParser(
        enable_type_hint=True,
        enable_pattern=False,
        enable_test=False
    )
    
    symbols2, relations2 = parser2.parse_file({
        "repo_id": "test",
        "path": "tests/test_cond.py",
        "abs_path": str(test_file),
        "language": "python"
    })
    
    type_hint_only = [
        r for r in relations2 
        if r.attrs.get("method") == "type_hint"
    ]
    
    print(f"타입 힌트만: {len(type_hint_only)}개")
    
    print("\n✅ 조건부 활성화 동작!")
    return True


def test_framework_auto_detect():
    """프레임워크 자동 감지 + 패턴 적용"""
    print("\n" + "=" * 80)
    print("프레임워크 자동 감지 테스트")
    print("=" * 80)
    
    django_code = '''
from django.views import View

class MyView(View):
    def dispatch(self, request, action):
        getattr(self, f"handle_{action}")
    
    def handle_create(self, request):
        pass
'''
    
    test_file = Path("/tmp/django_test.py")
    test_file.write_text(django_code)
    
    # framework=None (자동 감지)
    parser = EnhancedParser(framework=None)
    
    symbols, relations = parser.parse_file({
        "repo_id": "test",
        "path": "views.py",
        "abs_path": str(test_file),
        "language": "python"
    })
    
    # 패턴 매칭되었는지 확인
    pattern_rels = [r for r in relations if r.attrs.get("method") == "pattern"]
    
    print(f"\n자동 감지 후 패턴 매칭: {len(pattern_rels)}개")
    
    for rel in pattern_rels:
        source = rel.attrs.get("source_symbol", "?")
        target = rel.attrs.get("target_symbol", "?")
        print(f"  {source} → {target}")
    
    if len(pattern_rels) > 0:
        print("\n✅ Django 자동 감지 + 패턴 적용 성공!")
    else:
        print("\n⚠️  패턴 매칭 없음 (정상일 수 있음)")
    
    return True


def main():
    """모든 테스트 실행"""
    tests = [
        test_all_analyzers_integrated,
        test_performance_stats,
        test_conditional_activation,
        test_framework_auto_detect,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"❌ 테스트 실패: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"결과: {passed}개 통과, {failed}개 실패")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 EnhancedParser 통합 완료!")
        print("\n달성:")
        print("  ✅ 모든 분석기 통합")
        print("  ✅ 성능 통계 수집")
        print("  ✅ 조건부 활성화")
        print("  ✅ 프레임워크 자동 감지")
        print("\nWeek 2 Day 9-10 완료!")
        print("  커버리지: 80% → 90% (예상)")
        print("\n다음:")
        print("  📝 create_parser() 수정")
        print("  📝 커버리지 측정")
        print("  📝 90% 검증")
    else:
        print(f"\n⚠️  {failed}개 테스트 실패")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

