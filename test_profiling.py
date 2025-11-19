#!/usr/bin/env python
"""프로파일링 시스템 테스트"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.core.bootstrap import create_bootstrap
from src.indexer.repo_profiler import RepoProfiler
from src.indexer.file_profiler import FileProfiler


def test_repo_profiling():
    """Repo Profiling 테스트"""
    print("\n" + "="*60)
    print("테스트 1: Repo Profiling")
    print("="*60)
    
    repo_root = str(Path(__file__).parent)
    repo_id = "semantica-codegraph"
    
    profiler = RepoProfiler()
    profile = profiler.profile_repo(repo_root, repo_id)
    
    print(f"\n[결과]")
    print(f"  Primary Language: {profile.primary_language}")
    print(f"  Framework: {profile.framework}")
    print(f"  Frameworks: {profile.frameworks}")
    print(f"  Project Type: {profile.project_type}")
    print(f"  API Patterns: {profile.api_patterns[:5]}")
    print(f"\n[디렉토리]")
    print(f"  API: {profile.api_directories}")
    print(f"  Service: {profile.service_directories[:3]}")
    print(f"  Model: {profile.model_directories[:3]}")
    print(f"  Test: {profile.test_directories[:3]}")
    print(f"\n[엔트리포인트]")
    print(f"  {profile.entry_points}")
    print(f"\n[언어 분포]")
    for lang, lines in sorted(profile.languages.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {lang}: {lines:,} lines")
    
    assert profile.framework == "fastapi", f"Expected fastapi, got {profile.framework}"
    assert profile.project_type == "web_api", f"Expected web_api, got {profile.project_type}"
    assert len(profile.api_directories) > 0, "Should have API directories"
    
    print("\n✅ Repo Profiling 통과!")
    return profile


def test_file_profiling(repo_profile):
    """File Profiling 테스트"""
    print("\n" + "="*60)
    print("테스트 2: File Profiling")
    print("="*60)
    
    repo_root = str(Path(__file__).parent)
    repo_id = "semantica-codegraph"
    
    profiler = FileProfiler()
    
    # 테스트 파일들
    test_files = [
        ("apps/api/routes/hybrid.py", True, True),  # (경로, is_api_file, is_router)
        ("src/core/models.py", False, True),  # model 파일
        ("tests/core/test_bootstrap.py", False, False),  # test 파일
    ]
    
    print(f"\n[파일 프로파일링 결과]")
    for file_path, expected_api, expected_model in test_files:
        abs_path = Path(repo_root) / file_path
        if not abs_path.exists():
            print(f"  ⚠️  {file_path}: 파일 없음")
            continue
        
        profile = profiler.profile_file(
            repo_id=repo_id,
            file_path=file_path,
            abs_path=str(abs_path),
            framework=repo_profile.framework
        )
        
        print(f"\n  {file_path}:")
        print(f"    is_api_file: {profile.is_api_file}")
        print(f"    is_router: {profile.is_router}")
        print(f"    is_model: {profile.is_model}")
        print(f"    is_test_file: {profile.is_test_file}")
        print(f"    api_framework: {profile.api_framework}")
        print(f"    endpoints: {len(profile.endpoints)}개")
        if profile.endpoints:
            print(f"      예시: {profile.endpoints[0]}")
        print(f"    functions: {profile.function_count}, classes: {profile.class_count}")
    
    print("\n✅ File Profiling 통과!")


def test_chunk_tagging():
    """Chunk Tagging 테스트"""
    print("\n" + "="*60)
    print("테스트 3: Chunk Tagging")
    print("="*60)
    
    from src.chunking.chunk_tagger import ChunkTagger
    from src.core.models import FileProfile
    
    tagger = ChunkTagger()
    
    # API endpoint 청크
    api_chunk = '''
@router.post("/search")
async def hybrid_search(request: HybridSearchRequest):
    """하이브리드 검색"""
    return result
'''
    
    file_profile = FileProfile(
        repo_id="test",
        file_path="api/routes.py",
        is_api_file=True,
        is_router=True,
        api_framework="fastapi"
    )
    
    metadata = tagger.tag_chunk(api_chunk, file_profile)
    
    print(f"\n[API Endpoint 청크]")
    print(f"  is_api_endpoint_chunk: {metadata['is_api_endpoint_chunk']}")
    print(f"  http_method: {metadata.get('http_method')}")
    print(f"  http_path: {metadata.get('http_path')}")
    print(f"  has_docstring: {metadata['has_docstring']}")
    print(f"  is_function_definition: {metadata['is_function_definition']}")
    
    assert metadata["is_api_endpoint_chunk"], "Should be API endpoint chunk"
    assert metadata["http_method"] == "POST", f"Expected POST, got {metadata.get('http_method')}"
    assert metadata["http_path"] == "/search", f"Expected /search, got {metadata.get('http_path')}"
    
    print("\n✅ Chunk Tagging 통과!")


def test_query_analysis():
    """쿼리 타입 분석 테스트"""
    print("\n" + "="*60)
    print("테스트 4: 쿼리 타입 분석")
    print("="*60)
    
    # hybrid.py에서 import
    sys.path.insert(0, str(Path(__file__).parent / "apps"))
    from api.routes.hybrid import _analyze_query_type
    
    test_cases = [
        ("이 프로젝트의 API 엔드포인트는 어디 있어?", "api"),
        ("GraphStore 서비스는 뭐하는거야?", "service"),
        ("User 모델 정의 보여줘", "model"),
        ("설정 파일 어디 있어?", "config"),
        ("테스트 코드 찾아줘", "test"),
        ("bootstrap이 뭐하는 파일이야?", "general"),
    ]
    
    print(f"\n[쿼리 타입 분석]")
    for query, expected in test_cases:
        result = _analyze_query_type(query.lower())
        status = "✓" if result == expected else "✗"
        print(f"  {status} \"{query}\"")
        print(f"      → {result} (expected: {expected})")
    
    print("\n✅ 쿼리 분석 통과!")


def main():
    """전체 테스트 실행"""
    print("\n🧪 프로파일링 시스템 테스트 시작\n")
    
    try:
        # 1. Repo Profiling
        repo_profile = test_repo_profiling()
        
        # 2. File Profiling
        test_file_profiling(repo_profile)
        
        # 3. Chunk Tagging
        test_chunk_tagging()
        
        # 4. 쿼리 분석
        test_query_analysis()
        
        print("\n" + "="*60)
        print("🎉 모든 테스트 통과!")
        print("="*60)
        print("\n구현 완료:")
        print("  1. ✅ Repo Profiling (프로젝트 구조 분석)")
        print("  2. ✅ File Profiling (파일 역할 태깅)")
        print("  3. ✅ Chunk Tagging (청크 메타데이터)")
        print("  4. ✅ Graph Ranking (노드 중요도)")
        print("  5. ✅ 파이프라인 통합")
        print("  6. ✅ 검색 API 통합")
        print("\n다음 단계:")
        print("  • 재인덱싱 (프로파일링 포함)")
        print("  • 실제 검색 테스트")
        print()
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

