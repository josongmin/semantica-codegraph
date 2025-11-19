#!/usr/bin/env python
"""재인덱싱 및 검색 테스트"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.bootstrap import create_bootstrap
from src.core.models import RepoConfig

def test_reindex_with_profiling():
    """프로파일링 포함 재인덱싱 테스트"""
    print("\n" + "="*60)
    print("🔄 재인덱싱 테스트 (프로파일링 포함)")
    print("="*60)
    
    bootstrap = create_bootstrap()
    repo_id = "semantica-codegraph"
    repo_root = str(Path(__file__).parent)
    
    print(f"\n[1단계] 기존 저장소 삭제")
    try:
        bootstrap.repo_store.delete(repo_id)
        print(f"  ✓ {repo_id} 삭제 완료")
    except Exception as e:
        print(f"  ⚠️  삭제 실패 (없을 수 있음): {e}")
    
    print(f"\n[2단계] 재인덱싱 시작")
    print(f"  Repo: {repo_id}")
    print(f"  Path: {repo_root}")
    
    result = bootstrap.pipeline.index_repository(
        root_path=repo_root,
        repo_id=repo_id,
        name="semantica-codegraph",
        config=RepoConfig(
            languages=["python"],
            exclude_patterns=["*.pyc", "__pycache__", ".git", ".venv", "venv"],
        ),
    )
    
    print(f"\n[인덱싱 결과]")
    print(f"  상태: {result.status}")
    print(f"  파일: {result.total_files}개")
    print(f"  노드: {result.total_nodes}개")
    print(f"  청크: {result.total_chunks}개")
    print(f"  소요 시간: {result.duration_seconds:.1f}초")
    
    print(f"\n[3단계] Repo Profile 확인")
    repo_profile = bootstrap.repo_store.get_profile(repo_id)
    
    if repo_profile:
        print(f"  ✓ Repo Profile 생성됨!")
        print(f"    Framework: {repo_profile.framework}")
        print(f"    Project Type: {repo_profile.project_type}")
        print(f"    API Dirs: {len(repo_profile.api_directories)}개")
        print(f"    API Patterns: {repo_profile.api_patterns[:3]}")
        print(f"    Entry Points: {repo_profile.entry_points[:3]}")
    else:
        print(f"  ✗ Repo Profile 없음")
        return
    
    print(f"\n[4단계] File Profile 확인")
    api_files = bootstrap.repo_store.get_file_profiles_by_role(repo_id, "api")
    
    if api_files:
        print(f"  ✓ API 파일 {len(api_files)}개 발견!")
        for fp in api_files[:5]:
            print(f"    - {fp.file_path}")
            print(f"      endpoints: {len(fp.endpoints)}개")
            if fp.endpoints:
                print(f"        {fp.endpoints[0]}")
    else:
        print(f"  ⚠️  API 파일 프로파일 없음")
    
    print(f"\n[5단계] Chunk Metadata 확인")
    # 첫 번째 API 파일의 청크 확인
    if api_files:
        sample_file = api_files[0].file_path
        chunks = bootstrap.chunk_store.get_chunks_by_file(repo_id, sample_file)
        print(f"  파일: {sample_file}")
        print(f"  청크: {len(chunks)}개")
        
        if chunks:
            metadata = bootstrap.chunk_store.get_chunk_metadata(repo_id, chunks[0].id)
            if metadata:
                print(f"  샘플 메타데이터:")
                for key, value in list(metadata.items())[:8]:
                    print(f"    {key}: {value}")
            else:
                print(f"  ⚠️  메타데이터 없음")
    
    print("\n✅ 재인덱싱 테스트 완료!")
    return repo_profile


if __name__ == "__main__":
    try:
        test_reindex_with_profiling()
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

