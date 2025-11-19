#!/usr/bin/env python
"""프로파일링 포함 재인덱싱 스크립트"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.bootstrap import create_bootstrap
from src.core.models import RepoConfig

print("="*70)
print("🔄 프로파일링 포함 재인덱싱")
print("="*70)

bootstrap = create_bootstrap()
repo_id = "semantica-codegraph"
repo_root = str(Path(__file__).parent)

# 1. 기존 저장소 삭제
print(f"\n[1단계] 기존 저장소 삭제")
try:
    bootstrap.repo_store.delete(repo_id)
    bootstrap.graph_store.delete_repo(repo_id)
    print(f"  ✓ {repo_id} 삭제 완료")
except Exception as e:
    print(f"  ⚠️  삭제 실패 (없을 수 있음): {e}")

# 2. 재인덱싱
print(f"\n[2단계] 재인덱싱 시작")
print(f"  Repo ID: {repo_id}")
print(f"  Root: {repo_root}")
print(f"  시작 시간: {time.strftime('%H:%M:%S')}")

start_time = time.time()

try:
    result = bootstrap.pipeline.index_repository(
        root_path=repo_root,
        repo_id=repo_id,
        name="semantica-codegraph",
        config=RepoConfig(
            languages=["python"],
            exclude_patterns=["*.pyc", "__pycache__", ".git", ".venv", "venv", "*.egg-info"],
        ),
        parallel=False,  # 병렬 처리 끄기 (connection pool 문제 방지)
    )
    
    duration = time.time() - start_time
    
    print(f"\n[인덱싱 결과]")
    print(f"  ✓ 상태: {result.status}")
    print(f"  ✓ 파일: {result.total_files}개")
    print(f"  ✓ 노드: {result.total_nodes}개")
    print(f"  ✓ 청크: {result.total_chunks}개")
    print(f"  ✓ 소요 시간: {duration:.1f}초")
    
    # failed_files 속성 체크
    if hasattr(result, 'failed_files') and result.failed_files:
        print(f"  ⚠️  실패 파일: {len(result.failed_files)}개")

except Exception as e:
    print(f"\n❌ 인덱싱 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Repo Profile 확인
print(f"\n[3단계] Repo Profile 확인")
try:
    repo_profile = bootstrap.repo_store.get_profile(repo_id)
    
    if repo_profile:
        print(f"  ✓ Repo Profile 생성됨!")
        print(f"    • Primary Language: {repo_profile.primary_language}")
        print(f"    • Framework: {repo_profile.framework}")
        print(f"    • Project Type: {repo_profile.project_type}")
        print(f"    • API Dirs: {len(repo_profile.api_directories)}개")
        if repo_profile.api_directories:
            for d in repo_profile.api_directories[:3]:
                print(f"        - {d}")
        print(f"    • API Patterns: {repo_profile.api_patterns[:3]}")
        print(f"    • Entry Points: {repo_profile.entry_points[:3]}")
        print(f"    • Languages:")
        for lang, lines in sorted(repo_profile.languages.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"        - {lang}: {lines:,} lines")
    else:
        print(f"  ✗ Repo Profile 없음 - 프로파일링 실행 안됨")
        sys.exit(1)

except Exception as e:
    print(f"  ✗ Profile 조회 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. File Profile 확인
print(f"\n[4단계] File Profile 확인")
try:
    api_files = bootstrap.repo_store.get_file_profiles_by_role(repo_id, "api")
    test_files = bootstrap.repo_store.get_file_profiles_by_role(repo_id, "test")
    model_files = bootstrap.repo_store.get_file_profiles_by_role(repo_id, "model")
    
    print(f"  ✓ File Profiles 생성됨!")
    print(f"    • API 파일: {len(api_files)}개")
    print(f"    • Test 파일: {len(test_files)}개")
    print(f"    • Model 파일: {len(model_files)}개")
    
    if api_files:
        print(f"\n  [API 파일 샘플]")
        for fp in api_files[:3]:
            print(f"    • {fp.file_path}")
            print(f"        - framework: {fp.api_framework}")
            print(f"        - endpoints: {len(fp.endpoints)}개")
            if fp.endpoints:
                for ep in fp.endpoints[:2]:
                    print(f"            {ep['method']} {ep['path']}")

except Exception as e:
    print(f"  ✗ File Profile 조회 실패: {e}")

# 5. Chunk Metadata 확인
print(f"\n[5단계] Chunk Metadata 확인")
try:
    if api_files and api_files[0]:
        sample_file = api_files[0].file_path
        chunks = bootstrap.chunk_store.get_chunks_by_file(repo_id, sample_file)
        
        if chunks:
            sample_chunk = chunks[0]
            metadata = bootstrap.chunk_store.get_chunk_metadata(repo_id, sample_chunk.id)
            
            if metadata:
                print(f"  ✓ Chunk Metadata 생성됨!")
                print(f"    파일: {sample_file}")
                print(f"    청크 ID: {sample_chunk.id}")
                print(f"    메타데이터:")
                for key, value in list(metadata.items())[:10]:
                    print(f"        - {key}: {value}")
            else:
                print(f"  ⚠️  메타데이터 없음")
        else:
            print(f"  ⚠️  청크 없음")

except Exception as e:
    print(f"  ✗ Chunk Metadata 조회 실패: {e}")

print("\n" + "="*70)
print("🎉 재인덱싱 완료! (프로파일링 포함)")
print("="*70)
print("\n다음 단계: 검색 테스트")
print("  cd /Users/songmin/Documents/code-jo/semantica/semantica-copilot")
print("  uv run python example/code-question-test.py --llm \"이 프로젝트의 API 엔드포인트는 어디에 정의되어 있어?\"")
print()

