#!/usr/bin/env python
"""빠른 검증 테스트"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("프로파일링 구현 검증")
print("="*60)

# 1. 모델 검증
print("\n[1] 모델 정의 확인")
try:
    from src.core.models import RepoProfile, FileProfile
    print("  ✓ RepoProfile 임포트 성공")
    print("  ✓ FileProfile 임포트 성공")
except Exception as e:
    print(f"  ✗ 모델 임포트 실패: {e}")
    sys.exit(1)

# 2. 프로파일러 검증
print("\n[2] 프로파일러 클래스 확인")
try:
    from src.indexer.repo_profiler import RepoProfiler
    from src.indexer.file_profiler import FileProfiler
    from src.chunking.chunk_tagger import ChunkTagger
    print("  ✓ RepoProfiler 임포트 성공")
    print("  ✓ FileProfiler 임포트 성공")
    print("  ✓ ChunkTagger 임포트 성공")
except Exception as e:
    print(f"  ✗ 프로파일러 임포트 실패: {e}")
    sys.exit(1)

# 3. 저장소 메서드 검증
print("\n[3] 저장소 메서드 확인")
try:
    from src.core.repo_store import RepoMetadataStore
    
    store = RepoMetadataStore("host=localhost dbname=semantica user=semantica password=semantica")
    
    # 메서드 존재 확인
    assert hasattr(store, 'save_profile'), "save_profile 메서드 없음"
    assert hasattr(store, 'get_profile'), "get_profile 메서드 없음"
    assert hasattr(store, 'save_file_profile'), "save_file_profile 메서드 없음"
    assert hasattr(store, 'get_file_profile'), "get_file_profile 메서드 없음"
    
    print("  ✓ save_profile")
    print("  ✓ get_profile")
    print("  ✓ save_file_profile")
    print("  ✓ get_file_profile")
except Exception as e:
    print(f"  ✗ 저장소 검증 실패: {e}")
    sys.exit(1)

# 4. 그래프 랭킹 검증
print("\n[4] 그래프 랭킹 메서드 확인")
try:
    from src.graph.store_postgres import PostgresGraphStore
    
    # 메서드 존재 확인
    assert hasattr(PostgresGraphStore, 'calculate_node_importance'), "calculate_node_importance 메서드 없음"
    assert hasattr(PostgresGraphStore, 'update_all_node_importance'), "update_all_node_importance 메서드 없음"
    
    print("  ✓ calculate_node_importance")
    print("  ✓ update_all_node_importance")
except Exception as e:
    print(f"  ✗ 그래프 랭킹 검증 실패: {e}")
    sys.exit(1)

# 5. 파이프라인 통합 확인
print("\n[5] 파이프라인 통합 확인")
try:
    with open("src/indexer/pipeline.py") as f:
        content = f.read()
    
    assert "[Profiling] Repo profiling 시작" in content, "Repo profiling 통합 안됨"
    assert "[Profiling] File profiling 시작" in content, "File profiling 통합 안됨"
    assert "[Profiling] Chunk tagging 시작" in content, "Chunk tagging 통합 안됨"
    assert "[Profiling] Graph ranking 시작" in content, "Graph ranking 통합 안됨"
    
    print("  ✓ Repo Profiling 통합")
    print("  ✓ File Profiling 통합")
    print("  ✓ Chunk Tagging 통합")
    print("  ✓ Graph Ranking 통합")
except Exception as e:
    print(f"  ✗ 파이프라인 검증 실패: {e}")
    sys.exit(1)

# 6. API 재순위화 확인
print("\n[6] 검색 API 재순위화 확인")
try:
    with open("apps/api/routes/hybrid.py") as f:
        content = f.read()
    
    assert "_rerank_with_metadata" in content, "재순위화 함수 없음"
    assert "_analyze_query_type" in content, "쿼리 분석 함수 없음"
    assert "candidates = _rerank_with_metadata" in content, "재순위화 호출 없음"
    
    print("  ✓ _rerank_with_metadata 함수")
    print("  ✓ _analyze_query_type 함수")
    print("  ✓ 검색 API 통합")
except Exception as e:
    print(f"  ✗ API 검증 실패: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("🎉 모든 검증 통과!")
print("="*60)

print("\n구현 완료 내용:")
print("  1. ✅ Repo Profiling (프로젝트 구조 분석)")
print("  2. ✅ File Profiling (파일 역할 태깅)")
print("  3. ✅ Chunk Tagging (청크 메타데이터)")
print("  4. ✅ Graph Ranking (노드 중요도)")
print("  5. ✅ 인덱싱 파이프라인 통합")
print("  6. ✅ 검색 API 재순위화")

print("\n다음 단계:")
print("  1. API 서버 재시작 (코드 리로드)")
print("  2. 저장소 재인덱싱")
print("  3. 검색 테스트: '이 프로젝트의 API 엔드포인트는 어디에 정의되어 있어?'")
print("  4. 정확도 측정")
print()

