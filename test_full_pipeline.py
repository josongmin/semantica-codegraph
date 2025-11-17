#!/usr/bin/env python3
"""전체 파이프라인 테스트 스크립트"""

import os
import sys
from pathlib import Path

# 환경 변수 설정
os.environ["EMBEDDING_API_KEY"] = "1A1rgggB7kld772jEqCepElYdK5tgW9R"
os.environ["POSTGRES_PORT"] = "5433"

from src.core.bootstrap import create_bootstrap
from src.core.models import RepoConfig
from src.search.semantic.pgvector_adapter import PgVectorSemanticSearch
from src.search.graph.postgres_graph_adapter import PostgresGraphSearch
from src.search.retriever.hybrid_retriever import HybridRetriever
from src.search.ranking.ranker import Ranker


def test_indexing():
    """인덱싱 테스트"""
    print("\n" + "="*80)
    print("Phase 1: 저장소 인덱싱")
    print("="*80)
    
    bootstrap = create_bootstrap()
    
    # 테스트할 저장소 경로
    test_repo = Path(__file__).parent / "tests" / "fixtures"
    
    print(f"\n저장소 경로: {test_repo}")
    print("인덱싱 시작...")
    
    # 인덱싱 실행
    result = bootstrap.pipeline.index_repository(
        root_path=str(test_repo),
        name="test-fixtures"
    )
    
    print(f"\n인덱싱 결과:")
    print(f"  - 상태: {result.status}")
    print(f"  - 파일 수: {result.total_files}")
    print(f"  - 노드 수: {result.total_nodes}")
    print(f"  - 엣지 수: {result.total_edges}")
    print(f"  - 청크 수: {result.total_chunks}")
    print(f"  - 소요 시간: {result.duration_seconds:.2f}초")
    
    if result.status != "completed":
        print(f"\n❌ 인덱싱 실패: {result.error_message}")
        return None
    
    print("\n✅ 인덱싱 성공!")
    return result.repo_id


def test_lexical_search(repo_id: str):
    """Lexical 검색 테스트"""
    print("\n" + "="*80)
    print("Phase 2: Lexical 검색 (키워드 기반)")
    print("="*80)
    
    bootstrap = create_bootstrap()
    
    query = "User"  # 실제로 코드에 있는 클래스 이름
    print(f"\n검색 쿼리: '{query}'")
    
    try:
        results = bootstrap.lexical_search.search(
            repo_id=repo_id,
            query=query,
            k=5
        )
        
        print(f"\n검색 결과: {len(results)}개")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result.file_path}")
            print(f"    점수: {result.score:.4f}")
            print(f"    위치: Line {result.span[0]}-{result.span[2]}")
        
        print("\n✅ Lexical 검색 성공!")
    except Exception as e:
        print(f"\n⚠️  Lexical 검색 실패: {e}")


def test_semantic_search(repo_id: str):
    """Semantic 검색 테스트"""
    print("\n" + "="*80)
    print("Phase 3: Semantic 검색 (의미론적)")
    print("="*80)
    
    bootstrap = create_bootstrap()
    
    semantic_search = PgVectorSemanticSearch(
        embedding_service=bootstrap.embedding_service,
        embedding_store=bootstrap.embedding_store
    )
    
    query = "함수 정의"
    print(f"\n검색 쿼리: '{query}'")
    
    try:
        results = semantic_search.search(
            repo_id=repo_id,
            query=query,
            k=5
        )
        
        print(f"\n검색 결과: {len(results)}개")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result.file_path}")
            print(f"    점수: {result.score:.4f}")
            print(f"    위치: Line {result.span[0]}-{result.span[2]}")
        
        print("\n✅ Semantic 검색 성공!")
    except Exception as e:
        print(f"\n⚠️  Semantic 검색 실패: {e}")


def test_graph_search(repo_id: str):
    """Graph 검색 테스트"""
    print("\n" + "="*80)
    print("Phase 4: Graph 검색 (그래프 탐색)")
    print("="*80)
    
    bootstrap = create_bootstrap()
    
    graph_search = PostgresGraphSearch(
        graph_store=bootstrap.graph_store
    )
    
    # 샘플 파일의 첫 번째 함수 찾기
    print("\n위치 기반 노드 찾기...")
    node = graph_search.get_node_by_location(
        repo_id=repo_id,
        file_path="sample_python.py",
        line=5
    )
    
    if node:
        print(f"\n찾은 노드:")
        print(f"  - ID: {node.id}")
        print(f"  - 종류: {node.kind}")
        print(f"  - 이름: {node.name}")
        print(f"  - 파일: {node.file_path}")
        
        # 이웃 노드 확장
        print(f"\n이웃 노드 확장 (1-hop)...")
        neighbors = graph_search.expand_neighbors(
            repo_id=repo_id,
            node_id=node.id,
            k=1
        )
        
        print(f"\n이웃 노드: {len(neighbors)}개")
        for i, neighbor in enumerate(neighbors, 1):
            print(f"\n[{i}] {neighbor.name} ({neighbor.kind})")
            print(f"    파일: {neighbor.file_path}")
        
        print("\n✅ Graph 검색 성공!")
    else:
        print("\n⚠️  노드를 찾을 수 없습니다")


def test_hybrid_search(repo_id: str):
    """Hybrid 검색 테스트"""
    print("\n" + "="*80)
    print("Phase 5: Hybrid 검색 (통합)")
    print("="*80)
    
    bootstrap = create_bootstrap()
    
    semantic_search = PgVectorSemanticSearch(
        embedding_service=bootstrap.embedding_service,
        embedding_store=bootstrap.embedding_store
    )
    
    graph_search = PostgresGraphSearch(
        graph_store=bootstrap.graph_store
    )
    
    hybrid = HybridRetriever(
        lexical_search=bootstrap.lexical_search,
        semantic_search=semantic_search,
        graph_search=graph_search
    )
    
    query = "클래스 정의"
    print(f"\n검색 쿼리: '{query}'")
    
    try:
        candidates = hybrid.retrieve(
            repo_id=repo_id,
            query=query,
            k=5
        )
        
        print(f"\n검색 결과: {len(candidates)}개")
        for i, candidate in enumerate(candidates, 1):
            print(f"\n[{i}] {candidate.file_path}")
            print(f"    청크 ID: {candidate.chunk_id}")
            print(f"    특성:")
            for feature, value in candidate.features.items():
                if value > 0:
                    print(f"      - {feature}: {value:.4f}")
        
        # Ranker로 최종 랭킹
        print("\n\nRanker 적용...")
        ranker = Ranker()
        ranked = ranker.rank(candidates, max_items=3)
        
        print(f"\n최종 랭킹 (상위 3개):")
        for i, candidate in enumerate(ranked, 1):
            final_score = candidate.features.get("final_score", 0)
            print(f"\n[{i}] {candidate.file_path}")
            print(f"    최종 점수: {final_score:.4f}")
        
        print("\n✅ Hybrid 검색 + Ranker 성공!")
    except Exception as e:
        print(f"\n⚠️  Hybrid 검색 실패: {e}")


def check_environment():
    """환경 확인"""
    print("\n" + "="*80)
    print("환경 확인")
    print("="*80)
    
    issues = []
    
    # API 키 확인
    api_key = os.getenv("EMBEDDING_API_KEY")
    if api_key:
        print(f"\n✅ EMBEDDING_API_KEY 설정됨")
    else:
        print(f"\n❌ EMBEDDING_API_KEY 없음")
        issues.append("EMBEDDING_API_KEY")
    
    # PostgreSQL 확인
    postgres_port = os.getenv("POSTGRES_PORT", "5432")
    print(f"✅ POSTGRES_PORT: {postgres_port}")
    
    # DB 연결 테스트
    try:
        import psycopg2
        conn_str = f"host=localhost port={postgres_port} dbname=semantica_codegraph user=semantica password=semantica"
        psycopg2.connect(conn_str).close()
        print(f"✅ PostgreSQL 연결 성공")
    except Exception as e:
        print(f"❌ PostgreSQL 연결 실패: {e}")
        issues.append("PostgreSQL")
    
    if issues:
        print(f"\n⚠️  환경 문제: {', '.join(issues)}")
        return False
    
    print(f"\n✅ 모든 환경 확인 완료")
    return True


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("Semantica Codegraph 전체 파이프라인 테스트")
    print("="*80)
    
    # 환경 확인
    if not check_environment():
        print("\n환경 설정을 확인하세요:")
        print("  1. PostgreSQL 시작: docker-compose up -d postgres")
        print("  2. API 키 설정: export EMBEDDING_API_KEY=...")
        return
    
    # 인덱싱 테스트
    repo_id = test_indexing()
    if not repo_id:
        return
    
    # 검색 테스트
    test_lexical_search(repo_id)
    test_semantic_search(repo_id)
    test_graph_search(repo_id)
    test_hybrid_search(repo_id)
    
    print("\n" + "="*80)
    print("✅ 모든 테스트 완료!")
    print("="*80)


if __name__ == "__main__":
    main()

