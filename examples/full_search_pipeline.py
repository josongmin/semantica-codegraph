#!/usr/bin/env python3
"""전체 검색 파이프라인 사용 예시"""

import os

# 환경 변수 설정
os.environ["EMBEDDING_API_KEY"] = "1A1rgggB7kld772jEqCepElYdK5tgW9R"
os.environ["POSTGRES_PORT"] = "5433"

from src.core.bootstrap import create_bootstrap
from src.core.models import LocationContext


def main():
    """전체 검색 파이프라인 예시"""
    
    # 1. Bootstrap 생성
    print("Bootstrap 초기화...")
    bootstrap = create_bootstrap()
    
    # 2. 저장소 인덱싱
    print("\n1. 저장소 인덱싱")
    print("-" * 80)
    
    result = bootstrap.pipeline.index_repository(
        root_path="/Users/josongmin/Documents/jo-codes/semantica-codegraph/tests/fixtures",
        name="example-repo"
    )
    
    print(f"✅ 인덱싱 완료:")
    print(f"   - 파일: {result.total_files}개")
    print(f"   - 노드: {result.total_nodes}개")
    print(f"   - 청크: {result.total_chunks}개")
    print(f"   - 소요: {result.duration_seconds:.2f}초")
    
    repo_id = result.repo_id
    
    # 3. Hybrid 검색
    print("\n2. Hybrid 검색 (Lexical + Semantic + Graph)")
    print("-" * 80)
    
    query = "User 클래스"
    print(f"검색 쿼리: '{query}'")
    
    candidates = bootstrap.hybrid_retriever.retrieve(
        repo_id=repo_id,
        query=query,
        k=10
    )
    
    print(f"\n검색된 후보: {len(candidates)}개")
    for i, candidate in enumerate(candidates[:3], 1):
        print(f"\n[{i}] {candidate.file_path}")
        print(f"    청크 ID: {candidate.chunk_id}")
        print(f"    특성: {candidate.features}")
    
    # 4. Ranker 적용
    print("\n3. Ranker 적용")
    print("-" * 80)
    
    ranked = bootstrap.ranker.rank(candidates, max_items=5)
    
    print(f"랭킹된 결과 (상위 5개):")
    for i, candidate in enumerate(ranked, 1):
        final_score = candidate.features.get("final_score", 0)
        print(f"\n[{i}] {candidate.file_path}")
        print(f"    점수: {final_score:.4f}")
    
    # 5. Context Packer 적용
    print("\n4. Context Packer 적용 (LLM용 컨텍스트)")
    print("-" * 80)
    
    context = bootstrap.context_packer.pack(
        candidates=ranked,
        max_tokens=2000
    )
    
    print(f"\nPrimary snippet:")
    print(f"  파일: {context.primary.file_path}")
    print(f"  역할: {context.primary.role}")
    print(f"  위치: Line {context.primary.span[0]}-{context.primary.span[2]}")
    print(f"  텍스트 길이: {len(context.primary.text)} 글자")
    print(f"  코드 미리보기:")
    print("  " + "\n  ".join(context.primary.text.split("\n")[:5]))
    
    print(f"\nSupporting snippets: {len(context.supporting)}개")
    for i, snippet in enumerate(context.supporting[:3], 1):
        print(f"\n[{i}] {snippet.file_path}")
        print(f"    역할: {snippet.role}")
        print(f"    위치: Line {snippet.span[0]}-{snippet.span[2]}")
    
    # 6. 전체 컨텍스트 요약
    print("\n5. 최종 컨텍스트 요약")
    print("-" * 80)
    
    total_text = context.primary.text
    for snippet in context.supporting:
        total_text += "\n\n" + snippet.text
    
    estimated_tokens = len(total_text) // 4
    print(f"총 텍스트 길이: {len(total_text)} 글자")
    print(f"예상 토큰: ~{estimated_tokens} 토큰")
    print(f"Primary: 1개, Supporting: {len(context.supporting)}개")
    
    print("\n" + "="*80)
    print("✅ 전체 파이프라인 완료!")
    print("="*80)
    
    # LLM에 전달할 최종 형식 예시
    print("\nLLM에 전달할 컨텍스트 형식:")
    print(f"""
Primary Code ({context.primary.file_path}):
```{context.primary.text.split(chr(10))[0].strip() if context.primary.text else ''}
{context.primary.text[:200]}...
```

Supporting Code ({len(context.supporting)} snippets):
{chr(10).join(f'- {s.file_path} (Line {s.span[0]}-{s.span[2]})' for s in context.supporting[:3])}
""")


if __name__ == "__main__":
    main()

