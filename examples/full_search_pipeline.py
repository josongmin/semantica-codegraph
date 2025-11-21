#!/usr/bin/env python3
"""전체 검색 파이프라인 사용 예시"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.bootstrap import create_bootstrap  # noqa: E402


def main():
    """전체 검색 파이프라인 예시"""
    parser = argparse.ArgumentParser(description="전체 검색 파이프라인 실행 (인덱싱 + 검색 + LLM 호출)")
    parser.add_argument(
        "query",
        type=str,
        help="검색할 쿼리 (예: '와인정보', '함수 정의 찾기')",
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default="/Users/songmin/Documents/code-jo/perplexity-test",
        help="인덱싱할 저장소 경로 (기본값: /Users/songmin/Documents/code-jo/perplexity-test)",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="example-repo",
        help="저장소 이름 (기본값: example-repo)",
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="인덱싱 스킵 (기존 인덱싱된 저장소 사용)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="기존 저장소 ID (--skip-indexing 사용 시 필수)",
    )

    args = parser.parse_args()

    query = args.query

    # 1. Bootstrap 생성
    print("Bootstrap 초기화...")
    bootstrap = create_bootstrap()

    # 2. 저장소 인덱싱
    if not args.skip_indexing:
        print("\n1. 저장소 인덱싱")
        print("-" * 80)
        print("[정보] 인덱싱은 매번 새로 수행됩니다.")
        print("       - 파싱 단계만 캐시 사용 (ParseCache)")
        print("       - 그래프 저장, 청킹, 임베딩은 매번 재수행")

        test_repo_path = Path(args.repo_path)
        if not test_repo_path.exists():
            print(f"[오류] 저장소 경로가 존재하지 않습니다: {test_repo_path}")
            sys.exit(1)

        result = bootstrap.pipeline.index_repository(
            root_path=str(test_repo_path), name=args.repo_name
        )

        print("[OK] 인덱싱 완료:")
        print(f"   - 파일: {result.total_files}개")
        print(f"   - 노드: {result.total_nodes}개")
        print(f"   - 청크: {result.total_chunks}개")
        print(f"   - 소요: {result.duration_seconds:.2f}초")

        repo_id = result.repo_id
    else:
        if not args.repo_id:
            print("[오류] --skip-indexing 사용 시 --repo-id가 필요합니다.")
            sys.exit(1)
        repo_id = args.repo_id
        print(f"\n[정보] 인덱싱 스킵, 기존 저장소 사용: {repo_id}")

    # 3. Hybrid 검색
    print("\n2. Hybrid 검색 (Lexical + Semantic + Graph)")
    print("-" * 80)

    print(f"검색 쿼리: '{query}'")

    candidates = bootstrap.hybrid_retriever.retrieve(repo_id=repo_id, query=query, k=10)

    print(f"\n검색된 후보: {len(candidates)}개")
    for i, candidate in enumerate(candidates[:3], 1):
        print(f"\n[{i}] {candidate.file_path}")
        print(f"    청크 ID: {candidate.chunk_id}")
        print(f"    특성: {candidate.features}")

    # 4. Ranker 적용
    print("\n3. Ranker 적용")
    print("-" * 80)

    ranked = bootstrap.reranker.rank(candidates, max_items=5)

    print("랭킹된 결과 (상위 5개):")
    for i, candidate in enumerate(ranked, 1):
        final_score = candidate.features.get("final_score", 0)
        print(f"\n[{i}] {candidate.file_path}")
        print(f"    점수: {final_score:.4f}")

    # 5. Context Packer 적용
    print("\n4. Context Packer 적용 (LLM용 컨텍스트)")
    print("-" * 80)

    context = bootstrap.context_packer.pack(candidates=ranked, max_tokens=2000)

    print("\nPrimary snippet:")
    print(f"  파일: {context.primary.file_path}")
    print(f"  역할: {context.primary.role}")
    print(f"  위치: Line {context.primary.span[0]}-{context.primary.span[2]}")
    print(f"  텍스트 길이: {len(context.primary.text)} 글자")
    print("  코드 미리보기:")
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

    # 6. LLM 프롬프트 생성
    print("\n6. LLM 프롬프트 생성")
    print("-" * 80)

    # Markdown 형식 프롬프트
    prompt_markdown = bootstrap.context_packer.to_prompt(
        context=context, query=query, format="markdown"
    )

    print("\n[Markdown 형식 프롬프트]")
    print(prompt_markdown)

    # Plain 텍스트 형식 프롬프트
    prompt_plain = bootstrap.context_packer.to_prompt(context=context, query=query, format="plain")

    print("\n[Plain 텍스트 형식 프롬프트]")
    print(prompt_plain[:500] + "...")  # 일부만 표시

    # 7. LLM 호출
    print("\n7. LLM 호출")
    print("-" * 80)

    llm_response = call_llm(prompt_markdown, query)
    if llm_response:
        print("\n[LLM 응답]")
        print(llm_response)
    else:
        print("\n[경고] LLM 호출 실패 (API 키 확인 필요)")

    print("\n전체 파이프라인 완료!")


def call_llm(prompt: str, query: str) -> str | None:
    """
    오픈소스 LLM 호출 라이브러리 사용 (litellm)

    litellm은 오픈소스 라이브러리로 여러 LLM API를 통합 지원합니다.
    실제 모델은 OpenAI API를 사용하지만, 호출 라이브러리는 오픈소스입니다.

    사용법:
        1. 설치: pip install litellm
        2. 환경변수 설정: OPENAI_API_KEY

    Args:
        prompt: LLM에 전달할 프롬프트
        query: 원본 검색 쿼리

    Returns:
        LLM 응답 텍스트 또는 None
    """
    # OpenAI API 키 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("  [경고] OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return None

    # 시스템 프롬프트
    system_prompt = """You are a helpful code assistant. Answer questions about the codebase based on the provided context.
Be concise and accurate. If the context doesn't contain enough information, say so."""

    user_message = f"{prompt}\n\nQuestion: {query}\n\nAnswer:"

    try:
        import litellm

        # 모델 설정 (환경변수로 변경 가능)
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")  # 또는 "gpt-3.5-turbo"

        print(f"  [정보] litellm을 통해 OpenAI API 호출: {model}")

        # litellm을 통한 OpenAI 호출
        response = litellm.completion(
            model=f"openai/{model}",  # openai/ 접두사로 OpenAI API 지정
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1000,
            api_key=openai_key,
        )

        # 타입 안전성을 위해 명시적 변환
        content = response.choices[0].message.content
        if content is None:
            return None
        # litellm의 반환 타입이 Any이므로 명시적 변환 필요
        result: str = str(content)
        return result

    except ImportError:
        print("  [정보] litellm 라이브러리가 설치되지 않았습니다: pip install litellm")
        return None
    except Exception as e:
        print(f"  [오류] LLM 호출 실패: {e}")
        return None


if __name__ == "__main__":
    main()
