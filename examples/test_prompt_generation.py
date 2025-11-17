#!/usr/bin/env python3
"""프롬프트 생성 테스트 스크립트"""

from unittest.mock import MagicMock
from src.core.models import PackedContext, PackedSnippet
from src.context.packer import ContextPacker


def main():
    """프롬프트 생성 테스트"""
    print("=" * 80)
    print("프롬프트 생성 테스트")
    print("=" * 80)
    
    # Mock 스토어 생성
    chunk_store = MagicMock()
    graph_store = MagicMock()
    packer = ContextPacker(chunk_store, graph_store)
    
    # PackedContext 생성
    primary = PackedSnippet(
        repo_id="test",
        file_path="src/main.py",
        span=(10, 0, 20, 0),
        role="primary",
        text="""class User:
    def __init__(self, name: str):
        self.name = name
    
    def greet(self) -> str:
        return f"Hello, {self.name}!"
""",
        meta={"chunk_id": "chunk-1", "node_id": "node-1"}
    )
    
    supporting = [
        PackedSnippet(
            repo_id="test",
            file_path="src/utils.py",
            span=(5, 0, 10, 0),
            role="caller",
            text="""def create_user():
    user = User("Alice")
    return user.greet()
""",
            meta={"chunk_id": "chunk-2"}
        ),
        PackedSnippet(
            repo_id="test",
            file_path="src/types.py",
            span=(0, 0, 5, 0),
            role="type",
            text="""from typing import Optional

class UserProfile:
    name: Optional[str] = None
""",
            meta={"chunk_id": "chunk-3"}
        ),
        PackedSnippet(
            repo_id="test",
            file_path="tests/test_user.py",
            span=(0, 0, 8, 0),
            role="test",
            text="""def test_user_greet():
    user = User("Bob")
    assert user.greet() == "Hello, Bob!"
""",
            meta={"chunk_id": "chunk-4"}
        ),
    ]
    
    context = PackedContext(primary=primary, supporting=supporting)
    
    # 1. Markdown 형식 프롬프트
    print("\n[1] Markdown 형식 프롬프트")
    print("-" * 80)
    prompt_md = packer.to_prompt(context, query="User 클래스", format="markdown")
    print(prompt_md)
    
    # 2. Plain 텍스트 형식 프롬프트
    print("\n\n[2] Plain 텍스트 형식 프롬프트")
    print("-" * 80)
    prompt_plain = packer.to_prompt(context, query="User 클래스", format="plain")
    print(prompt_plain)
    
    # 3. 쿼리 없이 프롬프트 생성
    print("\n\n[3] 쿼리 없이 프롬프트 생성")
    print("-" * 80)
    prompt_no_query = packer.to_prompt(context, format="markdown")
    print(prompt_no_query)
    
    print("\n" + "=" * 80)
    print("✅ 테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

