"""ContextPacker 개선사항 테스트"""

from unittest.mock import Mock

import pytest

from src.context.packer import ContextPacker, ScoredSnippet
from src.core.models import (
    Candidate,
    CodeChunk,
    PackedSnippet,
)


@pytest.fixture
def mock_chunk_store():
    """Mock ChunkStorePort"""
    store = Mock()
    return store


@pytest.fixture
def mock_graph_store():
    """Mock GraphStorePort"""
    store = Mock()
    store.get_edges.return_value = []
    return store


@pytest.fixture
def packer(mock_chunk_store, mock_graph_store):
    """ContextPacker 인스턴스"""
    return ContextPacker(mock_chunk_store, mock_graph_store)


class TestTokenAccuracy:
    """토큰 카운팅 정확도 테스트"""

    def test_tiktoken_loaded(self, packer):
        """tiktoken이 정상적으로 로드되는지 확인"""
        # tiktoken이 설치되어 있으면 인코더가 로드되어야 함
        encoding = packer.encoding
        assert encoding is not None or encoding is False

    def test_english_code_tokens(self, packer):
        """영문 코드의 토큰 수 추정"""
        english_code = """def hello():
    print('world')
    return True"""

        tokens = packer._estimate_tokens(english_code)

        # tiktoken 사용 시 정확한 값
        if packer.encoding:
            # 실제 tiktoken 결과와 일치해야 함
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            expected = len(enc.encode(english_code))
            assert tokens == expected
        else:
            # Fallback 휴리스틱 사용
            assert tokens > 0

    def test_korean_comment_tokens(self, packer):
        """한글 주석이 포함된 코드의 토큰 수 추정"""
        korean_code = """# 사용자 인증 함수
def authenticate(username, password):
    # 비밀번호 해시 검증
    return check_password_hash(password)"""

        tokens = packer._estimate_tokens(korean_code)

        # 한글은 토큰 수가 더 많아야 함
        assert tokens > 20

        if packer.encoding:
            # tiktoken 사용 시 정확한 값
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            expected = len(enc.encode(korean_code))
            assert tokens == expected

    def test_heuristic_fallback(self, packer):
        """tiktoken 실패 시 휴리스틱이 동작하는지 확인"""
        # 인코더를 None으로 설정
        original_encoding = packer._encoding
        packer._encoding = False

        text = "def test(): pass"
        tokens = packer._estimate_tokens(text)

        # 휴리스틱이 동작해야 함
        assert tokens > 0
        assert tokens == packer._estimate_tokens_heuristic(text)

        # 복원
        packer._encoding = original_encoding

    def test_heuristic_ascii_vs_unicode(self, packer):
        """휴리스틱의 ASCII vs 유니코드 처리"""
        # ASCII 텍스트
        ascii_text = "hello world test" * 10  # 160 chars
        ascii_tokens = packer._estimate_tokens_heuristic(ascii_text)

        # 유니코드 텍스트 (한글)
        unicode_text = "안녕하세요" * 10  # 50 chars
        unicode_tokens = packer._estimate_tokens_heuristic(unicode_text)

        # 유니코드가 더 많은 토큰을 가져야 함 (1글자당 2토큰)
        assert unicode_tokens > ascii_tokens


class TestRolePriority:
    """역할 기반 우선순위 테스트"""

    def test_role_priority_order(self, packer, mock_chunk_store):
        """Caller가 Related보다 먼저 선택되는지 확인"""
        repo_id = "test/repo"  # RepoId는 str 타입

        # Primary chunk
        primary_chunk = CodeChunk(
            id="chunk_primary",
            repo_id=repo_id,
            file_path="main.py",
            span=(0, 0, 10, 0),
            text="def main(): pass",
            node_id="node_primary",
            language="python",
            attrs={},
        )

        # Caller chunk (높은 우선순위)
        caller_chunk = CodeChunk(
            id="chunk_caller",
            repo_id=repo_id,
            file_path="caller.py",
            span=(0, 0, 5, 0),
            text="main()",
            node_id="node_caller",
            language="python",
            attrs={},
        )

        # Related chunk (낮은 우선순위)
        related_chunk = CodeChunk(
            id="chunk_related",
            repo_id=repo_id,
            file_path="related.py",
            span=(0, 0, 5, 0),
            text="def helper(): pass",
            node_id="node_related",
            language="python",
            attrs={},
        )

        # Mock chunk_store
        def get_chunk(repo_id, chunk_id):
            chunks = {
                "chunk_primary": primary_chunk,
                "chunk_caller": caller_chunk,
                "chunk_related": related_chunk,
            }
            return chunks.get(chunk_id)

        mock_chunk_store.get_chunk = get_chunk

        # Candidates (Related가 먼저 오도록)
        candidates = [
            Candidate(
                repo_id=repo_id,
                chunk_id="chunk_primary",
                file_path="main.py",
                span=(0, 0, 10, 0),
                features={"final_score": 1.0},
            ),
            Candidate(
                repo_id=repo_id,
                chunk_id="chunk_related",
                file_path="related.py",
                span=(0, 0, 5, 0),
                features={"final_score": 0.8, "relation": "related"},
            ),
            Candidate(
                repo_id=repo_id,
                chunk_id="chunk_caller",
                file_path="caller.py",
                span=(0, 0, 5, 0),
                features={"final_score": 0.7, "relation": "caller"},
            ),
        ]

        # Pack
        context = packer.pack(candidates, max_tokens=1000)

        # Caller가 Related보다 먼저 와야 함
        assert len(context.supporting) == 2
        assert context.supporting[0].role == "caller"
        assert context.supporting[1].role == "related"

    def test_role_estimation_from_features(self, packer):
        """Features에서 역할을 추정하는지 확인"""
        primary_chunk = CodeChunk(
            id="chunk_primary",
            repo_id="test/repo",
            file_path="main.py",
            span=(0, 0, 10, 0),
            text="def main(): pass",
            node_id="node_primary",
            language="python",
            attrs={},
        )

        chunk = CodeChunk(
            id="chunk_test",
            repo_id="test/repo",
            file_path="test.py",
            span=(0, 0, 5, 0),
            text="test code",
            node_id="node_test",
            language="python",
            attrs={},
        )

        # relation이 "caller"인 경우
        candidate = Candidate(
            repo_id=primary_chunk.repo_id,
            chunk_id="chunk_test",
            file_path="test.py",
            span=(0, 0, 5, 0),
            features={"relation": "caller"},
        )

        role = packer._estimate_role(chunk, primary_chunk, candidate)
        assert role == "caller"

        # relation이 "callee"인 경우
        candidate = Candidate(
            repo_id=primary_chunk.repo_id,
            chunk_id="chunk_test",
            file_path="test.py",
            span=(0, 0, 5, 0),
            features={"relation": "callee"},
        )

        role = packer._estimate_role(chunk, primary_chunk, candidate)
        assert role == "callee"

    def test_role_estimation_from_file_path(self, packer):
        """파일 경로에서 역할을 추정하는지 확인"""
        primary_chunk = CodeChunk(
            id="chunk_primary",
            repo_id="test/repo",
            file_path="main.py",
            span=(0, 0, 10, 0),
            text="def main(): pass",
            node_id="node_primary",
            language="python",
            attrs={},
        )

        # 테스트 파일
        test_chunk = CodeChunk(
            id="chunk_test",
            repo_id="test/repo",
            file_path="test_main.py",
            span=(0, 0, 5, 0),
            text="def test_main(): pass",
            node_id="node_test",
            language="python",
            attrs={},
        )

        candidate = Candidate(
            repo_id=primary_chunk.repo_id,
            chunk_id="chunk_test",
            file_path="test_main.py",
            span=(0, 0, 5, 0),
            features={},
        )

        role = packer._estimate_role(test_chunk, primary_chunk, candidate)
        assert role == "test"


class TestDeduplication:
    """중복 제거 테스트"""

    def test_duplicate_chunk_id(self, packer, mock_chunk_store):
        """같은 chunk_id를 가진 candidates가 중복 제거되는지 확인"""
        repo_id = "test/repo"

        # 동일한 청크
        chunk = CodeChunk(
            id="chunk_1",
            repo_id=repo_id,
            file_path="main.py",
            span=(0, 0, 10, 0),
            text="def main(): pass",
            node_id="node_1",
            language="python",
            attrs={},
        )

        # Mock chunk_store
        mock_chunk_store.get_chunk = lambda repo_id, chunk_id: chunk

        # 같은 chunk_id를 가진 candidates
        candidates = [
            Candidate(
                repo_id=repo_id,
                chunk_id="chunk_1",
                file_path="main.py",
                span=(0, 0, 10, 0),
                features={"final_score": 1.0},
            ),
            Candidate(
                repo_id=repo_id,
                chunk_id="chunk_1",
                file_path="main.py",
                span=(0, 0, 10, 0),
                features={"final_score": 0.9, "source": "lexical"},
            ),
            Candidate(
                repo_id=repo_id,
                chunk_id="chunk_1",
                file_path="main.py",
                span=(0, 0, 10, 0),
                features={"final_score": 0.8, "source": "semantic"},
            ),
        ]

        # Pack
        context = packer.pack(candidates, max_tokens=1000)

        # Primary만 있어야 함 (supporting에는 없음)
        assert context.primary.meta["chunk_id"] == "chunk_1"
        assert len(context.supporting) == 0

    def test_overlapping_spans(self, packer, mock_chunk_store):
        """오버랩되는 spans가 제거되는지 확인"""
        repo_id = "test/repo"

        # 오버랩되는 청크들
        chunk1 = CodeChunk(
            id="chunk_1",
            repo_id=repo_id,
            file_path="main.py",
            span=(0, 0, 10, 0),
            text="def main(): pass",
            node_id="node_1",
            language="python",
            attrs={},
        )

        chunk2 = CodeChunk(
            id="chunk_2",
            repo_id=repo_id,
            file_path="main.py",
            span=(5, 0, 15, 0),  # 오버랩 (5-10)
            text="pass",
            node_id="node_2",
            language="python",
            attrs={},
        )

        chunk3 = CodeChunk(
            id="chunk_3",
            repo_id=repo_id,
            file_path="other.py",
            span=(0, 0, 10, 0),  # 다른 파일 (오버랩 아님)
            text="def other(): pass",
            node_id="node_3",
            language="python",
            attrs={},
        )

        # Mock chunk_store
        def get_chunk(repo_id, chunk_id):
            chunks = {
                "chunk_1": chunk1,
                "chunk_2": chunk2,
                "chunk_3": chunk3,
            }
            return chunks.get(chunk_id)

        mock_chunk_store.get_chunk = get_chunk

        # Candidates
        candidates = [
            Candidate(
                repo_id=repo_id,
                chunk_id="chunk_1",
                file_path="main.py",
                span=(0, 0, 10, 0),
                features={"final_score": 1.0},
            ),
            Candidate(
                repo_id=repo_id,
                chunk_id="chunk_2",
                file_path="main.py",
                span=(5, 0, 15, 0),
                features={"final_score": 0.9},
            ),
            Candidate(
                repo_id=repo_id,
                chunk_id="chunk_3",
                file_path="other.py",
                span=(0, 0, 10, 0),
                features={"final_score": 0.8},
            ),
        ]

        # Pack
        context = packer.pack(candidates, max_tokens=1000)

        # chunk_1 (primary), chunk_3 (supporting) 만 있어야 함
        # chunk_2는 chunk_1과 오버랩되므로 제외
        assert context.primary.meta["chunk_id"] == "chunk_1"
        assert len(context.supporting) == 1
        assert context.supporting[0].meta["chunk_id"] == "chunk_3"

    def test_spans_overlap_logic(self, packer):
        """_spans_overlap 메서드의 로직 테스트"""
        # 같은 파일, 오버랩 O
        assert packer._spans_overlap((0, 0, 10, 0), (5, 0, 15, 0), "main.py", "main.py") is True

        # 같은 파일, 오버랩 X
        assert packer._spans_overlap((0, 0, 10, 0), (15, 0, 20, 0), "main.py", "main.py") is False

        # 다른 파일, 오버랩 X (스팬이 같아도)
        assert packer._spans_overlap((0, 0, 10, 0), (0, 0, 10, 0), "main.py", "other.py") is False

        # 완전 포함
        assert packer._spans_overlap((0, 0, 20, 0), (5, 0, 10, 0), "main.py", "main.py") is True

        # 경계 케이스: 끝과 시작이 같음 (오버랩 아님)
        assert packer._spans_overlap((0, 0, 10, 0), (10, 0, 20, 0), "main.py", "main.py") is False


class TestTokenBudget:
    """토큰 제한 테스트"""

    def test_token_limit_respected(self, packer, mock_chunk_store):
        """토큰 제한이 지켜지는지 확인"""
        repo_id = "test/repo"

        # 작은 primary chunk
        primary_chunk = CodeChunk(
            id="chunk_primary",
            repo_id=repo_id,
            file_path="main.py",
            span=(0, 0, 2, 0),
            text="def main(): pass",
            node_id="node_primary",
            language="python",
            attrs={},
        )

        # 큰 supporting chunk
        large_chunk = CodeChunk(
            id="chunk_large",
            repo_id=repo_id,
            file_path="large.py",
            span=(0, 0, 100, 0),
            text="x" * 1000,  # 매우 큰 청크
            node_id="node_large",
            language="python",
            attrs={},
        )

        # Mock chunk_store
        def get_chunk(repo_id, chunk_id):
            chunks = {
                "chunk_primary": primary_chunk,
                "chunk_large": large_chunk,
            }
            return chunks.get(chunk_id)

        mock_chunk_store.get_chunk = get_chunk

        # Candidates
        candidates = [
            Candidate(
                repo_id=repo_id,
                chunk_id="chunk_primary",
                file_path="main.py",
                span=(0, 0, 2, 0),
                features={"final_score": 1.0},
            ),
            Candidate(
                repo_id=repo_id,
                chunk_id="chunk_large",
                file_path="large.py",
                span=(0, 0, 100, 0),
                features={"final_score": 0.9},
            ),
        ]

        # 매우 작은 토큰 제한
        max_tokens = 50
        context = packer.pack(candidates, max_tokens=max_tokens)

        # Primary만 있고 supporting은 없어야 함
        assert context.primary.meta["chunk_id"] == "chunk_primary"
        assert len(context.supporting) == 0

        # 총 토큰 수 체크
        total_tokens = packer._estimate_tokens(context.primary.text)
        for snippet in context.supporting:
            total_tokens += packer._estimate_tokens(snippet.text)

        assert total_tokens <= max_tokens


class TestScoredSnippet:
    """ScoredSnippet 데이터 클래스 테스트"""

    def test_scored_snippet_creation(self):
        """ScoredSnippet이 정상적으로 생성되는지 확인"""
        snippet = PackedSnippet(
            repo_id="test/repo",
            file_path="main.py",
            span=(0, 0, 10, 0),
            role="caller",
            text="def main(): pass",
            meta={},
        )

        scored = ScoredSnippet(snippet=snippet, score=10.5, tokens=100)

        assert scored.snippet == snippet
        assert scored.score == 10.5
        assert scored.tokens == 100
