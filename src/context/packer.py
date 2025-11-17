"""LLM 컨텍스트 패커 (개선 버전)"""

import logging
from dataclasses import dataclass

import tiktoken

from ..core.models import (
    Candidate,
    CodeChunk,
    LocationContext,
    PackedContext,
    PackedSnippet,
    RepoId,
)
from ..core.ports import ChunkStorePort, GraphStorePort

logger = logging.getLogger(__name__)


@dataclass
class ScoredSnippet:
    """점수가 매겨진 스니펫"""
    snippet: PackedSnippet
    score: float
    tokens: int


class ContextPacker:
    """
    개선된 LLM 컨텍스트 패커

    개선 사항:
    1. tiktoken 기반 정확한 토큰 카운팅
    2. 역할 기반 우선순위 정렬
    3. 중복 스니펫 제거 (chunk_id + span overlap)

    역할:
    - Primary snippet (최우선 코드)
    - Supporting snippets (주변 컨텍스트)
      - Caller: 이 코드를 호출하는 곳 (우선순위 최상)
      - Callee: 이 코드가 호출하는 곳
      - Type: 사용하는 타입 정의
      - Test: 테스트 코드
      - Related: 관련 코드
    - 토큰 제한 내 최적화
    """

    # 역할별 우선순위 (높을수록 중요)
    ROLE_PRIORITY = {
        "caller": 10,    # 사용 예시 (가장 중요)
        "callee": 8,     # 의존성
        "type": 7,       # 타입 정의
        "test": 5,       # 테스트 코드
        "related": 3,    # 관련 코드
        "other": 1,      # 기타
    }

    def __init__(
        self,
        chunk_store: ChunkStorePort,
        graph_store: GraphStorePort,
        model_type: str = "gpt-4",
    ):
        """
        Args:
            chunk_store: 청크 스토어
            graph_store: 그래프 스토어
            model_type: 모델 타입 (토크나이저 선택용)
        """
        self.chunk_store = chunk_store
        self.graph_store = graph_store
        self.model_type = model_type
        self._encoding = None

    @property
    def encoding(self):
        """tiktoken 인코더 (lazy loading + 캐싱)"""
        if self._encoding is None:
            try:
                # GPT-4, GPT-3.5, Claude 호환
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken: {e}, using heuristic")
                self._encoding = False
        return self._encoding if self._encoding else None

    def _estimate_tokens(self, text: str) -> int:
        """
        정확한 토큰 카운팅

        tiktoken 사용 시: 정확한 토큰 수
        실패 시: 개선된 휴리스틱
        """
        if self.encoding:
            return len(self.encoding.encode(text))

        # Fallback: 개선된 휴리스틱
        return self._estimate_tokens_heuristic(text)

    def _estimate_tokens_heuristic(self, text: str) -> int:
        """
        개선된 휴리스틱 토큰 추정

        규칙:
        - ASCII 문자: 4글자 ≈ 1토큰
        - 한글/한자/특수문자: 1글자 ≈ 2토큰
        """
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        non_ascii_chars = len(text) - ascii_chars

        # ASCII: 4글자당 1토큰, 비ASCII: 1글자당 2토큰
        estimated = (ascii_chars // 4) + (non_ascii_chars * 2)

        # 최소값 보장
        return max(estimated, len(text) // 6)

    def pack(
        self,
        candidates: list[Candidate],
        max_tokens: int,
        location_ctx: LocationContext | None = None,
    ) -> PackedContext:
        """
        개선된 패킹 알고리즘

        1. Primary 선택
        2. Supporting 점수화 (역할 + 검색 점수)
        3. 중복 제거 (chunk_id + span overlap)
        4. 토큰 제한 내 선택

        Args:
            candidates: Candidate 리스트 (이미 랭킹됨)
            max_tokens: 최대 토큰 수
            location_ctx: 위치 컨텍스트 (선택적)

        Returns:
            PackedContext (primary + supporting snippets)
        """
        if not candidates:
            raise ValueError("No candidates to pack")

        logger.debug(f"Packing {len(candidates)} candidates (max={max_tokens}t)")

        # 1. Primary snippet
        primary_candidate = candidates[0]
        primary_chunk = self.chunk_store.get_chunk(
            primary_candidate.repo_id,
            primary_candidate.chunk_id
        )

        if not primary_chunk:
            raise ValueError(f"Primary chunk not found: {primary_candidate.chunk_id}")

        primary = PackedSnippet(
            repo_id=primary_chunk.repo_id,
            file_path=primary_chunk.file_path,
            span=primary_chunk.span,
            role="primary",
            text=primary_chunk.text,
            meta={
                "chunk_id": primary_chunk.id,
                "node_id": primary_chunk.node_id,
                "features": primary_candidate.features
            }
        )

        primary_tokens = self._estimate_tokens(primary.text)
        remaining_tokens = max_tokens - primary_tokens

        logger.debug(f"Primary: {primary_tokens}t, Remaining: {remaining_tokens}t")

        # 2. Supporting 점수화
        scored_snippets = []

        for candidate in candidates[1:]:
            chunk = self.chunk_store.get_chunk(
                candidate.repo_id,
                candidate.chunk_id
            )

            if not chunk:
                continue

            # 역할 추정
            role = self._estimate_role(chunk, primary_chunk, candidate)

            # 점수 계산: 역할 우선순위 + 검색 점수
            search_score = candidate.features.get("final_score", 0)
            role_priority = self.ROLE_PRIORITY.get(role, 1)

            # 최종 점수 = 역할 가중치 * (1 + 검색 점수)
            final_score = role_priority * (1 + search_score)

            snippet = PackedSnippet(
                repo_id=chunk.repo_id,
                file_path=chunk.file_path,
                span=chunk.span,
                role=role,
                text=chunk.text,
                meta={
                    "chunk_id": chunk.id,
                    "node_id": chunk.node_id,
                    "features": candidate.features,
                    "role_priority": role_priority,
                    "search_score": search_score
                }
            )

            tokens = self._estimate_tokens(chunk.text)

            scored_snippets.append(
                ScoredSnippet(snippet=snippet, score=final_score, tokens=tokens)
            )

        # 3. 점수 기반 정렬 (높은 점수 우선)
        scored_snippets.sort(key=lambda x: x.score, reverse=True)

        # 4. 중복 제거 + 토큰 제한
        seen_chunks = {primary_chunk.id}
        selected_spans = [(primary.file_path, primary.span)]
        supporting = []

        for scored in scored_snippets:
            # Chunk ID 중복 체크
            chunk_id = scored.snippet.meta["chunk_id"]
            if chunk_id in seen_chunks:
                logger.debug(f"Skipping duplicate chunk: {chunk_id}")
                continue

            # Span overlap 체크
            has_overlap = any(
                self._spans_overlap(
                    scored.snippet.span,
                    span,
                    scored.snippet.file_path,
                    file_path
                )
                for file_path, span in selected_spans
            )

            if has_overlap:
                logger.debug(f"Skipping overlapping span: {scored.snippet.file_path}")
                continue

            # 토큰 체크
            if remaining_tokens - scored.tokens < 50:
                logger.debug(f"Token limit reached: {remaining_tokens}t remaining")
                break

            supporting.append(scored.snippet)
            seen_chunks.add(chunk_id)
            selected_spans.append((scored.snippet.file_path, scored.snippet.span))
            remaining_tokens -= scored.tokens

        # 5. 그래프 기반 추가 (caller/callee 우선)
        if location_ctx and remaining_tokens > 200:
            graph_snippets = self._get_graph_context(
                primary_chunk.repo_id,
                primary_chunk.node_id,
                remaining_tokens // 2,
                seen_chunks,
                selected_spans
            )
            supporting.extend(graph_snippets)

        total_supporting_tokens = sum(
            self._estimate_tokens(s.text) for s in supporting
        )

        logger.info(
            f"Packed: primary={primary_tokens}t, "
            f"supporting={len(supporting)} snippets ({total_supporting_tokens}t)"
        )

        return PackedContext(primary=primary, supporting=supporting)

    def _estimate_role(
        self,
        chunk: CodeChunk,
        primary_chunk: CodeChunk,
        candidate: Candidate
    ) -> str:
        """
        스니펫의 역할 추정

        전략:
        1. Candidate features 확인 (relation 정보)
        2. 그래프 관계 확인 (calls, uses 등)
        3. 파일 경로 확인 (test 파일 등)
        4. Features 확인 (graph_score)
        """
        # 1. 메타데이터에 이미 있으면 사용
        if "relation" in candidate.features:
            relation = candidate.features["relation"]
            if relation == "caller":
                return "caller"
            elif relation == "callee":
                return "callee"

        # 2. 그래프 관계 확인
        if primary_chunk.node_id and chunk.node_id:
            try:
                # Primary → chunk 방향 엣지 (Primary가 호출하는 것)
                outgoing_edges = self.graph_store.get_edges(
                    primary_chunk.repo_id,
                    primary_chunk.node_id
                )

                for edge in outgoing_edges:
                    if edge.target_id == chunk.node_id:
                        if edge.type == "calls":
                            return "callee"
                        elif edge.type in ["uses", "defines"]:
                            return "type"

                # chunk → Primary 방향 엣지 (Primary를 호출하는 것)
                incoming_edges = self.graph_store.get_edges(
                    chunk.repo_id,
                    chunk.node_id
                )

                for edge in incoming_edges:
                    if edge.target_id == primary_chunk.node_id:
                        if edge.type == "calls":
                            return "caller"
                        elif edge.type in ["uses", "defines"]:
                            return "type"

            except Exception as e:
                logger.debug(f"Failed to get graph edges: {e}")

        # 3. 파일 경로 기반
        if "test" in chunk.file_path.lower():
            return "test"

        # 4. Features 기반
        if candidate.features.get("graph_score", 0) > 0:
            return "related"

        # 5. 기본값
        return "related"

    def _spans_overlap(
        self,
        span1: tuple[int, int, int, int],
        span2: tuple[int, int, int, int],
        file_path1: str,
        file_path2: str
    ) -> bool:
        """
        두 스팬이 겹치는지 확인

        Args:
            span1: (start_line, start_col, end_line, end_col)
            span2: (start_line, start_col, end_line, end_col)
            file_path1, file_path2: 파일 경로

        Returns:
            True if overlap
        """
        # 다른 파일이면 겹치지 않음
        if file_path1 != file_path2:
            return False

        start1, _, end1, _ = span1
        start2, _, end2, _ = span2

        # 라인 기준 오버랩 체크
        # 겹치지 않는 경우: end1 <= start2 or end2 <= start1
        # 겹치는 경우: not (겹치지 않는 경우)
        return not (end1 <= start2 or end2 <= start1)

    def _get_graph_context(
        self,
        repo_id: RepoId,
        node_id: str,
        max_tokens: int,
        seen_chunks: set,
        selected_spans: list[tuple[str, tuple[int, int, int, int]]]
    ) -> list[PackedSnippet]:
        """
        그래프 기반 관련 코드 조회

        Args:
            repo_id: 저장소 ID
            node_id: 노드 ID
            max_tokens: 최대 토큰 수
            seen_chunks: 이미 선택된 청크 ID 집합
            selected_spans: 이미 선택된 스팬 목록

        Returns:
            PackedSnippet 리스트 (caller, callee 등)
        """
        snippets = []
        remaining_tokens = max_tokens

        # Edge type → role 매핑
        edge_role_map = {
            "calls": "callee",
            "called_by": "caller",
            "uses": "type",
            "used_by": "related",
            "defines": "type",
            "defined_by": "related",
            "inherits": "type",
            "inherited_by": "related",
            "imports": "related",
            "imported_by": "related",
        }

        try:
            # Outgoing edges (Primary가 사용하는 것들)
            outgoing_edges = self.graph_store.get_edges(repo_id, node_id)

            for edge in outgoing_edges[:5]:  # 최대 5개
                if remaining_tokens < 100:
                    break

                # 노드 → 청크 매핑
                chunks = self.chunk_store.get_chunks_by_node(repo_id, edge.target_id)

                if not chunks:
                    continue

                chunk = chunks[0]  # 첫 번째 청크 사용

                # 중복 체크
                if chunk.id in seen_chunks:
                    continue

                # Overlap 체크
                has_overlap = any(
                    self._spans_overlap(chunk.span, span, chunk.file_path, file_path)
                    for file_path, span in selected_spans
                )

                if has_overlap:
                    continue

                chunk_tokens = self._estimate_tokens(chunk.text)

                if chunk_tokens > remaining_tokens:
                    continue

                # Role 결정
                role = edge_role_map.get(edge.type, "related")

                snippets.append(
                    PackedSnippet(
                        repo_id=chunk.repo_id,
                        file_path=chunk.file_path,
                        span=chunk.span,
                        role=role,
                        text=chunk.text,
                        meta={
                            "chunk_id": chunk.id,
                            "node_id": chunk.node_id,
                            "relation": "graph_neighbor",
                            "edge_type": edge.type
                        }
                    )
                )

                seen_chunks.add(chunk.id)
                selected_spans.append((chunk.file_path, chunk.span))
                remaining_tokens -= chunk_tokens

        except Exception as e:
            logger.warning(f"Failed to get graph context: {e}")

        return snippets

    def to_prompt(
        self,
        context: PackedContext,
        query: str | None = None,
        format: str = "markdown"
    ) -> str:
        """
        PackedContext를 LLM 프롬프트 문자열로 변환

        Args:
            context: PackedContext 객체
            query: 사용자 쿼리 (선택적)
            format: 출력 형식 ("markdown" | "plain")

        Returns:
            LLM에 전달할 프롬프트 문자열
        """
        if format == "markdown":
            return self._to_markdown_prompt(context, query)
        else:
            return self._to_plain_prompt(context, query)

    def _to_markdown_prompt(
        self,
        context: PackedContext,
        query: str | None = None
    ) -> str:
        """Markdown 형식 프롬프트 생성"""
        lines = []

        # 헤더
        if query:
            lines.append(f"## 검색 쿼리\n\n{query}\n")

        lines.append("## 코드 컨텍스트\n")

        # Primary snippet
        primary = context.primary
        start_line, _, end_line, _ = primary.span
        language = self._detect_language(primary.file_path)

        lines.append(f"### Primary Code: `{primary.file_path}` (Line {start_line + 1}-{end_line + 1})")
        lines.append("")
        lines.append(f"```{language}")
        lines.append(primary.text)
        lines.append("```")
        lines.append("")

        # Supporting snippets
        if context.supporting:
            lines.append(f"### Supporting Code ({len(context.supporting)} snippets)\n")

            # Role별로 그룹화
            role_groups: dict[str, list[PackedSnippet]] = {}
            for snippet in context.supporting:
                role = snippet.role
                if role not in role_groups:
                    role_groups[role] = []
                role_groups[role].append(snippet)

            # Role 순서 정의 (우선순위 순)
            role_order = ["caller", "callee", "type", "test", "related"]

            for role in role_order:
                if role not in role_groups:
                    continue

                snippets = role_groups[role]
                role_label = {
                    "caller": "호출하는 코드",
                    "callee": "호출되는 코드",
                    "type": "타입 정의",
                    "test": "테스트 코드",
                    "related": "관련 코드"
                }.get(role, role)

                lines.append(f"#### {role_label}")
                lines.append("")

                for snippet in snippets:
                    start_line, _, end_line, _ = snippet.span
                    language = self._detect_language(snippet.file_path)

                    lines.append(f"**`{snippet.file_path}`** (Line {start_line + 1}-{end_line + 1})")
                    lines.append("")
                    lines.append(f"```{language}")
                    lines.append(snippet.text)
                    lines.append("```")
                    lines.append("")

            # 나머지 role들
            for role, snippets in role_groups.items():
                if role in role_order:
                    continue

                lines.append(f"#### {role}")
                lines.append("")

                for snippet in snippets:
                    start_line, _, end_line, _ = snippet.span
                    language = self._detect_language(snippet.file_path)

                    lines.append(f"**`{snippet.file_path}`** (Line {start_line + 1}-{end_line + 1})")
                    lines.append("")
                    lines.append(f"```{language}")
                    lines.append(snippet.text)
                    lines.append("```")
                    lines.append("")

        return "\n".join(lines)

    def _to_plain_prompt(
        self,
        context: PackedContext,
        query: str | None = None
    ) -> str:
        """Plain 텍스트 형식 프롬프트 생성"""
        lines = []

        if query:
            lines.append(f"검색 쿼리: {query}\n")

        lines.append("=" * 80)
        lines.append("코드 컨텍스트")
        lines.append("=" * 80)
        lines.append("")

        # Primary snippet
        primary = context.primary
        start_line, _, end_line, _ = primary.span

        lines.append(f"[Primary] {primary.file_path} (Line {start_line + 1}-{end_line + 1})")
        lines.append("-" * 80)
        lines.append(primary.text)
        lines.append("")

        # Supporting snippets
        if context.supporting:
            lines.append("=" * 80)
            lines.append(f"Supporting Code ({len(context.supporting)} snippets)")
            lines.append("=" * 80)
            lines.append("")

            for i, snippet in enumerate(context.supporting, 1):
                start_line, _, end_line, _ = snippet.span

                lines.append(f"[{i}] {snippet.role.upper()}: {snippet.file_path} (Line {start_line + 1}-{end_line + 1})")
                lines.append("-" * 80)
                lines.append(snippet.text)
                lines.append("")

        return "\n".join(lines)

    def _detect_language(self, file_path: str) -> str:
        """파일 경로에서 언어 감지"""
        ext = file_path.split(".")[-1].lower()

        lang_map = {
            "py": "python",
            "ts": "typescript",
            "tsx": "typescript",
            "js": "javascript",
            "jsx": "javascript",
            "java": "java",
            "go": "go",
            "rs": "rust",
            "cpp": "cpp",
            "c": "c",
            "cs": "csharp",
            "rb": "ruby",
            "php": "php",
            "swift": "swift",
            "kt": "kotlin",
            "scala": "scala",
            "sh": "bash",
            "bash": "bash",
            "zsh": "bash",
            "sql": "sql",
            "html": "html",
            "css": "css",
            "json": "json",
            "yaml": "yaml",
            "yml": "yaml",
            "toml": "toml",
            "md": "markdown",
            "xml": "xml",
        }

        return lang_map.get(ext, "text")
