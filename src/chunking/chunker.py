"""Chunker: CodeNode → CodeChunk 변환"""

import logging

from ..core.models import CodeChunk, CodeNode

logger = logging.getLogger(__name__)

# tiktoken 초기화 (임베딩 모델용)
try:
    import tiktoken

    # 대부분의 임베딩 모델에서 사용하는 cl100k_base 인코딩
    TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, falling back to simple token estimation")


class Chunker:
    """
    CodeNode를 검색에 최적화된 CodeChunk로 변환

    청킹 전략:
    1. 기본: 1 Node = 1 Chunk
    2. 큰 노드: 여러 청크로 분할 (max_lines)
    3. 작은 노드: 병합 (min_lines)
    4. 오버랩: 청크 간 컨텍스트 공유
    """

    def __init__(
        self,
        max_lines: int = 100,
        min_lines: int = 3,
        overlap_lines: int = 5,
        strategy: str = "node_based",
        max_tokens: int | None = 7000,  # 임베딩 API 제한보다 낮게 설정 (안전 마진)
    ):
        """
        Args:
            max_lines: 청크 최대 라인 수 (이상이면 분할)
            min_lines: 청크 최소 라인 수 (미만이면 병합 고려)
            overlap_lines: 청크 간 오버랩 라인 수
            strategy: 청킹 전략
                - "node_based": 1 Node = 1 Chunk (기본)
                - "size_based": 크기 기반 분할
                - "hierarchical": 계층적 (Class + 각 Method)
            max_tokens: 청크 최대 토큰 수 (None이면 토큰 제한 없음)
                - 기본값 7000은 대부분의 임베딩 모델의 8K 제한보다 낮게 설정 (안전 마진)
        """
        self.max_lines = max_lines
        self.min_lines = min_lines
        self.overlap_lines = overlap_lines
        self.strategy = strategy
        self.max_tokens = max_tokens

    def chunk(self, nodes: list[CodeNode]) -> list[CodeChunk]:
        """
        CodeNode 리스트를 CodeChunk 리스트로 변환

        Args:
            nodes: CodeNode 리스트

        Returns:
            CodeChunk 리스트
        """
        if not nodes:
            logger.warning("No nodes to chunk")
            return []

        if self.strategy == "node_based":
            chunks = self._chunk_node_based(nodes)
        elif self.strategy == "size_based":
            chunks = self._chunk_size_based(nodes)
        elif self.strategy == "hierarchical":
            chunks = self._chunk_hierarchical(nodes)
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, using node_based")
            chunks = self._chunk_node_based(nodes)

        logger.info(
            f"Chunked {len(nodes)} nodes into {len(chunks)} chunks (strategy={self.strategy})"
        )

        return chunks

    def _chunk_node_based(self, nodes: list[CodeNode]) -> list[CodeChunk]:
        """
        Node 기반 청킹: 1 Node = 1 Chunk (병렬 처리)

        가장 단순하고 빠른 방식
        """
        # 병렬 처리 임계값
        if len(nodes) < 100:
            # 노드 수가 적으면 순차 처리 (오버헤드 방지)
            return self._chunk_node_based_sequential(nodes)

        # 병렬 처리
        from concurrent.futures import ThreadPoolExecutor

        chunks = []

        def process_node(node: CodeNode) -> list[CodeChunk]:
            """단일 노드 처리 (병렬 실행용)"""
            # File 노드는 스킵
            if node.kind == "File":
                return []

            # 선택적 토큰 검증
            text_len = len(node.text)

            if self.max_tokens:
                safe_char_limit = self.max_tokens * 3

                if text_len > safe_char_limit:
                    token_count = self._count_tokens(node.text)
                    if token_count > self.max_tokens:
                        logger.debug(
                            f"Node {node.name} exceeds max_tokens ({token_count} > {self.max_tokens}), splitting"
                        )
                        return self._split_node_by_tokens(node)

            return [self._node_to_chunk(node)]

        # 병렬 실행
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_node, nodes)
            for node_chunks in results:
                chunks.extend(node_chunks)

        return chunks

    def _chunk_node_based_sequential(self, nodes: list[CodeNode]) -> list[CodeChunk]:
        """순차 처리 버전 (작은 프로젝트용)"""
        chunks = []

        for node in nodes:
            if node.kind == "File":
                continue

            text_len = len(node.text)

            if self.max_tokens:
                safe_char_limit = self.max_tokens * 3

                if text_len > safe_char_limit:
                    token_count = self._count_tokens(node.text)
                    if token_count > self.max_tokens:
                        logger.debug(
                            f"Node {node.name} exceeds max_tokens ({token_count} > {self.max_tokens}), splitting"
                        )
                        split_chunks = self._split_node_by_tokens(node)
                        chunks.extend(split_chunks)
                        continue

                chunk = self._node_to_chunk(node)
                chunks.append(chunk)

        return chunks

    def _chunk_size_based(self, nodes: list[CodeNode]) -> list[CodeChunk]:
        """
        크기 기반 청킹: 큰 노드는 분할

        예: 200줄 함수 → 2개 청크로 분할
        """
        chunks = []

        for node in nodes:
            if node.kind == "File":
                continue

            node_lines = self._get_line_count(node)

            # 작은 노드: 그대로
            if node_lines <= self.max_lines:
                chunks.append(self._node_to_chunk(node))

            # 큰 노드: 분할
            else:
                split_chunks = self._split_node(node)
                chunks.extend(split_chunks)

        return chunks

    def _chunk_hierarchical(self, nodes: list[CodeNode]) -> list[CodeChunk]:
        """
        계층적 청킹: Class + 각 Method

        예:
        - Class 전체 → 1 chunk (컨텍스트)
        - 각 Method → 개별 chunk (정밀 검색)
        """
        chunks = []

        # Node를 파일/클래스별로 그룹화
        file_nodes: dict[str, list[CodeNode]] = {}
        for node in nodes:
            if node.file_path not in file_nodes:
                file_nodes[node.file_path] = []
            file_nodes[node.file_path].append(node)

        # 파일별로 처리
        for _file_path, file_nodes_list in file_nodes.items():
            # Class 노드 찾기
            class_nodes = [n for n in file_nodes_list if n.kind == "Class"]
            method_nodes = [n for n in file_nodes_list if n.kind == "Method"]
            function_nodes = [n for n in file_nodes_list if n.kind == "Function"]

            # Class 전체 청크
            for class_node in class_nodes:
                chunks.append(self._node_to_chunk(class_node))

            # Method 개별 청크
            for method_node in method_nodes:
                chunks.append(self._node_to_chunk(method_node))

            # Function 청크
            for func_node in function_nodes:
                chunks.append(self._node_to_chunk(func_node))

        return chunks

    def _node_to_chunk(self, node: CodeNode) -> CodeChunk:
        """
        단일 Node를 Chunk로 변환

        Args:
            node: CodeNode

        Returns:
            CodeChunk
        """
        # Chunk ID 생성 (node_id 기반)
        chunk_id = self._generate_chunk_id(node)

        return CodeChunk(
            repo_id=node.repo_id,
            id=chunk_id,
            node_id=node.id,
            file_path=node.file_path,
            span=node.span,
            language=node.language,
            text=node.text,
            attrs={
                "node_kind": node.kind,
                "node_name": node.name,
                **node.attrs,  # Node attrs 상속
            },
        )

    def _split_node(self, node: CodeNode) -> list[CodeChunk]:
        """
        큰 노드를 여러 청크로 분할

        Args:
            node: 큰 CodeNode

        Returns:
            분할된 CodeChunk 리스트
        """
        chunks = []
        lines = node.text.split("\n")
        total_lines = len(lines)

        start_line_offset = node.span[0]
        chunk_idx = 0

        current_pos = 0
        while current_pos < total_lines:
            # 청크 크기 계산
            end_pos = min(current_pos + self.max_lines, total_lines)

            # 청크 텍스트
            chunk_lines = lines[current_pos:end_pos]
            chunk_text = "\n".join(chunk_lines)

            # 청크 span 계산
            chunk_span = (
                start_line_offset + current_pos,
                0,
                start_line_offset + end_pos - 1,
                len(chunk_lines[-1]) if chunk_lines else 0,
            )

            # 청크 생성
            chunk_id = f"{node.id}:chunk{chunk_idx}"
            chunks.append(
                CodeChunk(
                    repo_id=node.repo_id,
                    id=chunk_id,
                    node_id=node.id,
                    file_path=node.file_path,
                    span=chunk_span,
                    language=node.language,
                    text=chunk_text,
                    attrs={
                        "node_kind": node.kind,
                        "node_name": node.name,
                        "chunk_index": chunk_idx,
                        "is_split": True,
                    },
                )
            )

            # 오버랩 고려
            current_pos = end_pos - self.overlap_lines if end_pos < total_lines else end_pos
            chunk_idx += 1

        logger.debug(f"Split node {node.name} into {len(chunks)} chunks")
        return chunks

    def _generate_chunk_id(self, node: CodeNode) -> str:
        """
        Chunk ID 생성

        전략:
        - 1 Node = 1 Chunk: node_id 기반
        - 분할된 경우: node_id:chunk0, node_id:chunk1 등

        Args:
            node: CodeNode

        Returns:
            고유 chunk_id
        """
        # Node ID를 해시하여 짧게 만들기 (선택)
        # 또는 그대로 사용
        return f"chunk:{node.id}"

    def _get_line_count(self, node: CodeNode) -> int:
        """노드의 라인 수 계산"""
        start_line: int
        end_line: int
        start_line, _, end_line, _ = node.span
        return int(end_line - start_line + 1)

    def _count_tokens(self, text: str) -> int:
        """
        정확한 토큰 수 계산 (tiktoken 사용)

        tiktoken이 없으면 간단한 추정 사용

        Args:
            text: 텍스트

        Returns:
            토큰 수
        """
        if TIKTOKEN_AVAILABLE:
            try:
                tokens = TIKTOKEN_ENCODING.encode(text)
                return len(tokens)
            except Exception as e:
                logger.warning(f"tiktoken encoding failed: {e}, using fallback")

        # Fallback: 간단한 추정 (1토큰 ≈ 4글자)
        return len(text) // 4

    def _split_node_by_tokens(self, node: CodeNode) -> list[CodeChunk]:
        """
        토큰 수 기준으로 노드를 여러 청크로 분할

        Args:
            node: 큰 CodeNode

        Returns:
            분할된 CodeChunk 리스트
        """
        if not self.max_tokens:
            # max_tokens이 없으면 라인 기반 분할 사용
            return self._split_node(node)

        chunks = []
        lines = node.text.split("\n")
        total_lines = len(lines)

        start_line_offset = node.span[0]
        chunk_idx = 0

        current_pos = 0
        while current_pos < total_lines:
            # 토큰 제한 내에서 최대한 많은 라인 포함
            end_pos = current_pos + 1
            chunk_lines = lines[current_pos:end_pos]
            chunk_text = "\n".join(chunk_lines)

            # 토큰 수 확인하며 라인 추가
            while end_pos < total_lines:
                next_end = end_pos + 1
                next_chunk_lines = lines[current_pos:next_end]
                next_chunk_text = "\n".join(next_chunk_lines)

                if self._count_tokens(next_chunk_text) > self.max_tokens:
                    break

                end_pos = next_end
                chunk_lines = next_chunk_lines
                chunk_text = next_chunk_text

            # 청크 span 계산
            chunk_span = (
                start_line_offset + current_pos,
                0,
                start_line_offset + end_pos - 1,
                len(chunk_lines[-1]) if chunk_lines else 0,
            )

            # 청크 생성
            chunk_id = f"{node.id}:chunk{chunk_idx}"
            token_count = self._count_tokens(chunk_text)
            chunks.append(
                CodeChunk(
                    repo_id=node.repo_id,
                    id=chunk_id,
                    node_id=node.id,
                    file_path=node.file_path,
                    span=chunk_span,
                    language=node.language,
                    text=chunk_text,
                    attrs={
                        "node_kind": node.kind,
                        "node_name": node.name,
                        "chunk_index": chunk_idx,
                        "is_split": True,
                        "token_count": token_count,
                    },
                )
            )

            # 오버랩 고려 (라인 기반)
            current_pos = end_pos - self.overlap_lines if end_pos < total_lines else end_pos
            chunk_idx += 1

        logger.debug(f"Split node {node.name} into {len(chunks)} token-based chunks")
        return chunks

    def _estimate_token_count(self, text: str) -> int:
        """
        토큰 수 추정 (간단한 휴리스틱) - 하위 호환성을 위해 유지

        Args:
            text: 텍스트

        Returns:
            추정 토큰 수
        """
        return self._count_tokens(text)
