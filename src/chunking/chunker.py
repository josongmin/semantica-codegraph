"""Chunker: CodeNode → CodeChunk 변환"""

import logging

from ..core.models import CodeChunk, CodeNode

logger = logging.getLogger(__name__)


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
        """
        self.max_lines = max_lines
        self.min_lines = min_lines
        self.overlap_lines = overlap_lines
        self.strategy = strategy

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
        Node 기반 청킹: 1 Node = 1 Chunk

        가장 단순하고 빠른 방식
        """
        chunks = []

        for node in nodes:
            # File 노드는 스킵 (너무 큼)
            if node.kind == "File":
                continue

            # 기본 청크 생성
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

    def _estimate_token_count(self, text: str) -> int:
        """
        토큰 수 추정 (간단한 휴리스틱)

        실제 토큰화는 tiktoken 등 사용 권장
        여기서는 단어 수 * 1.3으로 추정

        Args:
            text: 텍스트

        Returns:
            추정 토큰 수
        """
        words = text.split()
        return int(len(words) * 1.3)
