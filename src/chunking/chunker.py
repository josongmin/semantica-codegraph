"""Chunker: CodeNode → CodeChunk 변환"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..core.models import CodeChunk, CodeNode

if TYPE_CHECKING:
    from .file_summary_builder import FileSummaryBuilder

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
    5. File 요약: 조건부 파일 요약 청크 생성
    """

    def __init__(
        self,
        max_lines: int = 100,
        min_lines: int = 3,
        overlap_lines: int = 5,
        strategy: str = "node_based",
        max_tokens: int | None = 7000,  # 임베딩 API 제한보다 낮게 설정 (안전 마진)
        enable_file_summary: bool = True,  # 파일 요약 청크 생성 활성화
        min_symbols_for_summary: int = 5,  # 파일 요약 생성 최소 심볼 개수
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
            enable_file_summary: 조건부 파일 요약 청크 생성 활성화
            min_symbols_for_summary: 파일 요약 청크를 생성할 최소 심볼 개수
        """
        self.max_lines = max_lines
        self.min_lines = min_lines
        self.overlap_lines = overlap_lines
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.enable_file_summary = enable_file_summary
        self.min_symbols_for_summary = min_symbols_for_summary

        # FileSummaryBuilder 초기화 (lazy)
        self._file_summary_builder: FileSummaryBuilder | None = None

    def chunk(
        self,
        nodes: list[CodeNode],
        source_files: dict[str, str] | None = None,
    ) -> tuple[list[CodeChunk], dict]:
        """
        CodeNode 리스트를 CodeChunk 리스트로 변환

        Args:
            nodes: CodeNode 리스트
            source_files: 파일별 소스 코드 (파일 요약 청크 생성용, optional)
                         {file_path: source_code}

        Returns:
            (CodeChunk 리스트, 메트릭 dict)
        """
        if not nodes:
            logger.warning("No nodes to chunk")
            return [], {}

        metrics: dict = {
            "total_nodes": len(nodes),
            "symbol_chunks": 0,
            "file_summary_chunks": 0,
            "split_nodes": 0,
        }

        if self.strategy == "node_based":
            chunks, chunk_metrics = self._chunk_node_based(nodes, source_files)
            metrics.update(chunk_metrics)
        elif self.strategy == "size_based":
            chunks = self._chunk_size_based(nodes)
            metrics["symbol_chunks"] = len(chunks)
        elif self.strategy == "hierarchical":
            chunks = self._chunk_hierarchical(nodes)
            metrics["symbol_chunks"] = len(chunks)
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, using node_based")
            chunks, chunk_metrics = self._chunk_node_based(nodes, source_files)
            metrics.update(chunk_metrics)

        logger.info(
            f"Chunked {len(nodes)} nodes into {len(chunks)} chunks (strategy={self.strategy}), "
            f"symbol:{metrics.get('symbol_chunks', 0)}, "
            f"file_summary:{metrics.get('file_summary_chunks', 0)}"
        )

        return chunks, metrics

    def _chunk_node_based(
        self,
        nodes: list[CodeNode],
        source_files: dict[str, str] | None = None,
    ) -> tuple[list[CodeChunk], dict]:
        """
        Node 기반 청킹: 1 Node = 1 Chunk (병렬 처리)
        + 조건부 파일 요약 청크 생성

        가장 단순하고 빠른 방식
        """
        # 병렬 처리 임계값
        if len(nodes) < 100:
            # 노드 수가 적으면 순차 처리 (오버헤드 방지)
            return self._chunk_node_based_sequential(nodes, source_files)

        # 1. 파일별로 노드 그룹화
        nodes_by_file: dict[str, list[CodeNode]] = {}
        file_nodes: dict[str, CodeNode] = {}  # file_path -> File 노드

        for node in nodes:
            file_path = node.file_path
            if file_path not in nodes_by_file:
                nodes_by_file[file_path] = []

            if node.kind == "File":
                file_nodes[file_path] = node
            else:
                nodes_by_file[file_path].append(node)

        # 2. Symbol 노드 청크 생성 (병렬)
        from concurrent.futures import ThreadPoolExecutor

        chunks = []

        def process_node(node: CodeNode) -> list[CodeChunk]:
            """단일 노드 처리 (병렬 실행용)"""
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

        # Symbol 노드만 병렬 처리
        symbol_nodes = [n for n in nodes if n.kind != "File"]
        split_count = 0

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_node, symbol_nodes)
            for node_chunks in results:
                chunks.extend(node_chunks)
                # 분할된 청크 감지
                if len(node_chunks) > 1:
                    split_count += 1

        # 3. 조건부 파일 요약 청크 생성
        file_summary_count = 0
        if self.enable_file_summary:
            summary_chunks = self._create_file_summary_chunks(
                file_nodes, nodes_by_file, source_files
            )
            chunks.extend(summary_chunks)
            file_summary_count = len(summary_chunks)

            if summary_chunks:
                logger.info(f"Created {file_summary_count} file summary chunks")

        # 메트릭 수집
        metrics = {
            "symbol_chunks": len(chunks) - file_summary_count,
            "file_summary_chunks": file_summary_count,
            "split_nodes": split_count,
        }

        return chunks, metrics

    def _chunk_node_based_sequential(
        self,
        nodes: list[CodeNode],
        source_files: dict[str, str] | None = None,
    ) -> tuple[list[CodeChunk], dict]:
        """순차 처리 버전 (작은 프로젝트용)"""
        chunks = []

        # 파일별로 노드 그룹화
        nodes_by_file: dict[str, list[CodeNode]] = {}
        file_nodes: dict[str, CodeNode] = {}
        split_count = 0

        for node in nodes:
            file_path = node.file_path
            if file_path not in nodes_by_file:
                nodes_by_file[file_path] = []

            if node.kind == "File":
                file_nodes[file_path] = node
                continue  # File 노드는 나중에 처리

            # Symbol 노드 청크 생성
            nodes_by_file[file_path].append(node)

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
                        split_count += 1
                        continue

            chunk = self._node_to_chunk(node)
            chunks.append(chunk)

        # 조건부 파일 요약 청크 생성
        file_summary_count = 0
        if self.enable_file_summary:
            summary_chunks = self._create_file_summary_chunks(
                file_nodes, nodes_by_file, source_files
            )
            chunks.extend(summary_chunks)
            file_summary_count = len(summary_chunks)

            if summary_chunks:
                logger.info(f"Created {file_summary_count} file summary chunks")

        # 메트릭 수집
        metrics = {
            "symbol_chunks": len(chunks) - file_summary_count,
            "file_summary_chunks": file_summary_count,
            "split_nodes": split_count,
        }

        return chunks, metrics

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

                # 토큰 수 체크 (hard limit 적용)
                if self._count_tokens(next_chunk_text) > self.max_tokens:
                    break

                # 문자 수 hard limit (안전장치)
                MAX_CHARS = self.max_tokens * 4  # 보수적 추정
                if len(next_chunk_text) > MAX_CHARS:
                    logger.debug(
                        f"Chunk exceeds char limit ({len(next_chunk_text)} > {MAX_CHARS}), splitting"
                    )
                    break

                end_pos = next_end
                chunk_lines = next_chunk_lines
                chunk_text = next_chunk_text

            # 최소 1줄은 포함 (무한 루프 방지)
            if end_pos == current_pos:
                end_pos = current_pos + 1
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
            token_count = self._count_tokens(chunk_text)

            # 재확인: 여전히 큰 경우 경고
            if token_count > self.max_tokens:
                logger.warning(
                    f"Chunk {chunk_id} still exceeds max_tokens after split: "
                    f"{token_count} > {self.max_tokens} (will be skipped during embedding)"
                )

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

    def _create_file_summary_chunks(
        self,
        file_nodes: dict[str, CodeNode],
        nodes_by_file: dict[str, list[CodeNode]],
        source_files: dict[str, str] | None,
    ) -> list[CodeChunk]:
        """
        조건부 파일 요약 청크 생성

        Args:
            file_nodes: file_path -> File 노드 매핑
            nodes_by_file: file_path -> Symbol 노드 리스트
            source_files: file_path -> 소스 코드 (optional)

        Returns:
            파일 요약 청크 리스트
        """
        # FileSummaryBuilder lazy 초기화
        if self._file_summary_builder is None:
            from .file_summary_builder import FileSummaryBuilder

            self._file_summary_builder = FileSummaryBuilder(
                min_symbols_for_summary=self.min_symbols_for_summary
            )

        summary_chunks = []

        for file_path, file_node in file_nodes.items():
            symbol_nodes = nodes_by_file.get(file_path, [])

            # 파일 확장자 추출
            from pathlib import Path

            file_ext = Path(file_path).suffix.lower()

            # 요약 청크 생성 여부 결정
            should_create = self._file_summary_builder.should_create_summary(
                file_node, symbol_nodes, file_ext
            )

            if should_create:
                # 소스 코드 가져오기
                file_content = None
                if source_files and file_path in source_files:
                    file_content = source_files[file_path]

                # 요약 청크 생성
                try:
                    summary_chunk = self._file_summary_builder.build_summary_chunk(
                        file_node, symbol_nodes, file_content
                    )
                    summary_chunks.append(summary_chunk)
                    logger.debug(f"Created file summary chunk for {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to create file summary for {file_path}: {e}")

        return summary_chunks
