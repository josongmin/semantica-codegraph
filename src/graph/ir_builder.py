"""IR Builder: RawSymbol/Relation → CodeNode/Edge 변환"""

import logging

from ..core.models import CodeEdge, CodeNode, RawRelation, RawSymbol, RepoId, Span

logger = logging.getLogger(__name__)


class IRBuilder:
    """
    언어별 파서 출력(Raw)을 언어 독립적인 IR(CodeNode/Edge)로 변환

    주요 역할:
    1. RawSymbol → CodeNode 변환
       - 고유 ID 생성
       - 텍스트 추출 (span 기반)
       - attrs 정규화

    2. RawRelation → CodeEdge 변환
       - src/dst 심볼 → ID 매핑
       - 관계 타입 정규화

    3. 그래프 검증
       - 순환 참조 체크
       - 고아 노드 정리
    """

    def __init__(self, normalize_ids: bool = True):
        """
        Args:
            normalize_ids: True면 ID를 정규화 (소문자, 특수문자 제거)
        """
        self.normalize_ids = normalize_ids

    def build(
        self,
        raw_symbols: list[RawSymbol],
        raw_relations: list[RawRelation],
        source_code: dict[str, str] | None = None,
    ) -> tuple[list[CodeNode], list[CodeEdge]]:
        """
        Raw 데이터를 CodeNode/Edge로 변환

        Args:
            raw_symbols: 파서에서 추출한 심볼 리스트
            raw_relations: 파서에서 추출한 관계 리스트
            source_code: 파일별 소스 코드 (span으로 텍스트 추출용)
                        {file_path: source_text}

        Returns:
            (CodeNode 리스트, CodeEdge 리스트)
        """
        if not raw_symbols:
            logger.warning("No symbols to build")
            return [], []

        # 1. RawSymbol → CodeNode 변환
        nodes = []
        symbol_to_id = {}  # RawSymbol → node_id 매핑

        for raw_sym in raw_symbols:
            node = self._raw_symbol_to_node(raw_sym, source_code)
            nodes.append(node)
            symbol_to_id[self._make_symbol_key(raw_sym)] = node.id

        logger.debug(f"Converted {len(raw_symbols)} symbols to {len(nodes)} nodes")

        # 2. RawRelation → CodeEdge 변환
        edges = []
        for raw_rel in raw_relations:
            edge = self._raw_relation_to_edge(raw_rel, symbol_to_id)
            if edge:
                edges.append(edge)

        logger.debug(f"Converted {len(raw_relations)} relations to {len(edges)} edges")

        # 3. 그래프 검증 및 정리
        nodes, edges = self._validate_graph(nodes, edges)

        logger.info(f"IR build completed: {len(nodes)} nodes, {len(edges)} edges")

        return nodes, edges

    def _raw_symbol_to_node(self, raw: RawSymbol, source_code: dict[str, str] | None) -> CodeNode:
        """
        RawSymbol → CodeNode 변환

        Args:
            raw: RawSymbol
            source_code: 파일별 소스 코드

        Returns:
            CodeNode
        """
        # 1. 고유 ID 생성
        node_id = self._generate_node_id(raw)

        # 2. 텍스트 추출
        text = self._extract_text(raw, source_code)

        # 3. CodeNode 생성
        return CodeNode(
            repo_id=raw.repo_id,
            id=node_id,
            kind=raw.kind,
            language=raw.language,
            file_path=raw.file_path,
            span=raw.span,
            name=raw.name,
            text=text,
            attrs=raw.attrs.copy(),  # attrs는 그대로 복사
        )

    def _raw_relation_to_edge(
        self, raw: RawRelation, symbol_to_id: dict[tuple, str]
    ) -> CodeEdge | None:
        """
        RawRelation → CodeEdge 변환

        Args:
            raw: RawRelation
            symbol_to_id: symbol key → node_id 매핑

        Returns:
            CodeEdge (매핑 실패 시 None)
        """
        # src/dst span으로 심볼 찾기

        # attrs에 target이 있으면 우선 사용
        target = raw.attrs.get("target") or raw.attrs.get("target_symbol")

        # src_id 찾기
        src_id = self._find_node_id_by_span(raw.repo_id, raw.file_path, raw.src_span, symbol_to_id)

        # dst_id 찾기
        dst_id = None
        if target:
            # target 이름으로 찾기
            dst_id = self._find_node_id_by_name(raw.repo_id, raw.file_path, target, symbol_to_id)
        else:
            # span으로 찾기
            dst_id = self._find_node_id_by_span(
                raw.repo_id, raw.file_path, raw.dst_span, symbol_to_id
            )

        if not src_id:
            logger.debug(f"Source node not found for relation: {raw.type}")
            return None

        if not dst_id:
            logger.debug(f"Destination node not found for relation: {raw.type} (target={target})")
            return None

        return CodeEdge(
            repo_id=raw.repo_id, src_id=src_id, dst_id=dst_id, type=raw.type, attrs=raw.attrs.copy()
        )

    def _generate_node_id(self, raw: RawSymbol) -> str:
        """
        고유 Node ID 생성

        형식: {repo_id}:{file_path}:{kind}:{name}

        예시:
        - myrepo:src/main.py:Function:calculate_total
        - myrepo:src/user.py:Class:User
        - myrepo:src/user.py:Method:User.save

        Args:
            raw: RawSymbol

        Returns:
            고유 ID
        """
        # 기본 형식
        id_parts = [raw.repo_id, raw.file_path, raw.kind, raw.name]

        node_id = ":".join(id_parts)

        # 정규화 (선택)
        if self.normalize_ids:
            # 소문자 변환, 경로 정규화
            node_id = node_id.replace("\\", "/")

        return node_id

    def _extract_text(self, raw: RawSymbol, source_code: dict[str, str] | None) -> str:
        """
        Span으로 텍스트 추출

        Note:
            Tree-sitter span은 0-based indexing:
            - (0, 0, 0, 5) = 첫 줄 0-5번 컬럼
            - (0, 0, 2, 0) = 0번 줄부터 2번 줄 시작까지

        Args:
            raw: RawSymbol
            source_code: 파일별 소스 코드

        Returns:
            추출된 텍스트 (실패 시 빈 문자열)
        """
        if not source_code or raw.file_path not in source_code:
            logger.debug(f"Source code not available for {raw.file_path}, text will be empty")
            return ""

        source = source_code[raw.file_path]
        lines = source.split("\n")

        start_line, start_col, end_line, end_col = raw.span

        try:
            # 범위 체크
            if start_line >= len(lines) or end_line > len(lines):
                logger.warning(
                    f"Span out of range for {raw.name}: span={raw.span}, total_lines={len(lines)}"
                )
                return ""

            # 단일 라인
            if start_line == end_line:
                line = lines[start_line]
                return line[start_col:end_col] if end_col > 0 else line[start_col:]

            # 여러 라인
            extracted_lines = []

            # 첫 라인 (start_col부터 끝까지)
            extracted_lines.append(lines[start_line][start_col:])

            # 중간 라인들 (전체)
            for line_idx in range(start_line + 1, end_line):
                extracted_lines.append(lines[line_idx])

            # 마지막 라인 (처음부터 end_col까지)
            if end_line < len(lines) and end_col > 0:
                extracted_lines.append(lines[end_line][:end_col])
            elif end_line < len(lines):
                # end_col이 0이면 해당 라인은 포함 안 함
                pass

            return "\n".join(extracted_lines)

        except Exception as e:
            logger.warning(f"Failed to extract text for {raw.name}: {e}")
            return ""

    def _find_node_id_by_span(
        self, repo_id: RepoId, file_path: str, span: Span, symbol_to_id: dict[tuple, str]
    ) -> str | None:
        """
        Span으로 node_id 찾기

        symbol_to_id의 키 형식: (file_path, name, kind, span)
        주어진 span과 매칭되는 심볼을 찾습니다.

        우선순위:
        1. 정확히 일치하는 span
        2. 주어진 span을 포함하는 심볼 (가장 작은 것)
        3. 주어진 span과 겹치는 심볼 (가장 작은 것)
        """

        def spans_equal(span1: Span, span2: Span) -> bool:
            """두 span이 정확히 일치하는지 확인"""
            return span1 == span2

        def span_contains(span1: Span, span2: Span) -> bool:
            """span1이 span2를 포함하는지 확인"""
            s1_start_line, s1_start_col, s1_end_line, s1_end_col = span1
            s2_start_line, s2_start_col, s2_end_line, s2_end_col = span2

            if s1_start_line > s2_start_line or s1_end_line < s2_end_line:
                return False

            if s1_start_line == s2_start_line and s1_start_col > s2_start_col:
                return False

            return not (s1_end_line == s2_end_line and s1_end_col < s2_end_col)

        def spans_overlap(span1: Span, span2: Span) -> bool:
            """두 span이 겹치는지 확인"""
            s1_start_line, s1_start_col, s1_end_line, s1_end_col = span1
            s2_start_line, s2_start_col, s2_end_line, s2_end_col = span2

            # 라인이 겹치는지 확인
            if s1_end_line < s2_start_line or s2_end_line < s1_start_line:
                return False

            # 같은 라인이면 컬럼도 확인
            if s1_start_line == s1_end_line == s2_start_line == s2_end_line and (
                s1_end_col < s2_start_col or s2_end_col < s1_start_col
            ):
                return False

            return True

        exact_match = None
        containing_candidates = []
        overlapping_candidates = []

        for key, node_id in symbol_to_id.items():
            # 키 형식: (file_path, name, kind, span)
            if len(key) == 4 and key[0] == file_path:
                symbol_span = key[3]

                # 1. 정확히 일치
                if spans_equal(span, symbol_span):
                    exact_match = node_id
                    break  # 정확히 일치하면 바로 반환

                # 2. 포함 관계 (symbol_span이 span을 포함)
                if span_contains(symbol_span, span):
                    span_size = (symbol_span[2] - symbol_span[0], symbol_span[3] - symbol_span[1])
                    containing_candidates.append((span_size, node_id))

                # 3. 겹침
                elif spans_overlap(span, symbol_span):
                    span_size = (symbol_span[2] - symbol_span[0], symbol_span[3] - symbol_span[1])
                    overlapping_candidates.append((span_size, node_id))

        # 우선순위에 따라 반환
        if exact_match:
            return exact_match

        if containing_candidates:
            containing_candidates.sort(key=lambda x: x[0])
            return containing_candidates[0][1]

        if overlapping_candidates:
            overlapping_candidates.sort(key=lambda x: x[0])
            return overlapping_candidates[0][1]

        return None

    def _find_node_id_by_name(
        self, repo_id: RepoId, file_path: str, name: str, symbol_to_id: dict[tuple, str]
    ) -> str | None:
        """
        이름으로 node_id 찾기

        symbol_to_id의 키 형식: (file_path, name, kind, span)
        주어진 이름과 정확히 일치하는 심볼을 찾습니다.
        """
        # 같은 파일에서 이름으로 찾기
        for key, node_id in symbol_to_id.items():
            # 키 형식: (file_path, name, kind, span)
            if len(key) == 4 and key[0] == file_path:
                symbol_name = key[1]
                # 정확히 일치하는 이름 찾기
                if symbol_name == name:
                    return node_id

        return None

    def _make_symbol_key(self, raw: RawSymbol) -> tuple:
        """
        심볼 키 생성 (중복 방지)

        Returns:
            (file_path, name, kind, span)
        """
        return (raw.file_path, raw.name, raw.kind, raw.span)

    def _validate_graph(
        self, nodes: list[CodeNode], edges: list[CodeEdge]
    ) -> tuple[list[CodeNode], list[CodeEdge]]:
        """
        그래프 검증 및 정리

        1. 존재하지 않는 노드를 참조하는 엣지 제거
        2. 중복 노드 제거
        3. 중복 엣지 제거

        Args:
            nodes: CodeNode 리스트
            edges: CodeEdge 리스트

        Returns:
            (정리된 nodes, 정리된 edges)
        """
        # 노드 ID 세트
        node_ids = {node.id for node in nodes}

        # 중복 노드 제거
        unique_nodes = {}
        for node in nodes:
            if node.id not in unique_nodes:
                unique_nodes[node.id] = node
            else:
                logger.debug(f"Duplicate node removed: {node.id}")

        # 유효한 엣지만 선택
        valid_edges = []
        edge_keys = set()

        for edge in edges:
            # src, dst가 모두 존재하는지 확인
            if edge.src_id not in node_ids:
                logger.debug(f"Edge skipped: src not found {edge.src_id}")
                continue

            if edge.dst_id not in node_ids:
                logger.debug(f"Edge skipped: dst not found {edge.dst_id}")
                continue

            # 중복 체크
            edge_key = (edge.src_id, edge.dst_id, edge.type)
            if edge_key in edge_keys:
                logger.debug(f"Duplicate edge removed: {edge_key}")
                continue

            edge_keys.add(edge_key)
            valid_edges.append(edge)

        logger.info(
            f"Graph validated: {len(unique_nodes)}/{len(nodes)} nodes, "
            f"{len(valid_edges)}/{len(edges)} edges"
        )

        return list(unique_nodes.values()), valid_edges
