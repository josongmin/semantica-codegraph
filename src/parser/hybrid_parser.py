"""Tree-sitter + SCIP 하이브리드 파서"""

import logging

from ..core.models import RawRelation, RawSymbol, Span
from ..core.ports import ParserPort
from .base import BaseTreeSitterParser
from .scip_parser import ScipParser
from .type_hint_analyzer import TypeHintAnalyzer

logger = logging.getLogger(__name__)


class HybridParser(ParserPort):
    """
    Tree-sitter + SCIP 하이브리드 파서

    전략:
    1. Tree-sitter로 구조 파싱 (span 정확도 ⭐⭐⭐⭐⭐)
    2. SCIP로 의미론적 정보 추가 (타입, 관계 ⭐⭐⭐⭐⭐)
    3. 두 결과를 병합

    장점:
    - Tree-sitter의 정확한 span
    - SCIP의 풍부한 타입/관계 정보
    - 최고 품질의 CodeNode/CodeEdge 생성

    단점:
    - 두 번 파싱 (느림)
    - SCIP 인덱스 사전 생성 필요

    사용 시나리오:
    - 정밀한 코드 분석 필요 시
    - 타입 정보가 중요한 프로젝트
    - 오프라인 인덱싱 (속도 덜 중요)
    """

    def __init__(
        self,
        tree_sitter_parser: BaseTreeSitterParser,
        scip_parser: ScipParser | None = None,
        prefer_tree_sitter_span: bool = True,
        enable_type_hint_analysis: bool = True
    ):
        """
        Args:
            tree_sitter_parser: Tree-sitter 파서 인스턴스
            scip_parser: SCIP 파서 인스턴스 (None이면 Tree-sitter만 사용)
            prefer_tree_sitter_span: True면 span은 Tree-sitter 우선
            enable_type_hint_analysis: Python 타입 힌트 분석 활성화
        """
        self.tree_sitter = tree_sitter_parser
        self.scip = scip_parser
        self.prefer_tree_sitter_span = prefer_tree_sitter_span
        self.enable_type_hint_analysis = enable_type_hint_analysis

        # 타입 힌트 분석기 (Python만 지원)
        if enable_type_hint_analysis:
            self.type_hint_analyzer = TypeHintAnalyzer()
        else:
            self.type_hint_analyzer = None

    def parse_file(
        self,
        file_meta: dict
    ) -> tuple[list[RawSymbol], list[RawRelation]]:
        """
        하이브리드 파싱

        Args:
            file_meta: 파일 메타데이터

        Returns:
            (병합된 RawSymbol 리스트, 병합된 RawRelation 리스트)
        """
        # 1. Tree-sitter로 구조 파싱
        logger.debug(f"Parsing with Tree-sitter: {file_meta['path']}")
        ts_symbols, ts_relations = self.tree_sitter.parse_file(file_meta)

        # 2. SCIP가 없으면 Tree-sitter 결과만 반환
        if self.scip is None:
            logger.debug("SCIP parser not available, using Tree-sitter only")
            symbols, relations = ts_symbols, ts_relations
        else:
            # 3. SCIP로 의미론적 정보 추가
            logger.debug(f"Parsing with SCIP: {file_meta['path']}")
            scip_symbols, scip_relations = self.scip.parse_file(file_meta)

            # 4. 병합
            symbols = self._merge_symbols(ts_symbols, scip_symbols)
            relations = self._merge_relations(ts_relations, scip_relations)

            logger.info(
                f"Hybrid parsing completed for {file_meta['path']}: "
                f"{len(symbols)} symbols ({len(ts_symbols)} TS + {len(scip_symbols)} SCIP), "
                f"{len(relations)} relations ({len(ts_relations)} TS + {len(scip_relations)} SCIP)"
            )

        # 5. 타입 힌트 분석 (Python 파일만)
        if self.type_hint_analyzer and file_meta.get('language') == 'python':
            type_hint_relations = self._analyze_type_hints(file_meta)
            if type_hint_relations:
                relations.extend(type_hint_relations)
                logger.info(
                    f"Type hint analysis: {len(type_hint_relations)} dynamic calls inferred"
                )

        return symbols, relations

    def _merge_symbols(
        self,
        ts_symbols: list[RawSymbol],
        scip_symbols: list[RawSymbol]
    ) -> list[RawSymbol]:
        """
        심볼 병합

        전략:
        - Tree-sitter span 우선 (더 정확)
        - SCIP attrs (타입 정보) 추가
        - 이름/kind 매칭으로 동일 심볼 판별

        Args:
            ts_symbols: Tree-sitter 심볼 리스트
            scip_symbols: SCIP 심볼 리스트

        Returns:
            병합된 RawSymbol 리스트
        """
        merged = {}

        # 1. Tree-sitter 심볼 먼저 (span 정확도)
        for sym in ts_symbols:
            key = self._make_symbol_key(sym)
            merged[key] = sym

        # 2. SCIP 정보 추가
        for scip_sym in scip_symbols:
            key = self._make_symbol_key(scip_sym)

            if key in merged:
                # Tree-sitter 심볼에 SCIP attrs 병합
                ts_sym = merged[key]

                # span은 Tree-sitter 우선
                if self.prefer_tree_sitter_span:
                    scip_attrs = scip_sym.attrs.copy()
                    ts_sym.attrs.update({
                        k: v for k, v in scip_attrs.items()
                        if k not in ts_sym.attrs or v is not None
                    })
                else:
                    # SCIP span 사용 (거의 안 씀)
                    ts_sym.span = scip_sym.span
                    ts_sym.attrs.update(scip_sym.attrs)

                logger.debug(f"Merged symbol: {sym.name}")
            else:
                # SCIP만 있는 심볼 (예: import된 타입, 외부 참조)
                merged[key] = scip_sym
                logger.debug(f"SCIP-only symbol: {scip_sym.name}")

        return list(merged.values())

    def _merge_relations(
        self,
        ts_relations: list[RawRelation],
        scip_relations: list[RawRelation]
    ) -> list[RawRelation]:
        """
        관계 병합

        전략:
        - SCIP 관계 우선 (더 정확하고 포괄적)
        - Tree-sitter 관계는 SCIP에 없는 것만 추가
        - 중복 제거

        Args:
            ts_relations: Tree-sitter 관계 리스트
            scip_relations: SCIP 관계 리스트

        Returns:
            병합된 RawRelation 리스트
        """
        merged = {}

        # 1. SCIP 관계 먼저 (더 정확)
        for rel in scip_relations:
            key = self._make_relation_key(rel)
            merged[key] = rel

        # 2. Tree-sitter 관계 추가 (중복 제거)
        for rel in ts_relations:
            key = self._make_relation_key(rel)
            if key not in merged:
                merged[key] = rel
                logger.debug(f"Added TS relation: {rel.type}")

        logger.debug(
            f"Merged relations: {len(scip_relations)} SCIP + "
            f"{len(ts_relations)} TS = {len(merged)} total"
        )

        return list(merged.values())

    def _make_symbol_key(self, symbol: RawSymbol) -> tuple:
        """
        심볼 키 생성 (중복 판별용)

        키: (file_path, name, kind)

        Note:
            span은 키에 포함하지 않음 (Tree-sitter와 SCIP span이 약간 다를 수 있음)
        """
        return (
            symbol.file_path,
            symbol.name,
            symbol.kind
        )

    def _make_relation_key(self, relation: RawRelation) -> tuple:
        """
        관계 키 생성 (중복 판별용)

        키: (file_path, type, src_span, target_symbol or dst_span)
        """
        # attrs에서 target_symbol 추출 (있으면)
        target = relation.attrs.get("target_symbol") or relation.attrs.get("target")

        return (
            relation.file_path,
            relation.type,
            relation.src_span,
            target if target else relation.dst_span
        )

    def _is_span_overlapping(self, span1: Span, span2: Span) -> bool:
        """
        두 span이 겹치는지 확인

        Args:
            span1: (start_line, start_col, end_line, end_col)
            span2: (start_line, start_col, end_line, end_col)

        Returns:
            겹치면 True
        """
        s1_start_line, s1_start_col, s1_end_line, s1_end_col = span1
        s2_start_line, s2_start_col, s2_end_line, s2_end_col = span2

        # 라인이 겹치는지 확인
        if s1_end_line < s2_start_line or s2_end_line < s1_start_line:
            return False

        # 같은 라인이면 컬럼도 확인
        if s1_start_line == s1_end_line == s2_start_line == s2_end_line:
            if s1_end_col < s2_start_col or s2_end_col < s1_start_col:
                return False

        return True

    def _find_matching_symbol_by_span(
        self,
        symbols: list[RawSymbol],
        target_span: Span,
        name_hint: str | None = None
    ) -> RawSymbol | None:
        """
        Span으로 매칭되는 심볼 찾기

        Args:
            symbols: 검색할 심볼 리스트
            target_span: 찾을 span
            name_hint: 이름 힌트 (있으면 우선 매칭)

        Returns:
            매칭된 심볼 (없으면 None)
        """
        # 이름 힌트가 있으면 먼저 이름으로 찾기
        if name_hint:
            for sym in symbols:
                if sym.name == name_hint and self._is_span_overlapping(sym.span, target_span):
                    return sym

        # span 겹치는 것 중 가장 작은 것 찾기 (가장 구체적인 심볼)
        candidates = [
            sym for sym in symbols
            if self._is_span_overlapping(sym.span, target_span)
        ]

        if not candidates:
            return None

        # 가장 작은 span (가장 구체적)
        return min(
            candidates,
            key=lambda s: (s.span[2] - s.span[0], s.span[3] - s.span[1])
        )

    def _analyze_type_hints(self, file_meta: dict) -> list[RawRelation]:
        """
        타입 힌트 기반 동적 호출 추론

        Args:
            file_meta: 파일 메타데이터

        Returns:
            추론된 RawRelation 리스트
        """
        # 파일 읽기
        try:
            with open(file_meta['abs_path'], encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.warning(f"Failed to read file for type hint analysis: {e}")
            return []

        # 타입 힌트 분석
        inferred_calls = self.type_hint_analyzer.analyze(code, file_meta['path'])

        # InferredCall을 RawRelation으로 변환
        relations = []
        for call in inferred_calls:
            relations.append(
                RawRelation(
                    source=call.source,
                    target=call.target,
                    type="calls",
                    attrs={
                        "confidence": call.confidence,
                        "inferred": True,
                        "method": "type_hint",
                        "line": call.line
                    }
                )
            )

        return relations

