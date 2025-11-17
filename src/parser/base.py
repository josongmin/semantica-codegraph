"""Tree-sitter 기반 파서 베이스 클래스"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

from tree_sitter import Language, Parser, Node

from ..core.models import RawRelation, RawSymbol, RepoId, Span
from ..core.ports import ParserPort

logger = logging.getLogger(__name__)


class BaseTreeSitterParser(ParserPort, ABC):
    """
    Tree-sitter 기반 파서 베이스 클래스
    
    Tree-sitter는 구문 분석(syntax parsing)을 담당합니다:
    - 정확한 span 정보 제공
    - 빠른 파싱 속도
    - 모든 언어 지원
    
    언어별 파서는 이 클래스를 상속받아 구현합니다.
    """

    def __init__(self, language: Language):
        """
        Args:
            language: Tree-sitter Language 객체
        """
        # tree-sitter 0.25+ API
        self.parser = Parser(language)

    def parse_file(self, file_meta: dict) -> Tuple[List[RawSymbol], List[RawRelation]]:
        """
        파일 파싱
        
        Args:
            file_meta: 파일 메타데이터
                - repo_id: RepoId
                - path: 상대 경로
                - abs_path: 절대 경로
                - language: 언어
        
        Returns:
            (RawSymbol 리스트, RawRelation 리스트)
        """
        abs_path = Path(file_meta["abs_path"])
        
        if not abs_path.exists():
            logger.warning(f"File not found: {abs_path}")
            return [], []
        
        try:
            # 파일 읽기
            with open(abs_path, "rb") as f:
                source_code = f.read()
            
            # Tree-sitter 파싱
            tree = self.parser.parse(source_code)
            
            if tree.root_node.has_error:
                logger.warning(f"Parse error in {file_meta['path']}")
                # 에러가 있어도 부분적으로 파싱된 결과 사용
            
            # 심볼 추출
            symbols = self.extract_symbols(
                tree.root_node,
                source_code,
                file_meta
            )
            
            # 관계 추출
            relations = self.extract_relations(
                tree.root_node,
                source_code,
                file_meta,
                symbols
            )
            
            logger.debug(
                f"Parsed {file_meta['path']}: "
                f"{len(symbols)} symbols, {len(relations)} relations"
            )
            
            return symbols, relations
            
        except Exception as e:
            logger.error(f"Failed to parse {file_meta['path']}: {e}")
            return [], []

    @abstractmethod
    def extract_symbols(
        self,
        root: Node,
        source: bytes,
        file_meta: dict
    ) -> List[RawSymbol]:
        """
        심볼 추출 (언어별 구현)
        
        Args:
            root: Tree-sitter 루트 노드
            source: 소스 코드 (bytes)
            file_meta: 파일 메타데이터
        
        Returns:
            RawSymbol 리스트
        """
        pass

    @abstractmethod
    def extract_relations(
        self,
        root: Node,
        source: bytes,
        file_meta: dict,
        symbols: List[RawSymbol]
    ) -> List[RawRelation]:
        """
        관계 추출 (언어별 구현)
        
        Note:
            Tree-sitter는 구문 분석만 하므로 제한적입니다.
            정확한 관계는 Phase 2에서 SCIP로 추가합니다.
        
        Args:
            root: Tree-sitter 루트 노드
            source: 소스 코드 (bytes)
            file_meta: 파일 메타데이터
            symbols: 추출된 심볼 리스트
        
        Returns:
            RawRelation 리스트
        """
        pass

    # === 유틸리티 메서드 ===

    def _node_to_span(self, node: Node) -> Span:
        """
        Tree-sitter Node → Span 변환
        
        Args:
            node: Tree-sitter 노드
        
        Returns:
            (start_line, start_col, end_line, end_col)
        """
        return (
            node.start_point[0],
            node.start_point[1],
            node.end_point[0],
            node.end_point[1]
        )

    def _get_node_text(self, node: Node | None, source: bytes) -> str:
        """
        Node의 텍스트 추출
        
        Args:
            node: Tree-sitter 노드
            source: 소스 코드
        
        Returns:
            노드의 텍스트 (UTF-8 디코딩)
        """
        if node is None:
            return ""
        
        try:
            return source[node.start_byte:node.end_byte].decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode node text at {self._node_to_span(node)}")
            return ""

    def _traverse(self, node: Node, callback):
        """
        노드 트리 순회 (DFS)
        
        Args:
            node: 시작 노드
            callback: 각 노드에 적용할 콜백 함수
        """
        callback(node)
        for child in node.children:
            self._traverse(child, callback)

    def _find_child_by_type(self, node: Node, child_type: str) -> Node | None:
        """
        특정 타입의 자식 노드 찾기
        
        Args:
            node: 부모 노드
            child_type: 찾을 자식 노드 타입
        
        Returns:
            첫 번째 매칭 노드 (없으면 None)
        """
        for child in node.children:
            if child.type == child_type:
                return child
        return None

    def _find_children_by_type(self, node: Node, child_type: str) -> List[Node]:
        """
        특정 타입의 모든 자식 노드 찾기
        
        Args:
            node: 부모 노드
            child_type: 찾을 자식 노드 타입
        
        Returns:
            매칭된 노드 리스트
        """
        return [child for child in node.children if child.type == child_type]

