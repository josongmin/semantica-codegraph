"""Python용 Tree-sitter 파서"""

import logging

import tree_sitter_python
from tree_sitter import Language, Node

from ..core.models import RawRelation, RawSymbol
from .base import BaseTreeSitterParser

logger = logging.getLogger(__name__)


class PythonTreeSitterParser(BaseTreeSitterParser):
    """
    Python용 Tree-sitter 파서

    추출하는 심볼:
    - File (파일 전체)
    - Class (클래스 정의)
    - Function (함수 정의)
    - Method (클래스 내 메서드)
    """

    def __init__(self):
        language = Language(tree_sitter_python.language())
        super().__init__(language)

    def extract_symbols(
        self,
        root: Node,
        source: bytes,
        file_meta: dict
    ) -> list[RawSymbol]:
        """Python 심볼 추출"""
        symbols = []

        # File 레벨 심볼
        symbols.append(RawSymbol(
            repo_id=file_meta["repo_id"],
            file_path=file_meta["path"],
            language="python",
            kind="File",
            name=file_meta["path"],
            span=self._node_to_span(root),
            attrs={}
        ))

        # 함수/클래스 추출 (재귀)
        self._extract_symbols_recursive(
            root,
            source,
            file_meta,
            symbols,
            parent_class=None
        )

        return symbols

    def _extract_symbols_recursive(
        self,
        node: Node,
        source: bytes,
        file_meta: dict,
        symbols: list[RawSymbol],
        parent_class: str | None
    ):
        """재귀적으로 노드 순회하며 심볼 추출"""

        # decorated_definition 처리 (데코레이터가 있는 함수/클래스)
        if node.type == "decorated_definition":
            # 데코레이터 추출
            decorators = []
            definition_node = None

            for child in node.children:
                if child.type == "decorator":
                    dec_text = self._get_node_text(child, source)
                    if dec_text.startswith("@"):
                        dec_text = dec_text[1:].strip()
                    decorators.append(dec_text)
                elif child.type in ("function_definition", "class_definition"):
                    definition_node = child

            # 실제 정의를 파싱하되, 데코레이터 정보를 전달
            if definition_node:
                self._extract_definition_with_decorators(
                    definition_node,
                    source,
                    file_meta,
                    symbols,
                    parent_class,
                    decorators
                )
            return

        # 클래스 정의
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = self._get_node_text(name_node, source)

                self._add_class_symbol(
                    node,
                    source,
                    file_meta,
                    symbols,
                    class_name,
                    decorators=[]
                )

                # 클래스 내부 재귀 (메서드 추출)
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        self._extract_symbols_recursive(
                            child,
                            source,
                            file_meta,
                            symbols,
                            parent_class=class_name
                        )

        # 함수 정의
        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = self._get_node_text(name_node, source)

                # 클래스 내부면 Method, 아니면 Function
                kind = "Method" if parent_class else "Function"
                full_name = f"{parent_class}.{func_name}" if parent_class else func_name

                self._add_function_symbol(
                    node,
                    source,
                    file_meta,
                    symbols,
                    full_name,
                    kind,
                    parent_class,
                    decorators=[]
                )

        # 다른 노드 타입은 자식만 순회
        else:
            for child in node.children:
                self._extract_symbols_recursive(
                    child,
                    source,
                    file_meta,
                    symbols,
                    parent_class
                )

    def _extract_parameters(self, func_node: Node, source: bytes) -> list[dict]:
        """함수 파라미터 추출"""
        params_node = func_node.child_by_field_name("parameters")
        if not params_node:
            return []

        params = []
        for child in params_node.children:
            if child.type == "identifier":
                params.append({
                    "name": self._get_node_text(child, source),
                    "type": None  # Tree-sitter는 타입 추론 못 함 (SCIP 필요)
                })
            elif child.type == "typed_parameter":
                name_node = self._find_child_by_type(child, "identifier")
                type_node = child.child_by_field_name("type")
                params.append({
                    "name": self._get_node_text(name_node, source) if name_node else "",
                    "type": self._get_node_text(type_node, source) if type_node else None
                })
            elif child.type == "default_parameter":
                name_node = self._find_child_by_type(child, "identifier")
                if name_node:
                    params.append({
                        "name": self._get_node_text(name_node, source),
                        "type": None,
                        "has_default": True
                    })

        return params

    def _extract_decorators(self, node: Node, source: bytes) -> list[str]:
        """데코레이터 추출"""
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                # @ 기호 제외하고 추출
                dec_text = self._get_node_text(child, source)
                if dec_text.startswith("@"):
                    dec_text = dec_text[1:].strip()
                decorators.append(dec_text)
        return decorators

    def _extract_base_classes(self, class_node: Node, source: bytes) -> list[str]:
        """상속 클래스 추출"""
        bases = []
        superclasses = class_node.child_by_field_name("superclasses")
        if superclasses:
            for child in superclasses.children:
                if child.type == "identifier":
                    bases.append(self._get_node_text(child, source))
                elif child.type == "attribute":
                    # module.ClassName 형태
                    bases.append(self._get_node_text(child, source))
        return bases

    def _extract_docstring(self, node: Node, source: bytes) -> str | None:
        """Docstring 추출"""
        body = node.child_by_field_name("body")
        if not body:
            return None

        # body의 첫 번째 expression_statement에서 string 찾기
        for child in body.children:
            if child.type == "expression_statement":
                string_node = self._find_child_by_type(child, "string")
                if string_node:
                    docstring = self._get_node_text(string_node, source)
                    # 따옴표 제거
                    docstring = docstring.strip('"""').strip("'''").strip('"').strip("'")
                    return docstring.strip()
                break

        return None

    def _is_async_function(self, func_node: Node) -> bool:
        """비동기 함수 여부 확인"""
        # async def로 시작하는지 확인
        for child in func_node.children:
            if child.type == "async" or child.type == "async_keyword":
                return True
        return False

    def extract_relations(
        self,
        root: Node,
        source: bytes,
        file_meta: dict,
        symbols: list[RawSymbol]
    ) -> list[RawRelation]:
        """
        Python 관계 추출

        Tree-sitter 레벨에서는 제한적:
        - 클래스 → 메서드 (defines)
        - 상속 관계 (inherits) - 단순 텍스트 매칭

        정확한 호출 관계, 타입 참조는 Phase 2 SCIP에서 추가
        """
        relations = []

        # 심볼 맵 생성 (빠른 조회)
        symbol_map = {s.name: s for s in symbols}

        # 클래스 → 메서드 관계
        for symbol in symbols:
            if symbol.kind == "Method" and symbol.attrs.get("parent_class"):
                parent_class = symbol.attrs["parent_class"]
                if parent_class in symbol_map:
                    relations.append(RawRelation(
                        repo_id=file_meta["repo_id"],
                        file_path=file_meta["path"],
                        language="python",
                        type="defines",
                        src_span=symbol_map[parent_class].span,
                        dst_span=symbol.span,
                        attrs={"target": symbol.name}
                    ))

        # 상속 관계
        for symbol in symbols:
            if symbol.kind == "Class":
                bases = symbol.attrs.get("bases", [])
                for base in bases:
                    # 단순 텍스트 매칭 (같은 파일 내)
                    if base in symbol_map:
                        relations.append(RawRelation(
                            repo_id=file_meta["repo_id"],
                            file_path=file_meta["path"],
                            language="python",
                            type="inherits",
                            src_span=symbol.span,
                            dst_span=symbol_map[base].span,
                            attrs={"base_class": base}
                        ))

        logger.debug(f"Extracted {len(relations)} relations for {file_meta['path']}")
        return relations

    def _extract_definition_with_decorators(
        self,
        node: Node,
        source: bytes,
        file_meta: dict,
        symbols: list[RawSymbol],
        parent_class: str | None,
        decorators: list[str]
    ):
        """decorated_definition에서 추출한 정의 처리"""
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = self._get_node_text(name_node, source)
                self._add_class_symbol(
                    node,
                    source,
                    file_meta,
                    symbols,
                    class_name,
                    decorators
                )

                # 클래스 내부 재귀
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        self._extract_symbols_recursive(
                            child,
                            source,
                            file_meta,
                            symbols,
                            parent_class=class_name
                        )

        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = self._get_node_text(name_node, source)
                kind = "Method" if parent_class else "Function"
                full_name = f"{parent_class}.{func_name}" if parent_class else func_name

                self._add_function_symbol(
                    node,
                    source,
                    file_meta,
                    symbols,
                    full_name,
                    kind,
                    parent_class,
                    decorators
                )

    def _add_class_symbol(
        self,
        node: Node,
        source: bytes,
        file_meta: dict,
        symbols: list[RawSymbol],
        class_name: str,
        decorators: list[str]
    ):
        """클래스 심볼 추가"""
        symbols.append(RawSymbol(
            repo_id=file_meta["repo_id"],
            file_path=file_meta["path"],
            language="python",
            kind="Class",
            name=class_name,
            span=self._node_to_span(node),
            attrs={
                "bases": self._extract_base_classes(node, source),
                "decorators": decorators,
                "docstring": self._extract_docstring(node, source)
            }
        ))

    def _add_function_symbol(
        self,
        node: Node,
        source: bytes,
        file_meta: dict,
        symbols: list[RawSymbol],
        full_name: str,
        kind: str,
        parent_class: str | None,
        decorators: list[str]
    ):
        """함수/메서드 심볼 추가"""
        symbols.append(RawSymbol(
            repo_id=file_meta["repo_id"],
            file_path=file_meta["path"],
            language="python",
            kind=kind,
            name=full_name,
            span=self._node_to_span(node),
            attrs={
                "parameters": self._extract_parameters(node, source),
                "decorators": decorators,
                "docstring": self._extract_docstring(node, source),
                "is_async": self._is_async_function(node),
                "parent_class": parent_class
            }
        ))

