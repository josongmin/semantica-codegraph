"""TypeScript/JavaScript용 Tree-sitter 파서"""

import logging

import tree_sitter_typescript
from tree_sitter import Language, Node

from ..core.models import RawRelation, RawSymbol
from .base import BaseTreeSitterParser

logger = logging.getLogger(__name__)


class TypeScriptTreeSitterParser(BaseTreeSitterParser):
    """
    TypeScript/JavaScript용 Tree-sitter 파서

    추출하는 심볼:
    - File
    - Class
    - Function
    - Method
    - Interface (TypeScript)
    - Type (TypeScript)
    """

    def __init__(self, use_tsx: bool = False):
        """
        Args:
            use_tsx: True면 TSX, False면 TypeScript
        """
        if use_tsx:
            language = Language(tree_sitter_typescript.language_tsx())
        else:
            language = Language(tree_sitter_typescript.language_typescript())
        super().__init__(language)

    def extract_symbols(
        self,
        root: Node,
        source: bytes,
        file_meta: dict
    ) -> list[RawSymbol]:
        """TypeScript 심볼 추출"""
        symbols = []

        # File 레벨 심볼
        symbols.append(RawSymbol(
            repo_id=file_meta["repo_id"],
            file_path=file_meta["path"],
            language=file_meta.get("language", "typescript"),
            kind="File",
            name=file_meta["path"],
            span=self._node_to_span(root),
            attrs={}
        ))

        # 심볼 추출
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

        # 추상 클래스 선언
        if node.type == "abstract_class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = self._get_node_text(name_node, source)

                symbols.append(RawSymbol(
                    repo_id=file_meta["repo_id"],
                    file_path=file_meta["path"],
                    language=file_meta.get("language", "typescript"),
                    kind="Class",
                    name=class_name,
                    span=self._node_to_span(node),
                    attrs={
                        "is_export": self._has_export_modifier(node),
                        "is_abstract": True,  # abstract class
                        "implements": self._extract_implements(node, source),
                        "extends": self._extract_extends(node, source)
                    }
                ))

                # 클래스 body 내부 순회
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

        # 클래스 선언
        elif node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = self._get_node_text(name_node, source)

                symbols.append(RawSymbol(
                    repo_id=file_meta["repo_id"],
                    file_path=file_meta["path"],
                    language=file_meta.get("language", "typescript"),
                    kind="Class",
                    name=class_name,
                    span=self._node_to_span(node),
                    attrs={
                        "is_export": self._has_export_modifier(node),
                        "is_abstract": self._has_modifier(node, "abstract"),
                        "implements": self._extract_implements(node, source),
                        "extends": self._extract_extends(node, source)
                    }
                ))

                # 클래스 body 내부 순회
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

        # 함수 선언
        elif node.type in ("function_declaration", "function"):
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = self._get_node_text(name_node, source)

                symbols.append(RawSymbol(
                    repo_id=file_meta["repo_id"],
                    file_path=file_meta["path"],
                    language=file_meta.get("language", "typescript"),
                    kind="Function",
                    name=func_name,
                    span=self._node_to_span(node),
                    attrs={
                        "is_export": self._has_export_modifier(node),
                        "is_async": self._has_modifier(node, "async"),
                        "parameters": self._extract_parameters(node, source)
                    }
                ))

        # 메서드 정의 (클래스 내부)
        elif node.type == "method_definition":
            name_node = node.child_by_field_name("name")
            if name_node and parent_class:
                method_name = self._get_node_text(name_node, source)
                full_name = f"{parent_class}.{method_name}"

                symbols.append(RawSymbol(
                    repo_id=file_meta["repo_id"],
                    file_path=file_meta["path"],
                    language=file_meta.get("language", "typescript"),
                    kind="Method",
                    name=full_name,
                    span=self._node_to_span(node),
                    attrs={
                        "parent_class": parent_class,
                        "is_static": self._has_modifier(node, "static"),
                        "is_async": self._has_modifier(node, "async"),
                        "visibility": self._get_visibility(node),
                        "parameters": self._extract_parameters(node, source)
                    }
                ))

        # 인터페이스 선언 (TypeScript)
        elif node.type == "interface_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                interface_name = self._get_node_text(name_node, source)

                symbols.append(RawSymbol(
                    repo_id=file_meta["repo_id"],
                    file_path=file_meta["path"],
                    language="typescript",
                    kind="Interface",
                    name=interface_name,
                    span=self._node_to_span(node),
                    attrs={
                        "is_export": self._has_export_modifier(node),
                        "extends": self._extract_interface_extends(node, source)
                    }
                ))

        # 타입 선언 (TypeScript)
        elif node.type == "type_alias_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                type_name = self._get_node_text(name_node, source)

                symbols.append(RawSymbol(
                    repo_id=file_meta["repo_id"],
                    file_path=file_meta["path"],
                    language="typescript",
                    kind="Type",
                    name=type_name,
                    span=self._node_to_span(node),
                    attrs={
                        "is_export": self._has_export_modifier(node)
                    }
                ))

        # 다른 노드는 자식만 순회
        else:
            for child in node.children:
                self._extract_symbols_recursive(
                    child,
                    source,
                    file_meta,
                    symbols,
                    parent_class
                )

    def _extract_parameters(self, node: Node, source: bytes) -> list[dict]:
        """파라미터 추출"""
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            return []

        params = []
        for child in params_node.children:
            if child.type in ("required_parameter", "optional_parameter"):
                pattern = child.child_by_field_name("pattern")
                type_node = child.child_by_field_name("type")

                param_name = ""
                if pattern and pattern.type == "identifier":
                    param_name = self._get_node_text(pattern, source)

                params.append({
                    "name": param_name,
                    "type": self._get_node_text(type_node, source) if type_node else None,
                    "is_optional": child.type == "optional_parameter"
                })

        return params

    def _has_modifier(self, node: Node, modifier: str) -> bool:
        """특정 modifier 존재 여부"""
        return any(child.type == modifier for child in node.children)

    def _has_export_modifier(self, node: Node) -> bool:
        """export modifier 존재 여부"""
        # export 키워드 찾기
        parent = node.parent
        if parent and parent.type == "export_statement":
            return True
        return self._has_modifier(node, "export")

    def _get_visibility(self, node: Node) -> str:
        """접근 제어자 추출 (public/private/protected)"""
        for child in node.children:
            if child.type in ("public", "private", "protected"):
                return child.type
        return "public"  # 기본값

    def _extract_extends(self, class_node: Node, source: bytes) -> str | None:
        """
        클래스 extends 추출

        TypeScript AST:
        class_heritage:
          extends_clause: "extends BaseRepository<User>"
        """
        # class_heritage 찾기
        heritage = self._find_child_by_type(class_node, "class_heritage")

        if heritage:
            # extends_clause 찾기
            extends_clause = self._find_child_by_type(heritage, "extends_clause")
            if extends_clause:
                # extends_clause 전체 텍스트에서 파싱
                clause_text = self._get_node_text(extends_clause, source)
                # "extends BaseRepository<User>" → "BaseRepository"
                clause_text = clause_text.replace("extends", "").strip()
                if "<" in clause_text:
                    clause_text = clause_text.split("<")[0]
                return clause_text

        return None

    def _extract_implements(self, class_node: Node, source: bytes) -> list[str]:
        """클래스 implements 추출"""
        implements = []

        # class_heritage 찾기
        heritage = self._find_child_by_type(class_node, "class_heritage")

        if heritage:
            # implements_clause 찾기
            implements_clause = self._find_child_by_type(heritage, "implements_clause")
            if implements_clause:
                # "implements Repository" → "Repository"
                clause_text = self._get_node_text(implements_clause, source)
                clause_text = clause_text.replace("implements", "").strip()
                # 여러 인터페이스가 있을 수 있음 (쉼표로 구분)
                for interface in clause_text.split(","):
                    interface = interface.strip()
                    if interface:
                        implements.append(interface)

        return implements

    def _extract_interface_extends(self, interface_node: Node, source: bytes) -> list[str]:
        """인터페이스 extends 추출"""
        extends = []
        heritage = interface_node.child_by_field_name("heritage")
        if heritage:
            for child in heritage.children:
                if child.type == "type_identifier":
                    extends.append(self._get_node_text(child, source))
        return extends

    def extract_relations(
        self,
        root: Node,
        source: bytes,
        file_meta: dict,
        symbols: list[RawSymbol]
    ) -> list[RawRelation]:
        """
        TypeScript 관계 추출

        Tree-sitter 레벨:
        - Class → Method (defines)
        - Class → Interface (implements)
        - Class → Class (extends)
        """
        relations = []

        symbol_map = {s.name: s for s in symbols}

        # Class → Method 관계
        for symbol in symbols:
            if symbol.kind == "Method" and symbol.attrs.get("parent_class"):
                parent_class = symbol.attrs["parent_class"]
                if parent_class in symbol_map:
                    relations.append(RawRelation(
                        repo_id=file_meta["repo_id"],
                        file_path=file_meta["path"],
                        language=file_meta.get("language", "typescript"),
                        type="defines",
                        src_span=symbol_map[parent_class].span,
                        dst_span=symbol.span,
                        attrs={"target": symbol.name}
                    ))

        # 상속/구현 관계 (같은 파일 내)
        for symbol in symbols:
            if symbol.kind == "Class":
                # extends
                extends = symbol.attrs.get("extends")
                if extends and extends in symbol_map:
                    relations.append(RawRelation(
                        repo_id=file_meta["repo_id"],
                        file_path=file_meta["path"],
                        language=file_meta.get("language", "typescript"),
                        type="extends",
                        src_span=symbol.span,
                        dst_span=symbol_map[extends].span,
                        attrs={"parent": extends}
                    ))

                # implements
                implements = symbol.attrs.get("implements", [])
                for interface in implements:
                    if interface in symbol_map:
                        relations.append(RawRelation(
                            repo_id=file_meta["repo_id"],
                            file_path=file_meta["path"],
                            language=file_meta.get("language", "typescript"),
                            type="implements",
                            src_span=symbol.span,
                            dst_span=symbol_map[interface].span,
                            attrs={"interface": interface}
                        ))

        logger.debug(f"Extracted {len(relations)} relations for {file_meta['path']}")
        return relations

