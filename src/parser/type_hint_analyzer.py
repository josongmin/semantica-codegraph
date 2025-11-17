"""타입 힌트 기반 동적 호출 추론

이 모듈은 Python 타입 힌트를 활용하여 
getattr 등의 동적 호출을 정적으로 추론합니다.

예시:
    def process(user: UserAuthenticator):
        method = getattr(user, "authenticate")
        # → UserAuthenticator.authenticate로 추론
"""

import ast
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TypeInfo:
    """타입 정보"""
    var_name: str
    type_name: str
    module: Optional[str] = None


@dataclass
class InferredCall:
    """추론된 호출 정보"""
    source: str  # "getattr(UserAuth, 'login')"
    target: str  # "UserAuth.login"
    confidence: float  # 0.0-1.0
    line: int = 0


class TypeHintAnalyzer:
    """
    타입 힌트를 활용한 동적 호출 추론
    
    지원 패턴:
    1. getattr(obj: Type, "method") → Type.method
    2. 변수 할당 추적: user: UserAuth = get_user()
    3. 함수 반환 타입: def get_auth() -> UserAuth
    
    개선사항 (v2):
    - 함수별 스코프 분리 (변수명 충돌 방지)
    - Import alias 해석 (from X import Y as Z)
    """
    
    def __init__(self):
        # 전역 스코프 (모듈 레벨)
        self.global_type_map: dict[str, str] = {}  # {var_name: type_name}
        self.function_returns: dict[str, str] = {}  # {func_name: return_type}
        
        # 함수별 스코프 (v2 추가)
        self.function_scopes: dict[str, dict[str, str]] = {}  # {func_name: {var: type}}
        self.current_function: Optional[str] = None  # 현재 분석 중인 함수
        
        # Import alias 매핑 (v2 추가)
        self.import_aliases: dict[str, str] = {}  # {alias: full_name}
    
    def analyze(self, code: str, file_path: str) -> list[InferredCall]:
        """
        코드 분석하여 추론된 호출 반환
        
        Args:
            code: Python 소스 코드
            file_path: 파일 경로 (로깅용)
        
        Returns:
            추론된 호출 리스트
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []
        
        # 1단계: Import 수집
        self._collect_imports(tree)
        
        # 2단계: 타입 정보 수집
        self._collect_type_hints(tree)
        
        # 2단계: getattr 호출 분석 (함수별로)
        inferred = []
        
        # 모듈 레벨 getattr 분석
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and not self._is_inside_function(node, tree):
                calls = self._analyze_getattr(node, None)
                inferred.extend(calls)
        
        # 함수별 getattr 분석
        for func_node in ast.walk(tree):
            if isinstance(func_node, ast.FunctionDef):
                self.current_function = func_node.name
                
                for node in ast.walk(func_node):
                    if isinstance(node, ast.Call):
                        calls = self._analyze_getattr(node, func_node.name)
                        inferred.extend(calls)
                
                self.current_function = None
        
        logger.debug(
            f"Type hint analysis: {len(inferred)} calls inferred from {file_path}"
        )
        
        return inferred
    
    def _is_inside_function(self, node: ast.AST, tree: ast.AST) -> bool:
        """노드가 함수 내부에 있는지 확인"""
        for func in ast.walk(tree):
            if isinstance(func, ast.FunctionDef):
                for child in ast.walk(func):
                    if child is node:
                        return True
        return False
    
    def _collect_imports(self, tree: ast.AST):
        """
        Import문 수집 (alias 해석용)
        
        from auth import UserAuth as UA
        → UA: auth.UserAuth
        
        import auth.models as models
        → models: auth.models
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # from X import Y [as Z]
                module = node.module or ""
                
                for alias in node.names:
                    name = alias.name
                    asname = alias.asname or name
                    
                    # 전체 경로 생성
                    if module:
                        full_name = f"{module}.{name}"
                    else:
                        full_name = name
                    
                    self.import_aliases[asname] = full_name
                    logger.debug(f"Import alias: {asname} → {full_name}")
            
            elif isinstance(node, ast.Import):
                # import X [as Y]
                for alias in node.names:
                    name = alias.name
                    asname = alias.asname or name
                    
                    self.import_aliases[asname] = name
                    logger.debug(f"Import: {asname} → {name}")
    
    def _collect_type_hints(self, tree: ast.AST):
        """타입 힌트 수집"""
        for node in ast.walk(tree):
            # 함수 인자 타입 힌트
            if isinstance(node, ast.FunctionDef):
                self._collect_from_function(node)
            
            # 변수 타입 힌트
            elif isinstance(node, ast.AnnAssign):
                self._collect_from_assignment(node)
    
    def _collect_from_function(self, node: ast.FunctionDef):
        """
        함수에서 타입 정보 수집 (스코프 분리)
        
        def process(user: UserAuth) -> Session:
            pass
        """
        # 함수별 스코프 생성
        func_scope = {}
        
        # 인자 타입 (함수 스코프에 저장)
        for arg in node.args.args:
            if arg.annotation:
                type_name = self._get_annotation_name(arg.annotation)
                if type_name:
                    func_scope[arg.arg] = type_name
                    logger.debug(f"[{node.name}] Found arg type: {arg.arg} -> {type_name}")
        
        # 함수 내부 변수 타입 어노테이션 수집
        for child in ast.walk(node):
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                type_name = self._get_annotation_name(child.annotation)
                if type_name:
                    func_scope[child.target.id] = type_name
                    logger.debug(f"[{node.name}] Found local var: {child.target.id} -> {type_name}")
        
        # 함수 스코프 저장
        if func_scope:
            self.function_scopes[node.name] = func_scope
        
        # 반환 타입 (전역)
        if node.returns:
            return_type = self._get_annotation_name(node.returns)
            if return_type:
                self.function_returns[node.name] = return_type
                logger.debug(f"Found return type: {node.name} -> {return_type}")
    
    def _collect_from_assignment(self, node: ast.AnnAssign):
        """
        할당문에서 타입 정보 수집 (모듈 레벨만)
        
        user: UserAuth = get_user()  # 모듈 레벨
        
        Note: 함수 내부 변수는 _collect_from_function에서 처리
        """
        if isinstance(node.target, ast.Name):
            type_name = self._get_annotation_name(node.annotation)
            if type_name:
                # 전역 스코프에 저장
                self.global_type_map[node.target.id] = type_name
                logger.debug(f"Found global assignment: {node.target.id} -> {type_name}")
    
    def _get_annotation_name(self, annotation: ast.AST) -> Optional[str]:
        """
        타입 어노테이션에서 이름 추출
        
        Args:
            annotation: AST 노드
        
        Returns:
            타입 이름 (예: "UserAuth", "auth.UserAuth")
        """
        if isinstance(annotation, ast.Name):
            # 단순 타입: UserAuth
            return annotation.id
        
        elif isinstance(annotation, ast.Attribute):
            # 모듈 포함: auth.UserAuth
            parts = []
            node = annotation
            while isinstance(node, ast.Attribute):
                parts.insert(0, node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.insert(0, node.id)
            return ".".join(parts)
        
        elif isinstance(annotation, ast.Subscript):
            # Generic 타입: Optional[UserAuth], List[UserAuth]
            # 내부 타입 추출 (slice가 실제 타입)
            return self._get_annotation_name(annotation.slice)
        
        return None
    
    def _analyze_getattr(self, node: ast.Call, function_name: Optional[str]) -> list[InferredCall]:
        """
        getattr 호출 분석 (스코프 고려)
        
        Args:
            node: getattr Call 노드
            function_name: 현재 함수 이름 (None이면 모듈 레벨)
        
        Returns:
            추론된 호출 리스트
        """
        if not self._is_getattr(node):
            return []
        
        if len(node.args) < 2:
            return []
        
        obj = node.args[0]
        attr = node.args[1]
        
        # 속성 이름이 문자열 리터럴이어야 추론 가능
        if not isinstance(attr, ast.Constant):
            logger.debug("getattr with non-constant attribute, skipping")
            return []
        
        method_name = attr.value
        if not isinstance(method_name, str):
            return []
        
        # 객체 타입 추론 (스코프 고려)
        obj_type = self._infer_type(obj, function_name)
        if not obj_type:
            logger.debug(f"Could not infer type for getattr(..., {method_name!r})")
            return []
        
        # Import alias 해석
        resolved_type = self._resolve_type_name(obj_type)
        
        # 추론된 호출 생성
        inferred = InferredCall(
            source=f"getattr({obj_type}, {method_name!r})",
            target=f"{resolved_type}.{method_name}",
            confidence=0.9,  # 타입 힌트가 있으면 높은 확률
            line=getattr(node, 'lineno', 0)
        )
        
        scope_info = f"[{function_name}]" if function_name else "[module]"
        logger.info(f"{scope_info} Inferred call: {inferred.source} → {inferred.target}")
        
        return [inferred]
    
    def _is_getattr(self, node: ast.Call) -> bool:
        """getattr 호출인지 확인"""
        return (
            isinstance(node.func, ast.Name) and
            node.func.id == "getattr"
        )
    
    def _infer_type(self, node: ast.AST, function_name: Optional[str] = None) -> Optional[str]:
        """
        노드의 타입 추론 (스코프 고려)
        
        전략:
        1. 현재 함수 스코프에서 찾기
        2. 전역 스코프에서 찾기
        3. 함수 호출이면 반환 타입 확인
        
        Args:
            node: AST 노드
            function_name: 현재 함수 이름 (None이면 모듈 레벨)
        
        Returns:
            타입 이름 (예: "UserAuth", "auth.UserAuth")
        """
        if isinstance(node, ast.Name):
            var_name = node.id
            
            # 1. 현재 함수 스코프 우선
            if function_name and function_name in self.function_scopes:
                func_scope = self.function_scopes[function_name]
                if var_name in func_scope:
                    logger.debug(f"Found {var_name} in function scope [{function_name}]")
                    return func_scope[var_name]
            
            # 2. 전역 스코프
            if var_name in self.global_type_map:
                logger.debug(f"Found {var_name} in global scope")
                return self.global_type_map[var_name]
            
            return None
        
        elif isinstance(node, ast.Call):
            # 함수 호출 결과 타입
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in self.function_returns:
                    logger.debug(f"Found return type of {func_name}()")
                    return self.function_returns[func_name]
        
        return None
    
    def _resolve_type_name(self, type_name: str) -> str:
        """
        Alias를 실제 타입 이름으로 변환
        
        Args:
            type_name: 타입 이름 (alias일 수 있음)
        
        Returns:
            실제 타입 이름
        
        Examples:
            UA → auth.UserAuth
            models.User → auth.models.User
        """
        # 단순 이름 (UA)
        if type_name in self.import_aliases:
            resolved = self.import_aliases[type_name]
            logger.debug(f"Resolved alias: {type_name} → {resolved}")
            return resolved
        
        # 복합 이름 (models.User)
        parts = type_name.split(".", 1)
        if len(parts) == 2:
            prefix, suffix = parts
            if prefix in self.import_aliases:
                resolved_prefix = self.import_aliases[prefix]
                resolved = f"{resolved_prefix}.{suffix}"
                logger.debug(f"Resolved prefix: {type_name} → {resolved}")
                return resolved
        
        # 변환 안됨 (원본 그대로)
        return type_name

