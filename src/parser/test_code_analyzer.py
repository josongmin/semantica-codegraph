"""테스트 코드에서 실제 호출 관계 추출

테스트 코드는 실제 사용 예시이므로,
테스트에서 호출하는 메서드 = 실제 API

전략:
- 테스트 파일만 분석 (성능)
- pytest, unittest 패턴 지원
- 높은 신뢰도 (0.95)
"""

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestCall:
    """테스트에서 추출된 호출"""
    test_function: str  # 테스트 함수 이름
    called_symbol: str  # 호출된 심볼
    line: int
    confidence: float = 0.95  # 테스트 코드는 신뢰도 높음


class TestCodeAnalyzer:
    """
    테스트 코드 분석

    지원:
    1. pytest 함수: def test_*()
    2. unittest 메서드: class Test* / def test_*()
    3. pytest 데코레이터: @pytest.mark.*

    제한:
    - 테스트 파일만 분석 (tests/, test_*.py, *_test.py)
    """

    @staticmethod
    def is_test_file(file_path: str) -> bool:
        """
        테스트 파일 판별

        Args:
            file_path: 파일 경로

        Returns:
            True if test file

        패턴:
        - tests/ 폴더
        - test_*.py
        - *_test.py
        - specs/ 폴더
        """
        path_parts = Path(file_path).parts
        file_path.lower()
        name_lower = Path(file_path).name.lower()

        return any([
            "tests" in path_parts,
            "test" in path_parts,
            name_lower.startswith("test_"),
            name_lower.endswith("_test.py"),
            "spec" in path_parts,
            "specs" in path_parts,
        ])

    def analyze(
        self,
        code: str,
        file_path: str,
        available_symbols: set[str] | None = None
    ) -> list[TestCall]:
        """
        테스트 코드에서 호출 관계 추출

        Args:
            code: 소스 코드
            file_path: 파일 경로
            available_symbols: 사용 가능한 심볼 (검증용)

        Returns:
            테스트 호출 리스트
        """
        # 테스트 파일이 아니면 스킵
        if not self.is_test_file(file_path):
            logger.debug(f"Not a test file, skipping: {file_path}")
            return []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []

        test_calls = []

        # 테스트 함수/메서드 찾기
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and self._is_test_function(node):
                # 테스트 내부의 호출 추출
                calls = self._extract_calls(node)

                for call_str, line in calls:
                    # 심볼 검증 (옵션)
                    if available_symbols and call_str not in available_symbols:
                        # 외부 라이브러리 호출일 수 있으므로 스킵
                        continue

                    test_calls.append(
                        TestCall(
                            test_function=node.name,
                            called_symbol=call_str,
                            line=line,
                            confidence=0.95
                        )
                    )

        logger.debug(
            f"Test analysis: {len(test_calls)} calls found in {file_path}"
        )

        return test_calls

    def _is_test_function(self, node: ast.FunctionDef) -> bool:
        """
        테스트 함수/메서드인지 확인

        패턴:
        1. def test_*()
        2. @pytest.mark.* 데코레이터
        3. @unittest.* 데코레이터
        """
        # 1. 함수명 패턴
        if node.name.startswith("test_"):
            return True

        # 2. 데코레이터 체크
        for decorator in node.decorator_list:
            # @pytest.mark.*
            if isinstance(decorator, ast.Attribute):
                if decorator.attr == "mark":
                    return True

            # @pytest.*, @unittest.*
            elif isinstance(decorator, ast.Name):
                dec_name = decorator.id.lower()
                if "test" in dec_name or "pytest" in dec_name:
                    return True

        return False

    def _extract_calls(self, node: ast.FunctionDef) -> list[tuple[str, int]]:
        """
        함수 내부의 모든 호출 추출

        Returns:
            [(call_string, line_number), ...]
        """
        calls = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_str = self._get_call_string(child)
                line = getattr(child, 'lineno', 0)

                if call_str:
                    calls.append((call_str, line))

        return calls

    def _get_call_string(self, call: ast.Call) -> str | None:
        """
        호출을 문자열로 변환

        Examples:
            obj.method() → "obj.method"
            function() → "function"
            Class() → "Class"
        """
        if isinstance(call.func, ast.Attribute):
            # obj.method()
            obj_name = self._get_name(call.func.value)
            method_name = call.func.attr

            # 외부 라이브러리 제외 (일반적 패턴)
            if obj_name in ["self", "cls"]:
                # self.method → method만
                return method_name

            return f"{obj_name}.{method_name}"

        elif isinstance(call.func, ast.Name):
            # function() or Class()
            return call.func.id

        return None

    def _get_name(self, node: ast.AST) -> str:
        """노드에서 이름 추출"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ""

    def to_relations(
        self,
        test_calls: list[TestCall],
        repo_id: str,
        file_path: str
    ) -> list[dict]:
        """
        테스트 호출을 RawRelation 형태로 변환

        Returns:
            RawRelation dict 리스트
        """
        relations = []

        for call in test_calls:
            relations.append({
                "source": f"test:{call.test_function}",
                "target": call.called_symbol,
                "type": "calls",
                "attrs": {
                    "confidence": call.confidence,
                    "inferred": True,
                    "method": "test_analysis",
                    "test_function": call.test_function,
                    "line": call.line
                }
            })

        return relations

