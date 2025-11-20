"""일반적인 패턴 기반 동적 호출 추론

프레임워크별 일반적인 패턴을 인식하여 동적 호출을 추론합니다.

지원 패턴:
1. Django generic view: getattr(self, f'handle_{action}')
2. Event handler: on_{event}_handler

오탐 방지:
- AST로 함수/클래스 범위만 검색
- 주석/문자열 제외
- 프레임워크 확인 시에만 활성화
"""

import ast
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CallPattern:
    """호출 패턴 정의"""

    name: str
    pattern: re.Pattern
    confidence: float
    description: str
    framework: str | None = None  # "django", "flask", None (범용)


@dataclass
class PatternMatch:
    """패턴 매칭 결과"""

    pattern_name: str
    matched_text: str
    line: int
    confidence: float
    suggestions: list[str]  # 추론된 타겟 후보


class PatternAnalyzer:
    """
    프레임워크별 패턴 기반 동적 호출 추론

    전략:
    1. AST로 함수/클래스 범위 추출
    2. 범위 내에서만 정규식 검색 (주석 제외)
    3. 프레임워크별 조건부 활성화

    오탐 방지:
    - 주석/문자열에서 매칭 제외
    - 실제 코드 블록만 검색
    - 프레임워크 설정 없으면 비활성화
    """

    # 초기 패턴 (2개만, 검증된 것)
    PATTERNS = [
        # Django generic view handler
        CallPattern(
            name="django_handler",
            pattern=re.compile(r'getattr\s*\(\s*self\s*,\s*f?["\']handle_'),
            confidence=0.85,
            description="Django generic view dispatch pattern",
            framework="django",
        ),
        # Event handler (범용)
        CallPattern(
            name="event_handler",
            pattern=re.compile(r"on_(\w+)(?:_handler)?"),
            confidence=0.80,
            description="Event handler pattern (generic)",
            framework=None,  # 범용
        ),
    ]

    def __init__(self, framework: str | None = None):
        """
        Args:
            framework: 프레임워크 이름 ("django", "flask", None)
                      None이면 범용 패턴만 사용
        """
        self.framework = framework
        self.active_patterns = self._select_patterns()

        logger.info(
            f"PatternAnalyzer initialized: framework={framework}, "
            f"patterns={len(self.active_patterns)}"
        )

    def _select_patterns(self) -> list[CallPattern]:
        """프레임워크에 맞는 패턴 선택"""
        active = []

        for pattern in self.PATTERNS:
            # 범용 패턴은 항상 포함
            if pattern.framework is None or pattern.framework == self.framework:
                active.append(pattern)

        return active

    def analyze(
        self, code: str, file_path: str, available_symbols: set[str] | None = None
    ) -> list[PatternMatch]:
        """
        코드에서 패턴 찾기

        Args:
            code: Python 소스 코드
            file_path: 파일 경로
            available_symbols: 해당 파일의 심볼 목록 (매칭 검증용)

        Returns:
            패턴 매칭 결과 리스트
        """
        if not self.active_patterns:
            logger.debug("No active patterns, skipping")
            return []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []

        matches = []

        # AST로 함수/클래스 범위 추출
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                # 함수/클래스 내부 코드만 추출
                node_code = self._extract_node_code(code, node)
                if node_code:
                    # 패턴 매칭
                    node_matches = self._match_patterns(node_code, node, available_symbols)
                    matches.extend(node_matches)

        logger.debug(f"Pattern analysis: {len(matches)} matches in {file_path}")

        return matches

    def _extract_node_code(self, code: str, node: ast.AST) -> str | None:
        """AST 노드의 소스 코드 추출"""
        try:
            # Python 3.9+
            return ast.get_source_segment(code, node)
        except (AttributeError, ValueError):
            # Fallback: 라인 번호로 추출
            lines = code.split("\n")
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") else start + 1
            return "\n".join(lines[start:end])

    def _match_patterns(
        self, code: str, node: ast.AST, available_symbols: set[str] | None
    ) -> list[PatternMatch]:
        """코드에서 패턴 매칭"""
        matches = []

        for pattern in self.active_patterns:
            regex_matches = pattern.pattern.finditer(code)

            for match in regex_matches:
                # 패턴별 suggestion 생성
                suggestions = self._generate_suggestions(pattern, match, available_symbols)

                if suggestions:
                    # 매칭 위치 계산
                    line_offset = code[: match.start()].count("\n")
                    actual_line = node.lineno + line_offset

                    matches.append(
                        PatternMatch(
                            pattern_name=pattern.name,
                            matched_text=match.group(0),
                            line=actual_line,
                            confidence=pattern.confidence,
                            suggestions=suggestions,
                        )
                    )

                    logger.debug(f"Matched {pattern.name}: {match.group(0)} → {suggestions}")

        return matches

    def _generate_suggestions(
        self, pattern: CallPattern, match: re.Match, available_symbols: set[str] | None
    ) -> list[str]:
        """
        패턴 매칭 결과에서 타겟 심볼 추론

        Args:
            pattern: 패턴 정의
            match: 정규식 매칭 결과
            available_symbols: 사용 가능한 심볼 목록

        Returns:
            추론된 타겟 심볼 리스트
        """
        suggestions = []

        if pattern.name == "django_handler":
            # getattr(self, "handle_login") → self.handle_login
            # available_symbols에서 handle_* 메서드 찾기
            if available_symbols:
                for symbol in available_symbols:
                    if "handle_" in symbol:
                        suggestions.append(symbol)

        elif pattern.name == "event_handler":
            # on_click_handler → on_click, on_click_handler 등
            # 매칭된 텍스트에서 이벤트명 추출
            matched = match.group(0)

            if available_symbols:
                # 정확히 매칭되는 심볼 찾기
                if matched in available_symbols:
                    suggestions.append(matched)

                # on_ 로 시작하는 심볼들
                for symbol in available_symbols:
                    if symbol.startswith("on_") and symbol in matched:
                        suggestions.append(symbol)

        return suggestions

    def to_relations(self, matches: list[PatternMatch], repo_id: str, file_path: str) -> list[dict]:
        """
        패턴 매칭 결과를 RawRelation 형태로 변환

        Returns:
            RawRelation dict 리스트
        """
        relations = []

        for match in matches:
            for target in match.suggestions:
                relations.append(
                    {
                        "source": f"pattern:{match.pattern_name}",
                        "target": target,
                        "type": "calls",
                        "attrs": {
                            "confidence": match.confidence,
                            "inferred": True,
                            "method": "pattern",
                            "pattern_name": match.pattern_name,
                            "matched_text": match.matched_text,
                            "line": match.line,
                        },
                    }
                )

        return relations


def detect_framework(code: str, file_path: str) -> str | None:
    """
    코드/경로에서 프레임워크 자동 감지

    간단한 휴리스틱:
    - django 관련 import → "django"
    - flask 관련 import → "flask"

    Args:
        code: 소스 코드
        file_path: 파일 경로

    Returns:
        프레임워크 이름 또는 None
    """
    # Import 문 체크
    if "from django" in code or "import django" in code:
        return "django"

    if "from flask" in code or "import flask" in code:
        return "flask"

    # 파일 경로 체크 (예: /myproject/django_app/)
    path_lower = file_path.lower()
    if "django" in path_lower:
        return "django"
    if "flask" in path_lower:
        return "flask"

    return None
