"""파일 요약 청크 생성"""

import logging
import re

from ..core.models import CodeChunk, CodeNode

logger = logging.getLogger(__name__)


class FileSummaryBuilder:
    """
    조건부 파일 요약 청크 생성

    목적:
    - "이 파일이 뭐 하는 파일인지" 질문에 답할 수 있는 요약 청크 생성
    - 전체 파일 내용이 아닌 메타데이터 + 구조 정보만 포함
    - Symbol 청크와 중복 최소화

    생성 조건:
    1. 심볼이 많은 허브 파일 (3개 이상)
    2. 설정/스키마 파일 (yaml, toml, json 등)
    3. API 라우터 파일 (endpoints 정보)
    """

    def __init__(
        self,
        min_symbols_for_summary: int = 5,  # 이 이상이면 요약 청크 생성
        max_summary_length: int = 2000,  # 요약 최대 길이
    ):
        """
        Args:
            min_symbols_for_summary: 파일 요약 청크를 생성할 최소 심볼 개수
            max_summary_length: 요약 텍스트 최대 길이 (chars)
        """
        self.min_symbols_for_summary = min_symbols_for_summary
        self.max_summary_length = max_summary_length

    def should_create_summary(
        self,
        file_node: CodeNode,
        symbol_nodes: list[CodeNode],
        file_type: str,
    ) -> bool:
        """
        파일 요약 청크를 생성할지 결정

        Args:
            file_node: File 노드
            symbol_nodes: 이 파일의 Symbol 노드들
            file_type: 파일 확장자 (py, ts, yaml 등)

        Returns:
            True면 요약 청크 생성
        """
        # 1. 설정 파일은 무조건 생성
        config_extensions = {".yaml", ".yml", ".toml", ".json", ".env", ".ini", ".conf"}
        if file_type in config_extensions:
            return True

        # 2. 코드 파일: 심볼 개수 기준
        code_extensions = {".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rs"}
        if file_type in code_extensions:
            symbol_count = len(symbol_nodes)
            # 심볼이 많으면 (허브 파일) 요약 생성
            if symbol_count >= self.min_symbols_for_summary:
                return True

        # 3. API 파일: 항상 생성
        if "api" in file_node.file_path or "route" in file_node.file_path:
            return True

        return False

    def build_summary_chunk(
        self,
        file_node: CodeNode,
        symbol_nodes: list[CodeNode],
        file_content: str | None = None,
    ) -> CodeChunk:
        """
        파일 요약 청크 생성

        Args:
            file_node: File 노드
            symbol_nodes: 이 파일의 Symbol 노드들
            file_content: 파일 원본 (docstring 추출용, 없으면 스킵)

        Returns:
            파일 요약 CodeChunk
        """
        parts = []

        # 1. 파일 경로 및 모듈 정보
        parts.append(f"# File: {file_node.file_path}")
        parts.append(f"# Language: {file_node.language}")
        parts.append("")

        # 2. 파일 docstring (있으면)
        if file_content:
            docstring = self._extract_file_docstring(file_content, file_node.language)
            if docstring:
                parts.append("# File Description:")
                parts.append(docstring)
                parts.append("")

        # 3. 주요 심볼 나열
        if symbol_nodes:
            parts.append(f"# Contains {len(symbol_nodes)} symbols:")
            parts.append("")

            # 심볼을 타입별로 그룹화
            by_kind: dict[str, list[str]] = {}
            for node in symbol_nodes:
                kind = node.kind
                if kind not in by_kind:
                    by_kind[kind] = []
                by_kind[kind].append(node.name)

            # Class, Function, Method 순으로 출력
            for kind in ["Class", "Function", "Method", "Interface", "Type"]:
                if kind in by_kind:
                    names = by_kind[kind][:20]  # 최대 20개만
                    parts.append(f"## {kind}s ({len(by_kind[kind])} total):")
                    for name in names:
                        parts.append(f"  - {name}")
                    if len(by_kind[kind]) > 20:
                        parts.append(f"  ... and {len(by_kind[kind]) - 20} more")
                    parts.append("")

        # 4. API 엔드포인트 (API 파일인 경우)
        endpoints = self._extract_api_endpoints(symbol_nodes, file_content)
        if endpoints:
            parts.append("# API Endpoints:")
            for method, path, handler in endpoints[:10]:  # 최대 10개
                parts.append(f"  {method} {path} -> {handler}")
            parts.append("")

        # 5. 텍스트 조합 및 길이 제한
        summary_text = "\n".join(parts)
        if len(summary_text) > self.max_summary_length:
            summary_text = summary_text[: self.max_summary_length] + "\n... (truncated)"

        # 6. CodeChunk 생성
        chunk_id = f"{file_node.repo_id}:{file_node.file_path}:File:summary"

        return CodeChunk(
            repo_id=file_node.repo_id,
            id=chunk_id,
            node_id=file_node.id,
            file_path=file_node.file_path,
            span=file_node.span,
            language=file_node.language,
            text=summary_text,
            attrs={
                "node_kind": "File",
                "node_name": file_node.name,
                "is_file_summary": True,
                "symbol_count": len(symbol_nodes),
            },
        )

    def _extract_file_docstring(self, content: str, language: str) -> str | None:
        """파일 상단 docstring/주석 추출"""
        lines = content.split("\n")
        docstring_lines = []

        if language == "python":
            # Python: 상단 """ 또는 '''
            in_docstring = False
            quote_char = None

            for line in lines[:50]:  # 상위 50줄만
                stripped = line.strip()

                # 빈 줄/주석은 스킵
                if (not stripped or stripped.startswith("#")) and not in_docstring:
                    continue

                # Docstring 시작
                if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                    quote_char = '"""' if '"""' in stripped else "'''"
                    in_docstring = True
                    docstring_lines.append(stripped.replace(quote_char, ""))
                    # 한 줄 docstring
                    if stripped.count(quote_char) >= 2:
                        break
                    continue

                # Docstring 끝
                if in_docstring and quote_char:
                    if quote_char in stripped:
                        docstring_lines.append(stripped.replace(quote_char, ""))
                        break
                    docstring_lines.append(stripped)

                # import 나오면 종료
                if not in_docstring and (
                    stripped.startswith("import ") or stripped.startswith("from ")
                ):
                    break

        elif language in ["typescript", "javascript"]:
            # TypeScript/JavaScript: 상단 /* */ 또는 //
            in_block_comment = False

            for line in lines[:50]:
                stripped = line.strip()

                if not stripped and not in_block_comment:
                    continue

                # 블록 주석 시작
                if stripped.startswith("/*"):
                    in_block_comment = True
                    docstring_lines.append(stripped.replace("/*", "").replace("*/", "").strip())
                    if "*/" in stripped:
                        break
                    continue

                # 블록 주석 끝
                if in_block_comment:
                    if "*/" in stripped:
                        docstring_lines.append(stripped.replace("*/", "").strip())
                        break
                    docstring_lines.append(stripped.replace("*", "").strip())
                    continue

                # 단일 줄 주석
                if stripped.startswith("//"):
                    docstring_lines.append(stripped.replace("//", "").strip())
                    continue

                # import 나오면 종료
                if stripped.startswith("import ") or stripped.startswith("export "):
                    break

        # 결과 조합
        if docstring_lines:
            result = "\n".join(docstring_lines).strip()
            # 최대 500자
            if len(result) > 500:
                result = result[:500] + "..."
            return result

        return None

    def _extract_api_endpoints(
        self,
        symbol_nodes: list[CodeNode],
        file_content: str | None,
    ) -> list[tuple[str, str, str]]:
        """
        API 엔드포인트 추출

        Returns:
            [(method, path, handler_name), ...]
        """
        if not file_content:
            return []

        endpoints = []

        # FastAPI: @router.get("/path")
        fastapi_pattern = r'@\w+\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']\)'
        matches = re.finditer(fastapi_pattern, file_content, re.IGNORECASE)

        for match in matches:
            method = match.group(1).upper()
            path = match.group(2)

            # 다음 줄에서 함수 이름 찾기
            start_pos = match.end()
            next_lines = file_content[start_pos : start_pos + 200]
            func_match = re.search(r"def\s+(\w+)", next_lines)
            handler = func_match.group(1) if func_match else "unknown"

            endpoints.append((method, path, handler))

        # Express: app.get("/path", handler)
        express_pattern = r'app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']\s*,\s*(\w+)'
        matches = re.finditer(express_pattern, file_content, re.IGNORECASE)

        for match in matches:
            method = match.group(1).upper()
            path = match.group(2)
            handler = match.group(3)
            endpoints.append((method, path, handler))

        return endpoints[:20]  # 최대 20개
