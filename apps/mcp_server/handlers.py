"""MCP 요청 핸들러"""

import logging
from typing import Any

from .service import MCPService

logger = logging.getLogger(__name__)


class MCPHandlers:
    """
    MCP 프로토콜 요청 핸들러

    Tools, Resources, Prompts 제공
    """

    def __init__(self, service: MCPService):
        """
        Args:
            service: MCP 서비스
        """
        self.service = service

    # ==================== Tools ====================

    async def handle_list_repositories(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        list_repositories 도구 핸들러

        Returns:
            {"repositories": [...]}
        """
        repositories = await self.service.list_repositories()
        return {"repositories": repositories}

    async def handle_search_code(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        search_code 도구 핸들러

        Args:
            arguments: {
                "query": str,
                "repo_ids": List[str] (optional),
                "limit": int (optional, default: 20),
                "file_path": str (optional),
                "language": str (optional)
            }

        Returns:
            {"results": [...]}
        """
        query = arguments.get("query")
        if not query:
            return {"error": "Missing required parameter: query"}

        repo_ids = arguments.get("repo_ids")
        limit = arguments.get("limit", 20)
        file_path = arguments.get("file_path")
        language = arguments.get("language")

        results = await self.service.search_code(
            query=query, repo_ids=repo_ids, limit=limit, file_path=file_path, language=language
        )

        return {"results": results}

    async def handle_find_definition(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        find_definition 도구 핸들러

        Args:
            arguments: {
                "symbol_name": str,
                "repo_ids": List[str] (optional),
                "kind": str (optional)
            }

        Returns:
            {"definitions": [...]}
        """
        symbol_name = arguments.get("symbol_name")
        if not symbol_name:
            return {"error": "Missing required parameter: symbol_name"}

        repo_ids = arguments.get("repo_ids")
        kind = arguments.get("kind")

        definitions = await self.service.find_definition(
            symbol_name=symbol_name, repo_ids=repo_ids, kind=kind
        )

        return {"definitions": definitions}

    async def handle_explore_graph(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        explore_graph 도구 핸들러

        Args:
            arguments: {
                "repo_id": str,
                "node_id": str,
                "direction": str (optional, default: "both"),
                "edge_types": List[str] (optional),
                "depth": int (optional, default: 1)
            }

        Returns:
            {"center": {...}, "neighbors": [...]}
        """
        repo_id = arguments.get("repo_id")
        node_id = arguments.get("node_id")

        if not repo_id or not node_id:
            return {"error": "Missing required parameters: repo_id, node_id"}

        direction = arguments.get("direction", "both")
        edge_types = arguments.get("edge_types")
        depth = arguments.get("depth", 1)

        result = await self.service.explore_graph(
            repo_id=repo_id,
            node_id=node_id,
            direction=direction,
            edge_types=edge_types,
            depth=depth,
        )

        return result

    async def handle_get_context(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        get_context 도구 핸들러

        Args:
            arguments: {
                "repo_id": str,
                "file_path": str,
                "line": int,
                "max_tokens": int (optional, default: 4000)
            }

        Returns:
            {"primary": {...}, "supporting": [...]}
        """
        repo_id = arguments.get("repo_id")
        file_path = arguments.get("file_path")
        line = arguments.get("line")

        if not repo_id or not file_path or line is None:
            return {"error": "Missing required parameters: repo_id, file_path, line"}

        max_tokens = arguments.get("max_tokens", 4000)

        result = await self.service.get_context(
            repo_id=repo_id, file_path=file_path, line=line, max_tokens=max_tokens
        )

        return result

    # ==================== Resources ====================

    async def handle_resource(self, uri: str) -> dict[str, Any] | None:
        """
        리소스 조회 핸들러

        지원 URI:
        - node://{repo_id}/{node_id}
        - file://{repo_id}/{file_path}

        Args:
            uri: 리소스 URI

        Returns:
            리소스 데이터 또는 None
        """
        logger.info(f"handle_resource: {uri}")

        if uri.startswith("node://"):
            # node://{repo_id}/{node_id}
            parts = uri[7:].split("/", 1)
            if len(parts) != 2:
                logger.error(f"Invalid node URI: {uri}")
                return None

            repo_id, node_id = parts
            node_data = await self.service.get_node(repo_id, node_id)

            if node_data:
                return {"uri": uri, "mimeType": "application/json", "data": node_data}
            return None

        elif uri.startswith("file://"):
            # file://{repo_id}/{file_path}
            parts = uri[7:].split("/", 1)
            if len(parts) != 2:
                logger.error(f"Invalid file URI: {uri}")
                return None

            repo_id, file_path = parts
            chunks = await self.service.get_file_chunks(repo_id, file_path)

            return {
                "uri": uri,
                "mimeType": "application/json",
                "data": {"file_path": file_path, "chunks": chunks},
            }

        else:
            logger.error(f"Unknown resource URI scheme: {uri}")
            return None

    # ==================== Tool Definitions ====================

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        도구 정의 반환 (MCP 클라이언트에게 제공)

        Returns:
            도구 정의 리스트
        """
        return [
            {
                "name": "list_repositories",
                "description": (
                    "인덱싱된 코드베이스 저장소 목록 조회\n\n"
                    "용도: 에이전트가 어떤 코드베이스가 있는지 먼저 알아야 함\n"
                    "시나리오:\n"
                    "  User: '어디에 인증 로직이 있어?'\n"
                    "  Agent: list_repositories() → [repo1, repo2, repo3]\n"
                    "         search_code('authentication', repo_ids=[repo1, repo2, repo3])"
                ),
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "search_code",
                "description": (
                    "쿼리로 관련 코드 찾기 (하이브리드 검색: 키워드 + 의미론 + 그래프 + 퍼지)\n\n"
                    "용도: 쿼리로 관련 코드 찾기\n"
                    "시나리오:\n"
                    "  User: 'API 에러 핸들링 어떻게 돼있어?'\n"
                    "  Agent: search_code('error handling API')\n"
                    "         → [chunk1: try-except 블록, chunk2: HTTPException, ...]"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "검색 쿼리"},
                        "repo_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "검색할 저장소 ID 목록 (옵션, 미지정 시 모든 저장소)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "반환 결과 수 (기본값: 20)",
                            "default": 20,
                        },
                        "file_path": {
                            "type": "string",
                            "description": "현재 파일 경로 (컨텍스트 용, 옵션)",
                        },
                        "language": {
                            "type": "string",
                            "description": "언어 필터 (python, typescript 등, 옵션)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "find_definition",
                "description": (
                    "심볼(함수, 클래스, 변수) 정의 위치 찾기\n\n"
                    "용도: 심볼 정의 위치 찾기\n"
                    "시나리오:\n"
                    "  User: 'calculate_total 함수 정의 어디 있어?'\n"
                    "  Agent: find_definition('calculate_total')\n"
                    "         → {file_path: 'utils.py', line: 45, text: 'def calculate_total...'}"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol_name": {
                            "type": "string",
                            "description": "찾을 심볼 이름 (함수명, 클래스명 등)",
                        },
                        "repo_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "검색할 저장소 ID 목록 (옵션, 미지정 시 모든 저장소)",
                        },
                        "kind": {
                            "type": "string",
                            "description": "심볼 종류 필터 (Function, Class, Method, Variable 등, 옵션)",
                        },
                    },
                    "required": ["symbol_name"],
                },
            },
            {
                "name": "explore_graph",
                "description": (
                    "코드 그래프 관계 탐색 (호출 관계, 의존성)\n\n"
                    "용도: 코드 관계 탐색 (호출 관계, 의존성)\n"
                    "시나리오:\n"
                    "  User: '이 함수 어디서 호출돼?'\n"
                    "  Agent: find_definition('process_payment')  # node_id 얻기\n"
                    "         explore_graph(node_id, direction='incoming', edge_types=['calls'])\n"
                    "         → [caller1, caller2, ...]"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_id": {"type": "string", "description": "저장소 ID"},
                        "node_id": {"type": "string", "description": "시작 노드 ID"},
                        "direction": {
                            "type": "string",
                            "enum": ["outgoing", "incoming", "both"],
                            "description": "탐색 방향 (outgoing: 이 노드가 호출하는 것, incoming: 이 노드를 호출하는 것, both: 양방향, 기본값: both)",
                            "default": "both",
                        },
                        "edge_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "엣지 타입 필터 (calls, defines, uses, imports 등, 옵션)",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "탐색 깊이 (기본값: 1)",
                            "default": 1,
                        },
                    },
                    "required": ["repo_id", "node_id"],
                },
            },
            {
                "name": "get_context",
                "description": (
                    "특정 위치의 관련 코드 컨텍스트 가져오기 (LLM에게 전달할 패킹된 컨텍스트)\n\n"
                    "용도: 특정 위치의 관련 컨텍스트 패킹 (LLM에게 전달)\n"
                    "시나리오:\n"
                    "  User: '이 코드 설명해줘' (커서 위치: main.py:150)\n"
                    "  Agent: get_context(repo_id, 'main.py', 150)\n"
                    "         → {primary: 해당 함수, supporting: 관련 클래스/함수}\n"
                    "         format_prompt(context) → LLM 호출"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_id": {"type": "string", "description": "저장소 ID"},
                        "file_path": {"type": "string", "description": "파일 경로"},
                        "line": {"type": "integer", "description": "라인 번호 (0-based)"},
                        "max_tokens": {
                            "type": "integer",
                            "description": "최대 토큰 수 (기본값: 4000)",
                            "default": 4000,
                        },
                    },
                    "required": ["repo_id", "file_path", "line"],
                },
            },
        ]

    def get_resource_templates(self) -> list[dict[str, Any]]:
        """
        리소스 템플릿 반환

        Returns:
            리소스 템플릿 리스트
        """
        return [
            {
                "uriTemplate": "node://{repo_id}/{node_id}",
                "name": "Code Node",
                "description": "코드 그래프 노드 정보",
                "mimeType": "application/json",
            },
            {
                "uriTemplate": "file://{repo_id}/{file_path}",
                "name": "File Chunks",
                "description": "파일의 모든 코드 청크",
                "mimeType": "application/json",
            },
        ]
