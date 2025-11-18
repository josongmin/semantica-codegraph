"""MCP 서버 메인"""

import asyncio
import logging
import sys
from typing import Any

from src.core.bootstrap import create_bootstrap

from .handlers import MCPHandlers
from .service import MCPService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # MCP는 stdio로 통신하므로 stderr로 로그 출력
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger(__name__)


class MCPServer:
    """
    MCP 서버

    stdio를 통해 MCP 프로토콜로 통신
    """

    def __init__(self):
        """
        서버 초기화

        Bootstrap을 통해 모든 컴포넌트 주입
        """
        logger.info("Initializing MCP Server...")

        # Bootstrap 생성 (환경변수에서 설정 로드)
        self.bootstrap = create_bootstrap()

        # Service & Handlers 생성
        self.service = MCPService(self.bootstrap)
        self.handlers = MCPHandlers(self.service)

        logger.info("MCP Server initialized")

    async def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        초기화 요청 처리

        Args:
            params: 초기화 파라미터

        Returns:
            서버 정보
        """
        logger.info("Handling initialize request")

        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "semantica-codegraph",
                "version": "0.1.0"
            },
            "capabilities": {
                "tools": {
                    "listChanged": False
                },
                "resources": {
                    "subscribe": False,
                    "listChanged": False
                }
            }
        }

    async def handle_tools_list(self) -> dict[str, Any]:
        """
        도구 목록 요청 처리

        Returns:
            도구 정의 목록
        """
        logger.info("Handling tools/list request")

        tools = self.handlers.get_tool_definitions()
        return {"tools": tools}

    async def handle_tools_call(
        self,
        name: str,
        arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        도구 호출 요청 처리

        Args:
            name: 도구 이름
            arguments: 도구 인자

        Returns:
            도구 실행 결과
        """
        logger.info(f"Handling tools/call: {name}")

        try:
            if name == "search_code":
                result = await self.handlers.handle_search_code(arguments)
            elif name == "find_definition":
                result = await self.handlers.handle_find_definition(arguments)
            elif name == "explore_graph":
                result = await self.handlers.handle_explore_graph(arguments)
            elif name == "get_context":
                result = await self.handlers.handle_get_context(arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}

            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(result)
                    }
                ]
            }

        except Exception as e:
            logger.error(f"Tool call failed: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {str(e)}"
                    }
                ],
                "isError": True
            }

    async def handle_resources_list(self) -> dict[str, Any]:
        """
        리소스 목록 요청 처리

        Returns:
            리소스 템플릿 목록
        """
        logger.info("Handling resources/list request")

        templates = self.handlers.get_resource_templates()
        return {"resourceTemplates": templates}

    async def handle_resources_read(self, uri: str) -> dict[str, Any]:
        """
        리소스 읽기 요청 처리

        Args:
            uri: 리소스 URI

        Returns:
            리소스 데이터
        """
        logger.info(f"Handling resources/read: {uri}")

        try:
            resource = await self.handlers.handle_resource(uri)

            if resource:
                return {
                    "contents": [
                        {
                            "uri": resource["uri"],
                            "mimeType": resource["mimeType"],
                            "text": str(resource["data"])
                        }
                    ]
                }
            else:
                return {
                    "contents": [],
                    "error": f"Resource not found: {uri}"
                }

        except Exception as e:
            logger.error(f"Resource read failed: {e}", exc_info=True)
            return {
                "contents": [],
                "error": str(e)
            }

    async def run_stdio(self):
        """
        Stdio 모드로 서버 실행

        JSON-RPC 2.0 메시지를 stdin에서 읽고 stdout으로 응답
        """
        logger.info("Starting MCP Server in stdio mode")

        # 참고: 실제 MCP SDK를 사용하면 이 부분은 자동 처리됨
        # 여기서는 구조만 제시

        print("MCP Server ready (stdio mode)", file=sys.stderr)
        print("Note: Use official MCP Python SDK for production", file=sys.stderr)

        # TODO: 실제 구현은 mcp 패키지 사용
        # from mcp.server import Server
        # from mcp.server.stdio import stdio_server
        #
        # server = Server("semantica-codegraph")
        #
        # @server.list_tools()
        # async def list_tools():
        #     return self.handlers.get_tool_definitions()
        #
        # @server.call_tool()
        # async def call_tool(name: str, arguments: dict):
        #     return await self.handle_tools_call(name, arguments)
        #
        # async with stdio_server() as streams:
        #     await server.run(streams[0], streams[1], server.create_initialization_options())

        # 임시 대기 (실제로는 위 SDK 코드 사용)
        await asyncio.Event().wait()


def main():
    """
    MCP 서버 엔트리포인트
    """
    try:
        server = MCPServer()
        asyncio.run(server.run_stdio())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
