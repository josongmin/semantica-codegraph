"""API Route 추출"""

import hashlib
import logging
import re
from dataclasses import dataclass, field

from ..core.models import CodeNode, FileProfile, RepoId

logger = logging.getLogger(__name__)


@dataclass
class RouteInfo:
    """Route 정보"""

    repo_id: RepoId
    http_method: str
    http_path: str
    handler_symbol_id: str
    handler_name: str
    file_path: str
    start_line: int
    end_line: int
    framework: str
    router_prefix: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """route_id 생성"""
        hash_input = (
            f"{self.repo_id}:{self.http_method}:{self.http_path}:{self.file_path}:{self.start_line}"
        )
        self.route_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]


class RouteExtractor:
    """
    CodeNode decorator에서 Route 정보 추출

    지원 프레임워크:
    - FastAPI: @router.post("/search"), @app.get("/health")
    - Django: @api_view(['POST'])
    - Spring: @PostMapping("/search"), @GetMapping("/users")
    - Express: (decorator 아님, 코드 패턴 분석 필요)
    """

    def extract_routes(
        self,
        nodes: list[CodeNode],
        file_profile: FileProfile | None = None,
    ) -> list[RouteInfo]:
        """
        CodeNode 리스트에서 Route 추출

        Args:
            nodes: 심볼 노드들
            file_profile: 파일 프로파일 (framework 정보)

        Returns:
            RouteInfo 리스트
        """
        routes = []
        framework = file_profile.api_framework if file_profile else None

        for node in nodes:
            if node.kind not in ("Function", "Method"):
                continue

            decorators = node.attrs.get("decorators", [])
            if not decorators:
                continue

            # Framework별 route 추출
            route_info = None
            if framework == "fastapi":
                route_info = self._extract_fastapi_route(node, decorators)
            elif framework == "django":
                route_info = self._extract_django_route(node, decorators)
            elif framework == "spring":
                route_info = self._extract_spring_route(node, decorators)
            # Express는 decorator가 아니므로 skip

            if route_info:
                route_info.framework = framework or "unknown"
                routes.append(route_info)

        logger.info(f"Extracted {len(routes)} routes from {len(nodes)} nodes")
        return routes

    def _extract_fastapi_route(self, node: CodeNode, decorators: list[str]) -> RouteInfo | None:
        """
        FastAPI decorator 파싱

        Examples:
            @router.post("/search")
            @app.get("/health")
            @router.get("/users/{user_id}")
        """
        for dec in decorators:
            # @router.post("/search") or @app.get("/health")
            match = re.match(
                r'(?:router|app)\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']\)', dec
            )
            if match:
                method = match.group(1).upper()
                path = match.group(2)

                return RouteInfo(
                    repo_id=node.repo_id,
                    http_method=method,
                    http_path=path,
                    handler_symbol_id=node.id,
                    handler_name=node.name,
                    file_path=node.file_path,
                    start_line=node.span[0],
                    end_line=node.span[2],
                    framework="fastapi",
                )
        return None

    def _extract_django_route(self, node: CodeNode, decorators: list[str]) -> RouteInfo | None:
        """
        Django @api_view 파싱

        Examples:
            @api_view(['POST'])
            @api_view(['GET', 'POST'])

        Note:
            Django는 path가 urls.py에 정의되므로 정확한 경로를 알 수 없음.
            함수명으로 추정.
        """
        for dec in decorators:
            if dec.startswith("api_view"):
                # @api_view(['POST'])
                match = re.search(r"'(GET|POST|PUT|DELETE|PATCH)'", dec)
                if match:
                    method = match.group(1)
                    # Path는 urls.py에서 정의되므로 함수명으로 추정
                    path = f"/{node.name}/"

                    return RouteInfo(
                        repo_id=node.repo_id,
                        http_method=method,
                        http_path=path,
                        handler_symbol_id=node.id,
                        handler_name=node.name,
                        file_path=node.file_path,
                        start_line=node.span[0],
                        end_line=node.span[2],
                        framework="django",
                        metadata={"path_inferred": True},
                    )
        return None

    def _extract_spring_route(self, node: CodeNode, decorators: list[str]) -> RouteInfo | None:
        """
        Spring @Mapping 파싱

        Examples:
            @PostMapping("/search")
            @GetMapping("/users/{id}")
            @RequestMapping(value="/api", method=RequestMethod.POST)
        """
        for dec in decorators:
            # @PostMapping("/search")
            match = re.match(r'(Get|Post|Put|Delete|Patch)Mapping\(["\']([^"\']+)["\']\)', dec)
            if match:
                method = match.group(1).upper()
                path = match.group(2)

                return RouteInfo(
                    repo_id=node.repo_id,
                    http_method=method,
                    http_path=path,
                    handler_symbol_id=node.id,
                    handler_name=node.name,
                    file_path=node.file_path,
                    start_line=node.span[0],
                    end_line=node.span[2],
                    framework="spring",
                )
        return None
