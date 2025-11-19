"""MCP 서버 비즈니스 로직"""

import logging

from src.core.bootstrap import Bootstrap
from src.core.models import LocationContext, RepoId

logger = logging.getLogger(__name__)


class MCPService:
    """
    MCP 서버의 비즈니스 로직

    Bootstrap을 통해 검색/그래프 기능 제공
    """

    def __init__(self, bootstrap: Bootstrap):
        """
        Args:
            bootstrap: Bootstrap 인스턴스 (DI)
        """
        self.bootstrap = bootstrap
        self.retriever = bootstrap.hybrid_retriever
        self.graph_search = bootstrap.graph_search
        self.graph_store = bootstrap.graph_store
        self.chunk_store = bootstrap.chunk_store
        self.context_packer = bootstrap.context_packer
        self.repo_store = bootstrap.repo_store

    async def search_code(
        self,
        query: str,
        repo_ids: list[RepoId] | None = None,
        limit: int = 20,
        file_path: str | None = None,
        language: str | None = None,
    ) -> list[dict]:
        """
        코드 검색 (하이브리드)

        Args:
            query: 검색 쿼리
            repo_ids: 검색할 저장소 ID 목록 (None이면 모든 repo)
            limit: 반환 결과 수
            file_path: 현재 파일 경로 (옵션)
            language: 언어 필터 (옵션)

        Returns:
            검색 결과 리스트
            [
                {
                    "repo_id": str,
                    "chunk_id": str,
                    "file_path": str,
                    "line_start": int,
                    "line_end": int,
                    "text": str,
                    "score": float,
                    "scores": {
                        "lexical": float,
                        "semantic": float,
                        "graph": float,
                        "fuzzy": float
                    }
                }
            ]
        """
        logger.info(f"search_code: query={query}, repos={repo_ids}, limit={limit}")

        # repo_ids가 없으면 모든 repo 검색
        if not repo_ids:
            repos = self.repo_store.list_repositories()
            repo_ids = [repo.repo_id for repo in repos]
            logger.debug(f"No repo_ids specified, searching all: {repo_ids}")

        # 각 repo에서 검색
        all_results = []
        for repo_id in repo_ids:
            try:
                # LocationContext 구성
                location_ctx = None
                if file_path:
                    filters = {}
                    if language:
                        filters["language"] = language

                    location_ctx = LocationContext(
                        repo_id=repo_id,
                        file_path=file_path,
                        line=0,  # 현재 라인 정보 없으면 0
                        filters=filters if filters else None,
                    )

                # 하이브리드 검색
                candidates = self.retriever.retrieve(
                    repo_id=repo_id, query=query, k=limit, location_ctx=location_ctx
                )

                # Candidate → 결과 딕셔너리 변환
                for candidate in candidates:
                    # ChunkStore에서 실제 텍스트 가져오기
                    try:
                        chunk = self.chunk_store.get_chunk(repo_id, candidate.chunk_id)
                        text = chunk.text if chunk else ""
                    except Exception as e:
                        logger.debug(f"Failed to get chunk text: {e}")
                        text = ""

                    result = {
                        "repo_id": candidate.repo_id,
                        "chunk_id": candidate.chunk_id,
                        "file_path": candidate.file_path,
                        "line_start": candidate.span[0],
                        "line_end": candidate.span[2],
                        "text": text,
                        "score": candidate.features.get("total_score", 0.0),
                        "scores": {
                            "lexical": candidate.features.get("lexical_score", 0.0),
                            "semantic": candidate.features.get("semantic_score", 0.0),
                            "graph": candidate.features.get("graph_score", 0.0),
                            "fuzzy": candidate.features.get("fuzzy_score", 0.0),
                        },
                    }
                    all_results.append(result)

            except Exception as e:
                logger.error(f"Search failed for repo {repo_id}: {e}")

        # 전체 결과 점수순 정렬 후 상위 limit개
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:limit]

    async def find_definition(
        self,
        symbol_name: str,
        repo_ids: list[RepoId] | None = None,
        kind: str | None = None,
    ) -> list[dict]:
        """
        심볼 정의 찾기

        Args:
            symbol_name: 심볼 이름
            repo_ids: 검색할 저장소 ID 목록
            kind: 심볼 종류 필터 ("Function", "Class", "Method" 등)

        Returns:
            정의 위치 리스트
            [
                {
                    "repo_id": str,
                    "node_id": str,
                    "kind": str,
                    "name": str,
                    "file_path": str,
                    "line_start": int,
                    "line_end": int,
                    "text": str
                }
            ]
        """
        logger.info(f"find_definition: symbol={symbol_name}, repos={repo_ids}")

        if not repo_ids:
            repos = self.repo_store.list_repositories()
            repo_ids = [repo.repo_id for repo in repos]

        results = []
        for repo_id in repo_ids:
            try:
                # GraphStore에서 노드 검색
                nodes = self.graph_store.find_nodes_by_name(
                    repo_id=repo_id, name=symbol_name, kind=kind
                )

                for node in nodes:
                    results.append(
                        {
                            "repo_id": node.repo_id,
                            "node_id": node.id,
                            "kind": node.kind,
                            "name": node.name,
                            "file_path": node.file_path,
                            "line_start": node.span[0],
                            "line_end": node.span[2],
                            "text": node.text,
                        }
                    )

            except Exception as e:
                logger.error(f"find_definition failed for repo {repo_id}: {e}")

        return results

    async def explore_graph(
        self,
        repo_id: RepoId,
        node_id: str,
        direction: str = "both",
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> dict:
        """
        그래프 탐색 (관계 확장)

        Args:
            repo_id: 저장소 ID
            node_id: 시작 노드 ID
            direction: 방향 ("outgoing", "incoming", "both")
            edge_types: 엣지 타입 필터 (["calls", "defines"] 등)
            depth: 탐색 깊이

        Returns:
            {
                "center": {노드 정보},
                "neighbors": [
                    {
                        "node": {노드 정보},
                        "edge_type": str,
                        "direction": str
                    }
                ]
            }
        """
        logger.info(f"explore_graph: repo={repo_id}, node={node_id}, depth={depth}")

        try:
            # 중심 노드
            center_node = self.graph_store.get_node(repo_id, node_id)
            if not center_node:
                return {"error": f"Node not found: {node_id}"}

            # 이웃 노드 확장
            neighbors = self.graph_search.expand_neighbors(
                repo_id=repo_id, node_id=node_id, k=depth
            )

            neighbor_results = []
            for neighbor in neighbors:
                # 엣지 정보 가져오기
                edges = []
                if direction in ("outgoing", "both"):
                    edges.extend(
                        self.graph_store.get_edges(repo_id, src_id=node_id, dst_id=neighbor.id)
                    )
                if direction in ("incoming", "both"):
                    edges.extend(
                        self.graph_store.get_edges(repo_id, src_id=neighbor.id, dst_id=node_id)
                    )

                # edge_types 필터링
                if edge_types:
                    edges = [e for e in edges if e.type in edge_types]

                for edge in edges:
                    neighbor_results.append(
                        {
                            "node": {
                                "node_id": neighbor.id,
                                "kind": neighbor.kind,
                                "name": neighbor.name,
                                "file_path": neighbor.file_path,
                                "line_start": neighbor.span[0],
                                "line_end": neighbor.span[2],
                            },
                            "edge_type": edge.type,
                            "direction": "outgoing" if edge.src_id == node_id else "incoming",
                        }
                    )

            return {
                "center": {
                    "node_id": center_node.id,
                    "kind": center_node.kind,
                    "name": center_node.name,
                    "file_path": center_node.file_path,
                    "line_start": center_node.span[0],
                    "line_end": center_node.span[2],
                    "text": center_node.text,
                },
                "neighbors": neighbor_results,
            }

        except Exception as e:
            logger.error(f"explore_graph failed: {e}")
            return {"error": str(e)}

    async def get_context(
        self,
        repo_id: RepoId,
        file_path: str,
        line: int,
        max_tokens: int = 4000,
    ) -> dict:
        """
        특정 위치의 관련 컨텍스트 가져오기

        Args:
            repo_id: 저장소 ID
            file_path: 파일 경로
            line: 라인 번호
            max_tokens: 최대 토큰 수

        Returns:
            {
                "primary": {
                    "file_path": str,
                    "line_start": int,
                    "line_end": int,
                    "text": str,
                    "role": "primary"
                },
                "supporting": [
                    {
                        "file_path": str,
                        "line_start": int,
                        "line_end": int,
                        "text": str,
                        "role": str  # "caller", "callee", "type" 등
                    }
                ]
            }
        """
        logger.info(f"get_context: repo={repo_id}, file={file_path}, line={line}")

        try:
            # 현재 위치의 노드 찾기
            node = self.graph_search.get_node_by_location(
                repo_id=repo_id, file_path=file_path, line=line, column=0
            )

            if not node:
                return {"error": "No node found at location"}

            # 노드에 해당하는 청크 찾기
            chunks = self.chunk_store.get_chunks_by_node(repo_id, node.id)
            if not chunks:
                return {"error": "No chunks found for node"}

            primary_chunk = chunks[0]

            # ContextPacker로 패킹
            packed = self.context_packer.pack(
                repo_id=repo_id, primary_chunk_id=primary_chunk.id, max_tokens=max_tokens
            )

            return {
                "primary": {
                    "file_path": packed.primary.file_path,
                    "line_start": packed.primary.span[0],
                    "line_end": packed.primary.span[2],
                    "text": packed.primary.text,
                    "role": packed.primary.role,
                },
                "supporting": [
                    {
                        "file_path": snippet.file_path,
                        "line_start": snippet.span[0],
                        "line_end": snippet.span[2],
                        "text": snippet.text,
                        "role": snippet.role,
                    }
                    for snippet in packed.supporting
                ],
            }

        except Exception as e:
            logger.error(f"get_context failed: {e}")
            return {"error": str(e)}

    async def get_node(self, repo_id: RepoId, node_id: str) -> dict | None:
        """
        노드 정보 조회 (Resource 용)

        Args:
            repo_id: 저장소 ID
            node_id: 노드 ID

        Returns:
            노드 정보 또는 None
        """
        try:
            node = self.graph_store.get_node(repo_id, node_id)
            if not node:
                return None

            return {
                "node_id": node.id,
                "kind": node.kind,
                "name": node.name,
                "language": node.language,
                "file_path": node.file_path,
                "line_start": node.span[0],
                "line_end": node.span[2],
                "text": node.text,
                "attrs": node.attrs,
            }
        except Exception as e:
            logger.error(f"get_node failed: {e}")
            return None

    async def get_file_chunks(self, repo_id: RepoId, file_path: str) -> list[dict]:
        """
        파일의 모든 청크 조회 (Resource 용)

        Args:
            repo_id: 저장소 ID
            file_path: 파일 경로

        Returns:
            청크 리스트
        """
        try:
            chunks = self.chunk_store.get_chunks_by_file(repo_id, file_path)

            return [
                {
                    "chunk_id": chunk.id,
                    "node_id": chunk.node_id,
                    "file_path": chunk.file_path,
                    "line_start": chunk.span[0],
                    "line_end": chunk.span[2],
                    "text": chunk.text,
                    "language": chunk.language,
                }
                for chunk in chunks
            ]
        except Exception as e:
            logger.error(f"get_file_chunks failed: {e}")
            return []

    async def list_repositories(self) -> list[dict]:
        """
        인덱싱된 저장소 목록 조회

        Returns:
            저장소 리스트
            [
                {
                    "repo_id": str,
                    "name": str,
                    "root_path": str,
                    "languages": list[str],
                    "total_files": int,
                    "total_nodes": int,
                    "total_edges": int,
                    "indexing_status": str,
                    "indexed_at": str | None
                }
            ]
        """
        logger.info("list_repositories")

        try:
            repos = self.repo_store.list_repositories()

            return [
                {
                    "repo_id": repo.repo_id,
                    "name": repo.name,
                    "root_path": repo.root_path,
                    "languages": repo.languages or [],
                    "total_files": repo.total_files,
                    "total_nodes": repo.total_nodes,
                    "total_edges": repo.total_edges,
                    "indexing_status": repo.indexing_status,
                    "indexed_at": repo.indexed_at.isoformat() if repo.indexed_at else None,
                }
                for repo in repos
            ]
        except Exception as e:
            logger.error(f"list_repositories failed: {e}")
            return []
