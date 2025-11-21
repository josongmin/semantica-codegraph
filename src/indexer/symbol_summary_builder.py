"""Symbol 요약 생성 (템플릿 기반)"""

import logging

from ..core.models import CodeNode, FileProfile
from ..core.ports import GraphStorePort

logger = logging.getLogger(__name__)


class SymbolSummaryBuilder:
    """
    Symbol 노드 요약 생성 (템플릿 기반)

    Phase 1: 템플릿 기반 메타텍스트
    Phase 2: LLM 기반 요약 (중요 심볼만)
    """

    def __init__(self, graph_store: GraphStorePort | None = None):
        """
        Args:
            graph_store: 그래프 스토어 (호출 관계 조회용, optional)
        """
        self.graph_store = graph_store

    def build(self, node: CodeNode) -> str:
        """
        Symbol 메타텍스트 생성

        Example:
            Function hybrid_search in routes/hybrid.py
            Purpose: Combines semantic and lexical search
            Signature: def hybrid_search(query: str, k: int = 10) -> List[Result]
            Calls: meilisearch.search, pgvector.similarity_search

        Args:
            node: CodeNode

        Returns:
            요약 텍스트
        """
        parts = []

        # 1. 기본 정보
        parts.append(f"{node.kind} {node.name} in {node.file_path}")

        # 2. Docstring (있으면)
        docstring = node.attrs.get("docstring")
        if docstring:
            # 첫 줄만 또는 200자까지
            first_line = docstring.split("\n")[0].strip()
            if len(first_line) > 200:
                first_line = first_line[:200] + "..."
            parts.append(f"Purpose: {first_line}")

        # 3. 시그니처 (있으면)
        signature = node.attrs.get("signature")
        if signature:
            parts.append(f"Signature: {signature}")

        # 4. 호출 관계 (graph_store 있을 때만)
        if self.graph_store:
            try:
                # Outgoing edges (calls)
                # Note: GraphStorePort에는 get_outgoing_edges가 없으므로 일단 스킵
                # Phase 2에서 추가 또는 neighbors 활용
                pass
            except Exception as e:
                logger.debug(f"Failed to get call graph for {node.id}: {e}")

        return "\n".join(parts)


def calculate_importance(
    node: CodeNode,
    file_profile: FileProfile | None = None,
) -> float:
    """
    Symbol 중요도 계산 (0~1)

    Phase 1: 간단한 heuristic
    Phase 2: PageRank + ML

    Args:
        node: CodeNode
        file_profile: FileProfile (optional)

    Returns:
        중요도 점수 (0~1)
    """
    score = 0.0

    # 1. Public (기본 30%)
    if not node.name.startswith("_"):
        score += 0.3

    # 2. API 핸들러 (40%)
    if node.attrs.get("is_api_handler"):
        score += 0.4

    # 3. Docstring 있음 (10%)
    if node.attrs.get("docstring"):
        score += 0.1

    # 4. 호출 관계 (최대 20%)
    # in_degree가 있으면 (나중에 graph ranking에서 계산)
    in_degree = node.attrs.get("in_degree", 0)
    if in_degree > 10:
        score += 0.2
    elif in_degree > 5:
        score += 0.15
    elif in_degree > 2:
        score += 0.1

    # 5. 파일 역할 (API/Router 파일) (10%)
    if file_profile:
        if file_profile.is_api_file or file_profile.is_router:
            score += 0.1
    else:
        # file_profile 없으면 attrs에서 추측
        if node.attrs.get("file_is_api") or node.attrs.get("file_is_router"):
            score += 0.1

    return min(score, 1.0)
