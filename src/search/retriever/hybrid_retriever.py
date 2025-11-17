"""하이브리드 리트리버 (Lexical + Semantic + Graph)"""

import logging
from typing import Dict, List, Optional

from ...core.models import Candidate, ChunkResult, LocationContext, RepoId
from ..ports.graph_search_port import GraphSearchPort
from ..ports.lexical_search_port import LexicalSearchPort
from ..ports.semantic_search_port import SemanticSearchPort

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    하이브리드 리트리버
    
    3가지 검색 방식을 병렬로 실행:
    1. Lexical (키워드 기반 - BM25)
    2. Semantic (임베딩 기반 - 벡터 유사도)
    3. Graph (그래프 탐색 - 관계 기반)
    
    각 검색 결과를 Candidate로 변환하여 병합
    """

    def __init__(
        self,
        lexical_search: LexicalSearchPort,
        semantic_search: SemanticSearchPort,
        graph_search: GraphSearchPort,
    ):
        """
        Args:
            lexical_search: Lexical 검색 포트
            semantic_search: Semantic 검색 포트
            graph_search: Graph 검색 포트
        """
        self.lexical_search = lexical_search
        self.semantic_search = semantic_search
        self.graph_search = graph_search

    def retrieve(
        self,
        repo_id: RepoId,
        query: str,
        k: int = 20,
        location_ctx: Optional[LocationContext] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Candidate]:
        """
        하이브리드 검색 실행
        
        Args:
            repo_id: 저장소 ID
            query: 검색 쿼리
            k: 반환할 결과 수
            location_ctx: 위치 컨텍스트 (현재 파일/라인 정보)
            weights: 각 검색 방식의 가중치
                예: {"lexical": 0.3, "semantic": 0.5, "graph": 0.2}
        
        Returns:
            Candidate 리스트 (중복 제거, 점수 통합)
        """
        if weights is None:
            weights = {
                "lexical": 0.3,
                "semantic": 0.5,
                "graph": 0.2
            }

        logger.info(f"Hybrid retrieval: {query} (k={k})")
        
        candidates = {}  # chunk_id -> Candidate
        
        # 1. Lexical 검색
        if weights.get("lexical", 0) > 0:
            try:
                lexical_k = int(k * 0.5)  # Lexical에서 k/2개
                lexical_results = self.lexical_search.search(
                    repo_id=repo_id,
                    query=query,
                    k=lexical_k,
                    filters=location_ctx.filters if location_ctx else None
                )
                
                for result in lexical_results:
                    candidates[result.chunk_id] = self._result_to_candidate(
                        result,
                        lexical_score=result.score * weights["lexical"]
                    )
                
                logger.debug(f"Lexical: {len(lexical_results)} results")
            except Exception as e:
                logger.error(f"Lexical search failed: {e}")
        
        # 2. Semantic 검색
        if weights.get("semantic", 0) > 0:
            try:
                semantic_k = int(k * 0.7)  # Semantic에서 k*0.7개
                semantic_results = self.semantic_search.search(
                    repo_id=repo_id,
                    query=query,
                    k=semantic_k,
                    filters=location_ctx.filters if location_ctx else None
                )
                
                for result in semantic_results:
                    if result.chunk_id in candidates:
                        # 이미 있으면 semantic_score 추가
                        candidates[result.chunk_id].features["semantic_score"] = \
                            result.score * weights["semantic"]
                    else:
                        candidates[result.chunk_id] = self._result_to_candidate(
                            result,
                            semantic_score=result.score * weights["semantic"]
                        )
                
                logger.debug(f"Semantic: {len(semantic_results)} results")
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
        
        # 3. Graph 검색 (위치 컨텍스트가 있을 때만)
        if weights.get("graph", 0) > 0 and location_ctx:
            try:
                # 현재 위치의 노드 찾기
                current_node = self.graph_search.get_node_by_location(
                    repo_id=repo_id,
                    file_path=location_ctx.file_path,
                    line=location_ctx.line,
                    column=location_ctx.column
                )
                
                if current_node:
                    # 이웃 노드 확장
                    neighbors = self.graph_search.expand_neighbors(
                        repo_id=repo_id,
                        node_id=current_node.id,
                        k=2  # 2-hop 이웃
                    )
                    
                    # 노드를 Candidate로 변환 (chunk와 1:1 매핑 가정)
                    for i, neighbor in enumerate(neighbors[:k]):
                        # 거리 기반 점수 (가까울수록 높음)
                        distance_score = 1.0 / (i + 1)
                        graph_score = distance_score * weights["graph"]
                        
                        # chunk_id는 node_id와 동일하다고 가정
                        # (실제로는 node -> chunk 매핑 필요)
                        chunk_id = f"chunk-{neighbor.id}"
                        
                        if chunk_id in candidates:
                            candidates[chunk_id].features["graph_score"] = graph_score
                        else:
                            candidates[chunk_id] = Candidate(
                                repo_id=repo_id,
                                chunk_id=chunk_id,
                                features={"graph_score": graph_score},
                                file_path=neighbor.file_path,
                                span=neighbor.span
                            )
                    
                    logger.debug(f"Graph: {len(neighbors)} neighbors")
            except Exception as e:
                logger.error(f"Graph search failed: {e}")
        
        # 4. Candidate 리스트로 변환
        candidate_list = list(candidates.values())
        
        # 5. 총 점수 계산
        for candidate in candidate_list:
            total_score = sum(candidate.features.values())
            candidate.features["total_score"] = total_score
        
        # 6. 점수 기준 정렬
        candidate_list.sort(key=lambda c: c.features.get("total_score", 0), reverse=True)
        
        # 7. 상위 k개 반환
        result = candidate_list[:k]
        logger.info(f"Retrieved {len(result)} candidates")
        
        return result

    def _result_to_candidate(
        self,
        result: ChunkResult,
        lexical_score: float = 0.0,
        semantic_score: float = 0.0,
        graph_score: float = 0.0
    ) -> Candidate:
        """ChunkResult를 Candidate로 변환"""
        features = {}
        if lexical_score > 0:
            features["lexical_score"] = lexical_score
        if semantic_score > 0:
            features["semantic_score"] = semantic_score
        if graph_score > 0:
            features["graph_score"] = graph_score
        
        return Candidate(
            repo_id=result.repo_id,
            chunk_id=result.chunk_id,
            features=features,
            file_path=result.file_path,
            span=result.span
        )

