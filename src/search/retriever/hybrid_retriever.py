"""하이브리드 리트리버 (Lexical + Semantic + Graph + Fuzzy)"""

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from ...core.config import Config
from ...core.models import Candidate, ChunkResult, LocationContext, RepoId
from ...core.ports import ChunkStorePort
from ..ports.fuzzy_search_port import FuzzySearchPort
from ..ports.graph_search_port import GraphSearchPort
from ..ports.lexical_search_port import LexicalSearchPort
from ..ports.semantic_search_port import SemanticSearchPort

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    하이브리드 리트리버
    
    4가지 검색 방식을 병렬로 실행:
    1. Lexical (키워드 기반 - BM25)
    2. Semantic (임베딩 기반 - 벡터 유사도)
    3. Graph (그래프 탐색 - 관계 기반)
    4. Fuzzy (퍼지 매칭 - 심볼명 유사도)
    
    각 검색 결과를 Candidate로 변환하여 병합
    """

    def __init__(
        self,
        lexical_search: LexicalSearchPort,
        semantic_search: SemanticSearchPort,
        graph_search: GraphSearchPort,
        fuzzy_search: Optional[FuzzySearchPort] = None,
        chunk_store: Optional[ChunkStorePort] = None,
        config: Optional[Config] = None,
    ):
        """
        Args:
            lexical_search: Lexical 검색 포트
            semantic_search: Semantic 검색 포트
            graph_search: Graph 검색 포트
            fuzzy_search: Fuzzy 검색 포트 (옵션)
            chunk_store: Chunk 저장소 (옵션, fuzzy 매핑용)
            config: 설정 (옵션, 없으면 기본값)
        """
        self.lexical_search = lexical_search
        self.semantic_search = semantic_search
        self.graph_search = graph_search
        self.fuzzy_search = fuzzy_search
        self.chunk_store = chunk_store
        self.config = config or Config()

    def retrieve(
        self,
        repo_id: RepoId,
        query: str,
        k: int = 20,
        location_ctx: Optional[LocationContext] = None,
        weights: Optional[Dict[str, float]] = None,
        parallel: bool = True,
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
            parallel: 병렬 검색 활성화 (기본값: True)
        
        Returns:
            Candidate 리스트 (중복 제거, 점수 통합)
        """
        if weights is None:
            weights = {
                "lexical": 0.25,
                "semantic": 0.45,
                "graph": 0.15,
                "fuzzy": 0.15
            }

        logger.info(f"Hybrid retrieval: {query} (k={k}, parallel={parallel})")
        
        if parallel and self.config.parallel_search_enabled:
            # 병렬 검색
            return self._retrieve_parallel(repo_id, query, k, location_ctx, weights)
        else:
            # 순차 검색 (기존 방식)
            return self._retrieve_sequential(repo_id, query, k, location_ctx, weights)

    def _retrieve_sequential(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        location_ctx: Optional[LocationContext],
        weights: Dict[str, float],
    ) -> List[Candidate]:
        """순차 검색 (기존 방식)"""
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
        
        # 4. Fuzzy 검색 (심볼명 퍼지 매칭)
        if weights.get("fuzzy", 0) > 0 and self.fuzzy_search:
            try:
                # 쿼리에서 심볼명 추출 (CamelCase, snake_case 분리)
                query_tokens = self._extract_symbol_tokens(query)
                
                for token in query_tokens:
                    # Config에서 최소 길이 읽기
                    if len(token) < self.config.fuzzy_min_token_length:
                        continue
                    
                    fuzzy_matches = self.fuzzy_search.search_symbols(
                        repo_id=repo_id,
                        query=token,
                        threshold=self.config.fuzzy_threshold,
                        k=self.config.fuzzy_results_per_token,
                    )
                    
                    for i, match in enumerate(fuzzy_matches):
                        # 순위 기반 점수 (첫 번째가 가장 높음)
                        rank_score = 1.0 / (i + 1)
                        fuzzy_score = match.score * rank_score * weights["fuzzy"]
                        
                        # node_id → chunk_id 매핑
                        if match.node_id and self.chunk_store:
                            # ChunkStore에서 실제 chunk 조회
                            try:
                                chunks = self.chunk_store.get_chunks_by_node(
                                    repo_id=repo_id,
                                    node_id=match.node_id
                                )
                                
                                if chunks:
                                    # 첫 번째 chunk 사용
                                    chunk = chunks[0]
                                    chunk_id = chunk.id
                                    file_path = chunk.file_path
                                    span = chunk.span
                                else:
                                    # chunk 없으면 임시 ID
                                    chunk_id = f"chunk-{match.node_id}"
                                    file_path = match.file_path or ""
                                    span = (0, 0, 0, 0)
                            except Exception as e:
                                logger.debug(f"Failed to get chunk for node {match.node_id}: {e}")
                                chunk_id = f"chunk-{match.node_id}"
                                file_path = match.file_path or ""
                                span = (0, 0, 0, 0)
                        else:
                            # ChunkStore 없으면 임시 ID
                            chunk_id = f"chunk-{match.node_id}" if match.node_id else f"chunk-fuzzy-{match.matched_text}"
                            file_path = match.file_path or ""
                            span = (0, 0, 0, 0)
                        
                        if chunk_id in candidates:
                            candidates[chunk_id].features["fuzzy_score"] = fuzzy_score
                        else:
                            # 새로운 candidate 추가
                            candidates[chunk_id] = Candidate(
                                repo_id=repo_id,
                                chunk_id=chunk_id,
                                features={"fuzzy_score": fuzzy_score},
                                file_path=file_path,
                                span=span
                            )
                    
                    if fuzzy_matches:
                        logger.debug(f"Fuzzy: '{token}' → {len(fuzzy_matches)} matches")
                        
            except Exception as e:
                logger.error(f"Fuzzy search failed: {e}")
        
        # 5. Candidate 리스트로 변환
        candidate_list = list(candidates.values())
        
        # 6. 총 점수 계산
        for candidate in candidate_list:
            total_score = sum(candidate.features.values())
            candidate.features["total_score"] = total_score
        
        # 7. 점수 기준 정렬
        candidate_list.sort(key=lambda c: c.features.get("total_score", 0), reverse=True)
        
        # 8. 상위 k개 반환
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

    def _extract_symbol_tokens(self, query: str) -> List[str]:
        """
        쿼리에서 심볼 토큰 추출
        
        - CamelCase 분리: UserService → User, Service
        - snake_case 분리: user_service → user, service
        - 공백 분리
        - 최소 길이 필터링 & 중복 제거
        
        Args:
            query: 검색 쿼리
        
        Returns:
            토큰 리스트
        """
        tokens = []
        
        # 1. 공백으로 분리
        words = query.split()
        
        for word in words:
            # 2. CamelCase 분리 (UserService → User, Service)
            # 대문자로 시작하는 단어들
            camel_tokens = re.findall(
                r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))',
                word
            )
            if camel_tokens:
                tokens.extend(camel_tokens)
            
            # 3. snake_case 분리 (user_service → user, service)
            snake_tokens = re.split(r'[_\-]', word)
            tokens.extend(snake_tokens)
            
            # 4. 원본 단어도 포함
            tokens.append(word)
        
        # 5. 최소 길이 필터 & 중복 제거
        min_length = self.config.fuzzy_min_token_length
        tokens = [t for t in tokens if len(t) >= min_length]
        tokens = list(dict.fromkeys(tokens))  # 순서 유지하면서 중복 제거
        
        return tokens

    def _retrieve_parallel(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        location_ctx: Optional[LocationContext],
        weights: Dict[str, float],
    ) -> List[Candidate]:
        """병렬 검색 (ThreadPoolExecutor 사용)"""
        candidates = {}  # chunk_id -> Candidate
        
        # ThreadPoolExecutor로 병렬 실행
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            # 1. Lexical 검색
            if weights.get("lexical", 0) > 0:
                future = executor.submit(
                    self._search_lexical,
                    repo_id, query, k, location_ctx, weights["lexical"]
                )
                futures.append(("lexical", future))
            
            # 2. Semantic 검색
            if weights.get("semantic", 0) > 0:
                future = executor.submit(
                    self._search_semantic,
                    repo_id, query, k, location_ctx, weights["semantic"]
                )
                futures.append(("semantic", future))
            
            # 3. Graph 검색
            if weights.get("graph", 0) > 0 and location_ctx:
                future = executor.submit(
                    self._search_graph,
                    repo_id, k, location_ctx, weights["graph"]
                )
                futures.append(("graph", future))
            
            # 4. Fuzzy 검색
            if weights.get("fuzzy", 0) > 0 and self.fuzzy_search:
                future = executor.submit(
                    self._search_fuzzy,
                    repo_id, query, k, weights["fuzzy"]
                )
                futures.append(("fuzzy", future))
            
            # 결과 수집
            for search_type, future in futures:
                try:
                    results = future.result()
                    for candidate in results:
                        chunk_id = candidate.chunk_id
                        if chunk_id in candidates:
                            # 기존 candidate에 점수 추가
                            for key, value in candidate.features.items():
                                if key not in candidates[chunk_id].features:
                                    candidates[chunk_id].features[key] = value
                        else:
                            candidates[chunk_id] = candidate
                    
                    logger.debug(f"{search_type}: {len(results)} results")
                except Exception as e:
                    logger.error(f"{search_type} search failed: {e}")
        
        # 총 점수 계산 및 정렬
        candidate_list = list(candidates.values())
        for candidate in candidate_list:
            total_score = sum(candidate.features.values())
            candidate.features["total_score"] = total_score
        
        candidate_list.sort(key=lambda c: c.features.get("total_score", 0), reverse=True)
        result = candidate_list[:k]
        
        logger.info(f"Retrieved {len(result)} candidates (parallel)")
        return result

    def _search_lexical(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        location_ctx: Optional[LocationContext],
        weight: float,
    ) -> List[Candidate]:
        """Lexical 검색"""
        try:
            lexical_k = int(k * 0.5)
            lexical_results = self.lexical_search.search(
                repo_id=repo_id,
                query=query,
                k=lexical_k,
                filters=location_ctx.filters if location_ctx else None
            )
            
            return [
                self._result_to_candidate(result, lexical_score=result.score * weight)
                for result in lexical_results
            ]
        except Exception as e:
            logger.error(f"Lexical search failed: {e}")
            return []

    def _search_semantic(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        location_ctx: Optional[LocationContext],
        weight: float,
    ) -> List[Candidate]:
        """Semantic 검색"""
        try:
            semantic_k = int(k * 0.7)
            semantic_results = self.semantic_search.search(
                repo_id=repo_id,
                query=query,
                k=semantic_k,
                filters=location_ctx.filters if location_ctx else None
            )
            
            return [
                self._result_to_candidate(result, semantic_score=result.score * weight)
                for result in semantic_results
            ]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _search_graph(
        self,
        repo_id: RepoId,
        k: int,
        location_ctx: LocationContext,
        weight: float,
    ) -> List[Candidate]:
        """Graph 검색"""
        try:
            current_node = self.graph_search.get_node_by_location(
                repo_id=repo_id,
                file_path=location_ctx.file_path,
                line=location_ctx.line,
                column=location_ctx.column
            )
            
            if not current_node:
                return []
            
            neighbors = self.graph_search.expand_neighbors(
                repo_id=repo_id,
                node_id=current_node.id,
                k=2
            )
            
            candidates = []
            for i, neighbor in enumerate(neighbors[:k]):
                distance_score = 1.0 / (i + 1)
                graph_score = distance_score * weight
                chunk_id = f"chunk-{neighbor.id}"
                
                candidates.append(Candidate(
                    repo_id=repo_id,
                    chunk_id=chunk_id,
                    features={"graph_score": graph_score},
                    file_path=neighbor.file_path,
                    span=neighbor.span
                ))
            
            return candidates
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    def _search_fuzzy(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        weight: float,
    ) -> List[Candidate]:
        """Fuzzy 검색"""
        try:
            query_tokens = self._extract_symbol_tokens(query)
            candidates = []
            
            for token in query_tokens:
                if len(token) < self.config.fuzzy_min_token_length:
                    continue
                
                fuzzy_matches = self.fuzzy_search.search_symbols(
                    repo_id=repo_id,
                    query=token,
                    threshold=self.config.fuzzy_threshold,
                    k=self.config.fuzzy_results_per_token,
                )
                
                for i, match in enumerate(fuzzy_matches):
                    rank_score = 1.0 / (i + 1)
                    fuzzy_score = match.score * rank_score * weight
                    
                    # node_id → chunk_id 매핑
                    if match.node_id and self.chunk_store:
                        try:
                            chunks = self.chunk_store.get_chunks_by_node(
                                repo_id=repo_id,
                                node_id=match.node_id
                            )
                            
                            if chunks:
                                chunk = chunks[0]
                                chunk_id = chunk.id
                                file_path = chunk.file_path
                                span = chunk.span
                            else:
                                chunk_id = f"chunk-{match.node_id}"
                                file_path = match.file_path or ""
                                span = (0, 0, 0, 0)
                        except Exception:
                            chunk_id = f"chunk-{match.node_id}"
                            file_path = match.file_path or ""
                            span = (0, 0, 0, 0)
                    else:
                        chunk_id = f"chunk-{match.node_id}" if match.node_id else f"chunk-fuzzy-{match.matched_text}"
                        file_path = match.file_path or ""
                        span = (0, 0, 0, 0)
                    
                    candidates.append(Candidate(
                        repo_id=repo_id,
                        chunk_id=chunk_id,
                        features={"fuzzy_score": fuzzy_score},
                        file_path=file_path,
                        span=span
                    ))
            
            return candidates
        except Exception as e:
            logger.error(f"Fuzzy search failed: {e}")
            return []

