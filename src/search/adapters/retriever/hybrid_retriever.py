"""하이브리드 리트리버 (Lexical + Semantic + Graph + Fuzzy)"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from src.core.config import Config
from src.core.models import Candidate, ChunkResult, LocationContext, RepoId
from src.core.ports import ChunkStorePort
from src.search.adapters.fusion import FusionStrategy, WeightedFusion
from src.search.ports.fuzzy_search_port import FuzzySearchPort
from src.search.ports.graph_search_port import GraphSearchPort
from src.search.ports.lexical_search_port import LexicalSearchPort
from src.search.ports.semantic_search_port import SemanticSearchPort

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
        fuzzy_search: FuzzySearchPort | None = None,
        chunk_store: ChunkStorePort | None = None,
        config: Config | None = None,
        fusion_strategy: FusionStrategy | None = None,
        query_log_store=None,  # QueryLogStore (Phase 2)
    ):
        """
        Args:
            lexical_search: Lexical 검색 포트
            semantic_search: Semantic 검색 포트
            graph_search: Graph 검색 포트
            fuzzy_search: Fuzzy 검색 포트 (옵션)
            chunk_store: Chunk 저장소 (옵션, fuzzy 매핑용)
            config: 설정 (옵션, 없으면 기본값)
            fusion_strategy: Fusion 전략 (옵션, 없으면 WeightedFusion)
        """
        self.lexical_search = lexical_search
        self.semantic_search = semantic_search
        self.graph_search = graph_search
        self.fuzzy_search = fuzzy_search
        self.chunk_store = chunk_store
        self.config = config or Config()
        self.fusion = fusion_strategy or WeightedFusion()
        self.query_log_store = query_log_store  # Phase 2: Query logging

        # Debug/logging 설정
        self.enable_query_logging = True  # Query logs 활성화
        self.log_query_embedding = False  # 쿼리 임베딩 저장 (용량 큼)

    def retrieve(
        self,
        repo_id: RepoId,
        query: str,
        k: int = 20,
        location_ctx: LocationContext | None = None,
        weights: dict[str, float] | None = None,
        parallel: bool = True,
        query_type: str | None = None,  # Phase 2: QueryClassifier 출력
    ) -> list[Candidate]:
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
            query_type: 쿼리 타입 (Phase 2 로깅용)

        Returns:
            Candidate 리스트 (중복 제거, 점수 통합)
        """
        import time

        if weights is None:
            weights = {"lexical": 0.25, "semantic": 0.45, "graph": 0.15, "fuzzy": 0.15}

        logger.info(f"Hybrid retrieval: {query} (k={k}, parallel={parallel})")

        # Phase 2: 로깅을 위한 시작 시간
        start_time = time.time()

        if parallel and self.config.parallel_search_enabled:
            # 병렬 검색
            results = self._retrieve_parallel(repo_id, query, k, location_ctx, weights)
        else:
            # 순차 검색 (기존 방식)
            results = self._retrieve_sequential(repo_id, query, k, location_ctx, weights)

        # Phase 2: Query logging
        if self.enable_query_logging and self.query_log_store:
            try:
                latency_ms = int((time.time() - start_time) * 1000)
                self._log_query(
                    repo_id=repo_id,
                    query=query,
                    query_type=query_type,
                    k=k,
                    weights=weights,
                    results=results,
                    latency_ms=latency_ms,
                )
            except Exception as e:
                logger.warning(f"Query logging failed: {e}")

        return results

    def _retrieve_sequential(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        location_ctx: LocationContext | None,
        weights: dict[str, float],
    ) -> list[Candidate]:
        """순차 검색 (기존 방식)"""
        candidates: dict[str, Candidate] = {}  # chunk_id -> Candidate

        # 1. Lexical 검색
        lexical_results_raw = []
        if weights.get("lexical", 0) > 0:
            try:
                lexical_k = int(k * self.config.lexical_fetch_multiplier)
                lexical_results_raw = self.lexical_search.search(
                    repo_id=repo_id,
                    query=query,
                    k=lexical_k,
                    filters=location_ctx.filters if location_ctx else None,
                )
                logger.debug(f"Lexical: {len(lexical_results_raw)} results")
            except Exception as e:
                logger.error(f"Lexical search failed: {e}")

        # 2. Semantic 검색
        semantic_results_raw = []
        if weights.get("semantic", 0) > 0:
            try:
                semantic_k = int(k * self.config.semantic_fetch_multiplier)
                semantic_results_raw = self.semantic_search.search(
                    repo_id=repo_id,
                    query=query,
                    k=semantic_k,
                    filters=location_ctx.filters if location_ctx else None,
                )
                logger.debug(f"Semantic: {len(semantic_results_raw)} results")
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
                    column=location_ctx.column,
                )

                if current_node:
                    # 이웃 노드를 edge 정보와 함께 확장
                    neighbors_with_edges = self.graph_search.expand_neighbors_with_edges(
                        repo_id=repo_id,
                        node_id=current_node.id,
                        k=2,  # 2-hop 이웃
                    )

                    # 노드를 Candidate로 변환
                    graph_neighbors = neighbors_with_edges[
                        : int(k * self.config.graph_fetch_multiplier)
                    ]
                    for neighbor_node, edge_type, depth in graph_neighbors:
                        # Edge 타입별 가중치 적용
                        edge_weights = self.config.graph_edge_weights or {}
                        edge_weight = edge_weights.get(edge_type, 0.5)

                        # Depth decay 적용 (깊이가 깊어질수록 점수 감소)
                        depth_decay = self.config.graph_depth_decay ** (depth - 1)

                        # 최종 graph score = edge_weight * depth_decay
                        graph_score = edge_weight * depth_decay
                        normalized_score = self._normalize_graph_score(graph_score)

                        # chunk_id는 node_id와 동일하다고 가정
                        # (실제로는 node -> chunk 매핑 필요)
                        chunk_id = f"chunk-{neighbor_node.id}"

                        if chunk_id in candidates:
                            # 기존 점수보다 높으면 업데이트 (여러 경로로 도달 가능)
                            existing_score = candidates[chunk_id].features.get("graph_score", 0)
                            candidates[chunk_id].features["graph_score"] = max(
                                existing_score, normalized_score
                            )
                        else:
                            candidates[chunk_id] = Candidate(
                                repo_id=repo_id,
                                chunk_id=chunk_id,
                                features={"graph_score": normalized_score},
                                file_path=neighbor_node.file_path,
                                span=neighbor_node.span,
                            )

                    logger.debug(f"Graph: {len(neighbors_with_edges)} neighbors with edges")
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
                        normalized_match_score = self._normalize_fuzzy_score(match.score)
                        base_fuzzy_score = normalized_match_score * rank_score

                        # node_id → chunk_id 매핑 (개선: 모든 청크 반환)
                        if match.node_id and self.chunk_store:
                            # ChunkStore에서 실제 chunk 조회
                            try:
                                chunks = self.chunk_store.get_chunks_by_node(
                                    repo_id=repo_id, node_id=match.node_id
                                )

                                if chunks:
                                    # 모든 청크에 점수 분배 (최대 max_chunks_per_node개)
                                    max_chunks = self.config.fuzzy_max_chunks_per_node
                                    selected_chunks = chunks[:max_chunks]
                                    num_chunks = len(selected_chunks)

                                    # 청크 수에 따라 점수 분배
                                    distributed_score = base_fuzzy_score / num_chunks

                                    for chunk in selected_chunks:
                                        chunk_id = chunk.id

                                        if chunk_id in candidates:
                                            # 기존 점수보다 높으면 업데이트
                                            existing_score = candidates[chunk_id].features.get(
                                                "fuzzy_score", 0
                                            )
                                            candidates[chunk_id].features["fuzzy_score"] = max(
                                                existing_score, distributed_score
                                            )
                                        else:
                                            # 새로운 candidate 추가
                                            candidates[chunk_id] = Candidate(
                                                repo_id=repo_id,
                                                chunk_id=chunk_id,
                                                features={"fuzzy_score": distributed_score},
                                                file_path=chunk.file_path,
                                                span=chunk.span,
                                            )
                                else:
                                    # chunk 없으면 임시 ID
                                    chunk_id = f"chunk-{match.node_id}"
                                    file_path = match.file_path or ""
                                    span = (0, 0, 0, 0)

                                    if chunk_id in candidates:
                                        candidates[chunk_id].features[
                                            "fuzzy_score"
                                        ] = base_fuzzy_score
                                    else:
                                        candidates[chunk_id] = Candidate(
                                            repo_id=repo_id,
                                            chunk_id=chunk_id,
                                            features={"fuzzy_score": base_fuzzy_score},
                                            file_path=file_path,
                                            span=span,
                                        )
                            except Exception as e:
                                logger.debug(f"Failed to get chunk for node {match.node_id}: {e}")
                                chunk_id = f"chunk-{match.node_id}"
                                file_path = match.file_path or ""
                                span = (0, 0, 0, 0)

                                if chunk_id in candidates:
                                    candidates[chunk_id].features["fuzzy_score"] = base_fuzzy_score
                                else:
                                    candidates[chunk_id] = Candidate(
                                        repo_id=repo_id,
                                        chunk_id=chunk_id,
                                        features={"fuzzy_score": base_fuzzy_score},
                                        file_path=file_path,
                                        span=span,
                                    )
                        else:
                            # ChunkStore 없으면 임시 ID
                            chunk_id = (
                                f"chunk-{match.node_id}"
                                if match.node_id
                                else f"chunk-fuzzy-{match.matched_text}"
                            )
                            file_path = match.file_path or ""
                            span = (0, 0, 0, 0)

                            if chunk_id in candidates:
                                candidates[chunk_id].features["fuzzy_score"] = base_fuzzy_score
                            else:
                                # 새로운 candidate 추가
                                candidates[chunk_id] = Candidate(
                                    repo_id=repo_id,
                                    chunk_id=chunk_id,
                                    features={"fuzzy_score": base_fuzzy_score},
                                    file_path=file_path,
                                    span=span,
                                )

                    if fuzzy_matches:
                        logger.debug(f"Fuzzy: '{token}' → {len(fuzzy_matches)} matches")

            except Exception as e:
                logger.error(f"Fuzzy search failed: {e}")

        # 5. Backend별 점수 정규화 적용
        normalization_method = self.config.score_normalization_method

        # Lexical 정규화
        if lexical_results_raw:
            lexical_normalized = self._normalize_scores(
                [(r.chunk_id, r.score) for r in lexical_results_raw], method=normalization_method
            )
            for result in lexical_results_raw:
                normalized_score = lexical_normalized.get(result.chunk_id, 0)
                if result.chunk_id in candidates:
                    candidates[result.chunk_id].features["lexical_score"] = normalized_score
                else:
                    candidates[result.chunk_id] = self._result_to_candidate(
                        result, lexical_score=normalized_score
                    )

        # Semantic 정규화
        if semantic_results_raw:
            semantic_normalized = self._normalize_scores(
                [(r.chunk_id, r.score) for r in semantic_results_raw], method=normalization_method
            )
            for result in semantic_results_raw:
                normalized_score = semantic_normalized.get(result.chunk_id, 0)
                if result.chunk_id in candidates:
                    candidates[result.chunk_id].features["semantic_score"] = normalized_score
                else:
                    candidates[result.chunk_id] = self._result_to_candidate(
                        result, semantic_score=normalized_score
                    )

        # 6. Candidate 리스트로 변환
        candidate_list = list(candidates.values())

        # 7. Weighted Fusion 적용 (점수 결합 및 정렬)
        fused_results: list[Candidate] = self.fusion.fuse_and_sort(candidate_list, weights, k=k)

        logger.info(f"Retrieved {len(fused_results)} candidates")
        return fused_results

    def _result_to_candidate(
        self,
        result: ChunkResult,
        lexical_score: float = 0.0,
        semantic_score: float = 0.0,
        graph_score: float = 0.0,
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
            span=result.span,
        )

    def _normalize_scores(
        self, results: list[tuple[str, float]], method: str = "rank"
    ) -> dict[str, float]:
        """
        검색 결과 점수를 정규화

        Args:
            results: (id, score) 튜플 리스트
            method: 정규화 방법 ("minmax" | "rank" | "zscore")

        Returns:
            {id: normalized_score} 딕셔너리
        """
        if not results:
            return {}

        if method == "rank":
            # Rank-based: 순위를 역순으로 정규화 (1등이 1.0)
            n = len(results)
            return {id_: (n - rank) / n for rank, (id_, _) in enumerate(results)}

        elif method == "minmax":
            # Min-Max: 최소-최대 범위로 정규화
            scores = [s for _, s in results]
            min_score = min(scores)
            max_score = max(scores)

            if max_score == min_score:
                return {id_: 1.0 for id_, _ in results}

            return {id_: (score - min_score) / (max_score - min_score) for id_, score in results}

        elif method == "zscore":
            # Z-score: 평균과 표준편차로 정규화
            import statistics

            scores = [s for _, s in results]
            mean = statistics.mean(scores)

            try:
                stdev = statistics.stdev(scores)
                if stdev == 0:
                    return {id_: 1.0 for id_, _ in results}

                normalized = {id_: (score - mean) / stdev for id_, score in results}

                # Z-score는 음수일 수 있으므로 0~1로 재조정
                z_scores = list(normalized.values())
                min_z = min(z_scores)
                max_z = max(z_scores)

                if max_z == min_z:
                    return dict.fromkeys(normalized, 1.0)

                return {id_: (z - min_z) / (max_z - min_z) for id_, z in normalized.items()}
            except statistics.StatisticsError:
                return {id_: 1.0 for id_, _ in results}

        else:
            # 기본값: rank
            return self._normalize_scores(results, method="rank")

    def _normalize_lexical_score(self, score: float) -> float:
        """
        Lexical(BM25) 점수를 0~1로 정규화 (레거시 메서드)

        Args:
            score: 원본 BM25 점수

        Returns:
            0~1 범위로 정규화된 점수
        """
        if not self.config.enable_score_normalization:
            return score

        # 최대값으로 나누고 1.0으로 clipping
        normalized = score / self.config.lexical_score_max
        return min(1.0, max(0.0, normalized))

    def _normalize_semantic_score(self, score: float) -> float:
        """
        Semantic(cosine similarity) 점수를 0~1로 정규화

        Cosine similarity는 이미 0~1 범위이므로 그대로 반환.
        (음수가 나올 수도 있지만 코드 임베딩에서는 거의 없음)

        Args:
            score: 원본 cosine similarity 점수

        Returns:
            0~1 범위로 정규화된 점수
        """
        if not self.config.enable_score_normalization:
            return score

        # Cosine similarity는 이미 0~1 범위
        return min(1.0, max(0.0, score))

    def _normalize_graph_score(self, score: float) -> float:
        """
        Graph 점수를 0~1로 정규화

        Graph 점수는 이미 거리 기반으로 0~1 범위로 계산되므로 그대로 반환.
        (distance_score = 1.0 / (i + 1) 형태)

        Args:
            score: 원본 graph 점수

        Returns:
            0~1 범위로 정규화된 점수
        """
        if not self.config.enable_score_normalization:
            return score

        # Graph 점수는 이미 0~1 범위
        return min(1.0, max(0.0, score))

    def _normalize_fuzzy_score(self, score: float) -> float:
        """
        Fuzzy 점수를 0~1로 정규화

        Fuzzy 라이브러리(rapidfuzz 등)는 일반적으로 0~100 범위를 반환하지만,
        설정된 최대값으로 정규화.

        Args:
            score: 원본 fuzzy 점수

        Returns:
            0~1 범위로 정규화된 점수
        """
        if not self.config.enable_score_normalization:
            return score

        # 최대값으로 나누고 1.0으로 clipping
        normalized = score / self.config.fuzzy_score_max
        return min(1.0, max(0.0, normalized))

    def _extract_symbol_tokens(self, query: str) -> list[str]:
        """
        쿼리에서 심볼 토큰 추출 (개선 버전)

        - CamelCase 분리: UserService → User, Service
        - 약어 처리: HTTPSConnection → HTTPS, Connection
        - snake_case 분리: user_service → user, service
        - 불용어 필터링
        - 최소 길이 필터링 & 중복 제거

        Args:
            query: 검색 쿼리

        Returns:
            토큰 리스트
        """
        tokens = []
        stopwords = set(self.config.fuzzy_stopwords or [])

        # 1. 공백으로 분리
        words = query.split()

        for word in words:
            # 원본 단어를 소문자로 변환하여 불용어 체크
            word_lower = word.lower()

            # 2. 불용어는 건너뛰기
            if word_lower in stopwords:
                continue

            # 3. CamelCase 분리 (개선)
            # HTTPSConnection → HTTPS, Connection
            # getUserById → get, User, By, Id
            # getURLFromAPI → get, URL, From, API

            # 연속된 대문자는 약어로 처리
            # 예: HTTPSConnection → [HTTPS, Connection]
            camel_tokens = []
            temp_token = ""

            for i, char in enumerate(word):
                if char.isupper():
                    # 다음 문자가 소문자면 새로운 토큰 시작
                    if i + 1 < len(word) and word[i + 1].islower():
                        if temp_token:
                            camel_tokens.append(temp_token)
                        temp_token = char
                    # 이전에 쌓인 대문자가 있고, 현재가 대문자면
                    elif temp_token and temp_token[-1].isupper():
                        temp_token += char
                    else:
                        if temp_token:
                            camel_tokens.append(temp_token)
                        temp_token = char
                elif char.islower() or char.isdigit():
                    temp_token += char
                else:
                    # 특수문자는 토큰 분리
                    if temp_token:
                        camel_tokens.append(temp_token)
                        temp_token = ""

            if temp_token:
                camel_tokens.append(temp_token)

            if camel_tokens:
                tokens.extend(camel_tokens)

            # 4. snake_case, kebab-case 분리
            snake_tokens = re.split(r"[_\-.]", word)
            for token in snake_tokens:
                if token and token.lower() not in stopwords:
                    tokens.append(token)

            # 5. 원본 단어도 포함 (불용어가 아닌 경우만)
            if word_lower not in stopwords:
                tokens.append(word)

        # 6. 최소 길이 필터 & 불용어 재필터 & 중복 제거
        min_length = self.config.fuzzy_min_token_length
        tokens = [t for t in tokens if len(t) >= min_length and t.lower() not in stopwords]
        tokens = list(dict.fromkeys(tokens))  # 순서 유지하면서 중복 제거

        return tokens

    def _retrieve_parallel(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        location_ctx: LocationContext | None,
        weights: dict[str, float],
    ) -> list[Candidate]:
        """병렬 검색 (ThreadPoolExecutor 사용)"""
        candidates: dict[str, Candidate] = {}  # chunk_id -> Candidate

        # ThreadPoolExecutor로 병렬 실행
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            # 1. Lexical 검색
            if weights.get("lexical", 0) > 0:
                future = executor.submit(
                    self._search_lexical, repo_id, query, k, location_ctx, weights["lexical"]
                )
                futures.append(("lexical", future))

            # 2. Semantic 검색
            if weights.get("semantic", 0) > 0:
                future = executor.submit(
                    self._search_semantic, repo_id, query, k, location_ctx, weights["semantic"]
                )
                futures.append(("semantic", future))

            # 3. Graph 검색
            if weights.get("graph", 0) > 0 and location_ctx:
                future = executor.submit(
                    self._search_graph, repo_id, k, location_ctx, weights["graph"]
                )
                futures.append(("graph", future))

            # 4. Fuzzy 검색
            if weights.get("fuzzy", 0) > 0 and self.fuzzy_search:
                future = executor.submit(self._search_fuzzy, repo_id, query, k, weights["fuzzy"])
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

        # Weighted Fusion 적용 (점수 결합 및 정렬)
        candidate_list = list(candidates.values())
        result = self.fusion.fuse_and_sort(candidate_list, weights, k=k)

        logger.info(f"Retrieved {len(result)} candidates (parallel)")
        return result

    def _search_lexical(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        location_ctx: LocationContext | None,
        weight: float,
    ) -> list[Candidate]:
        """Lexical 검색"""
        try:
            lexical_k = int(k * self.config.lexical_fetch_multiplier)
            lexical_results = self.lexical_search.search(
                repo_id=repo_id,
                query=query,
                k=lexical_k,
                filters=location_ctx.filters if location_ctx else None,
            )

            return [
                self._result_to_candidate(
                    result, lexical_score=self._normalize_lexical_score(result.score)
                )
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
        location_ctx: LocationContext | None,
        weight: float,
    ) -> list[Candidate]:
        """Semantic 검색"""
        try:
            semantic_k = int(k * self.config.semantic_fetch_multiplier)
            semantic_results = self.semantic_search.search(
                repo_id=repo_id,
                query=query,
                k=semantic_k,
                filters=location_ctx.filters if location_ctx else None,
            )

            return [
                self._result_to_candidate(
                    result, semantic_score=self._normalize_semantic_score(result.score)
                )
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
    ) -> list[Candidate]:
        """Graph 검색"""
        try:
            current_node = self.graph_search.get_node_by_location(
                repo_id=repo_id,
                file_path=location_ctx.file_path,
                line=location_ctx.line,
                column=location_ctx.column,
            )

            if not current_node:
                return []

            neighbors_with_edges = self.graph_search.expand_neighbors_with_edges(
                repo_id=repo_id, node_id=current_node.id, k=2
            )

            candidates = []
            graph_neighbors = neighbors_with_edges[: int(k * self.config.graph_fetch_multiplier)]
            for neighbor_node, edge_type, depth in graph_neighbors:
                # Edge 타입별 가중치 적용
                edge_weights = self.config.graph_edge_weights or {}
                edge_weight = edge_weights.get(edge_type, 0.5)

                # Depth decay 적용
                depth_decay = self.config.graph_depth_decay ** (depth - 1)

                # 최종 graph score
                graph_score = edge_weight * depth_decay
                normalized_score = self._normalize_graph_score(graph_score)
                chunk_id = f"chunk-{neighbor_node.id}"

                candidates.append(
                    Candidate(
                        repo_id=repo_id,
                        chunk_id=chunk_id,
                        features={"graph_score": normalized_score},
                        file_path=neighbor_node.file_path,
                        span=neighbor_node.span,
                    )
                )

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
    ) -> list[Candidate]:
        """Fuzzy 검색 (개선: 모든 청크 반환 + 점수 분배)"""
        if not self.fuzzy_search:
            return []
        try:
            query_tokens = self._extract_symbol_tokens(query)
            candidates_map: dict[str, Candidate] = {}  # chunk_id -> Candidate (중복 방지)

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
                    normalized_match_score = self._normalize_fuzzy_score(match.score)
                    base_fuzzy_score = normalized_match_score * rank_score

                    # node_id → chunk_id 매핑 (개선: 모든 청크 반환)
                    if match.node_id and self.chunk_store:
                        try:
                            chunks = self.chunk_store.get_chunks_by_node(
                                repo_id=repo_id, node_id=match.node_id
                            )

                            if chunks:
                                # 모든 청크에 점수 분배
                                max_chunks = self.config.fuzzy_max_chunks_per_node
                                selected_chunks = chunks[:max_chunks]
                                num_chunks = len(selected_chunks)
                                distributed_score = base_fuzzy_score / num_chunks

                                for chunk in selected_chunks:
                                    chunk_id = chunk.id

                                    if chunk_id in candidates_map:
                                        # 기존 점수보다 높으면 업데이트
                                        existing_score = candidates_map[chunk_id].features.get(
                                            "fuzzy_score", 0
                                        )
                                        candidates_map[chunk_id].features["fuzzy_score"] = max(
                                            existing_score, distributed_score
                                        )
                                    else:
                                        candidates_map[chunk_id] = Candidate(
                                            repo_id=repo_id,
                                            chunk_id=chunk_id,
                                            features={"fuzzy_score": distributed_score},
                                            file_path=chunk.file_path,
                                            span=chunk.span,
                                        )
                            else:
                                chunk_id = f"chunk-{match.node_id}"
                                file_path = match.file_path or ""
                                span = (0, 0, 0, 0)

                                if chunk_id not in candidates_map:
                                    candidates_map[chunk_id] = Candidate(
                                        repo_id=repo_id,
                                        chunk_id=chunk_id,
                                        features={"fuzzy_score": base_fuzzy_score},
                                        file_path=file_path,
                                        span=span,
                                    )
                        except Exception:
                            chunk_id = f"chunk-{match.node_id}"
                            file_path = match.file_path or ""
                            span = (0, 0, 0, 0)

                            if chunk_id not in candidates_map:
                                candidates_map[chunk_id] = Candidate(
                                    repo_id=repo_id,
                                    chunk_id=chunk_id,
                                    features={"fuzzy_score": base_fuzzy_score},
                                    file_path=file_path,
                                    span=span,
                                )
                    else:
                        chunk_id = (
                            f"chunk-{match.node_id}"
                            if match.node_id
                            else f"chunk-fuzzy-{match.matched_text}"
                        )
                        file_path = match.file_path or ""
                        span = (0, 0, 0, 0)

                        if chunk_id not in candidates_map:
                            candidates_map[chunk_id] = Candidate(
                                repo_id=repo_id,
                                chunk_id=chunk_id,
                                features={"fuzzy_score": base_fuzzy_score},
                                file_path=file_path,
                                span=span,
                            )

            return list(candidates_map.values())
        except Exception as e:
            logger.error(f"Fuzzy search failed: {e}")
            return []

    def _log_query(
        self,
        repo_id: RepoId,
        query: str,
        query_type: str | None,
        k: int,
        weights: dict[str, float],
        results: list[Candidate],
        latency_ms: int,
    ) -> None:
        """
        쿼리 로그 저장 (Phase 2)

        Args:
            repo_id: 저장소 ID
            query: 쿼리 텍스트
            query_type: 쿼리 타입
            k: 요청한 결과 수
            weights: 사용한 가중치
            results: 검색 결과
            latency_ms: 레이턴시
        """
        from ...query_log_store import QueryLog

        # Top 10 결과만 로깅 (용량 절약)
        top_results = []
        for i, candidate in enumerate(results[:10], 1):
            top_results.append(
                {
                    "rank": i,
                    "node_id": candidate.chunk_id,
                    "file_path": candidate.file_path,
                    "score": candidate.features.get("final_score", 0.0),
                    "signals": {
                        k: v
                        for k, v in candidate.features.items()
                        if k
                        in [
                            "lexical",
                            "semantic_small_code",
                            "semantic_small_node",
                            "semantic_large_node",
                            "graph",
                            "fuzzy",
                        ]
                    },
                }
            )

        # 쿼리 임베딩 (optional)
        query_embedding = None
        if self.log_query_embedding:
            try:
                query_embedding = self.semantic_search.embed_text(query)
            except Exception as e:
                logger.debug(f"Failed to embed query for logging: {e}")

        log = QueryLog(
            repo_id=repo_id,
            query_text=query,
            query_type=query_type,
            query_embedding=query_embedding,
            weights=weights,
            k=k,
            result_count=len(results),
            top_results=top_results,
            latency_ms=latency_ms,
        )

        self.query_log_store.log_query(log)
        logger.debug(f"Query logged: {query[:30]}... ({len(results)} results, {latency_ms}ms)")
