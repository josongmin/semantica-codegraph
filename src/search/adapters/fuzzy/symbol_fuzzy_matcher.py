"""심볼명 퍼지 매칭 구현"""

import logging
from collections import defaultdict
from functools import lru_cache

from rapidfuzz import fuzz, process

from ....core.config import Config
from ....core.models import RepoId
from ....core.ports import GraphStorePort
from ...ports.fuzzy_search_port import FuzzyMatch, FuzzySearchPort

logger = logging.getLogger(__name__)


class SymbolFuzzyMatcher(FuzzySearchPort):
    """
    심볼명 퍼지 매칭 구현

    특징:
    - rapidfuzz를 사용한 고속 퍼지 매칭
    - LRU 캐시를 통한 성능 최적화
    - 저장소별 심볼 인덱스 관리
    - 대소문자 무시, 축약형 허용

    알고리즘:
    - Token Set Ratio: 순서 무시, 부분 매칭 허용
    - Threshold 기반 필터링
    """

    def __init__(self, graph_store: GraphStorePort, config: Config):
        """
        Args:
            graph_store: 그래프 저장소 (심볼 정보 조회용)
            config: 설정
        """
        self.graph_store = graph_store
        self.config = config

        # 저장소별 심볼 인덱스 캐시
        # repo_id -> {symbol_name: [(node_id, file_path, kind), ...]}
        self._symbol_index: dict[RepoId, dict[str, list[tuple[str, str, str]]]] = {}

        # 심볼 매칭 결과 캐시 (LRU)
        self._build_search_cache()

    def _build_search_cache(self):
        """검색 결과 캐시 생성 (LRU)"""
        cache_size = self.config.fuzzy_cache_size

        @lru_cache(maxsize=cache_size)
        def _cached_search(repo_id: RepoId, query: str, threshold: float, k: int) -> tuple:
            """캐시된 검색 (내부용)"""
            return tuple(self._search_internal(repo_id, query, threshold, k))

        self._cached_search = _cached_search

    def search_symbols(
        self,
        repo_id: RepoId,
        query: str,
        threshold: float = 0.8,
        k: int = 10,
        kinds: list[str] | None = None,
    ) -> list[FuzzyMatch]:
        """심볼명에 대한 퍼지 매칭"""
        if not self.config.fuzzy_matching_enabled:
            logger.debug("Fuzzy matching disabled")
            return []

        if not query or not query.strip():
            return []

        # 실제 threshold는 config와 인자 중 높은 값 사용
        actual_threshold = max(threshold, self.config.fuzzy_threshold)

        # 캐시에서 검색 (kinds는 캐시 키에 포함 안 됨)
        try:
            cached_results = self._cached_search(repo_id, query.strip(), actual_threshold, k * 2)
            results = list(cached_results)
        except Exception as e:
            logger.error(f"Fuzzy search error: {e}")
            return []

        # kinds 필터링 (캐시 이후)
        if kinds:
            results = [r for r in results if r.kind in kinds]

        # k개로 제한
        return results[:k]

    def _search_internal(
        self,
        repo_id: RepoId,
        query: str,
        threshold: float,
        k: int,
    ) -> list[FuzzyMatch]:
        """내부 검색 로직 (캐시 안 됨)"""
        # 심볼 인덱스 로드 (필요시)
        if repo_id not in self._symbol_index:
            self._load_symbol_index(repo_id)

        symbol_index = self._symbol_index.get(repo_id, {})
        if not symbol_index:
            logger.debug(f"No symbols found for repo {repo_id}")
            return []

        # 심볼명 리스트
        symbol_names = list(symbol_index.keys())

        # 너무 많으면 샘플링 (성능 보호)
        max_candidates = self.config.fuzzy_max_candidates
        if len(symbol_names) > max_candidates:
            logger.debug(f"Too many symbols ({len(symbol_names)}), limiting to {max_candidates}")
            symbol_names = self._smart_filter(symbol_names, query, max_candidates)

        # rapidfuzz 퍼지 매칭
        # token_set_ratio: 순서 무시, 부분 매칭 허용
        # 대소문자 무시를 위해 모두 소문자로 변환
        query_lower = query.lower()
        symbol_names_lower = [s.lower() for s in symbol_names]

        matches = process.extract(
            query_lower,
            symbol_names_lower,
            scorer=fuzz.token_set_ratio,
            limit=k,
            score_cutoff=threshold * 100,  # rapidfuzz는 0~100 스케일
        )

        # FuzzyMatch 객체로 변환
        results = []
        for _matched_name_lower, score, index in matches:
            # 정규화 (0~100 → 0~1)
            normalized_score = score / 100.0

            # 원본 심볼명 복원 (대소문자 유지)
            original_name = symbol_names[index]

            # 해당 심볼의 첫 번째 인스턴스 사용 (여러 개 있을 수 있음)
            instances = symbol_index[original_name]
            if instances:
                node_id, file_path, kind = instances[0]
                results.append(
                    FuzzyMatch(
                        matched_text=original_name,
                        score=normalized_score,
                        node_id=node_id,
                        file_path=file_path,
                        kind=kind,
                    )
                )

        logger.debug(f"Fuzzy search: '{query}' → {len(results)} matches")
        return results

    def _smart_filter(
        self,
        symbols: list[str],
        query: str,
        max_count: int
    ) -> list[str]:
        """
        스마트 필터링: prefix 매칭 우선, 그 다음 길이 유사도

        Args:
            symbols: 전체 심볼 리스트
            query: 검색 쿼리
            max_count: 반환할 최대 개수

        Returns:
            필터링된 심볼 리스트
        """
        query_lower = query.lower()

        # 1. Prefix 매칭 (대소문자 무시)
        prefix_matches = [
            s for s in symbols
            if s.lower().startswith(query_lower)
        ]

        # 2. 길이 비슷한 것
        length_matches = sorted(
            symbols,
            key=lambda s: abs(len(s) - len(query))
        )

        # 3. 결합 (prefix 우선, 중복 제거)
        seen = set()
        result = []

        for symbol in prefix_matches + length_matches:
            if symbol not in seen:
                seen.add(symbol)
                result.append(symbol)
                if len(result) >= max_count:
                    break

        return result

    def _load_symbol_index(self, repo_id: RepoId) -> None:
        """
        저장소의 심볼 인덱스 로드

        GraphStore에서 모든 심볼을 가져와서 인메모리 인덱스 구축
        """
        logger.info(f"Loading symbol index for repo {repo_id}")

        try:
            # GraphStore의 list_nodes() 사용
            nodes = self.graph_store.list_nodes(repo_id)

            # 심볼 인덱스 구축
            symbol_index: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
            for node in nodes:
                symbol_index[node.name].append((node.id, node.file_path, node.kind))

            self._symbol_index[repo_id] = dict(symbol_index)
            logger.info(f"Loaded {len(symbol_index)} unique symbols for {repo_id}")

        except Exception as e:
            logger.error(f"Failed to load symbol index: {e}")
            self._symbol_index[repo_id] = {}

    def refresh_cache(self, repo_id: RepoId) -> None:
        """저장소의 퍼지 매칭 캐시 갱신"""
        logger.info(f"Refreshing fuzzy cache for {repo_id}")

        # 심볼 인덱스 재로드
        if repo_id in self._symbol_index:
            del self._symbol_index[repo_id]

        self._load_symbol_index(repo_id)

        # LRU 캐시 클리어
        self._cached_search.cache_clear()

    def clear_cache(self, repo_id: RepoId | None = None) -> None:
        """퍼지 매칭 캐시 삭제"""
        if repo_id:
            logger.info(f"Clearing fuzzy cache for {repo_id}")
            if repo_id in self._symbol_index:
                del self._symbol_index[repo_id]
        else:
            logger.info("Clearing all fuzzy caches")
            self._symbol_index.clear()

        # LRU 캐시 클리어
        self._cached_search.cache_clear()

