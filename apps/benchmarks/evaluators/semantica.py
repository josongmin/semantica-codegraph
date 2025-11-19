"""Semantica 리트리버 평가기"""

import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.bootstrap import Bootstrap
from src.core.config import Config
from src.core.models import LocationContext, RepoId

from .metrics import SearchResult


class SemanticaEvaluator:
    """Semantica 하이브리드 리트리버 평가기"""

    def __init__(self, config: Config | None = None):
        """
        Args:
            config: 설정 (None이면 환경변수에서 로드)
        """
        if config is None:
            config = Config.from_env()

        self.config = config
        self.bootstrap = Bootstrap(config)
        self.retriever = self.bootstrap.hybrid_retriever

    def search(
        self, repo_id: RepoId, query: str, k: int = 5, location_ctx: LocationContext | None = None
    ) -> SearchResult:
        """
        검색 실행 및 결과 반환

        Args:
            repo_id: 저장소 ID
            query: 검색 쿼리
            k: 반환할 결과 수
            location_ctx: 위치 컨텍스트 (옵션)

        Returns:
            SearchResult (파일 경로 리스트 + 지연시간)
        """
        # 검색 실행 및 시간 측정
        start = time.perf_counter()
        candidates = self.retriever.retrieve(
            repo_id=repo_id, query=query, k=k, location_ctx=location_ctx
        )
        end = time.perf_counter()
        latency_ms = (end - start) * 1000

        # Candidate -> 파일 경로 변환
        file_paths = [c.file_path for c in candidates]

        return SearchResult(
            query=query, results=file_paths, latency_ms=latency_ms, retriever_name="semantica"
        )

    def batch_search(self, repo_id: RepoId, queries: list[str], k: int = 5) -> list[SearchResult]:
        """
        배치 검색

        Args:
            repo_id: 저장소 ID
            queries: 쿼리 리스트
            k: 결과 수

        Returns:
            SearchResult 리스트
        """
        results = []
        for query in queries:
            result = self.search(repo_id, query, k)
            results.append(result)
        return results
