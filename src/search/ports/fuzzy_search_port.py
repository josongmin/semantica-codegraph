"""퍼지 검색 포트 정의"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ...core.models import RepoId


@dataclass
class FuzzyMatch:
    """퍼지 매칭 결과"""
    matched_text: str  # 실제 매칭된 심볼명
    score: float  # 유사도 점수 (0.0~1.0)
    node_id: str | None = None  # 매칭된 노드 ID (있는 경우)
    file_path: str | None = None  # 파일 경로
    kind: str | None = None  # 심볼 종류 (Function, Class 등)


class FuzzySearchPort(ABC):
    """
    퍼지 검색 포트

    심볼명/함수명/클래스명에 대한 퍼지 매칭을 제공.
    오타, 대소문자 차이, 축약형 등을 허용하여 검색.

    주요 사용처:
    - 사용자 입력 검색 (UX 개선)
    - 에이전트 심볼 선택 (안정성 향상)
    - RAG retrieval fallback (recall 증가)
    """

    @abstractmethod
    def search_symbols(
        self,
        repo_id: RepoId,
        query: str,
        threshold: float = 0.8,
        k: int = 10,
        kinds: list[str] | None = None,
    ) -> list[FuzzyMatch]:
        """
        심볼명에 대한 퍼지 매칭

        Args:
            repo_id: 저장소 ID
            query: 검색 쿼리 (심볼명)
            threshold: 유사도 임계값 (0.0~1.0, 높을수록 엄격)
            k: 반환할 최대 결과 수
            kinds: 필터링할 심볼 종류 (예: ["Function", "Class"])

        Returns:
            매칭된 심볼 리스트 (유사도 순 정렬)
        """
        pass

    @abstractmethod
    def refresh_cache(self, repo_id: RepoId) -> None:
        """
        저장소의 퍼지 매칭 캐시 갱신

        Args:
            repo_id: 저장소 ID
        """
        pass

    @abstractmethod
    def clear_cache(self, repo_id: RepoId | None = None) -> None:
        """
        퍼지 매칭 캐시 삭제

        Args:
            repo_id: 저장소 ID (None이면 전체 캐시 삭제)
        """
        pass

