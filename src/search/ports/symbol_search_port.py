"""심볼 검색 포트"""

from typing import Protocol

from ...core.models import CodeNode, RepoId


class SymbolSearchPort(Protocol):
    """
    심볼 검색 인터페이스

    code_nodes 테이블을 직접 조회하여 함수/클래스/메서드를 검색
    """

    def search_by_name(
        self,
        repo_id: RepoId,
        query: str,
        kind: str | None = None,
        fuzzy: bool = True,
        k: int = 20,
    ) -> list[CodeNode]:
        """
        심볼 이름으로 검색

        Args:
            repo_id: 저장소 ID
            query: 검색 쿼리 (함수/클래스 이름)
            kind: 심볼 종류 필터 ("Function", "Class", "Method", None=전체)
            fuzzy: 퍼지 매칭 여부 (True: LIKE %, False: 정확 매칭)
            k: 결과 개수

        Returns:
            CodeNode 리스트 (이름 길이 순 정렬)

        Examples:
            search_by_name("my-repo", "HybridRetriever", kind="Class")
            search_by_name("my-repo", "search", fuzzy=True)  # hybrid_search, search_symbols 등
        """
        ...

    def search_by_decorator(
        self,
        repo_id: RepoId,
        decorator_pattern: str,
        k: int = 50,
    ) -> list[CodeNode]:
        """
        Decorator로 검색

        Args:
            repo_id: 저장소 ID
            decorator_pattern: Decorator 패턴 ("@router.post", "router.get", "app.get" 등)
            k: 결과 개수

        Returns:
            CodeNode 리스트

        Examples:
            search_by_decorator("my-repo", "@router.post")
            search_by_decorator("my-repo", "app.get")

        Note:
            @ 기호는 있어도 없어도 됨 (자동 처리)
        """
        ...

    def search_by_file(
        self,
        repo_id: RepoId,
        file_path: str,
        kind: str | None = None,
        k: int = 100,
    ) -> list[CodeNode]:
        """
        파일 내 심볼 검색

        Args:
            repo_id: 저장소 ID
            file_path: 파일 경로
            kind: 심볼 종류 필터
            k: 결과 개수

        Returns:
            CodeNode 리스트 (라인 번호 순 정렬)
        """
        ...

    def get_symbol_by_location(
        self,
        repo_id: RepoId,
        file_path: str,
        line: int,
    ) -> CodeNode | None:
        """
        위치로 심볼 찾기

        Args:
            repo_id: 저장소 ID
            file_path: 파일 경로
            line: 라인 번호 (0-based)

        Returns:
            해당 위치의 가장 작은 심볼 (가장 구체적인 노드)

        Note:
            GraphSearchPort.get_node_by_location()과 동일
        """
        ...
