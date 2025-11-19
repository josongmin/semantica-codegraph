import logging
from typing import Any

from meilisearch import Client

from ....core.models import ChunkResult, CodeChunk, RepoId
from ...ports.lexical_search_port import LexicalSearchPort

try:
    from meilisearch.errors import MeilisearchApiError
except ImportError:
    # meilisearch 패키지가 설치되지 않은 경우 fallback
    MeilisearchApiError = Exception  # type: ignore

logger = logging.getLogger(__name__)


class MeiliSearchAdapter(LexicalSearchPort):
    """MeiliSearch를 사용한 BM25 검색 어댑터"""

    def __init__(
        self,
        client: Client,
        index_prefix: str = "chunks",
        batch_size: int = 1000,
    ):
        self.client = client
        self.index_prefix = index_prefix
        self.batch_size = batch_size

    def _get_index_name(self, repo_id: RepoId) -> str:
        """저장소별 인덱스 이름 생성"""
        # repo_id에서 특수문자 제거 (MeiliSearch 인덱스명 제약)
        safe_repo_id = repo_id.replace("/", "_").replace(":", "_").replace("-", "_")
        return f"{self.index_prefix}_{safe_repo_id}"

    def _sanitize_id(self, chunk_id: str) -> str:
        """
        Chunk ID를 MeiliSearch 호환 형식으로 변환

        MeiliSearch는 ID에 alphanumeric, -, _ 만 허용
        콜론(:), 슬래시(/) 등을 언더스코어로 변경
        """
        return chunk_id.replace(":", "_").replace("/", "_").replace(".", "_")

    def _ensure_index(self, repo_id: RepoId) -> Any:
        """인덱스 생성 또는 가져오기"""
        index_name = self._get_index_name(repo_id)

        try:
            index = self.client.get_index(index_name)
            logger.debug(f"Using existing index: {index_name}")
            return index
        except MeilisearchApiError as e:
            if e.code == "index_not_found":
                logger.info(f"Creating new index: {index_name}")
                task = self.client.create_index(index_name, {"primaryKey": "id"})
                # 작업 완료 대기
                task_uid = task.task_uid if hasattr(task, "task_uid") else task.uid
                self.client.wait_for_task(task_uid)
                index = self.client.get_index(index_name)

                # 검색 설정
                self._configure_index(index)
                return index
            else:
                raise

    def _configure_index(self, index: Any) -> None:
        """인덱스 검색 설정"""
        # 검색 가능한 속성 (우선순위 순서)
        task1 = index.update_searchable_attributes(
            [
                "text",  # 코드 텍스트가 가장 중요
                "file_path",  # 파일 경로로도 검색
            ]
        )

        # 필터 가능한 속성
        task2 = index.update_filterable_attributes(["repo_id", "language", "file_path", "node_id"])

        # 정렬 가능한 속성
        task3 = index.update_sortable_attributes(["file_path"])

        # 랭킹 규칙 (BM25 우선)
        task4 = index.update_ranking_rules(
            [
                "words",  # 단어 매칭 수
                "typo",  # 오타 허용
                "proximity",  # 단어 간 거리
                "attribute",  # 속성 우선순위
                "sort",  # 정렬
                "exactness",  # 정확도
            ]
        )

        # 작업 완료 대기
        for task in [task1, task2, task3, task4]:
            task_uid = task.task_uid if hasattr(task, "task_uid") else task.uid
            self.client.wait_for_task(task_uid)

        logger.debug(f"Configured index: {index.uid}")

    def index_chunks(self, chunks: list[CodeChunk]) -> None:
        """청크 인덱싱"""
        if not chunks:
            logger.warning("No chunks to index")
            return

        repo_id = chunks[0].repo_id
        index = self._ensure_index(repo_id)

        # 문서 변환
        documents = [
            {
                "id": self._sanitize_id(chunk.id),  # MeiliSearch 호환 ID
                "original_id": chunk.id,  # 원본 ID 보존
                "repo_id": chunk.repo_id,
                "node_id": chunk.node_id,
                "file_path": chunk.file_path,
                "language": chunk.language,
                # search_text 우선, 없으면 raw text
                "text": chunk.attrs.get("search_text", chunk.text),
                "span": list(chunk.span),  # tuple -> list (JSON 직렬화)
                **chunk.attrs,
            }
            for chunk in chunks
        ]

        # 배치 처리
        total_indexed = 0
        last_task = None
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            last_task = index.add_documents(batch)
            total_indexed += len(batch)
            logger.debug(f"Indexed {total_indexed}/{len(documents)} documents")

        # 마지막 작업 완료 대기 (검색 전에 인덱싱이 완료되도록)
        if last_task:
            task_uid = last_task.task_uid if hasattr(last_task, "task_uid") else last_task.uid
            self.client.wait_for_task(task_uid)

        logger.info(f"Indexed {len(documents)} chunks for repo: {repo_id}")

    def search(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        filters: dict | None = None,
    ) -> list[ChunkResult]:
        """검색 실행"""
        index_name = self._get_index_name(repo_id)

        try:
            index = self.client.get_index(index_name)
        except MeilisearchApiError as e:
            if e.code == "index_not_found":
                logger.warning(f"Index not found: {index_name}")
                return []
            else:
                raise

        # 필터 구성 (MeiliSearch 필터 문법)
        filter_str = None
        if filters:
            filter_parts = []
            if "language" in filters:
                # 문자열 값은 따옴표 필요
                filter_parts.append(f'language = "{filters["language"]}"')
            if "file_path" in filters:
                filter_parts.append(f'file_path = "{filters["file_path"]}"')
            if "node_id" in filters:
                filter_parts.append(f'node_id = "{filters["node_id"]}"')
            if filter_parts:
                filter_str = " AND ".join(filter_parts)

        # 검색 실행
        try:
            results = index.search(
                query,
                {
                    "limit": k,
                    "filter": filter_str,
                    "showRankingScore": True,
                },
            )
        except MeilisearchApiError as e:
            logger.error(f"Search error: {e}")
            return []

        # 결과 변환
        chunk_results = []
        for hit in results.get("hits", []):
            try:
                # original_id가 있으면 사용, 없으면 sanitized ID 사용
                chunk_id = hit.get("original_id", hit["id"])

                chunk_results.append(
                    ChunkResult(
                        repo_id=repo_id,
                        chunk_id=chunk_id,
                        score=hit.get("_rankingScore", 0.0),
                        source="meilisearch",
                        file_path=hit["file_path"],
                        span=tuple(hit["span"]),
                    )
                )
            except (KeyError, TypeError) as e:
                logger.warning(f"Failed to parse search result: {e}")
                continue

        logger.debug(f"Found {len(chunk_results)} results for query: {query}")
        return chunk_results

    def delete_repo_index(self, repo_id: RepoId) -> None:
        """저장소 인덱스 삭제"""
        index_name = self._get_index_name(repo_id)
        try:
            task = self.client.delete_index(index_name)
            task_uid = task.task_uid if hasattr(task, "task_uid") else task.uid
            self.client.wait_for_task(task_uid)
            logger.info(f"Deleted index: {index_name}")
        except MeilisearchApiError as e:
            if e.code == "index_not_found":
                logger.debug(f"Index already deleted: {index_name}")
            else:
                logger.error(f"Failed to delete index {index_name}: {e}")
                raise
