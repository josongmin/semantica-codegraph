"""Zoekt 검색 어댑터 with Chunk 매핑"""

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

from ....core.models import ChunkResult, CodeChunk, RepoId
from ....core.ports import ChunkStorePort
from ...ports.lexical_search_port import LexicalSearchPort

logger = logging.getLogger(__name__)


class ZoektAdapter(LexicalSearchPort):
    """
    Zoekt를 사용한 코드 검색 어댑터

    핵심 차이점:
    - MeiliSearch: chunk 기반 인덱스 → 직접 매핑
    - Zoekt: 파일/라인 기반 인덱스 → chunk 매핑 레이어 필요

    Zoekt는 trigram 인덱싱으로 파일:라인 결과를 반환합니다.
    이를 Semantica의 CodeChunk로 변환하기 위해 ChunkStore와 연동합니다.

    검색 흐름:
    1. Zoekt에서 파일:라인 검색
    2. ChunkStore에서 해당 위치의 CodeChunk 조회
    3. ChunkResult로 변환 반환
    """

    def __init__(
        self,
        zoekt_url: str,
        chunk_store: ChunkStorePort,
        timeout: int = 30,
    ):
        """
        Args:
            zoekt_url: Zoekt 서버 URL (예: http://localhost:7713)
            chunk_store: CodeChunk 조회를 위한 저장소
            timeout: HTTP 요청 타임아웃 (초)
        """
        self.zoekt_url = zoekt_url.rstrip("/")
        self.chunk_store = chunk_store
        self.timeout = timeout
        self._session_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _http_request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
    ) -> dict[str, Any]:
        """HTTP 요청 실행"""
        url = urljoin(self.zoekt_url, endpoint)

        try:
            req_data = json.dumps(data).encode("utf-8") if data else None

            request = Request(
                url,
                data=req_data,
                headers=self._session_headers,
                method=method,
            )

            with urlopen(request, timeout=self.timeout) as response:
                response_data = response.read().decode("utf-8")
                return json.loads(response_data) if response_data else {}

        except HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            logger.error(f"Zoekt HTTP error {e.code}: {error_body}")
            raise
        except URLError as e:
            logger.error(f"Zoekt connection error: {e.reason}")
            raise
        except Exception as e:
            logger.error(f"Zoekt request failed: {e}")
            raise

    def index_chunks(self, chunks: list[CodeChunk]) -> None:
        """
        청크 인덱싱

        Warning: Zoekt는 Git 저장소 단위로 인덱싱하는 시스템이므로,
                청크 단위 인덱싱은 비표준 방식입니다.

        대안:
        1. Zoekt 서버에 Git 저장소 URL 등록 (zoekt-mirror-github 등 사용)
        2. 커스텀 문서 저장소로 파일 생성 후 인덱싱
        3. 다른 검색 엔진 사용 (MeiliSearch 권장)

        현재 구현: 로그만 남기고 skip
        """
        if not chunks:
            logger.warning("No chunks to index")
            return

        repo_id = chunks[0].repo_id
        logger.warning(
            f"Zoekt는 Git 저장소 단위 인덱싱을 지원합니다. "
            f"repo_id={repo_id}의 청크 {len(chunks)}개를 인덱싱하려면 "
            f"Zoekt 서버에 Git 저장소를 직접 등록해주세요."
        )

        # TODO: Git 저장소 등록 API가 있다면 여기서 호출
        # 예: self._http_request("POST", "/api/index", {"repo": repo_url})

    def _find_chunk_by_location(
        self,
        repo_id: RepoId,
        file_path: str,
        line: int,
    ) -> CodeChunk | None:
        """
        파일:라인 위치로 해당하는 CodeChunk 찾기

        ChunkStore를 통해 해당 위치를 포함하는 chunk를 조회합니다.
        """
        # TODO: ChunkStore에 위치 기반 조회 메서드 추가 필요
        # 현재는 임시로 fallback 처리
        logger.debug(
            f"Looking up chunk at {repo_id}:{file_path}:{line} "
            "(chunk_store location query not implemented yet)"
        )
        return None

    def search(
        self,
        repo_id: RepoId,
        query: str,
        k: int,
        filters: dict | None = None,
    ) -> list[ChunkResult]:
        """
        검색 실행 with Chunk 매핑

        Zoekt 검색 흐름:
        1. Zoekt에 검색 쿼리 전송
        2. 파일:라인 기반 결과 수신
        3. 각 결과를 CodeChunk로 매핑
        4. ChunkResult 반환

        Zoekt 쿼리 문법:
        - 단순 텍스트: "function"
        - 정규식: "r:func.*"
        - 파일 필터: "file:*.py"
        - 저장소 필터: "repo:myrepo"
        """
        # 쿼리 구성
        search_query = query

        # 저장소 필터 추가
        if repo_id:
            search_query = f"repo:{repo_id} {query}"

        # 추가 필터 적용
        if filters:
            if "language" in filters:
                # 언어별 파일 확장자 매핑
                lang_ext_map = {
                    "python": "*.py",
                    "javascript": "*.js",
                    "typescript": "*.ts",
                    "go": "*.go",
                    "java": "*.java",
                }
                ext = lang_ext_map.get(filters["language"], f"*.{filters['language']}")
                search_query = f"file:{ext} {search_query}"

            if "file_path" in filters:
                search_query = f"file:{filters['file_path']} {search_query}"

        # 검색 요청 (URL 인코딩)
        try:
            encoded_query = quote(search_query)
            response = self._http_request(
                "GET",
                f"/search?q={encoded_query}&num={k * 2}",  # 매핑 실패 대비 2배
            )
        except Exception as e:
            logger.error(f"Zoekt search failed: {e}")
            return []

        # Zoekt JSON 응답 파싱
        chunk_results = []

        # Zoekt 응답 구조:
        # {
        #   "SearchResult": {
        #     "Files": [{
        #       "FileName": "path/to/file.py",
        #       "Repository": "repo-name",
        #       "Branches": ["main"],
        #       "LineMatches": [{
        #         "LineNumber": 42,
        #         "Line": "def function():",
        #         "LineFragments": [{
        #           "LineOffset": 4,
        #           "MatchLength": 8,
        #           "Offset": 100
        #         }]
        #       }],
        #       "ChunkMatches": [...],
        #       "Score": 123.45
        #     }],
        #     "Stats": {...}
        #   }
        # }
        search_result = response.get("SearchResult", {})
        files = search_result.get("Files", [])

        logger.debug(f"Zoekt returned {len(files)} file matches")

        for file_result in files:
            try:
                # 파일 정보 추출
                file_path = file_result.get("FileName", "")
                repo_name = file_result.get("Repository", repo_id)
                zoekt_score = file_result.get("Score", 0.0)

                # LineMatches에서 매칭된 라인 정보 추출
                line_matches = file_result.get("LineMatches", [])
                if not line_matches:
                    logger.debug(f"No line matches in {file_path}")
                    continue

                # 각 라인 매치를 CodeChunk로 매핑
                for line_match in line_matches:
                    line_number = line_match.get("LineNumber", 1)
                    line_match.get("Line", "")
                    line_fragments = line_match.get("LineFragments", [])

                    # Fragment 개수로 relevance 계산
                    fragment_score = len(line_fragments) if line_fragments else 1

                    # ChunkStore에서 해당 위치의 Chunk 조회
                    chunk = self._find_chunk_by_location(
                        repo_name,
                        file_path,
                        line_number,
                    )

                    if chunk:
                        # Chunk 찾음 → 정확한 매핑
                        chunk_results.append(
                            ChunkResult(
                                repo_id=chunk.repo_id,
                                chunk_id=chunk.id,
                                score=zoekt_score * fragment_score,
                                source="zoekt",
                                file_path=chunk.file_path,
                                span=chunk.span,
                            )
                        )
                    else:
                        # Chunk 없음 → Fallback (파일:라인 기반 임시 chunk)
                        logger.debug(
                            f"Chunk not found for {file_path}:{line_number}, "
                            "creating fallback ChunkResult"
                        )
                        chunk_results.append(
                            ChunkResult(
                                repo_id=repo_name,
                                chunk_id=f"zoekt:{repo_name}:{file_path}:{line_number}",
                                score=zoekt_score * fragment_score,
                                source="zoekt",
                                file_path=file_path,
                                span=(line_number, 0, line_number + 1, 0),
                            )
                        )

                    # k개 수집했으면 중단
                    if len(chunk_results) >= k:
                        break

                if len(chunk_results) >= k:
                    break

            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Failed to parse Zoekt search result: {e}")
                continue

        logger.info(
            f"Zoekt search completed: {len(files)} files, {len(chunk_results)} chunks mapped"
        )
        return chunk_results[:k]

    def delete_repo_index(self, repo_id: RepoId) -> None:
        """
        저장소 인덱스 삭제

        Warning: Zoekt는 저장소 삭제 API를 표준으로 제공하지 않습니다.
                서버 측에서 인덱스 디렉토리를 직접 삭제해야 합니다.

        TODO: Zoekt 서버 관리 API가 있다면 여기서 호출
        """
        logger.warning(
            f"Zoekt는 저장소 삭제 API를 제공하지 않습니다. "
            f"repo_id={repo_id}의 인덱스를 삭제하려면 "
            f"Zoekt 서버의 인덱스 디렉토리에서 직접 삭제해주세요."
        )

        # TODO: 관리 API 구현 시 호출
        # self._http_request("DELETE", f"/api/repos/{repo_id}")
