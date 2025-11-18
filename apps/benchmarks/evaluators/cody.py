"""Cody (Sourcegraph) 평가기"""

import os

import requests

from .metrics import SearchResult, time_search


class CodyEvaluator:
    """Sourcegraph API를 통한 Cody 평가기"""

    def __init__(self, api_token: str | None = None, endpoint: str | None = None):
        """
        Args:
            api_token: Sourcegraph API 토큰 (None이면 환경변수에서 로드)
            endpoint: API 엔드포인트 (기본: sourcegraph.com)
        """
        self.api_token = api_token or os.getenv("SOURCEGRAPH_TOKEN")
        if not self.api_token:
            raise ValueError(
                "Sourcegraph API token is required. "
                "Set SOURCEGRAPH_TOKEN env var or pass api_token."
            )

        self.endpoint = endpoint or "https://sourcegraph.com/.api/graphql"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {self.api_token}",
            "Content-Type": "application/json"
        })

    def search(
        self,
        query: str,
        repo: str,
        k: int = 5
    ) -> SearchResult:
        """
        Sourcegraph 검색 실행

        Args:
            query: 검색 쿼리
            repo: 저장소 이름 (예: github.com/owner/repo)
            k: 반환할 결과 수

        Returns:
            SearchResult (파일 경로 리스트 + 지연시간)
        """
        gql_query = """
        query Search($query: String!) {
          search(query: $query) {
            results {
              results {
                ... on FileMatch {
                  file {
                    path
                    url
                  }
                  lineMatches {
                    lineNumber
                    preview
                  }
                }
              }
            }
            matchCount
          }
        }
        """

        # repo: 필터 추가
        full_query = f"{query} repo:{repo} count:{k}"

        variables = {"query": full_query}

        # 검색 실행 및 시간 측정
        def _search():
            response = self.session.post(
                self.endpoint,
                json={"query": gql_query, "variables": variables},
                timeout=30
            )
            response.raise_for_status()
            return response.json()

        data, latency_ms = time_search(_search)

        # 결과 파싱
        file_paths = []
        try:
            results = data.get("data", {}).get("search", {}).get("results", {}).get("results", [])
            for result in results[:k]:
                if "file" in result:
                    file_paths.append(result["file"]["path"])
        except Exception as e:
            print(f"Error parsing Sourcegraph response: {e}")

        return SearchResult(
            query=query,
            results=file_paths,
            latency_ms=latency_ms,
            retriever_name="cody"
        )

    def batch_search(
        self,
        queries: list[str],
        repo: str,
        k: int = 5
    ) -> list[SearchResult]:
        """
        배치 검색

        Args:
            queries: 쿼리 리스트
            repo: 저장소 이름
            k: 결과 수

        Returns:
            SearchResult 리스트
        """
        results = []
        for query in queries:
            result = self.search(query, repo, k)
            results.append(result)
        return results

