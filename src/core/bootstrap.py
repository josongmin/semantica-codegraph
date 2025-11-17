"""의존성 주입 및 포트 초기화"""

from typing import Optional

from meilisearch import Client

from .config import Config
from .enums import LexicalSearchBackend
from ..search.ports.lexical_search_port import LexicalSearchPort
from ..search.lexical.meili_adapter import MeiliSearchAdapter
from ..search.lexical.zoekt_adapter import ZoektAdapter


class Bootstrap:
    """포트 인스턴스 생성 및 의존성 주입"""

    def __init__(self, config: Config):
        self.config = config
        self._lexical_search: LexicalSearchPort | None = None

    @property
    def lexical_search(self) -> LexicalSearchPort:
        """Lexical 검색 포트 인스턴스 반환"""
        if self._lexical_search is None:
            self._lexical_search = self._create_lexical_search()
        return self._lexical_search

    def _create_lexical_search(self) -> LexicalSearchPort:
        """BM25 검색 포트 생성"""
        backend = self.config.lexical_search_backend

        if backend == LexicalSearchBackend.MEILISEARCH:
            client = Client(
                self.config.meilisearch_url,
                api_key=self.config.meilisearch_master_key,
            )
            return MeiliSearchAdapter(client)
        elif backend == LexicalSearchBackend.ZOEKT:
            return ZoektAdapter(self.config.zoekt_url)
        else:
            raise ValueError(f"Unknown lexical search backend: {backend}")


def create_bootstrap(config: Optional[Config] = None) -> Bootstrap:
    """Bootstrap 인스턴스 생성"""
    if config is None:
        config = Config.from_env()
    return Bootstrap(config)

