"""의존성 주입 및 포트 초기화"""

from typing import Optional

from meilisearch import Client

from .config import Config
from .enums import LexicalSearchBackend
from .ports import ChunkStorePort
from ..search.ports.lexical_search_port import LexicalSearchPort
from ..search.lexical.meili_adapter import MeiliSearchAdapter
from ..search.lexical.zoekt_adapter import ZoektAdapter


class Bootstrap:
    """포트 인스턴스 생성 및 의존성 주입"""

    def __init__(self, config: Config, chunk_store: Optional[ChunkStorePort] = None):
        self.config = config
        self._chunk_store = chunk_store
        self._lexical_search: LexicalSearchPort | None = None

    @property
    def lexical_search(self) -> LexicalSearchPort:
        """Lexical 검색 포트 인스턴스 반환"""
        if self._lexical_search is None:
            self._lexical_search = self._create_lexical_search()
        return self._lexical_search

    def _create_lexical_search(self) -> LexicalSearchPort:
        """Lexical 검색 포트 생성"""
        backend = self.config.lexical_search_backend

        if backend == LexicalSearchBackend.MEILISEARCH:
            client = Client(
                self.config.meilisearch_url,
                api_key=self.config.meilisearch_master_key,
            )
            return MeiliSearchAdapter(client)
        elif backend == LexicalSearchBackend.ZOEKT:
            # Zoekt는 chunk 매핑을 위해 ChunkStore 필요
            if self._chunk_store is None:
                raise ValueError(
                    "ZoektAdapter requires ChunkStore for chunk mapping. "
                    "Please provide chunk_store when creating Bootstrap."
                )
            return ZoektAdapter(
                self.config.zoekt_url,
                chunk_store=self._chunk_store,
                timeout=self.config.zoekt_timeout,
            )
        else:
            raise ValueError(f"Unknown lexical search backend: {backend}")


def create_bootstrap(
    config: Optional[Config] = None,
    chunk_store: Optional[ChunkStorePort] = None,
) -> Bootstrap:
    """
    Bootstrap 인스턴스 생성
    
    Args:
        config: 애플리케이션 설정 (None이면 환경변수에서 로드)
        chunk_store: ChunkStore 구현체 (Zoekt 사용 시 필수)
    
    Returns:
        Bootstrap 인스턴스
    
    Note:
        Zoekt 백엔드를 사용할 경우, chunk_store는 필수입니다.
        파일:라인 기반 Zoekt 결과를 CodeChunk로 매핑하기 위해 필요합니다.
    """
    if config is None:
        config = Config.from_env()
    return Bootstrap(config, chunk_store)

