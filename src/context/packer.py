"""LLM 컨텍스트 패커"""

import logging
from typing import List, Optional

from ..core.models import (
    Candidate,
    CodeChunk,
    CodeNode,
    LocationContext,
    PackedContext,
    PackedSnippet,
    RepoId,
)
from ..core.ports import ChunkStorePort, GraphStorePort

logger = logging.getLogger(__name__)


class ContextPacker:
    """
    LLM 컨텍스트 패커
    
    역할:
    - Primary snippet (최우선 코드)
    - Supporting snippets (주변 컨텍스트)
      - Caller: 이 코드를 호출하는 곳
      - Callee: 이 코드가 호출하는 곳
      - Type: 사용하는 타입 정의
      - Related: 관련 코드
    - 토큰 제한 내 최적화
    """

    def __init__(
        self,
        chunk_store: ChunkStorePort,
        graph_store: GraphStorePort,
    ):
        """
        Args:
            chunk_store: 청크 스토어
            graph_store: 그래프 스토어
        """
        self.chunk_store = chunk_store
        self.graph_store = graph_store

    def pack(
        self,
        candidates: List[Candidate],
        max_tokens: int,
        location_ctx: Optional[LocationContext] = None,
    ) -> PackedContext:
        """
        후보를 토큰 제한 내에서 컨텍스트로 패킹
        
        Args:
            candidates: Candidate 리스트 (이미 랭킹됨)
            max_tokens: 최대 토큰 수
            location_ctx: 위치 컨텍스트 (선택적)
        
        Returns:
            PackedContext (primary + supporting snippets)
        """
        if not candidates:
            raise ValueError("No candidates to pack")

        logger.debug(f"Packing {len(candidates)} candidates (max_tokens={max_tokens})")
        
        # 1. Primary snippet 선택 (첫 번째 candidate)
        primary_candidate = candidates[0]
        primary_chunk = self.chunk_store.get_chunk(
            primary_candidate.repo_id,
            primary_candidate.chunk_id
        )
        
        if not primary_chunk:
            raise ValueError(f"Primary chunk not found: {primary_candidate.chunk_id}")
        
        primary = PackedSnippet(
            repo_id=primary_chunk.repo_id,
            file_path=primary_chunk.file_path,
            span=primary_chunk.span,
            role="primary",
            text=primary_chunk.text,
            meta={
                "chunk_id": primary_chunk.id,
                "node_id": primary_chunk.node_id,
                "features": primary_candidate.features
            }
        )
        
        # Primary 토큰 카운트
        primary_tokens = self._estimate_tokens(primary.text)
        remaining_tokens = max_tokens - primary_tokens
        
        logger.debug(f"Primary: {primary_tokens} tokens, Remaining: {remaining_tokens}")
        
        # 2. Supporting snippets 추가
        supporting = []
        
        if remaining_tokens > 100:  # 최소 100 토큰 여유가 있을 때만
            # 2-1. 나머지 candidates를 supporting으로
            for candidate in candidates[1:]:
                chunk = self.chunk_store.get_chunk(
                    candidate.repo_id,
                    candidate.chunk_id
                )
                
                if not chunk:
                    continue
                
                chunk_tokens = self._estimate_tokens(chunk.text)
                
                # 토큰 제한 체크
                if remaining_tokens - chunk_tokens < 50:  # 최소 50 토큰 여유 유지
                    break
                
                supporting.append(
                    PackedSnippet(
                        repo_id=chunk.repo_id,
                        file_path=chunk.file_path,
                        span=chunk.span,
                        role="related",
                        text=chunk.text,
                        meta={
                            "chunk_id": chunk.id,
                            "node_id": chunk.node_id,
                            "features": candidate.features
                        }
                    )
                )
                
                remaining_tokens -= chunk_tokens
            
            # 2-2. 그래프 기반 관련 코드 추가 (caller/callee)
            if location_ctx and remaining_tokens > 200:
                graph_snippets = self._get_graph_context(
                    primary_chunk.repo_id,
                    primary_chunk.node_id,
                    remaining_tokens // 2  # 남은 토큰의 절반만 사용
                )
                supporting.extend(graph_snippets)
        
        logger.info(
            f"Packed context: primary={primary_tokens}t, "
            f"supporting={len(supporting)} snippets"
        )
        
        return PackedContext(
            primary=primary,
            supporting=supporting
        )

    def _get_graph_context(
        self,
        repo_id: RepoId,
        node_id: str,
        max_tokens: int
    ) -> List[PackedSnippet]:
        """
        그래프 기반 관련 코드 조회
        
        Args:
            repo_id: 저장소 ID
            node_id: 노드 ID
            max_tokens: 최대 토큰 수
        
        Returns:
            PackedSnippet 리스트 (caller, callee 등)
        """
        snippets = []
        remaining_tokens = max_tokens
        
        try:
            # 이웃 노드 조회 (1-hop)
            neighbors = self.graph_store.neighbors(
                repo_id,
                node_id,
                edge_types=["calls", "called_by", "uses"]
            )
            
            for neighbor in neighbors[:5]:  # 최대 5개
                if remaining_tokens < 100:
                    break
                
                # 노드 → 청크 매핑 (간단하게 node_id 기반 조회)
                chunks = self.chunk_store.get_chunks_by_node(repo_id, neighbor.id)
                
                if not chunks:
                    continue
                
                chunk = chunks[0]  # 첫 번째 청크 사용
                chunk_tokens = self._estimate_tokens(chunk.text)
                
                if chunk_tokens > remaining_tokens:
                    continue
                
                # Role 결정 (edge type 기반)
                role = "caller"  # 기본값
                # TODO: edge type에 따라 role 결정
                
                snippets.append(
                    PackedSnippet(
                        repo_id=chunk.repo_id,
                        file_path=chunk.file_path,
                        span=chunk.span,
                        role=role,
                        text=chunk.text,
                        meta={
                            "chunk_id": chunk.id,
                            "node_id": chunk.node_id,
                            "relation": "graph_neighbor"
                        }
                    )
                )
                
                remaining_tokens -= chunk_tokens
        
        except Exception as e:
            logger.warning(f"Failed to get graph context: {e}")
        
        return snippets

    def _estimate_tokens(self, text: str) -> int:
        """
        토큰 수 추정
        
        간단한 방식: 4글자 = 1토큰
        정확한 방식은 tiktoken 사용
        """
        # 간단한 추정
        return len(text) // 4
    
    def _count_tokens_tiktoken(self, text: str) -> int:
        """
        tiktoken을 사용한 정확한 토큰 카운팅
        
        Optional: tiktoken 설치 시 사용
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 인코딩
            return len(encoding.encode(text))
        except ImportError:
            # tiktoken이 없으면 간단한 추정 사용
            return self._estimate_tokens(text)

