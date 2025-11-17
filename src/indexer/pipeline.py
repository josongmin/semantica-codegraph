"""인덱싱 파이프라인"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

from ..chunking.chunker import Chunker
from ..core.models import (
    CodeChunk,
    CodeEdge,
    CodeNode,
    FileMetadata,
    IndexingResult,
    RawRelation,
    RawSymbol,
    RepoConfig,
    RepoId,
    RepoMetadata,
)
from ..core.ports import ChunkStorePort, EmbeddingStorePort, GraphStorePort
from ..core.repo_store import RepoMetadataStore
from ..embedding.service import EmbeddingService
from ..graph.ir_builder import IRBuilder
from ..parser import create_parser
from ..search.ports.lexical_search_port import LexicalSearchPort
from .repo_scanner import RepoScanner

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """
    인덱싱 파이프라인
    
    전체 플로우:
    1. 파일 스캔
    2. 파싱 (Parser)
    3. IR 변환 (IRBuilder)
    4. 그래프 저장 (GraphStore)
    5. 청킹 (Chunker)
    6. 청크 저장 (ChunkStore)
    7. Lexical 인덱싱 (MeiliSearch/Zoekt)
    8. 임베딩 생성 (EmbeddingService)
    9. 임베딩 저장 (EmbeddingStore)
    """

    def __init__(
        self,
        repo_store: RepoMetadataStore,
        graph_store: GraphStorePort,
        chunk_store: ChunkStorePort,
        embedding_service: EmbeddingService,
        embedding_store: EmbeddingStorePort,
        lexical_search: LexicalSearchPort,
        ir_builder: IRBuilder,
        chunker: Chunker,
        scanner: RepoScanner,
    ):
        """
        Args:
            repo_store: 저장소 메타데이터 스토어
            graph_store: 코드 그래프 스토어
            chunk_store: 청크 스토어
            embedding_service: 임베딩 서비스
            embedding_store: 임베딩 스토어
            lexical_search: Lexical 검색 포트
            ir_builder: IR 빌더
            chunker: 청커
            scanner: 저장소 스캐너
        """
        self.repo_store = repo_store
        self.graph_store = graph_store
        self.chunk_store = chunk_store
        self.embedding_service = embedding_service
        self.embedding_store = embedding_store
        self.lexical_search = lexical_search
        self.ir_builder = ir_builder
        self.chunker = chunker
        self.scanner = scanner

    def index_repository(
        self,
        root_path: str,
        repo_id: Optional[RepoId] = None,
        name: Optional[str] = None,
        config: Optional[RepoConfig] = None,
    ) -> IndexingResult:
        """
        저장소 인덱싱
        
        Args:
            root_path: 저장소 루트 경로
            repo_id: 저장소 ID (None이면 자동 생성)
            name: 저장소 이름 (None이면 디렉토리 이름)
            config: 저장소 설정 (None이면 기본값)
        
        Returns:
            IndexingResult
        """
        start_time = time.time()
        
        # 1. Repo ID 생성
        if repo_id is None:
            repo_id = self._generate_repo_id(root_path)
        
        if name is None:
            name = Path(root_path).name
        
        logger.info(f"Starting indexing: {repo_id} ({root_path})")
        
        try:
            # 2. 메타데이터 생성 및 저장
            metadata = RepoMetadata(
                repo_id=repo_id,
                name=name,
                root_path=str(Path(root_path).resolve())
            )
            self.repo_store.save(metadata)
            
            # 3. 상태: indexing 시작
            self.repo_store.update_indexing_status(repo_id, "indexing", progress=0.0)
            
            # 4. 파일 스캔
            files = self.scanner.scan(root_path, config)
            logger.info(f"Found {len(files)} files to index")
            
            if len(files) == 0:
                logger.warning("No files found to index")
                return IndexingResult(
                    repo_id=repo_id,
                    status="completed",
                    total_files=0,
                    duration_seconds=time.time() - start_time
                )
            
            # 5. 파일별 파싱 + IR 변환 + 그래프 저장
            all_nodes = []
            all_edges = []
            
            for i, file_meta in enumerate(files, 1):
                try:
                    logger.debug(f"Processing [{i}/{len(files)}]: {file_meta.file_path}")
                    
                    # 파싱
                    raw_symbols, raw_relations = self._parse_file(repo_id, file_meta)
                    
                    # IR 변환
                    nodes, edges = self.ir_builder.build(
                        file_content=self._read_file(file_meta.abs_path),
                        raw_symbols=raw_symbols,
                        raw_relations=raw_relations
                    )
                    
                    all_nodes.extend(nodes)
                    all_edges.extend(edges)
                    
                    # 진행률 업데이트 (파싱 단계: 0-50%)
                    progress = (i / len(files)) * 0.5
                    self.repo_store.update_indexing_status(
                        repo_id, "indexing", progress=progress
                    )
                
                except Exception as e:
                    logger.error(f"Failed to parse {file_meta.file_path}: {e}")
                    continue
            
            logger.info(f"Parsed {len(all_nodes)} nodes, {len(all_edges)} edges")
            
            # 6. 그래프 저장
            if all_nodes:
                self.graph_store.save_graph(all_nodes, all_edges)
                logger.info("Saved graph to database")
            
            # 7. 청킹
            chunks = self.chunker.chunk(all_nodes)
            logger.info(f"Generated {len(chunks)} chunks")
            
            # 8. 청크 저장
            if chunks:
                self.chunk_store.save_chunks(chunks)
                logger.info("Saved chunks to database")
            
            # 진행률 50% (청킹 완료)
            self.repo_store.update_indexing_status(repo_id, "indexing", progress=0.5)
            
            # 9. Lexical 인덱싱
            if chunks:
                try:
                    self.lexical_search.index_chunks(chunks)
                    logger.info("Indexed chunks in lexical search")
                except Exception as e:
                    logger.error(f"Lexical indexing failed: {e}")
            
            # 진행률 70% (Lexical 완료)
            self.repo_store.update_indexing_status(repo_id, "indexing", progress=0.7)
            
            # 10. 임베딩 생성 및 저장
            if chunks:
                try:
                    logger.info("Generating embeddings...")
                    vectors = self.embedding_service.embed_chunks(chunks)
                    
                    chunk_ids = [chunk.id for chunk in chunks]
                    self.embedding_store.save_embeddings(repo_id, chunk_ids, vectors)
                    logger.info(f"Saved {len(vectors)} embeddings")
                except Exception as e:
                    logger.error(f"Embedding failed: {e}")
            
            # 11. 메타데이터 업데이트
            metadata.total_files = len(files)
            metadata.total_nodes = len(all_nodes)
            metadata.total_chunks = len(chunks)
            metadata.languages = list(set(f.language for f in files))
            self.repo_store.save(metadata)
            
            # 12. 상태: completed
            self.repo_store.update_indexing_status(repo_id, "completed", progress=1.0)
            
            duration = time.time() - start_time
            logger.info(f"Indexing completed in {duration:.2f}s")
            
            return IndexingResult(
                repo_id=repo_id,
                status="completed",
                total_files=len(files),
                processed_files=len(files),
                total_nodes=len(all_nodes),
                total_edges=len(all_edges),
                total_chunks=len(chunks),
                duration_seconds=duration
            )
        
        except Exception as e:
            logger.error(f"Indexing failed: {e}", exc_info=True)
            
            # 상태: failed
            self.repo_store.update_indexing_status(
                repo_id, "failed", error=str(e)
            )
            
            return IndexingResult(
                repo_id=repo_id,
                status="failed",
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def _generate_repo_id(self, root_path: str) -> RepoId:
        """저장소 ID 생성 (간단한 전략: 디렉토리 이름)"""
        return Path(root_path).resolve().name
    
    def _parse_file(
        self,
        repo_id: RepoId,
        file_meta: FileMetadata
    ) -> Tuple[List[RawSymbol], List[RawRelation]]:
        """파일 파싱"""
        parser = create_parser(file_meta.language)
        if parser is None:
            logger.warning(f"No parser for {file_meta.language}")
            return [], []
        
        return parser.parse_file({
            "repo_id": repo_id,
            "file_path": file_meta.file_path,
            "abs_path": file_meta.abs_path,
            "language": file_meta.language
        })
    
    def _read_file(self, abs_path: str) -> str:
        """파일 읽기"""
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {abs_path}: {e}")
            return ""

