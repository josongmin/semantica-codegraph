"""인덱싱 파이프라인"""

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        
        # 성능 최적화: 파서 캐시
        self._parser_cache = {}

    def index_repository(
        self,
        root_path: str,
        repo_id: Optional[RepoId] = None,
        name: Optional[str] = None,
        config: Optional[RepoConfig] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        parallel_threshold: Optional[int] = None,
    ) -> IndexingResult:
        """
        저장소 인덱싱
        
        Args:
            root_path: 저장소 루트 경로
            repo_id: 저장소 ID (None이면 자동 생성)
            name: 저장소 이름 (None이면 디렉토리 이름)
            config: 저장소 설정 (None이면 기본값)
            parallel: 병렬 처리 활성화 (기본값: True)
            max_workers: 병렬 처리 워커 수 (None이면 CPU 코어 수)
            parallel_threshold: 병렬 처리 시작 임계값 (None이면 자동: max(3, cpu_count // 2))
        
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
            failed_files = []  # 실패한 파일 추적
            
            # 병렬 처리 여부 결정
            use_parallel = parallel and hasattr(self, 'config') and getattr(self.config, 'parallel_indexing_enabled', True)
            
            # 병렬 처리 임계값 계산
            if parallel_threshold is None:
                parallel_threshold = max(3, multiprocessing.cpu_count() // 2)
            
            if use_parallel and len(files) > parallel_threshold:
                # 병렬 파싱
                logger.info(f"Using parallel parsing (threshold={parallel_threshold}, files={len(files)})")
                all_nodes, all_edges, failed_files = self._parse_files_parallel(
                    repo_id, files, max_workers
                )
            else:
                # 순차 파싱 (기존 방식)
                logger.info("Using sequential parsing")
                for i, file_meta in enumerate(files, 1):
                    try:
                        logger.debug(f"Processing [{i}/{len(files)}]: {file_meta.file_path}")
                        
                        # 파싱 (캐시된 파서 사용)
                        raw_symbols, raw_relations = self._parse_file(repo_id, file_meta)
                        
                        # IR 변환
                        file_content = self._read_file(file_meta.abs_path)
                        nodes, edges = self.ir_builder.build(
                            raw_symbols=raw_symbols,
                            raw_relations=raw_relations,
                            source_code={file_meta.file_path: file_content}
                        )
                        
                        all_nodes.extend(nodes)
                        all_edges.extend(edges)
                        
                        # 진행률 업데이트 최적화: 매 10개 파일 또는 10%마다
                        if i % 10 == 0 or i == len(files) or (i / len(files)) % 0.1 < (1 / len(files)):
                            progress = (i / len(files)) * 0.5
                            self.repo_store.update_indexing_status(
                                repo_id, "indexing", progress=progress
                            )
                    
                    except Exception as e:
                        logger.error(f"Failed to parse {file_meta.file_path}: {e}")
                        failed_files.append((file_meta.file_path, str(e)))
                        continue
            
            logger.info(f"Parsed {len(all_nodes)} nodes, {len(all_edges)} edges")
            
            # 실패한 파일 로깅
            if failed_files:
                logger.warning(f"Failed to parse {len(failed_files)} files:")
                for file_path, error in failed_files[:10]:  # 최대 10개만 출력
                    logger.warning(f"  - {file_path}: {error}")
                if len(failed_files) > 10:
                    logger.warning(f"  ... and {len(failed_files) - 10} more")
            
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
            
            # 10. 임베딩 생성 및 저장 (배치 병렬화)
            if chunks:
                try:
                    logger.info("Generating embeddings...")
                    vectors = self._embed_chunks_in_batches(chunks, repo_id)
                    
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
        """파일 파싱 (캐시된 파서 사용)"""
        # 파서 캐싱: 언어별로 재사용
        if file_meta.language not in self._parser_cache:
            parser = create_parser(file_meta.language)
            if parser is None:
                logger.warning(f"No parser for {file_meta.language}")
                return [], []
            self._parser_cache[file_meta.language] = parser
        else:
            parser = self._parser_cache[file_meta.language]
        
        return parser.parse_file({
            "repo_id": repo_id,
            "path": file_meta.file_path,  # 상대 경로 (base.py에서 사용)
            "file_path": file_meta.file_path,  # 호환성을 위해 유지
            "abs_path": file_meta.abs_path,  # 절대 경로
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
    
    def _embed_chunks_in_batches(
        self,
        chunks: List[CodeChunk],
        repo_id: RepoId,
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        청크를 배치로 나눠 임베딩 생성 (진행률 업데이트 포함)
        
        Args:
            chunks: 청크 리스트
            repo_id: 저장소 ID
            batch_size: 배치 크기 (API rate limit 고려)
        
        Returns:
            임베딩 벡터 리스트
        """
        all_vectors = []
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_vectors = self.embedding_service.embed_chunks(batch)
            all_vectors.extend(batch_vectors)
            
            # 진행률 업데이트 (70-100%)
            progress = 0.7 + (min(i + batch_size, total_chunks) / total_chunks) * 0.3
            self.repo_store.update_indexing_status(
                repo_id, "indexing", progress=progress
            )
            
            logger.debug(
                f"Embedded batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} "
                f"({min(i + batch_size, total_chunks)}/{total_chunks} chunks)"
            )
        
        return all_vectors
    
    def _parse_files_parallel(
        self,
        repo_id: RepoId,
        files: List[FileMetadata],
        max_workers: Optional[int] = None,
    ) -> Tuple[List[CodeNode], List[CodeEdge], List[Tuple[str, str]]]:
        """
        파일들을 병렬로 파싱
        
        Args:
            repo_id: 저장소 ID
            files: 파일 메타데이터 리스트
            max_workers: 워커 수 (None이면 CPU 코어 수)
        
        Returns:
            (모든 노드, 모든 엣지, 실패한 파일 리스트)
        """
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        all_nodes = []
        all_edges = []
        failed_files = []
        
        # ProcessPoolExecutor 사용 (CPU 집약적 작업)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 각 파일에 대한 future 생성
            future_to_file = {
                executor.submit(_parse_file_worker, repo_id, file_meta): file_meta
                for file_meta in files
            }
            
            # 완료된 작업부터 처리
            for i, future in enumerate(as_completed(future_to_file), 1):
                file_meta = future_to_file[future]
                try:
                    nodes, edges, error = future.result()
                    
                    if error:
                        # 워커에서 에러 발생
                        logger.error(f"Failed to parse {file_meta.file_path}: {error}")
                        failed_files.append((file_meta.file_path, error))
                    else:
                        all_nodes.extend(nodes)
                        all_edges.extend(edges)
                        logger.debug(f"Parsed [{i}/{len(files)}]: {file_meta.file_path}")
                    
                    # 진행률 업데이트 최적화: 매 10개 파일 또는 10%마다
                    if i % 10 == 0 or i == len(files) or (i / len(files)) % 0.1 < (1 / len(files)):
                        progress = (i / len(files)) * 0.5
                        self.repo_store.update_indexing_status(
                            repo_id, "indexing", progress=progress
                        )
                    
                except Exception as e:
                    logger.error(f"Failed to get result for {file_meta.file_path}: {e}")
                    failed_files.append((file_meta.file_path, str(e)))
        
        return all_nodes, all_edges, failed_files


def _parse_file_worker(
    repo_id: RepoId,
    file_meta: FileMetadata
) -> Tuple[List[CodeNode], List[CodeEdge], Optional[str]]:
    """
    워커 프로세스에서 실행되는 파일 파싱 함수
    
    Note: multiprocessing을 위해 top-level 함수로 정의
    
    Returns:
        (노드 리스트, 엣지 리스트, 에러 메시지 or None)
    """
    try:
        # 파싱
        parser = create_parser(file_meta.language)
        if parser is None:
            return [], [], f"No parser for {file_meta.language}"
        
        raw_symbols, raw_relations = parser.parse_file({
            "repo_id": repo_id,
            "path": file_meta.file_path,
            "file_path": file_meta.file_path,
            "abs_path": file_meta.abs_path,
            "language": file_meta.language
        })
        
        # 파일 읽기
        try:
            with open(file_meta.abs_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception as read_error:
            file_content = ""
            # 파일 읽기 실패는 계속 진행 (빈 문자열로)
        
        # IR 변환
        from ..graph.ir_builder import IRBuilder
        ir_builder = IRBuilder()
        nodes, edges = ir_builder.build(
            raw_symbols=raw_symbols,
            raw_relations=raw_relations,
            source_code={file_meta.file_path: file_content}
        )
        
        return nodes, edges, None  # 성공
    
    except Exception as e:
        # 에러 정보를 반환 (로깅은 메인 프로세스에서)
        return [], [], str(e)

