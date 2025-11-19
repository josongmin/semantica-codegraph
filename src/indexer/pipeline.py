"""인덱싱 파이프라인"""

import asyncio
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
from ..parser.cache import ParseCache
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
        parse_cache: ParseCache | None = None,
        route_store=None,  # RouteStore (optional)
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
            parse_cache: 파싱 캐시 (None이면 비활성화)
            route_store: Route 인덱스 스토어 (optional)
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
        self.route_store = route_store

        # Route 추출기
        if route_store:
            from .route_extractor import RouteExtractor
            self.route_extractor = RouteExtractor()
        else:
            self.route_extractor = None

        # 성능 최적화: 파서 캐시
        from typing import Any

        self._parser_cache: dict[str, Any] = {}

        # 파싱 결과 캐시
        self.parse_cache = parse_cache if parse_cache else ParseCache()

    def index_repository(
        self,
        root_path: str,
        repo_id: RepoId | None = None,
        name: str | None = None,
        config: RepoConfig | None = None,
        parallel: bool = True,
        max_workers: int | None = None,
        parallel_threshold: int | None = None,
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
                repo_id=repo_id, name=name, root_path=str(Path(root_path).resolve())
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
                    duration_seconds=time.time() - start_time,
                )
            
            # 4-1. Repo Profiling (프로젝트 구조 분석)
            logger.info("[Profiling] Repo profiling 시작...")
            repo_profile = None
            try:
                from .repo_profiler import RepoProfiler
                
                repo_profiler = RepoProfiler()
                repo_profile = repo_profiler.profile_repo(root_path, repo_id)
                self.repo_store.save_profile(repo_profile)
                
                logger.info(f"[Profiling] Repo: framework={repo_profile.framework}, type={repo_profile.project_type}")
                logger.info(f"[Profiling] API dirs: {len(repo_profile.api_directories)}, Test dirs: {len(repo_profile.test_directories)}")
            except Exception as e:
                logger.warning(f"[Profiling] Repo profiling 실패 (계속 진행): {e}")

            # 5. 파일별 파싱 + IR 변환 + 그래프 저장
            all_nodes: list[CodeNode] = []
            all_edges: list[CodeEdge] = []
            failed_files: list[tuple[str, str]] = []  # 실패한 파일 추적
            last_progress_update = 0.0  # 마지막 업데이트 진행률

            # 병렬 처리 여부 결정
            use_parallel = parallel

            # 병렬 처리 임계값 계산
            if parallel_threshold is None:
                parallel_threshold = 5  # 파일 5개부터 병렬 처리

            if use_parallel and len(files) > parallel_threshold:
                # 병렬 파싱
                logger.info(
                    f"Using parallel parsing (threshold={parallel_threshold}, files={len(files)})"
                )
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
                            source_code={file_meta.file_path: file_content},
                        )

                        all_nodes.extend(nodes)
                        all_edges.extend(edges)

                        # 진행률 업데이트 최적화: 5% 단위로만 업데이트
                        progress = (i / len(files)) * 0.5
                        if progress - last_progress_update >= 0.05 or i == len(files):
                            self.repo_store.update_indexing_status(
                                repo_id, "indexing", progress=progress
                            )
                            last_progress_update = progress

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
            
            # 6-1. File Profiling (파일 역할 태깅)
            logger.info("[Profiling] File profiling 시작...")
            file_profiles = []
            try:
                from .file_profiler import FileProfiler
                
                file_profiler = FileProfiler()
                
                for file_meta in files:
                    file_profile = file_profiler.profile_file(
                        repo_id=repo_id,
                        file_path=file_meta.file_path,
                        abs_path=file_meta.abs_path,
                        framework=repo_profile.framework if repo_profile else None,
                    )
                    file_profiles.append(file_profile)
                
                # 배치 저장
                if file_profiles:
                    self.repo_store.save_file_profiles_batch(file_profiles)
                    api_count = sum(1 for p in file_profiles if p.is_api_file)
                    logger.info(f"[Profiling] File: {len(file_profiles)}개 (API: {api_count}개)")
            except Exception as e:
                logger.warning(f"[Profiling] File profiling 실패 (계속 진행): {e}")
            
            # 6-2. Graph Ranking (노드 중요도 계산)
            logger.info("[Profiling] Graph ranking 시작...")
            try:
                if all_nodes:
                    updated_count = self.graph_store.update_all_node_importance(repo_id, batch_size=100)
                    logger.info(f"[Profiling] Graph: {updated_count}개 노드 중요도 계산 완료")
            except Exception as e:
                logger.warning(f"[Profiling] Graph ranking 실패 (계속 진행): {e}")

            # 7. 청킹
            chunks = self.chunker.chunk(all_nodes)
            logger.info(f"Generated {len(chunks)} chunks")

            # 8. 청크 저장
            if chunks:
                self.chunk_store.save_chunks(chunks)
                logger.info("Saved chunks to database")
            
            # 8-1. Chunk Tagging (청크 메타데이터 태깅)
            logger.info("[Profiling] Chunk tagging 시작...")
            try:
                from ..chunking.chunk_tagger import ChunkTagger
                
                chunk_tagger = ChunkTagger()
                metadata_updates = []
                
                # 파일별로 프로파일 매핑
                file_profile_map = {p.file_path: p for p in file_profiles}
                
                for chunk in chunks:
                    file_profile = file_profile_map.get(chunk.file_path)
                    chunk_metadata = chunk_tagger.tag_chunk(chunk.text, file_profile)
                    metadata_updates.append((repo_id, chunk.id, chunk_metadata))
                
                # 배치 업데이트
                if metadata_updates:
                    self.chunk_store.update_chunk_metadata_batch(metadata_updates)
                    api_chunks = sum(1 for _, _, m in metadata_updates if m.get("is_api_endpoint_chunk"))
                    logger.info(f"[Profiling] Chunk: {len(metadata_updates)}개 (API: {api_chunks}개)")
            except Exception as e:
                logger.warning(f"[Profiling] Chunk tagging 실패 (계속 진행): {e}")

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

            # 10. 임베딩 생성 및 저장 (비동기 병렬 + 중복 제거)
            if chunks:
                try:
                    import hashlib

                    logger.info("Generating embeddings (async + dedup)...")
                    vectors = asyncio.run(self._embed_chunks_async(chunks, repo_id))

                    # content_hash 계산
                    content_hashes = []
                    for chunk in chunks:
                        text = self.embedding_service._prepare_chunk_text(chunk)
                        content_hash = hashlib.md5(text.encode()).hexdigest()
                        content_hashes.append(content_hash)

                    chunk_ids = [chunk.id for chunk in chunks]
                    self.embedding_store.save_embeddings(
                        repo_id, chunk_ids, vectors, content_hashes=content_hashes
                    )
                    logger.info(f"Saved {len(vectors)} embeddings")
                except Exception as e:
                    logger.error(f"Embedding failed: {e}")

            # 10.5. Route 추출 및 저장 (API 엔드포인트 인덱싱)
            if self.route_store and self.route_extractor:
                try:
                    logger.info("Extracting API routes...")
                    all_routes = []
                    
                    # API 파일만 처리
                    for file_path, profile in file_profiles.items():
                        if not profile or not profile.is_api_file:
                            continue
                        
                        # 해당 파일의 nodes 가져오기
                        file_nodes = [n for n in all_nodes if n.file_path == file_path]
                        
                        if not file_nodes:
                            continue
                        
                        # Route 추출
                        routes = self.route_extractor.extract_routes(file_nodes, profile)
                        all_routes.extend(routes)
                    
                    # Route 저장
                    if all_routes:
                        self.route_store.save_routes(all_routes)
                        logger.info(f"Indexed {len(all_routes)} API routes")
                    else:
                        logger.info("No API routes found")
                        
                except Exception as e:
                    logger.error(f"Route extraction failed: {e}", exc_info=True)

            # 11. 메타데이터 업데이트
            metadata.total_files = len(files)
            metadata.total_nodes = len(all_nodes)
            metadata.total_chunks = len(chunks)
            metadata.languages = list({f.language for f in files})
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
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Indexing failed: {e}", exc_info=True)

            # 상태: failed
            self.repo_store.update_indexing_status(repo_id, "failed", error=str(e))

            return IndexingResult(
                repo_id=repo_id,
                status="failed",
                error_message=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def index_repository_async(
        self,
        root_path: str,
        repo_id: RepoId | None = None,
        name: str | None = None,
        config: RepoConfig | None = None,
        parallel: bool = True,
        max_workers: int | None = None,
        parallel_threshold: int | None = None,
    ) -> IndexingResult:
        """
        저장소 인덱싱 (Async 환경용: FastAPI 등)

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

        logger.info(f"Starting indexing (async): {repo_id} ({root_path})")

        try:
            # 2. 메타데이터 생성 및 저장
            metadata = RepoMetadata(
                repo_id=repo_id, name=name, root_path=str(Path(root_path).resolve())
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
                    duration_seconds=time.time() - start_time,
                )

            # 5. 파일별 파싱 + IR 변환 + 그래프 저장
            all_nodes: list[CodeNode] = []
            all_edges: list[CodeEdge] = []
            failed_files: list[tuple[str, str]] = []  # 실패한 파일 추적
            last_progress_update = 0.0  # 마지막 업데이트 진행률

            # 병렬 처리 여부 결정
            use_parallel = parallel

            # 병렬 처리 임계값 계산
            if parallel_threshold is None:
                parallel_threshold = 5  # 파일 5개부터 병렬 처리

            if use_parallel and len(files) > parallel_threshold:
                # 병렬 파싱
                logger.info(
                    f"Using parallel parsing (threshold={parallel_threshold}, files={len(files)})"
                )
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
                            source_code={file_meta.file_path: file_content},
                        )

                        all_nodes.extend(nodes)
                        all_edges.extend(edges)

                        # 진행률 업데이트 최적화: 5% 단위로만 업데이트
                        progress = (i / len(files)) * 0.5
                        if progress - last_progress_update >= 0.05 or i == len(files):
                            self.repo_store.update_indexing_status(
                                repo_id, "indexing", progress=progress
                            )
                            last_progress_update = progress

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
            
            # 6-1. File Profiling (파일 역할 태깅)
            logger.info("[Profiling] File profiling 시작...")
            file_profiles = []
            try:
                from .file_profiler import FileProfiler
                
                file_profiler = FileProfiler()
                
                for file_meta in files:
                    file_profile = file_profiler.profile_file(
                        repo_id=repo_id,
                        file_path=file_meta.file_path,
                        abs_path=file_meta.abs_path,
                        framework=repo_profile.framework if repo_profile else None,
                    )
                    file_profiles.append(file_profile)
                
                # 배치 저장
                if file_profiles:
                    self.repo_store.save_file_profiles_batch(file_profiles)
                    api_count = sum(1 for p in file_profiles if p.is_api_file)
                    logger.info(f"[Profiling] File: {len(file_profiles)}개 (API: {api_count}개)")
            except Exception as e:
                logger.warning(f"[Profiling] File profiling 실패 (계속 진행): {e}")
            
            # 6-2. Graph Ranking (노드 중요도 계산)
            logger.info("[Profiling] Graph ranking 시작...")
            try:
                if all_nodes:
                    updated_count = self.graph_store.update_all_node_importance(repo_id, batch_size=100)
                    logger.info(f"[Profiling] Graph: {updated_count}개 노드 중요도 계산 완료")
            except Exception as e:
                logger.warning(f"[Profiling] Graph ranking 실패 (계속 진행): {e}")

            # 7. 청킹
            chunks = self.chunker.chunk(all_nodes)
            logger.info(f"Generated {len(chunks)} chunks")

            # 8. 청크 저장
            if chunks:
                self.chunk_store.save_chunks(chunks)
                logger.info("Saved chunks to database")
            
            # 8-1. Chunk Tagging (청크 메타데이터 태깅)
            logger.info("[Profiling] Chunk tagging 시작...")
            try:
                from ..chunking.chunk_tagger import ChunkTagger
                
                chunk_tagger = ChunkTagger()
                metadata_updates = []
                
                # 파일별로 프로파일 매핑
                file_profile_map = {p.file_path: p for p in file_profiles}
                
                for chunk in chunks:
                    file_profile = file_profile_map.get(chunk.file_path)
                    chunk_metadata = chunk_tagger.tag_chunk(chunk.text, file_profile)
                    metadata_updates.append((repo_id, chunk.id, chunk_metadata))
                
                # 배치 업데이트
                if metadata_updates:
                    self.chunk_store.update_chunk_metadata_batch(metadata_updates)
                    api_chunks = sum(1 for _, _, m in metadata_updates if m.get("is_api_endpoint_chunk"))
                    logger.info(f"[Profiling] Chunk: {len(metadata_updates)}개 (API: {api_chunks}개)")
            except Exception as e:
                logger.warning(f"[Profiling] Chunk tagging 실패 (계속 진행): {e}")

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

            # 10. 임베딩 생성 및 저장 (비동기 병렬 + 중복 제거)
            if chunks:
                try:
                    import hashlib

                    logger.info("Generating embeddings (async + dedup)...")
                    # ✅ await 사용 (asyncio.run 대신)
                    vectors = await self._embed_chunks_async(chunks, repo_id)

                    # content_hash 계산
                    content_hashes = []
                    for chunk in chunks:
                        text = self.embedding_service._prepare_chunk_text(chunk)
                        content_hash = hashlib.md5(text.encode()).hexdigest()
                        content_hashes.append(content_hash)

                    chunk_ids = [chunk.id for chunk in chunks]
                    self.embedding_store.save_embeddings(
                        repo_id, chunk_ids, vectors, content_hashes=content_hashes
                    )
                    logger.info(f"Saved {len(vectors)} embeddings")
                except Exception as e:
                    logger.error(f"Embedding failed: {e}")

            # 10.5. Route 추출 및 저장 (API 엔드포인트 인덱싱)
            if self.route_store and self.route_extractor:
                try:
                    logger.info("Extracting API routes...")
                    all_routes = []
                    
                    # API 파일만 처리
                    for file_path, profile in file_profiles.items():
                        if not profile or not profile.is_api_file:
                            continue
                        
                        # 해당 파일의 nodes 가져오기
                        file_nodes = [n for n in all_nodes if n.file_path == file_path]
                        
                        if not file_nodes:
                            continue
                        
                        # Route 추출
                        routes = self.route_extractor.extract_routes(file_nodes, profile)
                        all_routes.extend(routes)
                    
                    # Route 저장
                    if all_routes:
                        self.route_store.save_routes(all_routes)
                        logger.info(f"Indexed {len(all_routes)} API routes")
                    else:
                        logger.info("No API routes found")
                        
                except Exception as e:
                    logger.error(f"Route extraction failed: {e}", exc_info=True)

            # 11. 메타데이터 업데이트
            metadata.total_files = len(files)
            metadata.total_nodes = len(all_nodes)
            metadata.total_chunks = len(chunks)
            metadata.languages = list({f.language for f in files})
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
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Indexing failed: {e}", exc_info=True)

            # 상태: failed
            self.repo_store.update_indexing_status(repo_id, "failed", error=str(e))

            return IndexingResult(
                repo_id=repo_id,
                status="failed",
                error_message=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _generate_repo_id(self, root_path: str) -> RepoId:
        """저장소 ID 생성 (간단한 전략: 디렉토리 이름)"""
        return Path(root_path).resolve().name

    def _parse_file(
        self, repo_id: RepoId, file_meta: FileMetadata
    ) -> tuple[list[RawSymbol], list[RawRelation]]:
        """파일 파싱 (파서 + 파싱 결과 캐시 사용)"""
        # 1. 파싱 결과 캐시 확인
        if self.parse_cache:
            cached: tuple[list[RawSymbol], list[RawRelation]] | None = self.parse_cache.get(
                repo_id, Path(file_meta.abs_path)
            )
            if cached:
                logger.debug(f"Parse cache HIT: {file_meta.file_path}")
                return cached

        # 2. 캐시 미스 - 파싱 실행
        # 파서 캐싱: 언어별로 재사용
        if file_meta.language not in self._parser_cache:
            parser = create_parser(file_meta.language)
            if parser is None:
                logger.warning(f"No parser for {file_meta.language}")
                return [], []
            self._parser_cache[file_meta.language] = parser
        else:
            parser = self._parser_cache[file_meta.language]

        symbols, relations = parser.parse_file(
            {
                "repo_id": repo_id,
                "path": file_meta.file_path,  # 상대 경로 (base.py에서 사용)
                "file_path": file_meta.file_path,  # 호환성을 위해 유지
                "abs_path": file_meta.abs_path,  # 절대 경로
                "language": file_meta.language,
            }
        )

        # 3. 파싱 결과 캐시 저장
        if self.parse_cache:
            self.parse_cache.save(repo_id, Path(file_meta.abs_path), symbols, relations)
            logger.debug(f"Parse cache SAVED: {file_meta.file_path}")

        return symbols, relations

    def _read_file(self, abs_path: str) -> str:
        """파일 읽기"""
        try:
            with Path(abs_path).open(encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {abs_path}: {e}")
            return ""

    def _embed_chunks_in_batches(
        self, chunks: list[CodeChunk], repo_id: RepoId, batch_size: int | None = None
    ) -> list[list[float]]:
        """
        청크를 배치로 나눠 임베딩 생성 (진행률 업데이트 포함)

        Args:
            chunks: 청크 리스트
            repo_id: 저장소 ID
            batch_size: 배치 크기 (None이면 모델별 자동 설정)

        Returns:
            임베딩 벡터 리스트
        """
        # 모델별 최적 배치 크기 설정
        if batch_size is None:
            from ..core.enums import EmbeddingModel

            model = self.embedding_service.model

            if model == EmbeddingModel.CODESTRAL_EMBED:
                batch_size = 50  # Mistral - 배치 전체 토큰 제한 고려
            elif model in (EmbeddingModel.OPENAI_3_SMALL, EmbeddingModel.OPENAI_3_LARGE):
                batch_size = 100  # OpenAI
            else:
                batch_size = 100  # 기본값
        all_vectors = []
        total_chunks = len(chunks)
        last_progress_update = 0.7  # 시작 진행률

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            batch_vectors = self.embedding_service.embed_chunks(batch)
            all_vectors.extend(batch_vectors)

            # 진행률 업데이트 최적화: 5% 단위로만 업데이트
            progress = 0.7 + (min(i + batch_size, total_chunks) / total_chunks) * 0.3
            if progress - last_progress_update >= 0.05 or i + batch_size >= total_chunks:
                self.repo_store.update_indexing_status(repo_id, "indexing", progress=progress)
                last_progress_update = progress

            logger.debug(
                f"Embedded batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} "
                f"({min(i + batch_size, total_chunks)}/{total_chunks} chunks)"
            )

        return all_vectors

    async def _embed_chunks_async(
        self,
        chunks: list[CodeChunk],
        repo_id: RepoId,
        batch_size: int | None = None,
        max_concurrent: int = 3,
    ) -> list[list[float]]:
        """
        청크를 비동기 병렬로 임베딩 생성 (중복 제거 포함)

        여러 배치를 동시에 API 호출하여 대기 시간 단축
        중복 청크는 기존 임베딩 재사용

        성능 튜닝 포인트:
        - batch_size: 모델별 최적값 (Mistral:200, OpenAI:150, 기본:100)
        - max_concurrent: API rate limit/네트워크 환경에 따라 2~5 조정
        - batch_size × max_concurrent 조합으로 RPS/latency 최적화
        """
        import hashlib

        # 배치 크기 설정
        if batch_size is None:
            from ..core.enums import EmbeddingModel

            model = self.embedding_service.model

            if model == EmbeddingModel.CODESTRAL_EMBED:
                batch_size = 200
            elif model in (EmbeddingModel.OPENAI_3_SMALL, EmbeddingModel.OPENAI_3_LARGE):
                batch_size = 150
            else:
                batch_size = 100

        # 1. 청크 텍스트 해시 계산
        chunk_hashes = []
        for chunk in chunks:
            text = self.embedding_service._prepare_chunk_text(chunk)
            content_hash = hashlib.md5(text.encode()).hexdigest()
            chunk_hashes.append(content_hash)

        # 2. 기존 임베딩 조회 (배치)
        model_name = self.embedding_service.model.value
        existing_embeddings = self.embedding_store.get_embeddings_by_content_hashes(
            chunk_hashes, model_name
        )

        cache_hits = len(existing_embeddings)
        logger.info(
            f"Embedding cache: {cache_hits}/{len(chunks)} hits ({cache_hits / len(chunks) * 100:.1f}%)"
        )

        # 3. 임베딩이 없는 청크만 필터링
        chunks_to_embed = []
        chunk_indices = []  # 원래 위치 기록

        for i, (chunk, content_hash) in enumerate(zip(chunks, chunk_hashes, strict=False)):
            if content_hash not in existing_embeddings:
                chunks_to_embed.append(chunk)
                chunk_indices.append(i)

        # 4. 새로운 청크만 임베딩 생성
        new_vectors = []
        if chunks_to_embed:
            logger.info(f"Generating {len(chunks_to_embed)} new embeddings...")

            # 배치 생성
            batches = [
                chunks_to_embed[i : i + batch_size]
                for i in range(0, len(chunks_to_embed), batch_size)
            ]
            total_batches = len(batches)

            # Semaphore로 동시 실행 제어
            semaphore = asyncio.Semaphore(max_concurrent)
            completed_batches = 0
            last_progress = 0.7

            async def embed_batch_with_tracking(batch_idx: int, batch: list[CodeChunk]):
                """배치 임베딩 + 진행률 추적"""
                nonlocal completed_batches, last_progress

                async with semaphore:
                    loop = asyncio.get_running_loop()

                    # 동기 함수를 비동기로 실행
                    vectors = await loop.run_in_executor(
                        None, self.embedding_service.embed_chunks, batch
                    )

                    completed_batches += 1

                    # 진행률 업데이트 (70-100%)
                    progress = 0.7 + (completed_batches / total_batches) * 0.3
                    if progress - last_progress >= 0.05 or completed_batches == total_batches:
                        self.repo_store.update_indexing_status(
                            repo_id, "indexing", progress=progress
                        )
                        last_progress = progress

                    logger.debug(f"Batch {completed_batches}/{total_batches} done")

                    return vectors

            # 모든 배치를 비동기로 실행
            tasks = [embed_batch_with_tracking(i, batch) for i, batch in enumerate(batches)]

            results = await asyncio.gather(*tasks)

            # 결과 평면화
            for vectors in results:
                new_vectors.extend(vectors)
        else:
            logger.info("All embeddings found in cache, skipping API calls")

        # 5. 결과 병합 (캐시 + 새로 생성)
        all_vectors = [None] * len(chunks)

        # 캐시된 임베딩 배치
        for i, content_hash in enumerate(chunk_hashes):
            if content_hash in existing_embeddings:
                all_vectors[i] = existing_embeddings[content_hash]

        # 새로 생성된 임베딩 배치
        for idx, vector in zip(chunk_indices, new_vectors, strict=False):
            all_vectors[idx] = vector

        logger.info(
            f"Total embeddings: {len(all_vectors)} (cached: {cache_hits}, new: {len(new_vectors)})"
        )

        # 타입 체크: 모든 벡터가 채워졌는지 확인
        from typing import cast

        return cast("list[list[float]]", all_vectors)

    def _parse_files_parallel(
        self,
        repo_id: RepoId,
        files: list[FileMetadata],
        max_workers: int | None = None,
    ) -> tuple[list[CodeNode], list[CodeEdge], list[tuple[str, str]]]:
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

        # parse_cache 경로 전달 (워커가 자체 인스턴스 생성)
        cache_root = self.parse_cache.cache_root if self.parse_cache else None

        # ProcessPoolExecutor 사용 (CPU 집약적 작업)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 각 파일에 대한 future 생성
            future_to_file = {
                executor.submit(_parse_file_worker, repo_id, file_meta, cache_root): file_meta
                for file_meta in files
            }

            # 완료된 작업부터 처리
            for i, future in enumerate(as_completed(future_to_file), 1):
                file_meta = future_to_file[future]
                try:
                    raw_symbols, raw_relations, file_content, error = future.result()

                    if error:
                        # 워커에서 에러 발생
                        logger.error(f"Failed to parse {file_meta.file_path}: {error}")
                        failed_files.append((file_meta.file_path, error))
                    else:
                        # ✅ IR 변환은 메인 프로세스에서 (DI 패턴 유지)
                        nodes, edges = self.ir_builder.build(
                            raw_symbols=raw_symbols,
                            raw_relations=raw_relations,
                            source_code={file_meta.file_path: file_content},
                        )
                        all_nodes.extend(nodes)
                        all_edges.extend(edges)
                        logger.debug(f"Parsed [{i}/{len(files)}]: {file_meta.file_path}")

                    # 진행률 업데이트: 매 20개 파일마다 또는 마지막 파일
                    if i % 20 == 0 or i == len(files):
                        progress = (i / len(files)) * 0.5
                        self.repo_store.update_indexing_status(
                            repo_id, "indexing", progress=progress
                        )

                except Exception as e:
                    logger.error(f"Failed to get result for {file_meta.file_path}: {e}")
                    failed_files.append((file_meta.file_path, str(e)))

        return all_nodes, all_edges, failed_files


def _parse_file_worker(
    repo_id: RepoId, file_meta: FileMetadata, cache_root: Path | None = None
) -> tuple[list[RawSymbol], list[RawRelation], str, str | None]:
    """
    워커 프로세스에서 실행되는 파일 파싱 함수 (파싱만 수행)

    IR 변환은 메인 프로세스에서 DI된 IRBuilder로 처리하여 일관성 유지

    Note: multiprocessing을 위해 top-level 함수로 정의

    Args:
        repo_id: 저장소 ID
        file_meta: 파일 메타데이터
        cache_root: 파싱 캐시 루트 경로 (None이면 캐시 미사용)

    Returns:
        (raw_symbols, raw_relations, file_content, 에러 메시지 or None)
    """
    try:
        # ✅ parse_cache 사용 (병렬 경로에서도 캐시 효과)
        parse_cache = None
        if cache_root:
            from ..parser.cache import ParseCache

            parse_cache = ParseCache(cache_root)

        # 캐시 확인
        if parse_cache:
            cached = parse_cache.get(repo_id, Path(file_meta.abs_path))
            if cached:
                raw_symbols, raw_relations = cached
                # 파일 내용도 읽기
                try:
                    with Path(file_meta.abs_path).open(encoding="utf-8") as f:
                        file_content = f.read()
                except Exception:
                    file_content = ""
                return raw_symbols, raw_relations, file_content, None

        # 캐시 미스 - 파싱 실행
        parser = create_parser(file_meta.language)
        if parser is None:
            return [], [], "", f"No parser for {file_meta.language}"

        raw_symbols, raw_relations = parser.parse_file(
            {
                "repo_id": repo_id,
                "path": file_meta.file_path,
                "file_path": file_meta.file_path,
                "abs_path": file_meta.abs_path,
                "language": file_meta.language,
            }
        )

        # 파일 읽기
        try:
            with Path(file_meta.abs_path).open(encoding="utf-8") as f:
                file_content = f.read()
        except Exception:
            file_content = ""
            # 파일 읽기 실패는 계속 진행 (빈 문자열로)

        # 캐시 저장
        if parse_cache:
            parse_cache.save(repo_id, Path(file_meta.abs_path), raw_symbols, raw_relations)

        # ✅ IR 변환은 메인 프로세스에서 수행 (DI 패턴 유지)
        return raw_symbols, raw_relations, file_content, None  # 성공

    except Exception as e:
        # 에러 정보를 반환 (로깅은 메인 프로세스에서)
        return [], [], "", str(e)
