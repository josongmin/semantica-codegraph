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
    FileProfile,
    IndexingResult,
    RawRelation,
    RawSymbol,
    RepoConfig,
    RepoId,
    RepoMetadata,
)
from ..core.ports import ChunkStorePort, EmbeddingStorePort, GraphStorePort
from ..core.repo_store import RepoMetadataStore
from ..core.telemetry import get_tracer
from ..embedding.service import EmbeddingService
from ..graph.ir_builder import IRBuilder
from ..parser import create_parser
from ..parser.cache import ParseCache
from ..search.ports.lexical_search_port import LexicalSearchPort
from .repo_scanner import RepoScanner

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


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
        semantic_node_store=None,  # SemanticNodeStore (Phase 1)
        embedding_service_small: EmbeddingService | None = None,  # Phase 1
        embedding_service_large: EmbeddingService | None = None,  # Phase 2
        profiler=None,  # 프로파일러 (optional)
    ):
        """
        Args:
            repo_store: 저장소 메타데이터 스토어
            graph_store: 코드 그래프 스토어
            chunk_store: 청크 스토어
            embedding_service: 임베딩 서비스 (코드 청크용)
            embedding_store: 임베딩 스토어
            lexical_search: Lexical 검색 포트
            ir_builder: IR 빌더
            chunker: 청커
            scanner: 저장소 스캐너
            parse_cache: 파싱 캐시 (None이면 비활성화)
            route_store: Route 인덱스 스토어 (optional)
            semantic_node_store: Semantic Node 스토어 (Phase 1)
            embedding_service_small: 임베딩 서비스 3-small (Phase 1)
            embedding_service_large: 임베딩 서비스 3-large (Phase 2)
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
        self.profiler = profiler  # 프로파일러 (optional)

        # Phase 1: Semantic nodes
        self.semantic_node_store = semantic_node_store
        self.embedding_service_small = embedding_service_small
        self.embedding_service_large = embedding_service_large

        # Route 추출기
        self.route_extractor = None  # type: ignore[assignment]
        if route_store:
            from .route_extractor import RouteExtractor

            self.route_extractor = RouteExtractor()

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
        use_embedding_cache: bool = False,
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
            if hasattr(self, "profiler") and self.profiler:
                self.profiler.start_sub_phase("scan_files")
            files = self.scanner.scan(root_path, config)
            logger.info(f"Found {len(files)} files to index")
            if hasattr(self, "profiler") and self.profiler:
                scan_data = self.profiler.end_sub_phase()
                if scan_data:
                    self.profiler.add_phase_counter("files_scanned", len(files))

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

                logger.info(
                    f"[Profiling] Repo: framework={repo_profile.framework}, type={repo_profile.project_type}"
                )
                logger.info(
                    f"[Profiling] API dirs: {len(repo_profile.api_directories)}, Test dirs: {len(repo_profile.test_directories)}"
                )
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
                    file_start_time = time.time()
                    file_nodes_count = 0
                    file_edges_count = 0
                    phase_breakdown = {}

                    try:
                        logger.debug(f"Processing [{i}/{len(files)}]: {file_meta.file_path}")

                        # 파싱 (캐시된 파서 사용)
                        if hasattr(self, "profiler") and self.profiler:
                            self.profiler.start_sub_phase(f"parse:{file_meta.file_path}")
                        raw_symbols, raw_relations = self._parse_file(repo_id, file_meta)
                        if hasattr(self, "profiler") and self.profiler:
                            parse_data = self.profiler.end_sub_phase()
                            if parse_data:
                                phase_breakdown["parse_and_ast"] = parse_data["elapsed_ms"]

                        # IR 변환
                        if hasattr(self, "profiler") and self.profiler:
                            self.profiler.start_sub_phase(f"build:{file_meta.file_path}")
                        file_content = self._read_file(file_meta.abs_path)
                        nodes, edges = self.ir_builder.build(
                            raw_symbols=raw_symbols,
                            raw_relations=raw_relations,
                            source_code={file_meta.file_path: file_content},
                        )
                        if hasattr(self, "profiler") and self.profiler:
                            build_data = self.profiler.end_sub_phase()
                            if build_data:
                                phase_breakdown["build_graph"] = build_data["elapsed_ms"]

                        file_nodes_count = len(nodes)
                        file_edges_count = len(edges)
                        all_nodes.extend(nodes)
                        all_edges.extend(edges)

                        # 진행률 업데이트 최적화: 5% 단위로만 업데이트
                        progress = (i / len(files)) * 0.5
                        if progress - last_progress_update >= 0.05 or i == len(files):
                            self.repo_store.update_indexing_status(
                                repo_id, "indexing", progress=progress
                            )
                            last_progress_update = progress

                        # 파일별 프로파일링 (성공)
                        if hasattr(self, "profiler") and self.profiler:
                            file_elapsed_ms = (time.time() - file_start_time) * 1000

                            # LOC 계산
                            try:
                                loc = len(
                                    Path(file_meta.abs_path)
                                    .read_text(encoding="utf-8")
                                    .splitlines()
                                )
                            except Exception:
                                loc = 0

                            # Flags 계산
                            file_path_lower = file_meta.file_path.lower()
                            flags = {
                                "is_test": "test" in file_path_lower or "spec" in file_path_lower,
                                "is_generated": "generated" in file_path_lower
                                or "__pycache__" in file_path_lower,
                                "is_config": "config" in file_path_lower
                                or "settings" in file_path_lower,
                                "skipped": False,
                            }

                            self.profiler.record_file(
                                file_path=file_meta.file_path,
                                language=file_meta.language,
                                elapsed_ms=file_elapsed_ms,
                                stats={
                                    "nodes": file_nodes_count,
                                    "edges": file_edges_count,
                                    "loc": loc,
                                },
                                phase_breakdown_ms=phase_breakdown,
                                flags=flags,
                            )

                    except Exception as e:
                        logger.error(f"Failed to parse {file_meta.file_path}: {e}")
                        failed_files.append((file_meta.file_path, str(e)))
                        # 파일별 프로파일링 (실패)
                        if hasattr(self, "profiler") and self.profiler:
                            file_elapsed_ms = (time.time() - file_start_time) * 1000
                            self.profiler.record_file(
                                file_path=file_meta.file_path,
                                language=file_meta.language,
                                elapsed_ms=file_elapsed_ms,
                                stats={"nodes": 0, "edges": 0},
                                flags={
                                    "is_test": "test" in file_meta.file_path.lower(),
                                    "skipped": True,
                                },
                            )
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
            if hasattr(self, "profiler") and self.profiler:
                self.profiler.start_sub_phase("db_flush")
            if all_nodes:
                self.graph_store.save_graph(all_nodes, all_edges)
                logger.info("Saved graph to database")
            if hasattr(self, "profiler") and self.profiler:
                db_flush_data = self.profiler.end_sub_phase()
                if db_flush_data:
                    self.profiler.add_phase_counter(
                        "db_flush_nodes_ms", db_flush_data["elapsed_ms"]
                    )

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
                    # 타입 체크 우회: 실제 구현체에는 update_all_node_importance가 있음
                    updated_count = self.graph_store.update_all_node_importance(  # type: ignore[attr-defined]
                        repo_id, batch_size=100
                    )
                    logger.info(f"[Profiling] Graph: {updated_count}개 노드 중요도 계산 완료")
            except Exception as e:
                logger.warning(f"[Profiling] Graph ranking 실패 (계속 진행): {e}")

            # 7. 청킹 (파일 요약 청크 포함)
            if hasattr(self, "profiler") and self.profiler:
                self.profiler.start_sub_phase("chunking")

            # source_files 수집 (파일 요약 청크용)
            source_files = {}
            for node in all_nodes:
                if node.kind == "File" and node.file_path not in source_files:
                    # File 노드의 text는 파일 전체 내용
                    source_files[node.file_path] = node.text

            chunks, chunk_metrics = self.chunker.chunk(all_nodes, source_files)
            logger.info(
                f"Generated {len(chunks)} chunks "
                f"(symbol:{chunk_metrics.get('symbol_chunks', 0)}, "
                f"file_summary:{chunk_metrics.get('file_summary_chunks', 0)}, "
                f"split_nodes:{chunk_metrics.get('split_nodes', 0)})"
            )
            if hasattr(self, "profiler") and self.profiler:
                chunking_data = self.profiler.end_sub_phase()
                if chunking_data:
                    self.profiler.add_phase_counter("chunking_ms", chunking_data["elapsed_ms"])
                    self.profiler.add_phase_counter("chunks_created", len(chunks))
                    # 청킹 상세 메트릭
                    for key, value in chunk_metrics.items():
                        self.profiler.add_counter(f"chunk_{key}", value)

            # 8. 청크 저장
            if hasattr(self, "profiler") and self.profiler:
                self.profiler.start_sub_phase("db_flush")
            if chunks:
                self.chunk_store.save_chunks(chunks)
                logger.info("Saved chunks to database")
            if hasattr(self, "profiler") and self.profiler:
                db_flush_data = self.profiler.end_sub_phase()
                if db_flush_data:
                    self.profiler.add_phase_counter(
                        "db_flush_chunks_ms", db_flush_data["elapsed_ms"]
                    )

            # 8-1. Chunk Tagging + search_text 생성
            logger.info("[Profiling] Chunk tagging 및 search_text 생성 시작...")
            try:
                from ..chunking.chunk_tagger import ChunkTagger
                from ..chunking.search_text_builder import SearchTextBuilder

                chunk_tagger = ChunkTagger()
                search_text_builder = SearchTextBuilder()
                metadata_updates = []

                # 파일별로 프로파일 매핑
                file_profile_map = {p.file_path: p for p in file_profiles}

                for chunk in chunks:
                    chunk_file_profile: FileProfile | None = file_profile_map.get(chunk.file_path)

                    # 1. 메타데이터 태깅
                    chunk_metadata = chunk_tagger.tag_chunk(chunk.text, chunk_file_profile)

                    # 2. search_text 생성
                    search_text = search_text_builder.build(chunk, file_profile, chunk_metadata)

                    # 3. chunk.attrs에 추가 (나중에 embedding/lexical에서 사용)
                    chunk.attrs["search_text"] = search_text
                    chunk.attrs["metadata"] = chunk_metadata

                    metadata_updates.append((repo_id, chunk.id, chunk_metadata))

                # 배치 업데이트
                if metadata_updates:
                    # 타입 체크 우회: 실제 구현체에는 update_chunk_metadata_batch가 있음
                    self.chunk_store.update_chunk_metadata_batch(metadata_updates)  # type: ignore[attr-defined]
                    api_chunks = sum(
                        1 for _, _, m in metadata_updates if m.get("is_api_endpoint_chunk")
                    )
                    logger.info(
                        f"[Profiling] Chunk: {len(metadata_updates)}개 (API: {api_chunks}개)"
                    )
                    logger.info(f"[Profiling] search_text 생성 완료 ({len(chunks)}개 chunk)")
            except Exception as e:
                logger.warning(f"[Profiling] Chunk tagging/search_text 실패 (계속 진행): {e}")

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
                    vectors = asyncio.run(
                        self._embed_chunks_async(chunks, repo_id, use_cache=use_embedding_cache)
                    )

                    # content_hash 계산
                    content_hashes = []
                    for chunk in chunks:
                        text = self.embedding_service._prepare_chunk_text(chunk)
                        content_hash = hashlib.md5(text.encode()).hexdigest()
                        content_hashes.append(content_hash)

                    chunk_ids = [chunk.id for chunk in chunks]
                    self.embedding_store.save_embeddings(repo_id, chunk_ids, vectors)
                    logger.info(f"Saved {len(vectors)} embeddings")
                except Exception as e:
                    logger.error(f"Embedding failed: {e}", exc_info=True)

            # 10.5. Route 추출 및 저장 (API 엔드포인트 인덱싱)
            all_routes = []
            if self.route_store and self.route_extractor:
                try:
                    logger.info("Extracting API routes...")

                    # API 파일만 처리
                    for profile in file_profiles:
                        if not profile or not profile.is_api_file:
                            continue

                        # 해당 파일의 nodes 가져오기
                        file_nodes = [n for n in all_nodes if n.file_path == profile.file_path]

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

            # 10.6. Route Semantic 인덱싱 (Phase 1)
            if self.semantic_node_store and self.embedding_service_small and all_routes:
                try:
                    logger.info("[Semantic] Route indexing...")
                    # 기존 route semantic 삭제
                    self.semantic_node_store.clear_repo(repo_id, node_types=["route"])

                    # 타입 체크 우회: 실제로는 메서드가 존재함
                    route_semantic_count = asyncio.run(
                        self._index_route_semantics(repo_id, all_routes)  # type: ignore[attr-defined]
                    )
                    logger.info(f"[Semantic] Indexed {route_semantic_count} route summaries")
                except Exception as e:
                    logger.error(f"[Semantic] Route indexing failed: {e}")

            # 10.7. Symbol Semantic 인덱싱 (Phase 1)
            if hasattr(self, "profiler") and self.profiler:
                self.profiler.start_sub_phase("semantic_nodes")
            if self.semantic_node_store and self.embedding_service_small and all_nodes:
                try:
                    logger.info("[Semantic] Symbol indexing...")
                    # 기존 symbol semantic 삭제
                    self.semantic_node_store.clear_repo(repo_id, node_types=["symbol"])

                    # 타입 체크 우회: 실제로는 메서드가 존재함
                    symbol_semantic_count = asyncio.run(
                        self._index_symbol_semantics(repo_id, all_nodes, file_profiles)  # type: ignore[attr-defined]
                    )
                    logger.info(f"[Semantic] Indexed {symbol_semantic_count} symbol summaries")
                except Exception as e:
                    logger.error(f"[Semantic] Symbol indexing failed: {e}")
            if hasattr(self, "profiler") and self.profiler:
                semantic_data = self.profiler.end_sub_phase()
                if semantic_data:
                    self.profiler.add_phase_counter(
                        "semantic_nodes_ms", semantic_data["elapsed_ms"]
                    )

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
        use_embedding_cache: bool = False,
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
            use_embedding_cache: 임베딩 캐시 사용 여부 (기본값: False, 테스트 시 False 권장)

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
            if hasattr(self, "profiler") and self.profiler:
                self.profiler.start_sub_phase("scan_files")

            step_start = time.time()

            # OpenTelemetry span
            if tracer:
                with tracer.start_as_current_span("scan_files") as scan_span:
                    scan_span.set_attribute("repo.id", repo_id)
                    scan_span.set_attribute("repo.path", root_path)

                    files = self.scanner.scan(root_path, config)

                    scan_span.set_attribute("files.count", len(files))
                    scan_span.set_attribute(
                        "files.total_size",
                        sum(
                            Path(f.abs_path).stat().st_size
                            for f in files
                            if Path(f.abs_path).exists()
                        ),
                    )
            else:
                files = self.scanner.scan(root_path, config)

            logger.info(
                f"[Step 1/11] File scan: {len(files)} files found ({time.time() - step_start:.2f}s)"
            )
            if hasattr(self, "profiler") and self.profiler:
                scan_data = self.profiler.end_sub_phase()
                if scan_data:
                    self.profiler.add_phase_counter("files_scanned", len(files))

            if len(files) == 0:
                logger.warning("No files found to index")
                return IndexingResult(
                    repo_id=repo_id,
                    status="completed",
                    total_files=0,
                    duration_seconds=time.time() - start_time,
                )

            # 4-1. Repo Profiling (프로젝트 구조 분석)
            repo_profile = None
            try:
                from .repo_profiler import RepoProfiler

                repo_profiler = RepoProfiler()
                repo_profile = repo_profiler.profile_repo(root_path, repo_id)
                self.repo_store.save_profile(repo_profile)
            except Exception as e:
                logger.warning(f"Repo profiling failed: {e}")

            # 5. 파일별 파싱 + IR 변환 + 그래프 저장
            step_start = time.time()
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

            logger.info(
                f"[Step 2/11] File parsing: {len(all_nodes)} nodes, {len(all_edges)} edges ({time.time() - step_start:.2f}s)"
            )

            # 실패한 파일 로깅
            if failed_files:
                logger.warning(f"Failed to parse {len(failed_files)} files:")
                for file_path, error in failed_files[:10]:  # 최대 10개만 출력
                    logger.warning(f"  - {file_path}: {error}")
                if len(failed_files) > 10:
                    logger.warning(f"  ... and {len(failed_files) - 10} more")

            # 6. 그래프 저장
            step_start = time.time()
            if all_nodes:
                self.graph_store.save_graph(all_nodes, all_edges)
            logger.info(
                f"[Step 3/11] Graph save: {len(all_nodes)} nodes, {len(all_edges)} edges ({time.time() - step_start:.2f}s)"
            )

            # 6-1. File Profiling (파일 역할 태깅)
            step_start = time.time()
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
                    logger.info(
                        f"[Step 4/11] File profiling: {len(file_profiles)} files (API: {api_count}) ({time.time() - step_start:.2f}s)"
                    )
            except Exception as e:
                logger.warning(f"[Step 4/11] File profiling failed: {e}")

            # 6-2. Graph Ranking (노드 중요도 계산)
            step_start = time.time()
            try:
                if all_nodes:
                    # 타입 체크 우회: 실제 구현체에는 update_all_node_importance가 있음
                    updated_count = self.graph_store.update_all_node_importance(  # type: ignore[attr-defined]
                        repo_id, batch_size=100
                    )
                    logger.info(
                        f"[Step 5/11] Graph ranking: {updated_count} nodes ({time.time() - step_start:.2f}s)"
                    )
            except Exception as e:
                logger.warning(f"[Step 5/11] Graph ranking failed: {e}")

            # 7. 청킹 (파일 요약 청크 포함)
            step_start = time.time()

            # source_files 수집 (파일 요약 청크용)
            source_files = {}
            for node in all_nodes:
                if node.kind == "File" and node.file_path not in source_files:
                    # File 노드의 text는 파일 전체 내용
                    source_files[node.file_path] = node.text

            chunks, chunk_metrics = self.chunker.chunk(all_nodes, source_files)
            logger.info(
                f"[Step 6/11] Chunking: {len(chunks)} chunks ({time.time() - step_start:.2f}s) "
                f"(Symbol:{chunk_metrics.get('symbol_chunks', 0)}, "
                f"FileSummary:{chunk_metrics.get('file_summary_chunks', 0)})"
            )

            # 프로파일러에 청킹 메트릭 추가
            if hasattr(self, "profiler") and self.profiler:
                for key, value in chunk_metrics.items():
                    self.profiler.add_counter(f"chunk_{key}", value)

            # 8. 청크 저장
            step_start = time.time()
            if chunks:
                self.chunk_store.save_chunks(chunks)
            logger.info(
                f"[Step 7/11] Chunk save: {len(chunks)} chunks ({time.time() - step_start:.2f}s)"
            )

            # 8-1. Chunk Tagging + search_text 생성
            step_start = time.time()
            try:
                from ..chunking.chunk_tagger import ChunkTagger
                from ..chunking.search_text_builder import SearchTextBuilder

                chunk_tagger = ChunkTagger()
                search_text_builder = SearchTextBuilder()
                metadata_updates = []

                # 파일별로 프로파일 매핑
                file_profile_map = {p.file_path: p for p in file_profiles}

                for chunk in chunks:
                    chunk_file_profile: FileProfile | None = file_profile_map.get(chunk.file_path)

                    # 1. 메타데이터 태깅
                    chunk_metadata = chunk_tagger.tag_chunk(chunk.text, chunk_file_profile)

                    # 2. search_text 생성
                    search_text = search_text_builder.build(chunk, file_profile, chunk_metadata)

                    # 3. chunk.attrs에 추가 (나중에 embedding/lexical에서 사용)
                    chunk.attrs["search_text"] = search_text
                    chunk.attrs["metadata"] = chunk_metadata

                    metadata_updates.append((repo_id, chunk.id, chunk_metadata))

                # 배치 업데이트
                if metadata_updates:
                    # 타입 체크 우회: 실제 구현체에는 update_chunk_metadata_batch가 있음
                    self.chunk_store.update_chunk_metadata_batch(metadata_updates)  # type: ignore[attr-defined]
                    api_chunks = sum(
                        1 for _, _, m in metadata_updates if m.get("is_api_endpoint_chunk")
                    )
                    logger.info(
                        f"[Step 8/11] Chunk tagging: {len(metadata_updates)} chunks (API: {api_chunks}) ({time.time() - step_start:.2f}s)"
                    )
            except Exception as e:
                logger.warning(f"[Step 8/11] Chunk tagging failed: {e}")

            # 진행률 50% (청킹 완료)
            self.repo_store.update_indexing_status(repo_id, "indexing", progress=0.5)

            # 9. Lexical 인덱싱
            step_start = time.time()
            if chunks:
                try:
                    self.lexical_search.index_chunks(chunks)
                    logger.info(
                        f"[Step 9/11] Lexical indexing: {len(chunks)} chunks ({time.time() - step_start:.2f}s)"
                    )
                except Exception as e:
                    logger.error(f"[Step 9/11] Lexical indexing failed: {e}")

            # 진행률 70% (Lexical 완료)
            self.repo_store.update_indexing_status(repo_id, "indexing", progress=0.7)

            # 10. 임베딩 생성 및 저장 (비동기 병렬 + 중복 제거)
            step_start = time.time()
            if chunks:
                try:
                    import hashlib

                    # ✅ await 사용 (asyncio.run 대신)
                    vectors = await self._embed_chunks_async(
                        chunks, repo_id, use_cache=use_embedding_cache
                    )

                    # content_hash 계산
                    content_hashes = []
                    for chunk in chunks:
                        text = self.embedding_service._prepare_chunk_text(chunk)
                        content_hash = hashlib.md5(text.encode()).hexdigest()
                        content_hashes.append(content_hash)

                    chunk_ids = [chunk.id for chunk in chunks]
                    self.embedding_store.save_embeddings(repo_id, chunk_ids, vectors)
                    logger.info(
                        f"[Step 10/11] Embedding generation: {len(vectors)} vectors ({time.time() - step_start:.2f}s)"
                    )
                except Exception as e:
                    logger.error(f"[Step 10/11] Embedding generation failed: {e}")

            # 10.5. Route 추출 및 저장 (API 엔드포인트 인덱싱)
            step_start = time.time()
            route_count = 0
            all_routes = []
            if self.route_store and self.route_extractor:
                try:
                    # API 파일만 처리
                    for profile in file_profiles:
                        if not profile or not profile.is_api_file:
                            continue

                        # 해당 파일의 nodes 가져오기
                        file_nodes = [n for n in all_nodes if n.file_path == profile.file_path]

                        if not file_nodes:
                            continue

                        # Route 추출
                        routes = self.route_extractor.extract_routes(file_nodes, profile)
                        all_routes.extend(routes)

                    # Route 저장
                    if all_routes:
                        self.route_store.save_routes(all_routes)
                        route_count = len(all_routes)

                except Exception as e:
                    logger.error(f"Route extraction failed: {e}", exc_info=True)
            logger.info(
                f"[Step 10.5/13] Route extraction: {route_count} routes ({time.time() - step_start:.2f}s)"
            )

            # 11. Route Semantic 인덱싱 (Phase 1)
            if self.semantic_node_store and self.embedding_service_small and all_routes:
                step_start = time.time()
                try:
                    # 기존 route semantic 삭제
                    self.semantic_node_store.clear_repo(repo_id, node_types=["route"])

                    # 새로 생성
                    # 타입 체크 우회: 실제로는 메서드가 존재함
                    route_semantic_count = await self._index_route_semantics(repo_id, all_routes)  # type: ignore[attr-defined]

                    logger.info(
                        f"[Step 11/13] Route semantics: {route_semantic_count} routes ({time.time() - step_start:.2f}s)"
                    )
                except Exception as e:
                    logger.error(f"[Step 11/13] Route semantics failed: {e}")

            # 12. Symbol Semantic 인덱싱 (Phase 1)
            if self.semantic_node_store and self.embedding_service_small and all_nodes:
                step_start = time.time()
                try:
                    # 기존 symbol semantic 삭제
                    self.semantic_node_store.clear_repo(repo_id, node_types=["symbol"])

                    # 새로 생성
                    # 타입 체크 우회: 실제로는 메서드가 존재함
                    symbol_semantic_count = await self._index_symbol_semantics(  # type: ignore[attr-defined]
                        repo_id, all_nodes, file_profiles
                    )

                    logger.info(
                        f"[Step 12/13] Symbol semantics: {symbol_semantic_count} symbols ({time.time() - step_start:.2f}s)"
                    )
                except Exception as e:
                    logger.error(f"[Step 12/13] Symbol semantics failed: {e}")

            # 13. 메타데이터 업데이트
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

        if parser is None:
            logger.warning(f"No parser available for language: {file_meta.language}")
            return [], []
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
        max_concurrent: int = 20,
        use_cache: bool = False,
    ) -> list[list[float]]:
        """
        청크를 비동기 병렬로 임베딩 생성 (중복 제거 포함)

        여러 배치를 동시에 API 호출하여 대기 시간 단축
        중복 청크는 기존 임베딩 재사용

        성능 튜닝 포인트:
        - batch_size: 모델별 최적값 (Mistral:350, OpenAI:200, 기본:100)
        - max_concurrent: API rate limit/네트워크 환경에 따라 10~20 조정
        - batch_size × max_concurrent 조합으로 RPS/latency 최적화
        """
        import hashlib

        # 배치 크기 설정 (토큰 제한 고려)
        if batch_size is None:
            from ..core.enums import EmbeddingModel

            model = self.embedding_service.model

            if model == EmbeddingModel.CODESTRAL_EMBED:
                batch_size = 350  # 토큰 제한(8192) 안전 범위
            elif model in (EmbeddingModel.OPENAI_3_SMALL, EmbeddingModel.OPENAI_3_LARGE):
                batch_size = 200
            else:
                batch_size = 100

        # 1. 청크 텍스트 해시 계산
        chunk_hashes = []
        for chunk in chunks:
            text = self.embedding_service._prepare_chunk_text(chunk)
            content_hash = hashlib.md5(text.encode()).hexdigest()
            chunk_hashes.append(content_hash)

        # 2. 기존 임베딩 조회 (배치) - 캐시 사용 여부에 따라
        existing_embeddings = {}
        cache_hits = 0
        if use_cache:
            model_name = self.embedding_service.model.value
            # 타입 체크 우회: 실제 구현체에는 get_embeddings_by_content_hashes가 있음
            existing_embeddings = self.embedding_store.get_embeddings_by_content_hashes(  # type: ignore[attr-defined]
                chunk_hashes, model_name
            )

            cache_hits = len(existing_embeddings)
            logger.info(
                f"Embedding cache: {cache_hits}/{len(chunks)} hits ({cache_hits / len(chunks) * 100:.1f}%)"
            )
        else:
            logger.info("Embedding cache: disabled (generating all embeddings)")

        # 3. 임베딩이 없는 청크만 필터링
        chunks_to_embed = []
        chunk_indices = []  # 원래 위치 기록

        for i, (chunk, content_hash) in enumerate(zip(chunks, chunk_hashes, strict=False)):
            if content_hash not in existing_embeddings:
                chunks_to_embed.append(chunk)
                chunk_indices.append(i)

        # 4. 큰 청크 필터링 (Mistral Codestral Embed: 16K 토큰)
        MAX_TOKEN_LIMIT = 15000  # 안전 마진 포함 (16K - 1K 여유)
        CHARS_PER_TOKEN = 4  # 평균 토큰당 글자 수
        max_chars = MAX_TOKEN_LIMIT * CHARS_PER_TOKEN

        filtered_chunks = []
        filtered_indices = []
        skipped_chunks = []

        for chunk, idx in zip(chunks_to_embed, chunk_indices, strict=False):
            chunk_text = self.embedding_service._prepare_chunk_text(chunk)
            if len(chunk_text) > max_chars:
                skipped_chunks.append(
                    (chunk.id, len(chunk_text), len(chunk_text) // CHARS_PER_TOKEN)
                )
            else:
                filtered_chunks.append(chunk)
                filtered_indices.append(idx)

        if skipped_chunks:
            logger.warning(
                f"Skipped {len(skipped_chunks)} large chunks exceeding {MAX_TOKEN_LIMIT} tokens:"
            )
            for chunk_id, chars, tokens in skipped_chunks[:5]:
                logger.warning(f"  - {chunk_id}: ~{tokens} tokens ({chars} chars)")
            if len(skipped_chunks) > 5:
                logger.warning(f"  ... and {len(skipped_chunks) - 5} more")

        # 4. 새로운 청크만 임베딩 생성
        new_vectors = []
        if filtered_chunks:
            logger.info(
                f"Generating {len(filtered_chunks)} new embeddings (skipped {len(skipped_chunks)} large chunks)..."
            )

            # 배치 생성
            batches = [
                filtered_chunks[i : i + batch_size]
                for i in range(0, len(filtered_chunks), batch_size)
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
        for idx, vector in zip(filtered_indices, new_vectors, strict=False):
            all_vectors[idx] = vector

        logger.info(
            f"Total embeddings: {len(all_vectors)} (cached: {cache_hits}, new: {len(new_vectors)}, skipped: {len(skipped_chunks)})"
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


async def _index_route_semantics(
    self,
    repo_id: RepoId,
    routes: list,  # list[RouteInfo]
) -> int:
    """
    Route 템플릿 summary + 3-small 임베딩

    Args:
        repo_id: 저장소 ID
        routes: RouteInfo 리스트

    Returns:
        저장된 route 수
    """
    if not routes:
        return 0

    import time as time_module

    from ..core.enums import EmbeddingModel

    logger.info("[Semantic] Generating route summaries...")

    filter_start = time_module.time()
    semantic_nodes = []
    embeddings_to_generate = []
    summary_lengths = []

    for route in routes:
        # 템플릿 summary
        summary = (
            f"{route.http_method} {route.http_path}: {route.handler_name} in {route.file_path}"
        )
        embeddings_to_generate.append(summary)
        summary_lengths.append(len(summary))

        semantic_nodes.append(
            {
                "repo_id": repo_id,
                "node_id": route.route_id,
                "node_type": "route",
                "summary": summary,
                "summary_method": "template",
                "model": EmbeddingModel.OPENAI_3_SMALL.value,
                "embedding": None,  # 나중에 채움
                "source_table": "route_index",
                "source_id": route.route_id,
                "metadata": {
                    "http_method": route.http_method,
                    "http_path": route.http_path,
                    "framework": route.framework,
                    "importance_score": 0.8,  # route는 기본적으로 중요
                },
            }
        )

    # 배치 임베딩 생성
    import statistics
    import time as time_module

    logger.info(f"[Semantic] Generating {len(embeddings_to_generate)} route embeddings...")

    # Summary 생성 시간
    summary_time_ms = int((time_module.time() - filter_start) * 1000)

    embed_start = time_module.time()
    embeddings = self.embedding_service_small.embed_texts(embeddings_to_generate)
    embed_time_ms = int((time_module.time() - embed_start) * 1000)

    # 임베딩 실패 카운트
    embedding_failures = sum(1 for e in embeddings if e is None)

    # 임베딩 채우기
    for node, embedding in zip(semantic_nodes, embeddings, strict=False):
        node["embedding"] = embedding

    # 배치 저장
    save_start = time_module.time()
    batch_size = 1000
    saved = self.semantic_node_store.save_batch(
        semantic_nodes, batch_size=batch_size, on_conflict="replace"
    )
    save_time_ms = int((time_module.time() - save_start) * 1000)

    # 비용 및 통계 계산
    total_tokens = sum(len(s.split()) * 1.3 for s in embeddings_to_generate)
    estimated_cost = total_tokens / 1_000_000 * 0.02

    # Summary 길이 통계
    if summary_lengths:
        summary_stats = {
            "min": min(summary_lengths),
            "max": max(summary_lengths),
            "avg": int(statistics.mean(summary_lengths)),
            "median": int(statistics.median(summary_lengths)),
        }
    else:
        summary_stats = {}

    # 배치 통계
    num_batches = (len(semantic_nodes) + batch_size - 1) // batch_size

    logger.info(
        f"[Semantic] Routes: {saved} saved, ~{int(total_tokens)} tokens, "
        f"~${estimated_cost:.4f} cost (3-small), "
        f"summary:{summary_time_ms}ms, embed:{embed_time_ms}ms, save:{save_time_ms}ms, "
        f"failures:{embedding_failures}, batches:{num_batches}"
    )

    # 프로파일러에 상세 메트릭 추가
    if hasattr(self, "profiler") and self.profiler:
        self.profiler.add_counter("route_semantic_count", int(saved))
        self.profiler.add_counter("route_summary_time_ms", summary_time_ms)
        self.profiler.add_counter("route_embed_time_ms", embed_time_ms)
        self.profiler.add_counter("route_save_time_ms", save_time_ms)
        self.profiler.add_counter("route_tokens", int(total_tokens))
        self.profiler.add_counter("route_estimated_cost", round(estimated_cost, 6))
        self.profiler.add_counter("route_batches", num_batches)
        self.profiler.add_counter(
            "route_avg_batch_size", len(semantic_nodes) // num_batches if num_batches > 0 else 0
        )
        self.profiler.add_counter("route_api_calls", 1)
        self.profiler.add_counter("route_embedding_failures", embedding_failures)
        if summary_stats:
            self.profiler.add_counter("route_summary_lengths", summary_stats)

    return int(saved)


async def _index_symbol_semantics(
    self,
    repo_id: RepoId,
    nodes: list,  # list[CodeNode]
    file_profiles: list,  # list[FileProfile]
) -> int:
    """
    Symbol 템플릿 summary + 3-small 임베딩 (퍼블릭 심볼만)

    Args:
        repo_id: 저장소 ID
        nodes: CodeNode 리스트
        file_profiles: FileProfile 리스트

    Returns:
        저장된 symbol 수
    """
    from ..core.enums import EmbeddingModel
    from .symbol_summary_builder import SymbolSummaryBuilder, calculate_importance

    # 파일 프로파일 매핑
    file_profile_map = {p.file_path: p for p in file_profiles}

    # 필터링 통계
    filter_stats = {
        "total_nodes": len(nodes),
        "file_nodes": 0,
        "private": 0,
        "test_files": 0,
        "migrations": 0,
        "indexable": 0,
    }

    # 인덱싱 대상 필터링
    indexable = []
    for node in nodes:
        # 1. Function/Class/Method만
        if node.kind not in ("Function", "Class", "Method"):
            if node.kind == "File":
                filter_stats["file_nodes"] += 1
            continue

        # 2. Private 제외 (__init__ 제외)
        if node.name.startswith("_") and node.name != "__init__":
            filter_stats["private"] += 1
            continue

        # 3. 테스트 파일 제외 (선택적)
        file_profile = file_profile_map.get(node.file_path)
        if file_profile and file_profile.is_test_file:
            filter_stats["test_files"] += 1
            continue

        # 4. Migration 파일 제외
        if "migration" in node.file_path.lower() or "alembic" in node.file_path.lower():
            filter_stats["migrations"] += 1
            continue

        indexable.append(node)

    filter_stats["indexable"] = len(indexable)
    filter_stats["filtered_out"] = (
        filter_stats["file_nodes"]
        + filter_stats["private"]
        + filter_stats["test_files"]
        + filter_stats["migrations"]
    )

    logger.info(
        f"[Semantic] Symbol filtering: {len(indexable)} indexable from {len(nodes)} nodes "
        f"(filtered: File:{filter_stats['file_nodes']}, Private:{filter_stats['private']}, "
        f"Test:{filter_stats['test_files']}, Migration:{filter_stats['migrations']})"
    )

    # 상한선 (비용 제어)
    MAX_SYMBOLS_PER_REPO = 20000
    if len(indexable) > MAX_SYMBOLS_PER_REPO:
        # importance 순으로 정렬 후 상위만
        for node in indexable:
            file_profile = file_profile_map.get(node.file_path)
            node._temp_importance = calculate_importance(node, file_profile)
        indexable.sort(key=lambda n: n._temp_importance, reverse=True)
        indexable = indexable[:MAX_SYMBOLS_PER_REPO]
        logger.warning(
            f"[Semantic] Symbol limit reached, indexing top {MAX_SYMBOLS_PER_REPO} by importance"
        )

    logger.info(f"[Semantic] Generating symbol summaries for {len(indexable)} symbols...")

    builder = SymbolSummaryBuilder(self.graph_store)
    semantic_nodes = []
    embeddings_to_generate = []

    # 파일별 심볼 수 집계 (프로파일링용)
    symbols_by_file = {}
    for node in indexable:
        if node.file_path not in symbols_by_file:
            symbols_by_file[node.file_path] = 0
        symbols_by_file[node.file_path] += 1

    # Summary 생성
    import statistics
    import time as time_module
    from collections import defaultdict

    summary_start = time_module.time()
    summary_lengths = []
    importance_scores = []
    node_by_kind: dict = defaultdict(lambda: {"count": 0, "tokens": 0})

    for node in indexable:
        # Importance 계산
        file_profile = file_profile_map.get(node.file_path)
        importance = calculate_importance(node, file_profile)
        importance_scores.append(importance)

        # 템플릿 summary
        summary = builder.build(node)
        embeddings_to_generate.append(summary)
        summary_lengths.append(len(summary))

        # 노드 타입별 통계
        node_by_kind[node.kind]["count"] += 1
        node_by_kind[node.kind]["tokens"] += len(summary.split()) * 1.3

        semantic_nodes.append(
            {
                "repo_id": repo_id,
                "node_id": node.id,  # prefix 없이 원본 ID
                "node_type": "symbol",
                "summary": summary,
                "summary_method": "template",
                "model": EmbeddingModel.OPENAI_3_SMALL.value,
                "embedding": None,
                "source_table": "code_nodes",
                "source_id": node.id,
                "metadata": {
                    "importance_score": importance,
                    "is_api_handler": node.attrs.get("is_api_handler", False),
                    "kind": node.kind,
                    "name": node.name,
                    "file_path": node.file_path,
                    "line_count": node.attrs.get("line_count", 0),
                },
            }
        )

    summary_time_ms = int((time_module.time() - summary_start) * 1000)

    # 배치 임베딩 생성
    logger.info(f"[Semantic] Generating {len(embeddings_to_generate)} symbol embeddings...")

    # 프로파일링: 파일별 심볼 수 기록
    if hasattr(self, "profiler") and self.profiler:
        self.profiler.add_counter("symbols_by_file", symbols_by_file)
        self.profiler.add_counter("total_symbols", len(indexable))

    embed_start = time_module.time()
    embeddings = self.embedding_service_small.embed_texts(embeddings_to_generate)
    embed_time_ms = int((time_module.time() - embed_start) * 1000)

    # 임베딩 실패 카운트
    embedding_failures = sum(1 for e in embeddings if e is None)

    # 임베딩 채우기
    for node, embedding in zip(semantic_nodes, embeddings, strict=False):
        node["embedding"] = embedding

    # 배치 저장 (1000개씩)
    save_start = time_module.time()
    batch_size = 1000
    saved = self.semantic_node_store.save_batch(
        semantic_nodes, batch_size=batch_size, on_conflict="replace"
    )
    save_time_ms = int((time_module.time() - save_start) * 1000)

    # 비용 및 통계 계산
    total_tokens = sum(len(s.split()) * 1.3 for s in embeddings_to_generate)
    estimated_cost = total_tokens / 1_000_000 * 0.02
    num_batches = (len(semantic_nodes) + batch_size - 1) // batch_size

    # Summary 길이 통계
    summary_stats = {}
    if summary_lengths:
        summary_stats = {
            "min": min(summary_lengths),
            "max": max(summary_lengths),
            "avg": int(statistics.mean(summary_lengths)),
            "median": int(statistics.median(summary_lengths)),
        }

    # 중요도 분포
    importance_distribution = {
        "0.0-0.2": sum(1 for i in importance_scores if 0.0 <= i < 0.2),
        "0.2-0.4": sum(1 for i in importance_scores if 0.2 <= i < 0.4),
        "0.4-0.6": sum(1 for i in importance_scores if 0.4 <= i < 0.6),
        "0.6-0.8": sum(1 for i in importance_scores if 0.6 <= i < 0.8),
        "0.8-1.0": sum(1 for i in importance_scores if 0.8 <= i <= 1.0),
    }
    avg_importance = statistics.mean(importance_scores) if importance_scores else 0
    high_importance = sum(1 for i in importance_scores if i >= 0.8)

    # 파일별 semantic node 수
    files_with_nodes = len(symbols_by_file)
    files_without_nodes = len([p for p in file_profiles if p.file_path not in symbols_by_file])

    logger.info(
        f"[Semantic] Symbols: {saved} saved, ~{int(total_tokens)} tokens, "
        f"~${estimated_cost:.4f} cost (3-small), "
        f"summary:{summary_time_ms}ms, embed:{embed_time_ms}ms, save:{save_time_ms}ms, "
        f"failures:{embedding_failures}, batches:{num_batches}"
    )

    # 프로파일러에 상세 메트릭 추가
    if hasattr(self, "profiler") and self.profiler:
        # 기존 메트릭
        for key, value in filter_stats.items():
            self.profiler.add_counter(f"symbol_filter_{key}", value)
        self.profiler.add_counter("symbol_semantic_count", int(saved))
        self.profiler.add_counter("symbol_summary_time_ms", summary_time_ms)
        self.profiler.add_counter("symbol_embed_time_ms", embed_time_ms)
        self.profiler.add_counter("symbol_save_time_ms", save_time_ms)
        self.profiler.add_counter("symbol_tokens", int(total_tokens))
        self.profiler.add_counter("symbol_estimated_cost", round(estimated_cost, 6))

        # 배치 처리 통계
        self.profiler.add_counter("symbol_batches", num_batches)
        self.profiler.add_counter(
            "symbol_avg_batch_size", len(semantic_nodes) // num_batches if num_batches > 0 else 0
        )

        # API 호출 통계
        self.profiler.add_counter("symbol_api_calls", 1)
        self.profiler.add_counter("symbol_embedding_failures", embedding_failures)

        # Summary 길이 통계
        if summary_stats:
            self.profiler.add_counter("symbol_summary_lengths", summary_stats)

        # 중요도 분포
        self.profiler.add_counter("symbol_importance_distribution", importance_distribution)
        self.profiler.add_counter("symbol_avg_importance", round(avg_importance, 3))
        self.profiler.add_counter("symbol_high_importance_count", high_importance)

        # 노드 타입별 통계
        self.profiler.add_counter("symbol_by_kind", dict(node_by_kind))

        # 파일별 통계
        self.profiler.add_counter("files_with_semantic_nodes", files_with_nodes)
        self.profiler.add_counter("files_without_semantic_nodes", files_without_nodes)

    return int(saved)


# 메서드를 IndexingPipeline 클래스에 추가
IndexingPipeline._index_route_semantics = _index_route_semantics  # type: ignore[attr-defined]
IndexingPipeline._index_symbol_semantics = _index_symbol_semantics  # type: ignore[attr-defined]
