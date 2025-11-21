"""상세 프로파일러 (Waterfall 차트 포함)"""
import json
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


@dataclass
class PhaseMetrics:
    """Phase 메트릭"""

    name: str
    start_time: float
    end_time: float = 0.0
    mem_start_mb: float = 0.0
    mem_end_mb: float = 0.0
    counters: dict[str, Any] = field(default_factory=dict)
    sub_phases: list["PhaseMetrics"] = field(default_factory=list)

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def mem_delta_mb(self) -> float:
        return self.mem_end_mb - self.mem_start_mb


class DetailedProfiler:
    """상세 프로파일러 (메모리 추적, Waterfall 차트)"""

    def __init__(self, repo_id: str, repo_path: str):
        self.repo_id = repo_id
        self.repo_path = repo_path
        self.run_start_time = 0.0
        self.run_end_time = 0.0
        self.phases: list[PhaseMetrics] = []
        self.current_phase: PhaseMetrics | None = None
        self.phase_stack: list[PhaseMetrics] = []

        # 메모리 추적
        self.process = psutil.Process()
        self.mem_start_mb = 0.0
        self.mem_peak_mb = 0.0

        # 파일별 메트릭
        self.file_metrics: dict[str, dict] = {}

        # 환경 정보
        self.environment_info: dict[str, Any] = {}

    def _get_memory_mb(self) -> float:
        """현재 메모리 사용량 (MB)"""
        return float(self.process.memory_info().rss / 1024 / 1024)

    def start_run(self):
        """실행 시작"""
        self.run_start_time = time.time()
        self.mem_start_mb = self._get_memory_mb()
        tracemalloc.start()

    def end_run(self, results=None):
        """실행 종료"""
        self.run_end_time = time.time()
        self.mem_peak_mb = max(self.mem_peak_mb, self._get_memory_mb())
        tracemalloc.stop()

    def start_phase(self, name: str):
        """Phase 시작"""
        current_mem = self._get_memory_mb()
        self.mem_peak_mb = max(self.mem_peak_mb, current_mem)

        phase = PhaseMetrics(
            name=name,
            start_time=time.time(),
            mem_start_mb=current_mem,
        )

        if self.current_phase:
            self.phase_stack.append(self.current_phase)
            self.current_phase.sub_phases.append(phase)
        else:
            self.phases.append(phase)

        self.current_phase = phase

    def end_phase(self):
        """Phase 종료"""
        if self.current_phase:
            current_mem = self._get_memory_mb()
            self.mem_peak_mb = max(self.mem_peak_mb, current_mem)

            self.current_phase.end_time = time.time()
            self.current_phase.mem_end_mb = current_mem

            if self.phase_stack:
                self.current_phase = self.phase_stack.pop()
            else:
                self.current_phase = None

    def add_counter(self, key: str, value: Any):
        """카운터 추가"""
        if self.current_phase:
            self.current_phase.counters[key] = value

    def add_phase_counter(self, key: str, value: Any):
        """Phase 카운터 추가"""
        self.add_counter(key, value)

    def start_sub_phase(self, name: str):
        """Sub-phase 시작"""
        self.start_phase(name)

    def end_sub_phase(self):
        """Sub-phase 종료"""
        self.end_phase()
        return None

    def record_file(
        self,
        file_path: str,
        language: str = "",
        elapsed_ms: float = 0.0,
        stats: dict | None = None,
        flags: dict | None = None,
        phase_breakdown: dict | None = None,
        **kwargs,
    ):
        """파일별 메트릭 기록"""
        stats = stats or {}
        flags = flags or {}
        phase_breakdown = phase_breakdown or {}

        self.file_metrics[file_path] = {
            "elapsed_ms": elapsed_ms,
            "language": language,
            "loc": stats.get("loc", 0),
            "nodes": stats.get("nodes", 0),
            "edges": stats.get("edges", 0),
            "chunks": stats.get("chunks", 0),
            "phase_breakdown": phase_breakdown,
            "flags": flags,
        }

    def set_environment_info(self, info: dict[str, Any]):
        """환경 정보 설정"""
        self.environment_info.update(info)

    def save_report(self, output_dir: Path):
        """상세 리포트 저장"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = output_dir / "reports" / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 저장
        json_path = date_dir / f"index_metrics_{self.repo_id}_{timestamp}.json"
        with json_path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # 텍스트 리포트 (상세)
        txt_path = date_dir / f"index_profile_{self.repo_id}_{timestamp}.txt"
        with txt_path.open("w") as f:
            f.write(self.to_detailed_text())

        print(f"\nJSON 리포트 저장: {json_path}")
        print(f"텍스트 리포트 저장: {txt_path}")

        return json_path, txt_path

    def to_dict(self) -> dict:
        """JSON 직렬화"""
        total_elapsed_ms = (self.run_end_time - self.run_start_time) * 1000

        return {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "repo_id": self.repo_id,
            "repo_path": self.repo_path,
            "summary": {
                "total_elapsed_ms": total_elapsed_ms,
                "mem_start_mb": self.mem_start_mb,
                "mem_end_mb": self._get_memory_mb(),
                "mem_peak_mb": self.mem_peak_mb,
                "mem_delta_mb": self.mem_peak_mb - self.mem_start_mb,
            },
            "phases": [self._phase_to_dict(p) for p in self.phases],
        }

    def _phase_to_dict(self, phase: PhaseMetrics) -> dict:
        """Phase를 dict로 변환"""
        return {
            "name": phase.name,
            "elapsed_ms": phase.elapsed_ms,
            "mem_start_mb": phase.mem_start_mb,
            "mem_end_mb": phase.mem_end_mb,
            "mem_delta_mb": phase.mem_delta_mb,
            "counters": phase.counters,
            "sub_phases": [self._phase_to_dict(sp) for sp in phase.sub_phases],
        }

    def to_detailed_text(self) -> str:
        """상세 텍스트 리포트 생성"""
        lines = []
        lines.append("=" * 80)
        lines.append("인덱스 성능 프로파일링 리포트")
        lines.append("=" * 80)
        lines.append(f"생성 시간: {datetime.now().isoformat()}")
        lines.append(f"Repository ID: {self.repo_id}")
        lines.append(f"Repository Path: {self.repo_path}")
        lines.append(f"Run ID: idx_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{self.repo_id}")
        lines.append("")

        # 환경 설정 (성능 튜닝 참고용)
        if self.environment_info:
            lines.append("## 인덱싱 환경")
            lines.append("-" * 80)

            # 하드웨어
            if "hardware" in self.environment_info:
                hw = self.environment_info["hardware"]
                lines.append(f"CPU: {hw.get('cpu_count', 'N/A')}코어 ({hw.get('cpu_freq', 'N/A')})")
                lines.append(f"메모리: {hw.get('total_memory_gb', 'N/A')} GB")

            # 데이터베이스
            if "database" in self.environment_info:
                db = self.environment_info["database"]
                lines.append(
                    f"PostgreSQL: {db.get('postgres_host', 'localhost')}:{db.get('postgres_port', 7711)}"
                )
                lines.append(
                    f"  - Connection Pool: {db.get('pool_size', 'N/A')} ~ {db.get('pool_max', 'N/A')}"
                )

            # 벡터 저장소
            if "vector_store" in self.environment_info:
                vs = self.environment_info["vector_store"]
                backend = vs.get("backend", "N/A")
                lines.append(f"벡터 저장소: {backend}")
                if backend == "qdrant":
                    lines.append(
                        f"  - Qdrant: {vs.get('qdrant_host', 'localhost')}:{vs.get('qdrant_port', 7714)}"
                    )
                    lines.append(f"  - gRPC: {vs.get('qdrant_use_grpc', False)}")
                elif backend == "pgvector":
                    lines.append(f"  - Dimension: {vs.get('embedding_dimension', 'N/A')}")

            # Lexical 검색
            if "lexical_search" in self.environment_info:
                ls = self.environment_info["lexical_search"]
                lines.append(f"Lexical 검색: {ls.get('backend', 'N/A')}")
                if ls.get("backend") == "meilisearch":
                    lines.append(f"  - Meilisearch: {ls.get('meilisearch_url', 'N/A')}")

            # 임베딩 모델
            if "embedding" in self.environment_info:
                emb = self.environment_info["embedding"]
                lines.append(f"임베딩 모델: {emb.get('model', 'N/A')}")
                lines.append(f"  - Dimension: {emb.get('dimension', 'N/A')}")
                lines.append(f"  - Batch Size: {emb.get('batch_size', 'N/A')}")

            # 병렬 처리
            if "parallel" in self.environment_info:
                par = self.environment_info["parallel"]
                lines.append("병렬 처리:")
                lines.append(f"  - 검색 병렬화: {par.get('search_enabled', False)}")
                lines.append(f"  - 인덱싱 병렬화: {par.get('indexing_enabled', False)}")
                lines.append(f"  - Max Workers: {par.get('max_workers', 'N/A')}")

            # 캐시 설정
            if "cache" in self.environment_info:
                cache = self.environment_info["cache"]
                lines.append("캐시:")
                lines.append(f"  - 파싱 캐시: {cache.get('parse_cache_enabled', False)}")
                lines.append(f"  - 임베딩 캐시: {cache.get('embedding_cache_enabled', False)}")
                lines.append(f"  - 퍼지 캐시: {cache.get('fuzzy_cache_size', 'N/A')}")

            # Fusion 전략
            if "fusion" in self.environment_info:
                fusion = self.environment_info["fusion"]
                lines.append(f"Fusion 전략: {fusion.get('strategy', 'N/A')}")
                if "weights" in fusion:
                    weights = fusion["weights"]
                    lines.append(
                        f"  - Weights: L:{weights.get('lexical', 0):.2f} "
                        f"S:{weights.get('semantic', 0):.2f} "
                        f"G:{weights.get('graph', 0):.2f} "
                        f"F:{weights.get('fuzzy', 0):.2f}"
                    )

            lines.append("")

        # 1. 전체 요약
        total_ms = (self.run_end_time - self.run_start_time) * 1000
        mem_end = self._get_memory_mb()
        mem_delta = self.mem_peak_mb - self.mem_start_mb

        lines.append("## 1. 전체 요약")
        lines.append("-" * 80)
        lines.append(f"총 소요 시간: {total_ms/1000:.2f}초")
        lines.append(f"시작 메모리: {self.mem_start_mb:.1f} MB")
        lines.append(f"종료 메모리: {mem_end:.1f} MB")
        lines.append(f"피크 메모리: {self.mem_peak_mb:.1f} MB")
        lines.append(f"메모리 증가: +{mem_delta:.1f} MB")
        lines.append("")

        # 인덱싱 결과 (indexing_core phase에서 추출)
        indexing_phase = next((p for p in self.phases if p.name == "indexing_core"), None)
        if indexing_phase:
            files = indexing_phase.counters.get("files_parsed", 0)
            nodes = indexing_phase.counters.get("nodes_created", 0)
            chunks = indexing_phase.counters.get("chunks_created", 0)
            edges = indexing_phase.counters.get("edges_created", 0)

            lines.append("인덱싱 결과:")
            lines.append(f"  - 파일: {files}개")
            lines.append(f"  - 노드: {nodes}개")
            lines.append(f"  - 청크: {chunks}개")
            lines.append(f"  - 엣지: {edges}개")
            lines.append("")

        # 2. Waterfall 차트
        lines.append("## 2. Phase별 성능 (Waterfall)")
        lines.append("-" * 80)
        lines.append("")
        lines.append("시간 흐름:")
        lines.append("")

        self._add_waterfall(lines, total_ms)

        # Phase 요약 테이블
        lines.append("")
        lines.append("Phase 요약:")
        lines.append("-" * 80)
        lines.append(f"{'Phase':<35} {'시간(ms)':>10} {'비율(%)':>10} {'메모리(MB)':>15}")
        lines.append("-" * 80)

        for phase in self.phases:
            self._add_phase_summary(lines, phase, total_ms, indent=0)

        lines.append("")

        # 3. 느린 파일 Top 10
        lines.append("## 3. 느린 파일 Top 10")
        lines.append("-" * 80)
        if self.file_metrics:
            sorted_files = sorted(
                self.file_metrics.items(), key=lambda x: x[1].get("elapsed_ms", 0), reverse=True
            )[:10]
            for idx, (file_path, metrics) in enumerate(sorted_files, 1):
                lines.append(f"{idx}. {file_path}")
                lines.append(f"   시간: {metrics.get('elapsed_ms', 0):.0f}ms")
                lines.append(f"   언어: {metrics.get('language', 'N/A')}")
                lines.append(f"   LOC: {metrics.get('loc', 0)}줄")
                lines.append(f"   노드: {metrics.get('nodes', 0)}개")
                lines.append(f"   엣지: {metrics.get('edges', 0)}개")
                lines.append(f"   청크: {metrics.get('chunks', 0)}개")

                # Flags
                flags = metrics.get("flags", {})
                flag_labels = []
                if flags.get("is_test"):
                    flag_labels.append("테스트")
                if flags.get("is_config"):
                    flag_labels.append("설정")
                if flags.get("is_generated"):
                    flag_labels.append("생성됨")
                if flag_labels:
                    lines.append(f"   플래그: {', '.join(flag_labels)}")

                if "phase_breakdown" in metrics:
                    lines.append("   Phase breakdown:")
                    for phase_name, phase_time in metrics["phase_breakdown"].items():
                        pct = (
                            (phase_time / metrics["elapsed_ms"]) * 100
                            if metrics["elapsed_ms"] > 0
                            else 0
                        )
                        lines.append(f"     - {phase_name}: {phase_time:.0f}ms ({pct:.1f}%)")
                lines.append("")
        else:
            lines.append("(데이터 없음)")
            lines.append("")

        # 3-1. Semantic Nodes 상세 분석
        indexing_phase = next((p for p in self.phases if p.name == "indexing_core"), None)
        if indexing_phase:
            semantic_subphase = next(
                (sp for sp in indexing_phase.sub_phases if sp.name == "semantic_nodes"), None
            )
            
            # Symbol Semantics 상세 메트릭
            if semantic_subphase and semantic_subphase.counters:
                counters = semantic_subphase.counters
                
                # 필터링 상세
                if "symbol_filter_total_nodes" in counters:
                    lines.append("## 3-1-1. Symbol 필터링 상세")
                    lines.append("-" * 80)
                    lines.append(f"전체 노드: {counters.get('symbol_filter_total_nodes', 0)}개")
                    lines.append(f"  - File 노드: {counters.get('symbol_filter_file_nodes', 0)}개")
                    lines.append(f"  - Private 심볼: {counters.get('symbol_filter_private', 0)}개")
                    lines.append(f"  - 테스트 파일: {counters.get('symbol_filter_test_files', 0)}개")
                    lines.append(f"  - Migration 파일: {counters.get('symbol_filter_migrations', 0)}개")
                    lines.append(f"필터링 제외: {counters.get('symbol_filter_filtered_out', 0)}개")
                    lines.append(f"인덱싱 대상: {counters.get('symbol_filter_indexable', 0)}개")
                    lines.append("")
                
                # 시간 분해
                if "symbol_summary_time_ms" in counters:
                    lines.append("## 3-1-2. Symbol Semantics 시간 분해")
                    lines.append("-" * 80)
                    summary_ms = counters.get('symbol_summary_time_ms', 0)
                    embed_ms = counters.get('symbol_embed_time_ms', 0)
                    save_ms = counters.get('symbol_save_time_ms', 0)
                    total_ms = summary_ms + embed_ms + save_ms
                    
                    if total_ms > 0:
                        lines.append(f"Summary 생성: {summary_ms}ms ({summary_ms/total_ms*100:.1f}%)")
                        lines.append(f"임베딩 생성: {embed_ms}ms ({embed_ms/total_ms*100:.1f}%)")
                        lines.append(f"DB 저장: {save_ms}ms ({save_ms/total_ms*100:.1f}%)")
                        lines.append(f"합계: {total_ms}ms")
                        lines.append("")
                        
                        if save_ms / total_ms > 0.7:
                            lines.append("병목: DB 저장이 전체의 70% 이상 차지")
                            lines.append("최적화 권장: execute_batch 사용, 트랜잭션 최적화")
                            lines.append("")
                
                # 노드 타입별 통계
                if "symbol_by_kind" in counters:
                    lines.append("## 3-1-3. 노드 타입별 통계")
                    lines.append("-" * 80)
                    by_kind = counters.get('symbol_by_kind', {})
                    total_count = sum(stats['count'] for stats in by_kind.values())
                    total_tokens = sum(stats['tokens'] for stats in by_kind.values())
                    
                    for kind, stats in sorted(by_kind.items(), key=lambda x: x[1]['count'], reverse=True):
                        count = stats['count']
                        tokens = int(stats['tokens'])
                        count_pct = (count / total_count * 100) if total_count > 0 else 0
                        tokens_pct = (tokens / total_tokens * 100) if total_tokens > 0 else 0
                        
                        lines.append(f"{kind}: {count}개 ({count_pct:.1f}%), {tokens} tokens ({tokens_pct:.1f}%)")
                    
                    lines.append(f"\n총 {total_count}개 심볼, {int(total_tokens)} tokens")
                    lines.append("")
                
                # Summary 길이 통계
                if "symbol_summary_lengths" in counters:
                    lines.append("## 3-1-4. Summary 길이 통계")
                    lines.append("-" * 80)
                    lengths = counters['symbol_summary_lengths']
                    lines.append(f"최소: {lengths.get('min', 0)}자")
                    lines.append(f"최대: {lengths.get('max', 0)}자")
                    lines.append(f"평균: {lengths.get('avg', 0)}자")
                    lines.append(f"중간값: {lengths.get('median', 0)}자")
                    lines.append("")
                
                # 중요도 분포
                if "symbol_importance_distribution" in counters:
                    lines.append("## 3-1-5. 중요도 분포")
                    lines.append("-" * 80)
                    dist = counters['symbol_importance_distribution']
                    total_nodes = sum(dist.values())
                    
                    for range_key in ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']:
                        count = dist.get(range_key, 0)
                        pct = (count / total_nodes * 100) if total_nodes > 0 else 0
                        bar = '█' * int(pct / 2)
                        lines.append(f"{range_key}: {count:4}개 ({pct:5.1f}%) {bar}")
                    
                    lines.append(f"\n평균 중요도: {counters.get('symbol_avg_importance', 0):.3f}")
                    lines.append(f"고중요도 (>=0.8): {counters.get('symbol_high_importance_count', 0)}개")
                    lines.append("")
                
                # 비용 및 API
                if "symbol_tokens" in counters:
                    lines.append("## 3-1-6. 비용 및 API 통계")
                    lines.append("-" * 80)
                    lines.append(f"총 토큰: {counters.get('symbol_tokens', 0)}")
                    lines.append(f"예상 비용: ${counters.get('symbol_estimated_cost', 0):.6f} (OpenAI 3-small)")
                    lines.append(f"API 호출: {counters.get('symbol_api_calls', 0)}회")
                    lines.append(f"임베딩 실패: {counters.get('symbol_embedding_failures', 0)}개")
                    lines.append(f"배치 수: {counters.get('symbol_batches', 0)}개")
                    lines.append(f"평균 배치 크기: {counters.get('symbol_avg_batch_size', 0)}개")
                    lines.append("")
                
                # 파일별 통계
                if "files_with_semantic_nodes" in counters:
                    lines.append("## 3-1-7. 파일별 Semantic Node 통계")
                    lines.append("-" * 80)
                    with_nodes = counters.get('files_with_semantic_nodes', 0)
                    without_nodes = counters.get('files_without_semantic_nodes', 0)
                    total_files = with_nodes + without_nodes
                    
                    lines.append(f"Semantic node 있는 파일: {with_nodes}개 ({with_nodes/total_files*100:.1f}%)")
                    lines.append(f"Semantic node 없는 파일: {without_nodes}개 ({without_nodes/total_files*100:.1f}%)")
                    lines.append("")
            
            # Route Semantics 메트릭
            if "route_semantic_count" in indexing_phase.counters:
                lines.append("## 3-2. Route Semantics 통계")
                lines.append("-" * 80)
                counters = indexing_phase.counters
                
                lines.append(f"Route 수: {counters.get('route_semantic_count', 0)}개")
                
                if "route_summary_time_ms" in counters:
                    summary_ms = counters.get('route_summary_time_ms', 0)
                    embed_ms = counters.get('route_embed_time_ms', 0)
                    save_ms = counters.get('route_save_time_ms', 0)
                    total_ms = summary_ms + embed_ms + save_ms
                    
                    if total_ms > 0:
                        lines.append(f"\n시간 분해:")
                        lines.append(f"  Summary: {summary_ms}ms ({summary_ms/total_ms*100:.1f}%)")
                        lines.append(f"  임베딩: {embed_ms}ms ({embed_ms/total_ms*100:.1f}%)")
                        lines.append(f"  DB 저장: {save_ms}ms ({save_ms/total_ms*100:.1f}%)")
                
                if "route_summary_lengths" in counters:
                    lengths = counters['route_summary_lengths']
                    lines.append(f"\nSummary 길이:")
                    lines.append(f"  평균: {lengths.get('avg', 0)}자 (min:{lengths.get('min', 0)}, max:{lengths.get('max', 0)})")
                
                if "route_tokens" in counters:
                    lines.append(f"\n비용:")
                    lines.append(f"  토큰: {counters.get('route_tokens', 0)}")
                    lines.append(f"  예상 비용: ${counters.get('route_estimated_cost', 0):.6f}")
                    lines.append(f"  API 호출: {counters.get('route_api_calls', 0)}회")
                
                lines.append("")
            
            # 기존 symbols_by_file 정보
            if semantic_subphase and "symbols_by_file" in semantic_subphase.counters:
                lines.append("## 3-3. Semantic Nodes 파일별 심볼 수")
                lines.append("-" * 80)

                symbols_by_file = semantic_subphase.counters["symbols_by_file"]
                total_symbols = semantic_subphase.counters.get(
                    "total_symbols", sum(symbols_by_file.values())
                )
                total_time_ms = semantic_subphase.elapsed_ms

                sorted_files = sorted(symbols_by_file.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]

                for idx, (file_path, symbol_count) in enumerate(sorted_files, 1):
                    pct = (symbol_count / total_symbols) * 100 if total_symbols > 0 else 0
                    estimated_time = (
                        (symbol_count / total_symbols) * total_time_ms if total_symbols > 0 else 0
                    )
                    lines.append(f"{idx}. {file_path}")
                    lines.append(f"   심볼 수: {symbol_count}개 ({pct:.1f}%)")
                    lines.append(f"   예상 시간: {estimated_time:.0f}ms")
                    lines.append("")

                lines.append(f"총 {len(symbols_by_file)}개 파일, {total_symbols}개 심볼 처리")
                lines.append("")

        # 4. 데이터베이스 성능
        lines.append("## 4. 데이터베이스 성능")
        lines.append("-" * 80)
        lines.append("총 쿼리 수: 0개")
        lines.append("총 쿼리 시간: 0.00초")
        lines.append("")

        # 5. 성능 분석
        lines.append("## 5. 성능 분석")
        lines.append("-" * 80)

        if indexing_phase:
            files = indexing_phase.counters.get("files_parsed", 0)
            if files > 0:
                avg_time = total_ms / files
                lines.append(f"파일당 평균 처리 시간: {avg_time:.2f}ms")

        lines.append("")
        lines.append("병목 구간:")
        if self.phases:
            slowest = max(self.phases, key=lambda p: p.elapsed_ms)
            pct = (slowest.elapsed_ms / total_ms) * 100 if total_ms > 0 else 0
            lines.append(
                f"  가장 느린 Phase: {slowest.name} ({slowest.elapsed_ms/1000:.2f}초, {pct:.1f}%)"
            )

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _add_waterfall(self, lines: list[str], total_ms: float):
        """Waterfall 차트 추가"""
        width = 60

        for phase in self.phases:
            # 시작 위치 계산
            start_pos = int((phase.start_time - self.run_start_time) * 1000 / total_ms * width)
            duration_width = max(1, int(phase.elapsed_ms / total_ms * width))

            # 바 그리기
            bar = " " * start_pos + "█" * duration_width
            lines.append(f"{phase.name:<30}│{bar}")
            lines.append(
                f"{'':30}│  시작: {phase.start_time - self.run_start_time:6.2f}s, "
                f"종료: {phase.end_time - self.run_start_time:6.2f}s, "
                f"소요: {phase.elapsed_ms/1000:6.2f}s ({phase.elapsed_ms/total_ms*100:5.1f}%), "
                f"메모리: {phase.mem_delta_mb:+.1f}MB"
            )

            # 카운터 출력
            if phase.counters:
                counter_str = ", ".join([f"{k}: {v}" for k, v in phase.counters.items()])
                lines.append(f"{'':30}│  카운터: {counter_str}")

            # Sub-phases
            if phase.sub_phases:
                for sub in phase.sub_phases:
                    sub_pct = (
                        (sub.elapsed_ms / phase.elapsed_ms) * 100 if phase.elapsed_ms > 0 else 0
                    )
                    lines.append(
                        f"{'':30}│    └─ {sub.name}: {sub.elapsed_ms/1000:.2f}s ({sub_pct:.1f}%)"
                    )

            lines.append("")

        # 시간 축
        lines.append(f"{'':30}└{'─' * width}")
        tick_line = f"{'':30}|"
        for i in range(5):
            pos = i * width // 4
            time_val = i * total_ms / 1000 / 4
            tick_line += f"{' ' * (pos - len(tick_line) + 30)}|"
        lines.append(tick_line)

        time_labels = f"{'':30}"
        for i in range(5):
            time_val = i * total_ms / 1000 / 4
            label = f"{time_val:.1f}s"
            time_labels += f"{label:>14}"
        lines.append(time_labels)

    def _add_phase_summary(
        self, lines: list[str], phase: PhaseMetrics, total_ms: float, indent: int
    ):
        """Phase 요약 테이블 행 추가"""
        indent_str = "  " * indent
        if indent > 0:
            indent_str += "└─ "

        pct = (phase.elapsed_ms / total_ms) * 100 if total_ms > 0 else 0
        mem_str = (
            f"+{phase.mem_delta_mb:.1f}" if phase.mem_delta_mb >= 0 else f"{phase.mem_delta_mb:.1f}"
        )

        lines.append(
            f"{indent_str}{phase.name:<{35-len(indent_str)}} "
            f"{phase.elapsed_ms:>10.0f} {pct:>10.1f} {mem_str:>15}"
        )

        for sub in phase.sub_phases:
            self._add_phase_summary(lines, sub, phase.elapsed_ms, indent + 1)
