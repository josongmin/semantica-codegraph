"""Enhanced Parser - 모든 분석 기법 통합

정적 + 동적 분석을 모두 통합한 최종 Python 파서

구성:
1. Tree-sitter: 기본 구조 파싱 (80%)
2. TypeHintAnalyzer: 타입 힌트 기반 동적 호출 (+5%)
3. PatternAnalyzer: 프레임워크 패턴 (+3%)
4. TestCodeAnalyzer: 테스트 코드 분석 (+2%)

총 90% 커버리지 목표
"""

import logging
import time

from ..core.models import RawRelation, RawSymbol
from ..core.ports import ParserPort
from .pattern_analyzer import PatternAnalyzer, detect_framework
from .python_parser import PythonTreeSitterParser
from .test_code_analyzer import TestCodeAnalyzer
from .type_hint_analyzer import TypeHintAnalyzer

logger = logging.getLogger(__name__)


class EnhancedParser(ParserPort):
    """
    강화된 Python 파서

    통합:
    - Tree-sitter (정적 분석)
    - TypeHintAnalyzer (타입 힌트)
    - PatternAnalyzer (프레임워크 패턴)
    - TestCodeAnalyzer (테스트 코드)

    성능 가드:
    - 테스트 파일만 TestCodeAnalyzer 적용
    - 프레임워크 확인 시에만 PatternAnalyzer 적용
    - 성능 영향 10% 이내 목표

    측정:
    - attrs["method"] ∈ {type_hint, pattern, test_analysis}
    - confidence 평균 ≥ 0.85
    """

    def __init__(
        self,
        framework: str | None = None,
        enable_type_hint: bool = True,
        enable_pattern: bool = True,
        enable_test: bool = True,
    ):
        """
        Args:
            framework: 프레임워크 ("django", "flask", None=auto-detect)
            enable_type_hint: 타입 힌트 분석 활성화
            enable_pattern: 패턴 분석 활성화
            enable_test: 테스트 분석 활성화
        """
        # 기본 파서
        self.base_parser = PythonTreeSitterParser()

        # 동적 분석기들
        self.type_hint_analyzer = TypeHintAnalyzer() if enable_type_hint else None
        self.pattern_analyzer = None  # 나중에 초기화 (프레임워크 감지 후)
        self.test_analyzer = TestCodeAnalyzer() if enable_test else None

        self.framework = framework
        self.enable_pattern = enable_pattern

        # 성능 통계
        self.stats = {
            "base_time": 0.0,
            "type_hint_time": 0.0,
            "pattern_time": 0.0,
            "test_time": 0.0,
        }

        logger.info(
            f"EnhancedParser initialized: framework={framework}, "
            f"type_hint={enable_type_hint}, pattern={enable_pattern}, test={enable_test}"
        )

    def parse_file(
        self,
        file_meta: dict
    ) -> tuple[list[RawSymbol], list[RawRelation]]:
        """
        강화된 파싱

        Args:
            file_meta: 파일 메타데이터

        Returns:
            (RawSymbol 리스트, RawRelation 리스트)
        """
        file_path = file_meta.get("path", file_meta.get("file_path", ""))
        abs_path = file_meta.get("abs_path", "")

        # 1. 기본 정적 분석 (Tree-sitter)
        start = time.time()
        symbols, relations = self.base_parser.parse_file(file_meta)
        self.stats["base_time"] += time.time() - start

        # 파일 읽기 (동적 분석용)
        try:
            with open(abs_path, encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.warning(f"Failed to read file for dynamic analysis: {e}")
            return symbols, relations

        # 심볼 이름 세트 (검증용)
        symbol_names = {s.name for s in symbols}

        # 2. 타입 힌트 분석
        if self.type_hint_analyzer:
            start = time.time()
            type_hint_calls = self.type_hint_analyzer.analyze(code, file_path)
            self.stats["type_hint_time"] += time.time() - start

            # RawRelation으로 변환
            for call in type_hint_calls:
                # 추론된 호출은 심볼 이름만 있으므로
                # span은 더미, attrs에 심볼 정보 저장
                relations.append(
                    RawRelation(
                        repo_id=file_meta.get("repo_id", ""),
                        file_path=file_path,
                        language="python",
                        type="calls",
                        src_span=(call.line, 0, call.line, 0),  # 더미 span
                        dst_span=(0, 0, 0, 0),  # 더미 span
                        attrs={
                            "confidence": call.confidence,
                            "inferred": True,
                            "method": "type_hint",
                            "source_symbol": call.source,
                            "target_symbol": call.target,
                            "line": call.line
                        }
                    )
                )

            if type_hint_calls:
                logger.debug(
                    f"Type hint: {len(type_hint_calls)} calls inferred"
                )

        # 3. 패턴 분석 (프레임워크 있을 때만)
        if self.enable_pattern:
            # 프레임워크 감지 (설정 or 자동)
            detected_framework = self.framework or detect_framework(code, file_path)

            if detected_framework:
                if not self.pattern_analyzer:
                    # 첫 감지 시 PatternAnalyzer 초기화
                    self.pattern_analyzer = PatternAnalyzer(detected_framework)
                    logger.info(f"Framework detected: {detected_framework}")

                start = time.time()
                pattern_matches = self.pattern_analyzer.analyze(
                    code,
                    file_path,
                    symbol_names
                )
                self.stats["pattern_time"] += time.time() - start

                # RawRelation으로 변환
                pattern_relations = self.pattern_analyzer.to_relations(
                    pattern_matches,
                    file_meta.get("repo_id", ""),
                    file_path
                )

                for rel_dict in pattern_relations:
                    relations.append(
                        RawRelation(
                            repo_id=file_meta.get("repo_id", ""),
                            file_path=file_path,
                            language="python",
                            type=rel_dict["type"],
                            src_span=(rel_dict["attrs"].get("line", 0), 0, rel_dict["attrs"].get("line", 0), 0),
                            dst_span=(0, 0, 0, 0),
                            attrs={
                                **rel_dict["attrs"],
                                "source_symbol": rel_dict["source"],
                                "target_symbol": rel_dict["target"]
                            }
                        )
                    )

                if pattern_matches:
                    logger.debug(
                        f"Pattern: {len(pattern_matches)} matches found"
                    )

        # 4. 테스트 분석 (테스트 파일만)
        if self.test_analyzer and TestCodeAnalyzer.is_test_file(file_path):
            start = time.time()
            test_calls = self.test_analyzer.analyze(code, file_path, symbol_names)
            self.stats["test_time"] += time.time() - start

            # RawRelation으로 변환
            test_relations = self.test_analyzer.to_relations(
                test_calls,
                file_meta.get("repo_id", ""),
                file_path
            )

            for rel_dict in test_relations:
                relations.append(
                    RawRelation(
                        repo_id=file_meta.get("repo_id", ""),
                        file_path=file_path,
                        language="python",
                        type=rel_dict["type"],
                        src_span=(rel_dict["attrs"].get("line", 0), 0, rel_dict["attrs"].get("line", 0), 0),
                        dst_span=(0, 0, 0, 0),
                        attrs={
                            **rel_dict["attrs"],
                            "source_symbol": rel_dict["source"],
                            "target_symbol": rel_dict["target"]
                        }
                    )
                )

            if test_calls:
                logger.debug(
                    f"Test: {len(test_calls)} calls extracted"
                )

        # 통계 로깅
        inferred_count = sum(1 for r in relations if r.attrs.get("inferred"))
        if inferred_count > 0:
            logger.info(
                f"Enhanced parsing: {len(symbols)} symbols, "
                f"{len(relations)} relations ({inferred_count} inferred)"
            )

        return symbols, relations

    def get_performance_stats(self) -> dict:
        """
        성능 통계

        Returns:
            {"base_time": 0.5, "type_hint_time": 0.1, ...}
        """
        return self.stats.copy()

    def reset_stats(self):
        """통계 초기화"""
        for key in self.stats:
            self.stats[key] = 0.0

