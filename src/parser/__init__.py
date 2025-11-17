"""Parser 팩토리 및 공통 인터페이스"""

import logging
from pathlib import Path
from typing import Optional

from .base import BaseTreeSitterParser
from .hybrid_parser import HybridParser
from .python_parser import PythonTreeSitterParser
from .scip_parser import ScipParser
from .typescript_parser import TypeScriptTreeSitterParser
from ..core.ports import ParserPort

logger = logging.getLogger(__name__)


def create_parser(
    language: str,
    use_scip: bool = False,
    use_hybrid: bool = False,
    scip_index_path: Optional[Path] = None
) -> Optional[ParserPort]:
    """
    언어별 파서 생성
    
    Args:
        language: 언어 이름 (python, typescript, javascript 등)
        use_scip: True면 SCIP 파서만 사용
        use_hybrid: True면 하이브리드 파서 (Tree-sitter + SCIP)
        scip_index_path: SCIP 인덱스 파일 경로 (use_hybrid 시 사용)
    
    Returns:
        ParserPort 구현체 (지원 안 되는 언어면 None)
    
    Example:
        >>> # 1. Tree-sitter만 (빠름, 구조만)
        >>> parser = create_parser("python")
        >>> 
        >>> # 2. SCIP만 (타입/관계 포함)
        >>> scip_parser = create_parser("python", use_scip=True)
        >>> 
        >>> # 3. 하이브리드 (최고 품질)
        >>> hybrid = create_parser("python", use_hybrid=True)
    """
    # SCIP만 사용
    if use_scip:
        return ScipParser(scip_index_path=scip_index_path)
    
    # 하이브리드 사용
    if use_hybrid:
        ts_parser = _create_tree_sitter_parser(language)
        if ts_parser:
            scip = ScipParser(
                scip_index_path=scip_index_path,
                auto_index=(scip_index_path is None)
            )
            return HybridParser(ts_parser, scip)
        else:
            logger.warning(f"Language {language} not supported for hybrid parsing")
            return None
    
    # Tree-sitter만 사용
    return _create_tree_sitter_parser(language)


def _create_tree_sitter_parser(language: str) -> Optional[BaseTreeSitterParser]:

    """Tree-sitter 파서 생성 (내부 헬퍼)"""
    parsers = {
        "python": PythonTreeSitterParser,
        "typescript": lambda: TypeScriptTreeSitterParser(use_tsx=False),
        "javascript": lambda: TypeScriptTreeSitterParser(use_tsx=False),
        "tsx": lambda: TypeScriptTreeSitterParser(use_tsx=True),
        "jsx": lambda: TypeScriptTreeSitterParser(use_tsx=True),
    }

    parser_factory = parsers.get(language.lower())
    if parser_factory:
        if callable(parser_factory) and not isinstance(parser_factory, type):
            return parser_factory()
        else:
            return parser_factory()

    return None


def create_scip_parser(scip_index_path: Optional[Path] = None) -> ScipParser:
    """
    SCIP 파서 생성
    
    Args:
        scip_index_path: SCIP 인덱스 파일 경로 (.scip)
                        None이면 auto_index=True로 생성
    
    Returns:
        ScipParser 인스턴스
    """
    return ScipParser(
        scip_index_path=scip_index_path,
        auto_index=(scip_index_path is None)
    )


__all__ = [
    "BaseTreeSitterParser",
    "PythonTreeSitterParser",
    "TypeScriptTreeSitterParser",
    "ScipParser",
    "HybridParser",
    "create_parser",
    "create_scip_parser",
]

