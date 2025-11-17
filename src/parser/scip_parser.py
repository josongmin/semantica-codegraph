"""SCIP 기반 의미론적 파서"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.models import RawRelation, RawSymbol, RepoId, Span
from ..core.ports import ParserPort

logger = logging.getLogger(__name__)


class ScipParser(ParserPort):
    """
    SCIP (SCIP Code Intelligence Protocol) 기반 의미론적 파서
    
    SCIP는 Tree-sitter와 달리 의미론적(semantic) 정보를 제공합니다:
    - 심볼 정의와 참조
    - 타입 정보
    - 호출 관계
    - 크로스 파일 참조
    
    사용 방법:
    1. 사전에 SCIP 인덱스 생성 필요 (scip-python, scip-typescript 등)
    2. 생성된 index.scip 파일을 파싱
    
    Note:
        SCIP는 프로젝트 전체를 인덱싱하므로 느립니다.
        Tree-sitter로 기본 구조를 파악한 후, SCIP로 관계를 보강하는 방식 권장.
    """

    def __init__(
        self,
        scip_index_path: Optional[Path] = None,
        auto_index: bool = False
    ):
        """
        Args:
            scip_index_path: SCIP 인덱스 파일 경로 (.scip)
            auto_index: True면 인덱스가 없을 때 자동 생성
        """
        self.scip_index_path = scip_index_path
        self.auto_index = auto_index
        self._index_data: Optional[Dict] = None

    def parse_file(
        self,
        file_meta: dict
    ) -> Tuple[List[RawSymbol], List[RawRelation]]:
        """
        SCIP 인덱스에서 파일 정보 추출
        
        Args:
            file_meta: 파일 메타데이터
        
        Returns:
            (RawSymbol 리스트, RawRelation 리스트)
        """
        # SCIP 인덱스 로드
        if not self._ensure_index(file_meta):
            logger.warning(f"No SCIP index available for {file_meta['path']}")
            return [], []

        try:
            # SCIP 데이터 추출
            scip_data = self._load_file_data(file_meta["path"])
            
            if not scip_data:
                logger.debug(f"No SCIP data for {file_meta['path']}")
                return [], []

            # 심볼 추출
            symbols = self._extract_symbols_from_scip(scip_data, file_meta)
            
            # 관계 추출
            relations = self._extract_relations_from_scip(scip_data, file_meta)
            
            logger.debug(
                f"SCIP parsed {file_meta['path']}: "
                f"{len(symbols)} symbols, {len(relations)} relations"
            )
            
            return symbols, relations
            
        except Exception as e:
            logger.error(f"Failed to parse SCIP data for {file_meta['path']}: {e}")
            return [], []

    def _ensure_index(self, file_meta: dict) -> bool:
        """SCIP 인덱스가 존재하는지 확인하고 필요시 생성"""
        if self.scip_index_path and self.scip_index_path.exists():
            return True

        if self.auto_index:
            logger.info("Auto-indexing with SCIP...")
            return self._generate_scip_index(file_meta.get("repo_root", "."))

        return False

    def _generate_scip_index(self, repo_root: str) -> bool:
        """
        SCIP 인덱스 생성
        
        Args:
            repo_root: 저장소 루트 경로
        
        Returns:
            성공 여부
        
        Note:
            언어별로 다른 인덱서 필요:
            - Python: scip-python
            - TypeScript: scip-typescript
            - Go: scip-go
        """
        repo_path = Path(repo_root)
        index_path = repo_path / "index.scip"

        # Python 프로젝트 감지
        if (repo_path / "pyproject.toml").exists() or (repo_path / "setup.py").exists():
            try:
                logger.info("Indexing Python project with scip-python...")
                result = subprocess.run(
                    ["scip-python", "index", "--project-dir", str(repo_path)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5분 타임아웃
                )
                
                if result.returncode == 0:
                    self.scip_index_path = index_path
                    logger.info(f"SCIP index created: {index_path}")
                    return True
                else:
                    logger.error(f"scip-python failed: {result.stderr}")
                    return False
                    
            except FileNotFoundError:
                logger.error("scip-python not found. Install: pip install scip-python")
                return False
            except subprocess.TimeoutExpired:
                logger.error("SCIP indexing timed out")
                return False

        # TypeScript 프로젝트 감지
        elif (repo_path / "package.json").exists():
            try:
                logger.info("Indexing TypeScript project with scip-typescript...")
                result = subprocess.run(
                    ["scip-typescript", "index"],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    self.scip_index_path = index_path
                    return True
                else:
                    logger.error(f"scip-typescript failed: {result.stderr}")
                    return False
                    
            except FileNotFoundError:
                logger.error("scip-typescript not found. Install: npm install -g @sourcegraph/scip-typescript")
                return False

        logger.warning(f"Unknown project type in {repo_root}")
        return False

    def _load_file_data(self, file_path: str) -> Optional[Dict]:
        """
        SCIP 인덱스에서 특정 파일의 데이터 추출
        
        Args:
            file_path: 파일 경로
        
        Returns:
            파일의 SCIP 데이터 (Document)
        
        Note:
            SCIP는 protobuf 형식이지만, 여기서는 단순화를 위해
            JSON 변환 후 처리한다고 가정합니다.
            실제로는 protobuf 파싱이 필요합니다.
        """
        if not self.scip_index_path:
            return None

        # TODO: 실제 SCIP protobuf 파싱 구현
        # 현재는 placeholder
        logger.debug(f"Loading SCIP data for {file_path}")
        
        # scip snapshot 명령으로 JSON 변환 가능
        try:
            result = subprocess.run(
                ["scip", "snapshot", "--from", str(self.scip_index_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # JSON 파싱
                snapshot = json.loads(result.stdout)
                
                # 파일에 해당하는 Document 찾기
                for document in snapshot.get("documents", []):
                    if document.get("relative_path") == file_path:
                        return document
                        
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            logger.debug(f"Failed to load SCIP snapshot: {e}")

        return None

    def _extract_symbols_from_scip(
        self,
        scip_data: Dict,
        file_meta: dict
    ) -> List[RawSymbol]:
        """
        SCIP 데이터에서 심볼 추출
        
        Args:
            scip_data: SCIP Document 데이터
            file_meta: 파일 메타데이터
        
        Returns:
            RawSymbol 리스트 (타입 정보 포함)
        """
        symbols = []

        for occurrence in scip_data.get("occurrences", []):
            # Definition만 추출 (symbol_roles & 1)
            if not (occurrence.get("symbol_roles", 0) & 1):
                continue

            symbol_str = occurrence.get("symbol", "")
            scip_range = occurrence.get("range", [])

            # SymbolInformation 조회
            symbol_info = self._get_symbol_information(scip_data, symbol_str)

            symbols.append(RawSymbol(
                repo_id=file_meta["repo_id"],
                file_path=file_meta["path"],
                language=file_meta["language"],
                kind=self._scip_kind_to_kind(symbol_info),
                name=self._extract_symbol_name(symbol_str),
                span=self._scip_range_to_span(scip_range),
                attrs={
                    "scip_symbol": symbol_str,
                    "signature": symbol_info.get("signature_documentation", {}).get("text") if symbol_info else None,
                    "documentation": symbol_info.get("documentation", [None])[0] if symbol_info else None,
                    "kind_string": symbol_info.get("kind") if symbol_info else None
                }
            ))

        return symbols

    def _extract_relations_from_scip(
        self,
        scip_data: Dict,
        file_meta: dict
    ) -> List[RawRelation]:
        """
        SCIP 데이터에서 관계 추출
        
        Args:
            scip_data: SCIP Document 데이터
            file_meta: 파일 메타데이터
        
        Returns:
            RawRelation 리스트 (정확한 참조 관계)
        """
        relations = []

        for occurrence in scip_data.get("occurrences", []):
            symbol_roles = occurrence.get("symbol_roles", 0)
            
            # Reference (not definition)
            if not (symbol_roles & 1) and (symbol_roles > 0):
                relation_type = self._symbol_roles_to_relation_type(symbol_roles)
                
                relations.append(RawRelation(
                    repo_id=file_meta["repo_id"],
                    file_path=file_meta["path"],
                    language=file_meta["language"],
                    type=relation_type,
                    src_span=self._scip_range_to_span(occurrence.get("range", [])),
                    dst_span=(0, 0, 0, 0),  # 정의 위치는 심볼 조회 필요
                    attrs={
                        "target_symbol": occurrence.get("symbol", ""),
                        "symbol_roles": symbol_roles
                    }
                ))

        return relations

    def _get_symbol_information(
        self,
        scip_data: Dict,
        symbol: str
    ) -> Optional[Dict]:
        """심볼 정보 조회"""
        for symbol_info in scip_data.get("symbols", []):
            if symbol_info.get("symbol") == symbol:
                return symbol_info
        return None

    def _scip_range_to_span(self, scip_range: List[int]) -> Span:
        """
        SCIP range → Span 변환
        
        SCIP range: [start_line, start_char, end_line, end_char]
        """
        if len(scip_range) >= 3:
            return (
                scip_range[0],
                scip_range[1],
                scip_range[2],
                scip_range[3] if len(scip_range) > 3 else scip_range[1]
            )
        return (0, 0, 0, 0)

    def _scip_kind_to_kind(self, symbol_info: Optional[Dict]) -> str:
        """SCIP SymbolInformation.Kind → RawSymbol.kind"""
        if not symbol_info:
            return "Unknown"

        kind = symbol_info.get("kind", "")
        kind_map = {
            "Method": "Method",
            "Function": "Function",
            "Class": "Class",
            "Interface": "Interface",
            "Type": "Type",
            "Variable": "Variable",
            "Constant": "Constant",
            "Module": "Module",
            "Namespace": "Namespace"
        }
        return kind_map.get(kind, "Unknown")

    def _extract_symbol_name(self, scip_symbol: str) -> str:
        """
        SCIP 심볼 문자열에서 이름 추출
        
        SCIP symbol format: scip-typescript npm <package> <version> <path>`<name>`
        """
        if "`" in scip_symbol:
            # 마지막 ` 사이의 내용이 심볼 이름
            parts = scip_symbol.split("`")
            if len(parts) >= 2:
                return parts[-2]
        
        # 파싱 실패 시 전체 반환
        return scip_symbol.split("/")[-1] if "/" in scip_symbol else scip_symbol

    def _symbol_roles_to_relation_type(self, roles: int) -> str:
        """
        SCIP SymbolRole → RawRelation.type
        
        SymbolRole flags:
        - 1: Definition
        - 2: Import
        - 4: WriteAccess
        - 8: ReadAccess
        - 16: Generated
        - 32: Test
        """
        if roles & 2:
            return "imports"
        elif roles & 4:
            return "writes"
        elif roles & 8:
            return "reads"
        else:
            return "references"

