"""Tree-sitter 파싱 결과 캐시 저장소"""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from ..core.models import RawRelation, RawSymbol

logger = logging.getLogger(__name__)


class ParseCache:
    """
    Tree-sitter 파싱 결과를 JSON 파일로 캐시하는 저장소
    
    캐시 구조:
    .semantica-cache/
      {repo_id}/
        {file_hash}.json
    
    JSON 형식:
    {
        "file_path": "src/example.py",
        "file_hash": "abc123...",
        "symbols": [...],
        "relations": [...]
    }
    """

    def __init__(self, cache_root: Optional[Path] = None):
        """
        Args:
            cache_root: 캐시 루트 디렉토리 (None이면 .semantica-cache 사용)
        """
        if cache_root is None:
            cache_root = Path.cwd() / ".semantica-cache"
        
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Parse cache initialized at {self.cache_root}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산 (SHA256)"""
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute hash for {file_path}: {e}")
            return ""

    def _get_cache_path(self, repo_id: str, file_hash: str) -> Path:
        """캐시 파일 경로 생성"""
        repo_cache_dir = self.cache_root / repo_id
        repo_cache_dir.mkdir(parents=True, exist_ok=True)
        return repo_cache_dir / f"{file_hash}.json"

    def get(
        self,
        repo_id: str,
        file_path: Path,
        current_hash: Optional[str] = None
    ) -> Optional[Tuple[List[RawSymbol], List[RawRelation]]]:
        """
        캐시에서 파싱 결과 조회
        
        Args:
            repo_id: 저장소 ID
            file_path: 파일 경로
            current_hash: 현재 파일 해시 (None이면 계산)
        
        Returns:
            (symbols, relations) 튜플 또는 None (캐시 미스)
        """
        if not file_path.exists():
            return None

        if current_hash is None:
            current_hash = self._compute_file_hash(file_path)

        if not current_hash:
            return None

        cache_path = self._get_cache_path(repo_id, current_hash)
        
        if not cache_path.exists():
            logger.debug(f"Cache miss: {file_path}")
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 파일 경로 확인 (같은 해시지만 다른 파일일 수 있음)
            if data.get("file_path") != str(file_path):
                logger.debug(f"Cache path mismatch: {data.get('file_path')} != {file_path}")
                return None

            # 해시 확인
            if data.get("file_hash") != current_hash:
                logger.debug(f"Cache hash mismatch for {file_path}")
                return None

            # 역직렬화
            symbols = [self._deserialize_symbol(s) for s in data.get("symbols", [])]
            relations = [self._deserialize_relation(r) for r in data.get("relations", [])]

            logger.debug(f"Cache hit: {file_path} ({len(symbols)} symbols, {len(relations)} relations)")
            return symbols, relations

        except Exception as e:
            logger.warning(f"Failed to load cache for {file_path}: {e}")
            return None

    def save(
        self,
        repo_id: str,
        file_path: Path,
        symbols: List[RawSymbol],
        relations: List[RawRelation],
        file_hash: Optional[str] = None
    ) -> None:
        """
        파싱 결과를 캐시에 저장
        
        Args:
            repo_id: 저장소 ID
            file_path: 파일 경로
            symbols: 추출된 심볼 리스트
            relations: 추출된 관계 리스트
            file_hash: 파일 해시 (None이면 계산)
        """
        if not file_path.exists():
            logger.warning(f"Cannot cache non-existent file: {file_path}")
            return

        if file_hash is None:
            file_hash = self._compute_file_hash(file_path)

        if not file_hash:
            logger.warning(f"Cannot cache file without hash: {file_path}")
            return

        cache_path = self._get_cache_path(repo_id, file_hash)

        try:
            data = {
                "file_path": str(file_path),
                "file_hash": file_hash,
                "symbols": [self._serialize_symbol(s) for s in symbols],
                "relations": [self._serialize_relation(r) for r in relations],
            }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Cached: {file_path} -> {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save cache for {file_path}: {e}")

    def clear_repo(self, repo_id: str) -> None:
        """특정 저장소의 캐시 삭제"""
        repo_cache_dir = self.cache_root / repo_id
        if repo_cache_dir.exists():
            import shutil
            shutil.rmtree(repo_cache_dir)
            logger.info(f"Cleared cache for repo: {repo_id}")

    def clear_all(self) -> None:
        """전체 캐시 삭제"""
        if self.cache_root.exists():
            import shutil
            shutil.rmtree(self.cache_root)
            logger.info("Cleared all cache")

    def _serialize_symbol(self, symbol: RawSymbol) -> dict:
        """RawSymbol → dict 변환"""
        return {
            "repo_id": symbol.repo_id,
            "file_path": symbol.file_path,
            "language": symbol.language,
            "kind": symbol.kind,
            "name": symbol.name,
            "span": list(symbol.span),
            "attrs": symbol.attrs,
        }

    def _deserialize_symbol(self, data: dict) -> RawSymbol:
        """dict → RawSymbol 변환"""
        return RawSymbol(
            repo_id=data["repo_id"],
            file_path=data["file_path"],
            language=data["language"],
            kind=data["kind"],
            name=data["name"],
            span=tuple(data["span"]),
            attrs=data.get("attrs", {}),
        )

    def _serialize_relation(self, relation: RawRelation) -> dict:
        """RawRelation → dict 변환"""
        return {
            "repo_id": relation.repo_id,
            "file_path": relation.file_path,
            "language": relation.language,
            "type": relation.type,
            "src_span": list(relation.src_span),
            "dst_span": list(relation.dst_span),
            "attrs": relation.attrs,
        }

    def _deserialize_relation(self, data: dict) -> RawRelation:
        """dict → RawRelation 변환"""
        return RawRelation(
            repo_id=data["repo_id"],
            file_path=data["file_path"],
            language=data["language"],
            type=data["type"],
            src_span=tuple(data["src_span"]),
            dst_span=tuple(data["dst_span"]),
            attrs=data.get("attrs", {}),
        )

