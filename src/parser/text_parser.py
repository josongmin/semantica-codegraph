"""텍스트 파일 파서"""

import logging
from pathlib import Path

from ..core.models import RawRelation, RawSymbol
from ..core.ports import ParserPort

logger = logging.getLogger(__name__)


class TextParser(ParserPort):
    """
    텍스트/마크다운 파일 파서

    텍스트 파일을 단일 Document 노드로 변환
    """

    def parse_file(self, file_meta: dict) -> tuple[list[RawSymbol], list[RawRelation]]:
        """
        텍스트 파일을 Document 노드로 변환

        Args:
            file_meta: 파일 메타데이터

        Returns:
            ([Document 심볼], []) - 관계는 없음
        """
        abs_path = Path(file_meta["abs_path"])

        try:
            with open(abs_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {abs_path}: {e}")
            return [], []

        # 줄 수 계산 (빈 파일은 0줄)
        if not content:
            line_count = 0
        else:
            line_count = content.count('\n') + (0 if content.endswith('\n') else 1)

        # 파일 전체를 하나의 Document 노드로
        symbol = RawSymbol(
            repo_id=file_meta["repo_id"],
            file_path=file_meta["path"],
            language=file_meta["language"],
            kind="Document",
            name=abs_path.stem,
            span=(0, 0, line_count, len(content)),
            attrs={
                "text": content,  # 전체 텍스트 내용
                "file_type": abs_path.suffix,
                "size": len(content),
                "lines": line_count
            }
        )

        return [symbol], []  # 관계 없음
