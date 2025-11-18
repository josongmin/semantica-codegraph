"""CLI 진입점"""

import sys
from pathlib import Path

import click

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.bootstrap import create_bootstrap


def main() -> None:
    """CLI 진입점"""
    # 기본적으로 대화형 모드 실행
    from apps.cli.interactive import run_interactive
    run_interactive()


if __name__ == "__main__":
    main()

