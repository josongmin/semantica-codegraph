"""대화형 CLI 실행"""

import sys

from apps.cli.app2 import SemanticaApp
from src.core.bootstrap import create_bootstrap


def run_interactive() -> None:
    """대화형 CLI 실행"""
    try:
        bootstrap = create_bootstrap()
    except Exception as e:
        from rich.console import Console

        console = Console()
        console.print("[bold red]초기화 실패[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        console.print("\n환경변수를 확인해주세요.")
        sys.exit(1)

    # Textual 앱 실행
    app = SemanticaApp(bootstrap)
    app.run()
