"""CLI 진입점"""

import sys
from typing import Optional

import click

from src.core.bootstrap import create_bootstrap


@click.group()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="설정 파일 경로 (.env 파일)",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="대화형 모드 실행",
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], interactive: bool) -> None:
    """Semantica Codegraph CLI"""
    ctx.ensure_object(dict)
    
    # 설정 로드
    if config:
        from dotenv import load_dotenv
        load_dotenv(config)
    
    # Bootstrap 인스턴스 생성
    try:
        bootstrap = create_bootstrap()
        ctx.obj["bootstrap"] = bootstrap
    except Exception as e:
        click.echo(f"설정 로드 실패: {e}", err=True)
        sys.exit(1)
    
    # 대화형 모드
    if interactive:
        from .interactive import run_interactive
        run_interactive()
        sys.exit(0)


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--repo-id", help="저장소 ID (기본값: 자동 생성)")
@click.option("--name", help="저장소 이름 (기본값: 디렉토리 이름)")
@click.pass_context
def index(ctx: click.Context, repo_path: str, repo_id: Optional[str], name: Optional[str]) -> None:
    """저장소 인덱싱"""
    bootstrap = ctx.obj["bootstrap"]
    
    click.echo(f"인덱싱 시작: {repo_path}")
    try:
        result = bootstrap.pipeline.index_repository(
            root_path=repo_path,
            repo_id=repo_id,
            name=name,
        )
        click.echo(f"인덱싱 완료: {result.repo_id}")
        click.echo(f"  - 노드: {result.total_nodes}개")
        click.echo(f"  - 엣지: {result.total_edges}개")
        click.echo(f"  - 청크: {result.total_chunks}개")
        click.echo(f"  - 소요시간: {result.duration_seconds:.2f}초")
    except Exception as e:
        click.echo(f"인덱싱 실패: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--repo-id", required=True, help="저장소 ID")
@click.option("--limit", default=10, help="결과 개수")
@click.pass_context
def search(ctx: click.Context, query: str, repo_id: str, limit: int) -> None:
    """코드 검색"""
    bootstrap = ctx.obj["bootstrap"]
    
    click.echo(f"검색 중: {query}")
    # TODO: 검색 기능 구현
    click.echo("검색 기능은 아직 구현되지 않았습니다.")


def main() -> None:
    """CLI 진입점"""
    # -i 또는 --interactive 옵션이 있거나 인자가 없으면 대화형 모드로 실행
    if len(sys.argv) == 1 or "-i" in sys.argv or "--interactive" in sys.argv:
        # Bootstrap 초기화
        try:
            bootstrap = create_bootstrap()
        except Exception as e:
            from rich.console import Console
            console = Console()
            console.print(f"[bold red]초기화 실패[/bold red]")
            console.print(f"[red]{str(e)}[/red]")
            console.print("\n환경변수를 확인해주세요.")
            sys.exit(1)
        
        # 대화형 모드 실행
        from .interactive import run_interactive
        run_interactive()
    else:
        cli()


if __name__ == "__main__":
    main()

