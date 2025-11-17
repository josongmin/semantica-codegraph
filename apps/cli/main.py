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
@click.pass_context
def cli(ctx: click.Context, config: Optional[str]) -> None:
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
        click.echo(f"  - 노드: {result.nodes_count}개")
        click.echo(f"  - 엣지: {result.edges_count}개")
        click.echo(f"  - 청크: {result.chunks_count}개")
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
    cli()


if __name__ == "__main__":
    main()

