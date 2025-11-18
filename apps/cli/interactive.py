"""대화형 CLI 메인 로직"""

import sys
import threading
import time
from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt

from src.core.bootstrap import create_bootstrap
from src.core.models import LocationContext
from src.search.graph.postgres_graph_adapter import PostgresGraphSearch
from src.search.retriever.hybrid_retriever import HybridRetriever
from src.search.semantic.pgvector_adapter import PgVectorSemanticSearch

from .tui import (
    console,
    select_directory,
    select_repo_from_list,
    show_banner,
    show_menu,
    show_search_results,
)

bootstrap = create_bootstrap()


def handle_index_repo():
    """저장소 인덱싱 처리"""
    console.print("\n[bold cyan]저장소 인덱싱[/bold cyan]\n")

    repo_path = select_directory()
    if not repo_path:
        return

    default_name = Path(repo_path).name
    repo_name = Prompt.ask("저장소 이름", default=default_name)

    repo_id = Prompt.ask("저장소 ID (자동 생성: Enter)", default="")
    if not repo_id:
        repo_id = None

    console.print("\n[bold yellow]인덱싱 시작[/bold yellow]")
    console.print(f"경로: {repo_path}")
    console.print(f"이름: {repo_name}\n")

    result_container = {"result": None, "error": None, "done": False}

    def run_indexing():
        try:
            result = bootstrap.pipeline.index_repository(
                root_path=repo_path,
                repo_id=repo_id,
                name=repo_name,
            )
            result_container["result"] = result
        except Exception as e:
            result_container["error"] = e
        finally:
            result_container["done"] = True

    indexing_thread = threading.Thread(target=run_indexing, daemon=True)
    indexing_thread.start()

    actual_repo_id = repo_id if repo_id else Path(repo_path).resolve().name

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("·"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]인덱싱 중...", total=100)

        last_progress = 0.0
        while not result_container["done"]:
            try:
                repo = bootstrap.repo_store.get(actual_repo_id)
                if repo and repo.indexing_progress is not None:
                    current_progress = repo.indexing_progress * 100
                    if abs(current_progress - last_progress) > 0.1:
                        progress.update(task, completed=current_progress)
                        last_progress = current_progress

                        if current_progress < 50:
                            progress.update(task, description="[cyan]파일 파싱 중...")
                        elif current_progress < 70:
                            progress.update(task, description="[yellow]검색 인덱스 생성 중...")
                        elif current_progress < 100:
                            progress.update(task, description="[magenta]임베딩 생성 중...")
            except Exception:
                pass

            time.sleep(0.1)

        progress.update(task, completed=100, description="[bold green]완료!")

    if result_container["error"]:
        console.print("\n[bold red]인덱싱 실패[/bold red]")
        error = result_container["error"]
        console.print(f"[red]{str(error)}[/red]")
        if isinstance(error, Exception):
            import traceback
            console.print("[dim]Traceback:[/dim]")
            console.print(f"[dim]{''.join(traceback.format_exception(type(error), error, error.__traceback__))}[/dim]")
    elif result_container["result"]:
        result = result_container["result"]
        duration = result.duration_seconds

        console.print("\n[bold green]인덱싱 완료![/bold green]")
        console.print(f"  저장소 ID: [cyan]{result.repo_id}[/cyan]")
        console.print(f"  파일: [green]{result.total_files}개[/green]")
        console.print(f"  노드: [green]{result.total_nodes}개[/green]")
        console.print(f"  엣지: [green]{result.total_edges}개[/green]")
        console.print(f"  청크: [green]{result.total_chunks}개[/green]")
        console.print(f"  소요 시간: [yellow]{duration:.2f}초[/yellow]")


def handle_search():
    """코드 검색 처리"""
    console.print("\n[bold cyan]코드 검색[/bold cyan]\n")

    try:
        repos = bootstrap.repo_store.list_all()
        repo_list = [
            {
                "repo_id": repo.repo_id,
                "name": repo.name,
                "root_path": repo.root_path,
                "indexing_status": repo.indexing_status,
            }
            for repo in repos
        ]
    except Exception as e:
        console.print(f"[red]저장소 목록 조회 실패: {e}[/red]")
        return

    repo_id = select_repo_from_list(repo_list)
    if not repo_id:
        return

    query = Prompt.ask("\n검색 쿼리")
    if not query:
        console.print("[yellow]검색 쿼리를 입력해주세요.[/yellow]")
        return

    try:
        limit_str = Prompt.ask("결과 개수", default="20")
        limit = int(limit_str)
    except ValueError:
        limit = 20

    use_location = Confirm.ask("위치 컨텍스트 사용?", default=False)
    location_ctx = None

    if use_location:
        file_path = Prompt.ask("파일 경로")
        if file_path:
            try:
                line_str = Prompt.ask("라인 번호", default="0")
                column_str = Prompt.ask("컬럼 번호", default="0")
                line = int(line_str)
                column = int(column_str)
                location_ctx = LocationContext(
                    repo_id=repo_id,
                    file_path=file_path,
                    line=line,
                    column=column,
                )
            except ValueError:
                console.print("[yellow]잘못된 라인/컬럼 번호입니다.[/yellow]")

    console.print("\n[bold yellow]검색 중...[/bold yellow]")

    try:
        graph_search = PostgresGraphSearch(bootstrap.graph_store)
        semantic_search = PgVectorSemanticSearch(
            embedding_service=bootstrap.embedding_service,
            embedding_store=bootstrap.embedding_store,
            chunk_store=bootstrap.chunk_store,
        )

        retriever = HybridRetriever(
            lexical_search=bootstrap.lexical_search,
            semantic_search=semantic_search,
            graph_search=graph_search,
        )

        candidates = retriever.retrieve(
            repo_id=repo_id,
            query=query,
            k=limit,
            location_ctx=location_ctx,
        )

        results = []
        for candidate in candidates:
            chunk = bootstrap.chunk_store.get_chunk(repo_id, candidate.chunk_id)
            if chunk:
                results.append({
                    "chunk_id": candidate.chunk_id,
                    "file_path": candidate.file_path,
                    "span": list(candidate.span),
                    "score": candidate.features.get("total_score", 0.0),
                    "text": chunk.text[:200],
                })

        show_search_results(results, query)

    except Exception as e:
        console.print("\n[bold red]검색 실패[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        console.print_exception()


def handle_list_repos():
    """저장소 목록 표시"""
    console.print("\n[bold cyan]저장소 목록[/bold cyan]\n")

    try:
        repos = bootstrap.repo_store.list_all()
        repo_list = [
            {
                "repo_id": repo.repo_id,
                "name": repo.name,
                "root_path": repo.root_path,
                "indexing_status": repo.indexing_status,
                "total_files": repo.total_files,
                "total_nodes": repo.total_nodes,
            }
            for repo in repos
        ]

        select_repo_from_list(repo_list)

    except Exception as e:
        console.print(f"[red]저장소 목록 조회 실패: {e}[/red]")
        console.print_exception()


def handle_delete_repo():
    """저장소 삭제 처리"""
    console.print("\n[bold red]저장소 삭제[/bold red]\n")

    try:
        repos = bootstrap.repo_store.list_all()
        repo_list = [
            {
                "repo_id": repo.repo_id,
                "name": repo.name,
                "root_path": repo.root_path,
                "indexing_status": repo.indexing_status,
            }
            for repo in repos
        ]
    except Exception as e:
        console.print(f"[red]저장소 목록 조회 실패: {e}[/red]")
        return

    repo_id = select_repo_from_list(repo_list)
    if not repo_id:
        return

    if not Confirm.ask("\n[bold red]정말 삭제하시겠습니까?[/bold red]", default=False):
        console.print("[yellow]취소되었습니다.[/yellow]")
        return

    try:
        bootstrap.repo_store.delete(repo_id)
        bootstrap.graph_store.delete_repo(repo_id)
        console.print("\n[bold green]삭제 완료[/bold green]")
    except Exception as e:
        console.print("\n[bold red]삭제 실패[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        console.print_exception()


def run_interactive():
    """대화형 CLI 실행"""
    show_banner()

    try:
        _ = bootstrap
    except Exception as e:
        console.print("[bold red]초기화 실패[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        console.print("\n환경변수를 확인해주세요.")
        sys.exit(1)

    commands = [
        {"cmd": "index", "key": "1", "desc": "저장소 인덱싱"},
        {"cmd": "search", "key": "2", "desc": "코드 검색"},
        {"cmd": "list", "key": "3", "desc": "저장소 목록"},
        {"cmd": "delete", "key": "4", "desc": "저장소 삭제"},
        {"cmd": "help", "key": "h", "desc": "도움말"},
        {"cmd": "quit", "key": "q", "desc": "종료"},
    ]

    while True:
        try:
            choice = show_menu(commands)

            if choice == "1":
                handle_index_repo()
            elif choice == "2":
                handle_search()
            elif choice == "3":
                handle_list_repos()
            elif choice == "4":
                handle_delete_repo()
            elif choice == "q" or choice == "quit":
                console.print("\n[bold cyan]종료합니다.[/bold cyan]")
                break
            else:
                console.print("[yellow]잘못된 선택입니다.[/yellow]")

            if choice != "q":
                Prompt.ask("\n[dim]계속하려면 Enter를 누르세요...[/dim]", default="")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]종료합니다.[/yellow]")
            break
        except Exception:
            console.print("\n[bold red]오류 발생[/bold red]")
            console.print_exception()
            Prompt.ask("\n[dim]계속하려면 Enter를 누르세요...[/dim]", default="")

