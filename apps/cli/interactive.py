"""대화형 CLI 메인 로직"""

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from src.core.bootstrap import create_bootstrap
from src.core.models import LocationContext
from src.search.retriever.hybrid_retriever import HybridRetriever
from src.search.graph.postgres_graph_adapter import PostgresGraphSearch
from src.search.semantic.pgvector_adapter import PgVectorSemanticSearch

from .tui import (
    console,
    select_directory,
    select_repo_from_list,
    show_menu,
    show_indexing_progress,
    show_search_results,
)

bootstrap = create_bootstrap()


def handle_index_repo():
    """저장소 인덱싱 처리"""
    console.print("\n[bold cyan]저장소 인덱싱[/bold cyan]\n")
    
    # 디렉토리 선택
    repo_path = select_directory()
    if not repo_path:
        return
    
    # 저장소 이름 입력
    from rich.prompt import Prompt
    default_name = Path(repo_path).name
    repo_name = Prompt.ask(f"\n저장소 이름", default=default_name)
    
    # 저장소 ID 입력 (선택적)
    repo_id = Prompt.ask("저장소 ID (자동 생성: Enter)", default="")
    if not repo_id:
        repo_id = None
    
    # 인덱싱 시작
    console.print(f"\n[bold yellow]인덱싱 시작 중...[/bold yellow]")
    console.print(f"경로: {repo_path}")
    console.print(f"이름: {repo_name}")
    
    try:
        result = bootstrap.pipeline.index_repository(
            root_path=repo_path,
            repo_id=repo_id,
            name=repo_name,
        )
        
        console.print("\n[bold green]✓ 인덱싱 완료![/bold green]")
        console.print(f"  저장소 ID: [cyan]{result.repo_id}[/cyan]")
        console.print(f"  노드: [green]{result.nodes_count}개[/green]")
        console.print(f"  엣지: [green]{result.edges_count}개[/green]")
        console.print(f"  청크: [green]{result.chunks_count}개[/green]")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ 인덱싱 실패[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        console.print_exception()


def handle_search():
    """코드 검색 처리"""
    console.print("\n[bold cyan]코드 검색[/bold cyan]\n")
    
    # 저장소 목록 가져오기
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
    
    # 저장소 선택
    repo_id = select_repo_from_list(repo_list)
    if not repo_id:
        return
    
    # 검색 쿼리 입력
    from rich.prompt import Prompt, Confirm
    query = Prompt.ask("\n검색 쿼리")
    if not query:
        console.print("[yellow]검색 쿼리를 입력해주세요.[/yellow]")
        return
    
    # 검색 옵션
    try:
        limit_str = Prompt.ask("결과 개수", default="20")
        limit = int(limit_str)
    except ValueError:
        limit = 20
    
    # 위치 컨텍스트 (선택적)
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
    
    # 검색 실행
    console.print(f"\n[bold yellow]검색 중...[/bold yellow]")
    
    try:
        # HybridRetriever 생성
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
        
        # 검색 실행
        candidates = retriever.retrieve(
            repo_id=repo_id,
            query=query,
            k=limit,
            location_ctx=location_ctx,
        )
        
        # 결과 변환
        results = []
        for candidate in candidates:
            # 청크 조회
            chunk = bootstrap.chunk_store.get_chunk(repo_id, candidate.chunk_id)
            if chunk:
                results.append({
                    "chunk_id": candidate.chunk_id,
                    "file_path": candidate.file_path,
                    "span": list(candidate.span),
                    "score": candidate.features.get("total_score", 0.0),
                    "text": chunk.text[:200],  # 최대 200자
                })
        
        # 결과 표시
        show_search_results(results, query)
        
    except Exception as e:
        console.print(f"\n[bold red]✗ 검색 실패[/bold red]")
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
    
    # 저장소 목록 가져오기
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
    
    # 저장소 선택
    repo_id = select_repo_from_list(repo_list)
    if not repo_id:
        return
    
    # 확인
    from rich.prompt import Confirm
    if not Confirm.ask(f"\n[bold red]정말 삭제하시겠습니까?[/bold red]", default=False):
        console.print("[yellow]취소되었습니다.[/yellow]")
        return
    
    # 삭제 실행
    try:
        bootstrap.repo_store.delete(repo_id)
        bootstrap.graph_store.delete_repo(repo_id)
        console.print(f"\n[bold green]✓ 삭제 완료[/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]✗ 삭제 실패[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        console.print_exception()


def run_interactive():
    """대화형 CLI 실행"""
    try:
        # Bootstrap 초기화 확인
        _ = bootstrap
    except Exception as e:
        console.print(f"[bold red]초기화 실패[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        console.print("\n환경변수를 확인해주세요.")
        sys.exit(1)
    
    # 메인 루프
    while True:
        try:
            choice = show_menu()
            
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
            
            # 다음 작업을 위해 잠시 대기
            if choice != "q":
                from rich.prompt import Prompt
                Prompt.ask("\n[dim]계속하려면 Enter를 누르세요...[/dim]", default="")
        
        except KeyboardInterrupt:
            console.print("\n\n[yellow]종료합니다.[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]오류 발생[/bold red]")
            console.print_exception()
            from rich.prompt import Prompt
            Prompt.ask("\n[dim]계속하려면 Enter를 누르세요...[/dim]", default="")

