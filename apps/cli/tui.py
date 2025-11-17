"""대화형 TUI 컴포넌트"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

console = Console()


class DirectoryCompleter(Completer):
    """디렉토리 자동완성"""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        path = Path(text).expanduser()
        
        if path.is_dir():
            try:
                for item in sorted(path.iterdir()):
                    if item.is_dir():
                        item_path = str(item) + os.sep
                        yield Completion(item_path, start_position=-len(text))
            except PermissionError:
                pass


def select_directory(start_path: Optional[str] = None) -> Optional[str]:
    """
    디렉토리 선택 (대화형 탐색기)
    
    화살표 키로 이동, Enter로 선택/진입
    
    Args:
        start_path: 시작 경로 (None이면 현재 디렉토리)
    
    Returns:
        선택된 디렉토리 경로 또는 None
    """
    current_path = Path(start_path).expanduser() if start_path else Path.cwd()
    
    if not current_path.exists():
        current_path = Path.cwd()
    
    if not current_path.is_dir():
        current_path = current_path.parent
    
    selected_index = 0
    
    while True:
        try:
            # 화면 클리어
            console.clear()
            
            # 헤더
            console.print(Panel.fit(
                f"[bold cyan]디렉토리 탐색[/bold cyan]\n"
                f"현재 경로: [yellow]{current_path}[/yellow]",
                border_style="cyan",
            ))
            
            # 디렉토리 항목 가져오기
            items = []
            
            # 상위 디렉토리 항목 추가
            if current_path.parent != current_path:
                items.append({
                    "name": "..",
                    "path": current_path.parent,
                    "is_dir": True,
                    "is_parent": True,
                })
            
            # 현재 디렉토리 항목 추가
            items.append({
                "name": ".",
                "path": current_path,
                "is_dir": True,
                "is_current": True,
            })
            
            # 하위 디렉토리 및 파일 추가
            try:
                for item in sorted(current_path.iterdir()):
                    try:
                        is_dir = item.is_dir()
                        items.append({
                            "name": item.name,
                            "path": item,
                            "is_dir": is_dir,
                        })
                    except PermissionError:
                        continue
            except PermissionError:
                console.print("[red]권한이 없습니다.[/red]")
                return None
            
            # 선택 인덱스 범위 조정
            if selected_index >= len(items):
                selected_index = len(items) - 1
            if selected_index < 0:
                selected_index = 0
            
            # 항목 목록 표시
            console.print("\n[bold]항목:[/bold]")
            for i, item in enumerate(items):
                name = item["name"]
                is_dir = item["is_dir"]
                is_parent = item.get("is_parent", False)
                is_current = item.get("is_current", False)
                
                # 선택된 항목 강조
                if i == selected_index:
                    prefix = "[bold green]▶[/bold green] "
                    style = "bold"
                else:
                    prefix = "  "
                    style = ""
                
                # 디렉토리 표시
                if is_parent:
                    display_name = f"[blue]{name}[/blue] (상위 디렉토리)"
                elif is_current:
                    display_name = f"[green]{name}[/green] (현재 디렉토리 선택)"
                elif is_dir:
                    display_name = f"[blue]{name}/[/blue]"
                else:
                    display_name = name
                
                console.print(f"{prefix}[{style}]{display_name}[/{style}]")
            
            # 도움말
            console.print("\n[dim]조작법:[/dim]")
            console.print("  [cyan]↑/↓[/cyan] : 이동")
            console.print("  [cyan]Enter[/cyan] : 선택/진입")
            console.print("  [cyan]q[/cyan] : 취소")
            
            # 키 입력 받기 (prompt_toolkit 사용)
            from prompt_toolkit import Application
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.layout import Layout, Window
            
            kb = KeyBindings()
            action = [None]  # 리스트로 감싸서 nonlocal 대신 사용
            
            @kb.add("up")
            @kb.add("c-p")
            def go_up(event):
                nonlocal selected_index
                selected_index = max(0, selected_index - 1)
                action[0] = "refresh"
                event.app.exit()
            
            @kb.add("down")
            @kb.add("c-n")
            def go_down(event):
                nonlocal selected_index
                selected_index = min(len(items) - 1, selected_index + 1)
                action[0] = "refresh"
                event.app.exit()
            
            @kb.add("enter")
            def select_item(event):
                action[0] = "select"
                event.app.exit()
            
            @kb.add("q")
            @kb.add("c-c")
            def cancel(event):
                action[0] = "cancel"
                event.app.exit()
            
            # 간단한 Application으로 키 입력 받기
            try:
                # 빈 레이아웃으로 키 입력만 받기
                layout = Layout(Window())
                app = Application(
                    layout=layout,
                    key_bindings=kb,
                    full_screen=False,
                )
                app.run()
                
                if action[0] == "cancel" or action[0] is None:
                    return None
                
                if action[0] == "select":
                    selected_item = items[selected_index]
                    selected_path = selected_item["path"]
                    
                    # 현재 디렉토리 선택
                    if selected_item.get("is_current", False):
                        return str(current_path.resolve())
                    
                    # 상위 디렉토리로 이동
                    if selected_item.get("is_parent", False):
                        current_path = selected_path
                        selected_index = 0
                        action[0] = None
                        continue
                    
                    # 디렉토리 진입
                    if selected_item["is_dir"]:
                        current_path = selected_path.resolve()
                        selected_index = 0
                        action[0] = None
                        continue
                    else:
                        # 파일은 선택 불가
                        console.print("[yellow]파일입니다. 디렉토리를 선택해주세요.[/yellow]")
                        import time
                        time.sleep(0.5)
                        action[0] = None
                        continue
                
                if action[0] == "refresh":
                    action[0] = None
                    continue
                
            except KeyboardInterrupt:
                return None
            except Exception as e:
                # 키 입력 실패 시 간단한 입력으로 폴백
                user_input = input("\n경로 입력 (Enter: 현재 선택, ..: 상위, q: 취소): ").strip()
                
                if user_input == "q":
                    return None
                elif user_input == "":
                    # Enter만 누르면 현재 디렉토리 선택
                    return str(current_path.resolve())
                elif user_input == "..":
                    current_path = current_path.parent
                    selected_index = 0
                    continue
                elif user_input == ".":
                    return str(current_path.resolve())
                else:
                    # 경로 처리
                    if user_input.startswith("/"):
                        new_path = Path(user_input)
                    elif user_input.startswith("~"):
                        new_path = Path(user_input).expanduser()
                    else:
                        new_path = current_path / user_input
                    
                    if new_path.exists() and new_path.is_dir():
                        current_path = new_path.resolve()
                        selected_index = 0
                    elif new_path.exists() and new_path.is_file():
                        console.print("[yellow]파일입니다. 디렉토리를 선택해주세요.[/yellow]")
                        import time
                        time.sleep(1)
                    else:
                        console.print("[red]존재하지 않는 경로입니다.[/red]")
                        import time
                        time.sleep(1)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]취소되었습니다.[/yellow]")
            return None
        except Exception as e:
            console.print(f"[red]오류: {e}[/red]")
            import time
            time.sleep(1)


def select_repo_from_list(repos: List[dict]) -> Optional[str]:
    """
    저장소 목록에서 선택
    
    Args:
        repos: 저장소 정보 리스트
    
    Returns:
        선택된 repo_id 또는 None
    """
    if not repos:
        console.print("[yellow]등록된 저장소가 없습니다.[/yellow]")
        return None
    
    table = Table(title="등록된 저장소", show_header=True, header_style="bold magenta")
    table.add_column("번호", style="cyan", width=6)
    table.add_column("ID", style="green")
    table.add_column("이름", style="yellow")
    table.add_column("경로", style="blue")
    table.add_column("상태", style="magenta")
    
    for i, repo in enumerate(repos, 1):
        table.add_row(
            str(i),
            repo.get("repo_id", ""),
            repo.get("name", ""),
            repo.get("root_path", ""),
            repo.get("indexing_status", "unknown"),
        )
    
    console.print(table)
    
    try:
        choice = Prompt.ask(
            "\n저장소 번호 선택 (취소: Enter)",
            default="",
        )
        
        if not choice:
            return None
        
        idx = int(choice) - 1
        if 0 <= idx < len(repos):
            return repos[idx].get("repo_id")
        else:
            console.print("[red]잘못된 번호입니다.[/red]")
            return None
    except ValueError:
        console.print("[red]숫자를 입력해주세요.[/red]")
        return None
    except KeyboardInterrupt:
        return None


def show_menu() -> str:
    """메인 메뉴 표시"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Semantica Codegraph[/bold cyan]\n"
        "[dim]코드 그래프 인덱싱 및 검색 시스템[/dim]",
        border_style="cyan",
    ))
    
    console.print("\n[bold]메뉴:[/bold]")
    console.print("  [cyan]1[/cyan] 저장소 인덱싱")
    console.print("  [cyan]2[/cyan] 코드 검색")
    console.print("  [cyan]3[/cyan] 저장소 목록")
    console.print("  [cyan]4[/cyan] 저장소 삭제")
    console.print("  [cyan]q[/cyan] 종료")
    
    choice = Prompt.ask("\n선택", default="q")
    return choice.strip().lower()


def show_indexing_progress(repo_id: str, current: int, total: int):
    """인덱싱 진행 상황 표시"""
    if total > 0:
        percent = (current / total) * 100
        bar_length = 30
        filled = int(bar_length * current / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        console.print(f"\r[{bar}] {percent:.1f}% ({current}/{total})", end="")
    else:
        console.print(f"\r인덱싱 중... ({current}개 파일 처리됨)", end="")


def show_search_results(results: List[dict], query: str):
    """검색 결과 표시"""
    if not results:
        console.print(f"\n[yellow]'{query}'에 대한 결과가 없습니다.[/yellow]")
        return
    
    console.print(f"\n[bold green]검색 결과:[/bold green] '{query}' ({len(results)}개)")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("파일", style="cyan")
    table.add_column("라인", style="yellow")
    table.add_column("코드", style="white", overflow="fold")
    table.add_column("점수", style="green", justify="right")
    
    for result in results[:20]:  # 최대 20개만 표시
        file_path = result.get("file_path", "")
        span = result.get("span", [0, 0, 0, 0])
        score = result.get("score", 0.0)
        text = result.get("text", "")[:100]  # 최대 100자
        
        table.add_row(
            file_path,
            f"{span[0] + 1}-{span[2] + 1}" if len(span) >= 3 else str(span[0] + 1),
            text,
            f"{score:.3f}",
        )
    
    console.print(table)
    
    if len(results) > 20:
        console.print(f"\n[dim]... 외 {len(results) - 20}개 결과 (전체 보려면 API 사용)[/dim]")

