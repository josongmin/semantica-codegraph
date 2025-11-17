"""대화형 TUI 컴포넌트"""

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

try:
    import pyfiglet
    PYFIGLET_AVAILABLE = True
except ImportError:
    PYFIGLET_AVAILABLE = False

console = Console()


def escape_markup(text: str) -> str:
    """
    Rich markup 문자열 이스케이프
    
    Args:
        text: 이스케이프할 텍스트
        
    Returns:
        이스케이프된 텍스트
    """
    return text.replace("[", "\\[").replace("]", "\\]")


def show_error_and_wait(message: str, duration: float = 1.0) -> None:
    """
    에러 메시지 출력 후 대기
    
    Args:
        message: 에러 메시지
        duration: 대기 시간 (초)
    """
    console.print(f"[red]{message}[/red]")
    time.sleep(duration)


def show_warning_and_wait(message: str, duration: float = 0.5) -> None:
    """
    경고 메시지 출력 후 대기
    
    Args:
        message: 경고 메시지
        duration: 대기 시간 (초)
    """
    console.print(f"[yellow]{message}[/yellow]")
    time.sleep(duration)


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
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.layout.controls import FormattedTextControl
    
    current_path = Path(start_path).expanduser() if start_path else Path.cwd()
    
    if not current_path.exists():
        current_path = Path.cwd()
    
    if not current_path.is_dir():
        current_path = current_path.parent
    
    selected_index = [0]  # 리스트로 감싸서 참조 전달
    action = [None]  # "select", "cancel"
    
    def get_items():
        """현재 경로의 항목 목록 가져오기"""
        items = []
        
        # 현재 디렉토리 선택 항목 (맨 위)
        items.append({
            "name": "현재 디렉토리 선택",
            "path": current_path,
            "is_dir": True,
            "is_current": True,
        })
        
        # 상위 디렉토리 항목 추가
        if current_path.parent != current_path:
            items.append({
                "name": "..",
                "path": current_path.parent,
                "is_dir": True,
                "is_parent": True,
            })
        
        # 하위 디렉토리만 추가 (파일 제외)
        try:
            for item in sorted(current_path.iterdir()):
                try:
                    is_dir = item.is_dir()
                    # 폴더만 추가
                    if is_dir:
                        items.append({
                            "name": item.name,
                            "path": item,
                            "is_dir": is_dir,
                        })
                except PermissionError:
                    continue
        except PermissionError:
            pass
        
        return items
    
    def render_content() -> FormattedText:
        """화면 내용 렌더링"""
        items = get_items()
        
        # 선택 인덱스 범위 조정
        if selected_index[0] >= len(items):
            selected_index[0] = len(items) - 1
        if selected_index[0] < 0:
            selected_index[0] = 0
        
        lines = []
        
        # 헤더
        lines.append(("bold cyan", "디렉토리 탐색\n"))
        lines.append(("yellow", f"현재 경로: {current_path}\n\n"))
        
        # 항목 목록
        lines.append(("bold", "항목:\n"))
        for i, item in enumerate(items):
            name = item["name"]
            is_dir = item["is_dir"]
            is_parent = item.get("is_parent", False)
            is_current = item.get("is_current", False)
            is_selected = i == selected_index[0]
            
            # prefix
            if is_selected:
                prefix = ("bold green", "▶ ")
            else:
                prefix = ("", "  ")
            
            # name 부분
            if is_parent:
                if is_selected:
                    name_part = [("bold blue", f"{name}"), ("bold", " (상위 디렉토리)")]
                else:
                    name_part = [("blue", f"{name}"), ("", " (상위 디렉토리)")]
            elif is_current:
                # 현재 디렉토리 선택 항목 (이미 이름에 포함되어 있으므로 추가 라벨 불필요)
                if is_selected:
                    name_part = [("bold green", f"{name}")]
                else:
                    name_part = [("green", f"{name}")]
            elif is_dir:
                if is_selected:
                    name_part = [("bold blue", f"{name}/")]
                else:
                    name_part = [("blue", f"{name}/")]
            else:
                if is_selected:
                    name_part = [("bold", name)]
                else:
                    name_part = [("", name)]
            
            lines.append(prefix)
            lines.extend(name_part)
            lines.append(("", "\n"))
        
        # 도움말
        lines.append(("", "\n"))
        lines.append(("dim", "조작법: "))
        lines.extend([("cyan", "↑/↓"), ("dim", ":폴더이동  ")])
        lines.extend([("cyan", "Enter"), ("dim", ":선택/진입  ")])
        lines.extend([("cyan", "Backspace"), ("dim", ":뒤로가기")])
        lines.append(("", "\n"))
        
        return FormattedText(lines)
    
    # 키 바인딩
    kb = KeyBindings()
    
    @kb.add("up")
    @kb.add("c-p")
    def go_up(event):
        items = get_items()
        if selected_index[0] > 0:
            selected_index[0] -= 1
        # 레이아웃 업데이트
        control = FormattedTextControl(text=render_content)
        event.app.layout = Layout(Window(content=control))
        event.app.invalidate()
    
    @kb.add("down")
    @kb.add("c-n")
    def go_down(event):
        items = get_items()
        if selected_index[0] < len(items) - 1:
            selected_index[0] += 1
        # 레이아웃 업데이트
        control = FormattedTextControl(text=render_content)
        event.app.layout = Layout(Window(content=control))
        event.app.invalidate()
    
    @kb.add("enter")
    def select_item(event):
        action[0] = "select"
        event.app.exit()
    
    @kb.add("backspace")
    def go_back(event):
        nonlocal current_path
        # 상위 디렉토리로 이동
        if current_path.parent != current_path:
            current_path = current_path.parent
            selected_index[0] = 0
            # 레이아웃 업데이트
            control = FormattedTextControl(text=render_content)
            event.app.layout = Layout(Window(content=control))
            event.app.invalidate()
    
    @kb.add("q")
    @kb.add("c-c")
    def cancel(event):
        action[0] = "cancel"
        event.app.exit()
    
    # 메인 루프
    while True:
        try:
            # Application 생성 및 실행
            control = FormattedTextControl(text=render_content)
            layout = Layout(Window(content=control))
            app = Application(
                layout=layout,
                key_bindings=kb,
                full_screen=False,
            )
            app.run()
            
            if action[0] == "cancel" or action[0] is None:
                return None
            
            if action[0] == "select":
                items = get_items()
                selected_item = items[selected_index[0]]
                selected_path = selected_item["path"]
                
                # 현재 디렉토리 선택
                if selected_item.get("is_current", False):
                    return str(current_path.resolve())
                
                # 상위 디렉토리로 이동
                if selected_item.get("is_parent", False):
                    current_path = selected_path
                    selected_index[0] = 0
                    action[0] = None
                    continue
                
                # 디렉토리 진입
                if selected_item["is_dir"]:
                    current_path = selected_path.resolve()
                    selected_index[0] = 0
                    action[0] = None
                    continue
                else:
                    # 파일은 선택 불가
                    show_warning_and_wait("파일입니다. 디렉토리를 선택해주세요.")
                    action[0] = None
                    continue
        
        except KeyboardInterrupt:
            return None
        except Exception as e:
            # 에러 발생 시 간단한 입력으로 폴백
            error_msg = escape_markup(str(e))
            console.print(f"\n[red]오류:[/red] {error_msg}")
            console.print("[yellow]간단한 입력 모드로 전환합니다.[/yellow]\n")
            
            # 간단한 입력 방식으로 폴백
            try:
                user_input = input("경로 입력 (Enter: 현재 선택, ..: 상위, q: 취소): ").strip()
                
                if user_input == "q":
                    return None
                elif user_input == "":
                    return str(current_path.resolve())
                elif user_input == "..":
                    current_path = current_path.parent
                    selected_index = 0
                    continue
                elif user_input == ".":
                    return str(current_path.resolve())
                else:
                    if user_input.startswith("/"):
                        new_path = Path(user_input)
                    elif user_input.startswith("~"):
                        new_path = Path(user_input).expanduser()
                    else:
                        new_path = current_path / user_input
                    
                    if new_path.exists() and new_path.is_dir():
                        current_path = new_path.resolve()
                        selected_index = 0
                    else:
                        show_error_and_wait("존재하지 않는 경로입니다.")
            except KeyboardInterrupt:
                return None


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


def show_banner():
    """환영 화면 표시 (왼쪽 ASCII 아트 + 오른쪽 환영 메시지)"""
    from pathlib import Path
    from rich.columns import Columns
    from rich.panel import Panel
    
    # 왼쪽: ASCII 아트
    ascii_art = ""
    if PYFIGLET_AVAILABLE:
        try:
            semantica = pyfiglet.figlet_format("SEMANTICA", font="slant")
            code = pyfiglet.figlet_format("CODE", font="slant")
            
            semantica_lines = semantica.split('\n')
            code_lines = code.split('\n')
            max_lines = max(len(semantica_lines), len(code_lines))
            semantica_width = max(len(line) for line in semantica_lines) if semantica_lines else 0
            
            for i in range(max_lines):
                semantica_line = semantica_lines[i] if i < len(semantica_lines) else ""
                code_line = code_lines[i] if i < len(code_lines) else ""
                ascii_art += f"[bold cyan]{semantica_line.ljust(semantica_width)}[/bold cyan]  [bold green]{code_line}[/bold green]\n"
            
            ascii_art += "\n[dim]v0.1.0[/dim]"
        except Exception:
            ascii_art = "[bold cyan]SEMANTICA[/bold cyan] [bold green]CODE[/bold green]\n[dim]v0.1.0[/dim]"
    else:
        ascii_art = "[bold cyan]SEMANTICA[/bold cyan] [bold green]CODE[/bold green]\n[dim]v0.1.0[/dim]"
    
    # 오른쪽: 환영 메시지 및 안내
    welcome_text = """[bold]Welcome to Semantica Code[/bold]

[cyan]Ctrl+O[/cyan] to execute [yellow]commands[/yellow]
Type [dim]/[/dim] to open command palette
[cyan]Ctrl+C[/cyan] to exit

Use [yellow]/help[/yellow] command for more information

Use [cyan]Tab/Shift+Tab[/cyan] to navigate to previous messages"""
    
    # 두 컬럼으로 나란히 표시
    left_panel = Panel(ascii_art, border_style="green", padding=(1, 2))
    right_panel = Panel(welcome_text, border_style="cyan", padding=(1, 2))
    
    console.print(Columns([left_panel, right_panel], equal=True, expand=True))
    console.print("")


def show_menu() -> str:
    """메인 메뉴 표시 (일반 텍스트 입력, / 입력 시 Command Palette)"""
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window, HSplit, VSplit, FloatContainer, Float
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.history import InMemoryHistory
    import os
    
    # 명령어 목록
    commands = [
        {"cmd": "index", "key": "1", "desc": "저장소 선택 후 (재)인덱싱"},
        {"cmd": "search", "key": "2", "desc": "코드 검색"},
        {"cmd": "list", "key": "3", "desc": "저장소 목록"},
        {"cmd": "delete", "key": "4", "desc": "저장소 삭제"},
        {"cmd": "help", "key": "h", "desc": "도움말"},
        {"cmd": "quit", "key": "q", "desc": "종료"},
    ]
    
    # 상태 변수
    show_palette = [False]
    selected_index = [0]
    action = [None]
    user_input_text = [""]
    
    # 입력 버퍼
    input_buffer = Buffer(history=InMemoryHistory())
    
    # Command Palette 렌더링
    def render_palette() -> FormattedText:
        """Command Palette 렌더링 (박스 테두리 포함)"""
        if not show_palette[0]:
            return FormattedText("")
        
        lines = []
        
        # 박스 너비
        box_width = 60
        
        # 상단 테두리
        lines.append(("bold cyan", "╭" + "─" * (box_width - 2) + "╮\n"))
        
        # 제목
        title = "Command Palette"
        padding = (box_width - 2 - len(title)) // 2
        lines.append(("bold cyan", "│"))
        lines.append(("", " " * padding))
        lines.append(("bold green", title))
        lines.append(("", " " * (box_width - 2 - len(title) - padding)))
        lines.append(("bold cyan", "│\n"))
        
        # 구분선
        lines.append(("bold cyan", "├" + "─" * (box_width - 2) + "┤\n"))
        
        # 입력 필드
        filter_text = user_input_text[0] or ""
        lines.append(("bold cyan", "│"))
        lines.append(("", " > "))
        lines.append(("bold white", filter_text))
        lines.append(("", " " * (box_width - 6 - len(filter_text))))
        lines.append(("bold cyan", "│\n"))
        
        # 빈 줄
        lines.append(("bold cyan", "│"))
        lines.append(("", " " * (box_width - 2)))
        lines.append(("bold cyan", "│\n"))
        
        # 필터링된 명령어 목록
        filter_text_lower = filter_text.lower()
        filtered_commands = [
            cmd for cmd in commands
            if filter_text_lower == "" or cmd["cmd"].startswith(filter_text_lower)
        ]
        
        for i, cmd in enumerate(filtered_commands):
            is_selected = i == selected_index[0]
            lines.append(("bold cyan", "│"))
            
            # 명령어 텍스트
            cmd_text = f"  {cmd['cmd']:<15} {cmd['desc']}"
            if len(cmd_text) > box_width - 4:
                cmd_text = cmd_text[:box_width - 7] + "..."
            
            padding_right = box_width - 2 - len(cmd_text)
            
            if is_selected:
                lines.append(("bg:#2d5016 bold white", cmd_text))
                lines.append(("bg:#2d5016", " " * padding_right))
            else:
                lines.append(("", cmd_text))
                lines.append(("", " " * padding_right))
            
            lines.append(("bold cyan", "│\n"))
        
        # 빈 줄 추가 (최소 높이 유지)
        current_lines = 5 + len(filtered_commands)  # 헤더(4) + 명령어들 + 입력줄
        min_lines = 12
        for _ in range(max(0, min_lines - current_lines)):
            lines.append(("bold cyan", "│"))
            lines.append(("", " " * (box_width - 2)))
            lines.append(("bold cyan", "│\n"))
        
        # 하단 도움말
        lines.append(("bold cyan", "├" + "─" * (box_width - 2) + "┤\n"))
        lines.append(("bold cyan", "│"))
        help_text = " ↑↓:이동 Enter:선택 Esc:닫기"
        lines.append(("dim", help_text))
        lines.append(("", " " * (box_width - 2 - len(help_text))))
        lines.append(("bold cyan", "│\n"))
        
        # 하단 테두리
        lines.append(("bold cyan", "╰" + "─" * (box_width - 2) + "╯\n"))
        
        return FormattedText(lines)
    
    # 배너를 FormattedText로 렌더링
    def render_banner() -> FormattedText:
        """배너를 FormattedText로 렌더링"""
        lines = []
        
        # ASCII 아트 생성
        if PYFIGLET_AVAILABLE:
            try:
                semantica = pyfiglet.figlet_format("SEMANTICA", font="slant")
                code = pyfiglet.figlet_format("CODE", font="slant")
                
                semantica_lines = semantica.split('\n')
                code_lines = code.split('\n')
                max_lines = max(len(semantica_lines), len(code_lines))
                semantica_width = max(len(line) for line in semantica_lines) if semantica_lines else 0
                
                for i in range(max_lines):
                    semantica_line = semantica_lines[i] if i < len(semantica_lines) else ""
                    code_line = code_lines[i] if i < len(code_lines) else ""
                    # SEMANTICA 부분 (시안색)
                    lines.append(("bold cyan", semantica_line.ljust(semantica_width)))
                    lines.append(("", "  "))
                    # CODE 부분 (초록색)
                    lines.append(("bold green", f"{code_line}\n"))
                
                # 버전 정보
                lines.append(("", "\n"))
                lines.append(("dim", "v0.1.0\n"))
            except Exception:
                lines.append(("bold cyan", "SEMANTICA"))
                lines.append(("", " "))
                lines.append(("bold green", "CODE\n"))
                lines.append(("", "\n"))
                lines.append(("dim", "v0.1.0\n"))
        else:
            lines.append(("bold cyan", "SEMANTICA"))
            lines.append(("", " "))
            lines.append(("bold green", "CODE\n"))
            lines.append(("", "\n"))
            lines.append(("dim", "v0.1.0\n"))
        
        lines.append(("", "\n"))
        lines.append(("dim", "명령어를 입력하거나 / 를 눌러 Command Palette를 여세요.\n"))
        lines.append(("", "\n"))
        
        return FormattedText(lines)
    
    # 메인 콘텐츠 렌더링
    def render_content() -> FormattedText:
        """메인 콘텐츠 렌더링 (배너 포함)"""
        return render_banner()
    
    # 키 바인딩
    kb = KeyBindings()
    
    @kb.add("/")
    def open_palette(event):
        """/ 입력 시 Command Palette 열기 (입력창에는 / 추가하지 않음)"""
        show_palette[0] = True
        selected_index[0] = 0
        user_input_text[0] = ""  # 필터링 텍스트 초기화
        # 버퍼에는 /를 추가하지 않음
        event.app.invalidate()
    
    @kb.add("c-c")
    def exit_app(event):
        action[0] = "quit"
        event.app.exit()
    
    @kb.add("up")
    def move_up(event):
        if show_palette[0]:
            if selected_index[0] > 0:
                selected_index[0] -= 1
            event.app.invalidate()
    
    @kb.add("down")
    def move_down(event):
        if show_palette[0]:
            filter_text = (user_input_text[0] or "").lower()
            filtered_commands = [
                cmd for cmd in commands
                if filter_text == "" or cmd["cmd"].startswith(filter_text)
            ]
            if selected_index[0] < len(filtered_commands) - 1:
                selected_index[0] += 1
            event.app.invalidate()
    
    @kb.add("enter")
    def select_command(event):
        if show_palette[0]:
            filter_text = (user_input_text[0] or "").lower()
            filtered_commands = [
                cmd for cmd in commands
                if filter_text == "" or cmd["cmd"].startswith(filter_text)
            ]
            if filtered_commands:
                selected_cmd = filtered_commands[selected_index[0]]
                action[0] = "select"
                user_input_text[0] = selected_cmd["key"]
                event.app.exit()
        else:
            # 일반 텍스트 입력 완료
            action[0] = "text"
            user_input_text[0] = input_buffer.text
            event.app.exit()
    
    @kb.add("escape")
    def close_palette(event):
        if show_palette[0]:
            show_palette[0] = False
            user_input_text[0] = ""
            selected_index[0] = 0
            # 입력 버퍼도 비우기
            input_buffer.text = ""
            event.app.invalidate()
    
    # 레이아웃 업데이트 함수
    def update_layout():
        """레이아웃 업데이트"""
        if hasattr(update_layout, 'app'):
            update_layout.app.invalidate()
    
    # 입력 버퍼 업데이트 핸들러
    def on_text_changed(buffer):
        if show_palette[0]:
            # Command Palette가 열린 상태: 입력 텍스트를 필터링에 사용
            user_input_text[0] = buffer.text
            selected_index[0] = 0  # 필터링 시 선택 인덱스 초기화
        else:
            # Command Palette가 닫힌 상태: 일반 텍스트 입력
            if buffer.text == "":
                show_palette[0] = False
            user_input_text[0] = buffer.text
        update_layout()
    
    input_buffer.on_text_changed += on_text_changed
    
    try:
        # 레이아웃 구성
        main_content = Window(
            content=FormattedTextControl(text=render_content),
            wrap_lines=False,
        )
        
        # 입력창 (하단) - 항상 표시
        input_window = Window(
            content=BufferControl(buffer=input_buffer),
            height=1,
            style="",
            wrap_lines=False,
        )
        
        # Command Palette Window - 조건부 표시
        def get_palette_content():
            """Command Palette 내용을 동적으로 반환"""
            if show_palette[0]:
                return render_palette()
            return FormattedText("")
        
        # 조건부로 높이 설정하는 Window
        class ConditionalHeightWindow(Window):
            def get_height(self):
                if show_palette[0]:
                    return 18
                return 0
        
        palette_window = ConditionalHeightWindow(
            content=FormattedTextControl(text=get_palette_content),
            width=62,
            height=18,
            style="bg:#1e1e1e",  # 어두운 배경
            dont_extend_width=True,
            dont_extend_height=True,
        )
        
        # 커서 옆에 배치
        palette_float = Float(
            palette_window,
            xcursor=True,  # 커서 가로 위치
            ycursor=True,  # 커서 세로 위치
        )
        
        # 전체 레이아웃
        container = FloatContainer(
            HSplit([
                main_content,
                input_window,
            ]),
            floats=[palette_float],
        )
        
        layout = Layout(container)
        
        # Application 생성 및 실행
        app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            mouse_support=True,
            refresh_interval=0.1,  # 주기적 업데이트
        )
        
        # 업데이트 함수에 app 참조 저장
        update_layout.app = app
        
        # 초기 렌더링
        app.invalidate()
        
        app.run()
        
        if action[0] == "quit":
            return "q"
        elif action[0] == "select":
            return user_input_text[0]
        elif action[0] == "text":
            # 일반 텍스트 입력은 다시 메뉴로
            return show_menu()
        else:
            return "q"
    
    except (KeyboardInterrupt, EOFError):
        return "q"
    except Exception as e:
        # 에러 발생 시 간단한 입력으로 폴백
        from rich.prompt import Prompt
        console.print(f"\n[dim]에러: {e}[/dim]")
        console.print("[dim]명령어를 입력하세요: /index, /search, /list, /delete, /quit[/dim]")
        user_input = Prompt.ask("", default="")
        
        if user_input.startswith("/"):
            cmd = user_input[1:].strip().lower()
            cmd_map = {c["cmd"]: c["key"] for c in commands}
            return cmd_map.get(cmd, "q")
        else:
            return show_menu()


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

