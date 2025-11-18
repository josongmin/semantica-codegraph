"""TUI 컴포넌트"""

from pathlib import Path

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

try:
    import pyfiglet

    PYFIGLET_AVAILABLE = True
except ImportError:
    PYFIGLET_AVAILABLE = False

console = Console()


def show_banner():
    """3D 로고 배너 표시"""
    if PYFIGLET_AVAILABLE:
        try:
            fonts_to_try = ["3d", "3-d", "shadow", "isometric1", "isometric3", "larry3d", "slant"]
            semantica_text = None
            code_text = None

            for font in fonts_to_try:
                try:
                    semantica_text = pyfiglet.figlet_format("SEMANTICA", font=font)
                    code_text = pyfiglet.figlet_format("CODE", font=font)
                    break
                except Exception:
                    continue

            if not semantica_text:
                semantica_text = pyfiglet.figlet_format("SEMANTICA", font="slant")
                code_text = pyfiglet.figlet_format("CODE", font="slant")

            semantica_lines = semantica_text.split("\n")
            code_lines = code_text.split("\n")
            max_lines = max(len(semantica_lines), len(code_lines))
            semantica_width = max(len(line) for line in semantica_lines) if semantica_lines else 0

            # 3D 효과
            ascii_art = ""
            for i in range(max_lines):
                semantica_line = semantica_lines[i] if i < len(semantica_lines) else ""
                code_line = code_lines[i] if i < len(code_lines) else ""

                if semantica_line.strip():
                    shadow_offset = 2
                    shadow_line = " " * shadow_offset + semantica_line.rstrip()
                    ascii_art += (
                        f"[black]{shadow_line.ljust(semantica_width + shadow_offset)}[/black]\n"
                    )

                ascii_art += f"[bold bright_cyan]{semantica_line.ljust(semantica_width)}[/bold bright_cyan]  "

                if code_line.strip():
                    code_shadow_offset = 2
                    code_shadow_line = " " * code_shadow_offset + code_line.rstrip()
                    ascii_art += f"[black]{code_shadow_line}[/black]\n"
                    ascii_art += f"[bold bright_green]{code_line}[/bold bright_green]"
                else:
                    ascii_art += f"[bold bright_green]{code_line}[/bold bright_green]"
                ascii_art += "\n"

            ascii_art += "\n[dim]v0.1.0[/dim]"

            welcome_text = """[bold]Welcome to Semantica Code[/bold]

[cyan]Ctrl+C[/cyan] to exit
Type [dim]/[/dim] to see commands"""

            left_panel = Panel(ascii_art, border_style="green", padding=(1, 2))
            right_panel = Panel(welcome_text, border_style="cyan", padding=(1, 2))

            from rich.columns import Columns

            console.print(Columns([left_panel, right_panel], equal=True, expand=True))
            console.print("")
        except Exception:
            console.print("[bold cyan]SEMANTICA[/bold cyan] [bold green]CODE[/bold green]")
    else:
        console.print("[bold cyan]SEMANTICA[/bold cyan] [bold green]CODE[/bold green]")


def show_menu(commands: list[dict]) -> str:
    """메인 메뉴 표시 (명령어 자동완성 포함)"""

    class CommandCompleter(Completer):
        def get_completions(self, document: Document, complete_event):
            text = document.text_before_cursor
            if text.startswith("/"):
                cmd_prefix = text[1:].lower().strip()
                for cmd_info in commands:
                    cmd = cmd_info["cmd"]
                    if not cmd_prefix or cmd.startswith(cmd_prefix):
                        yield Completion(
                            f"/{cmd}",
                            start_position=-len(text),
                            display=f"/{cmd} - {cmd_info['desc']}",
                            display_meta=cmd_info["desc"],
                        )

    action = [None]
    user_input_text = [""]
    selected_completion_index = [0]

    input_buffer = Buffer(
        history=InMemoryHistory(),
        completer=CommandCompleter(),
        complete_while_typing=False,  # 커스텀 UI 사용
    )

    def render_banner() -> FormattedText:
        lines = []
        if PYFIGLET_AVAILABLE:
            try:
                fonts_to_try = [
                    "3d",
                    "3-d",
                    "shadow",
                    "isometric1",
                    "isometric3",
                    "larry3d",
                    "slant",
                ]
                semantica_text = None
                code_text = None

                for font in fonts_to_try:
                    try:
                        semantica_text = pyfiglet.figlet_format("SEMANTICA", font=font)
                        code_text = pyfiglet.figlet_format("CODE", font=font)
                        break
                    except Exception:
                        continue

                if not semantica_text:
                    semantica_text = pyfiglet.figlet_format("SEMANTICA", font="slant")
                    code_text = pyfiglet.figlet_format("CODE", font="slant")

                semantica_lines = semantica_text.split("\n")
                code_lines = code_text.split("\n")
                max_lines = max(len(semantica_lines), len(code_lines))
                semantica_width = (
                    max(len(line) for line in semantica_lines) if semantica_lines else 0
                )

                for i in range(max_lines):
                    semantica_line = semantica_lines[i] if i < len(semantica_lines) else ""
                    code_line = code_lines[i] if i < len(code_lines) else ""

                    if semantica_line.strip():
                        shadow_offset = 2
                        shadow_line = " " * shadow_offset + semantica_line.rstrip()
                        lines.append(
                            ("black", f"{shadow_line.ljust(semantica_width + shadow_offset)}\n")
                        )

                    lines.append(("bold bright_cyan", semantica_line.ljust(semantica_width)))
                    lines.append(("", "  "))

                    if code_line.strip():
                        code_shadow_offset = 2
                        code_shadow_line = " " * code_shadow_offset + code_line.rstrip()
                        lines.append(("black", f"{code_shadow_line}\n"))
                        lines.append(("bold bright_green", f"{code_line}\n"))
                    else:
                        lines.append(("bold bright_green", f"{code_line}\n"))

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
        lines.append(("dim", "명령어를 입력하세요 (/: 자동완성)\n"))
        lines.append(("", "\n"))

        return FormattedText(lines)

    # 필터링된 명령어 목록 가져오기
    def get_filtered_commands(text: str) -> list[dict]:
        if not text.startswith("/"):
            return []
        cmd_prefix = text[1:].lower().strip()
        filtered = []
        for cmd_info in commands:
            cmd = cmd_info["cmd"]
            if not cmd_prefix or cmd.startswith(cmd_prefix):
                filtered.append(cmd_info)
        return filtered

    # 한글 문자 폭 계산 (한글은 2칸, 영문/숫자는 1칸)
    def get_display_width(text: str) -> int:
        """텍스트의 실제 터미널 표시 폭 계산"""
        width = 0
        for char in text:
            # 한글, 한자, 일본어 등은 2칸
            # ASCII 범위를 벗어나는 문자는 2칸으로 계산
            if ord(char) > 127:
                width += 2
            else:
                width += 1
        return width

    # 자동완성 메뉴 렌더링
    def render_completions() -> FormattedText:
        text = input_buffer.text
        filtered = get_filtered_commands(text)

        if not filtered:
            return FormattedText("")

        lines = []
        max_width = 50  # 고정 폭

        # 상단 테두리 (파스텔 cyan 색상)
        lines.append(("fg:#87ceeb bold", "╭" + "─" * (max_width - 2) + "╮\n"))

        # 명령어 목록 (헤더 없이 바로 시작)
        for i, cmd_info in enumerate(filtered):
            is_selected = i == selected_completion_index[0]
            cmd = cmd_info["cmd"]
            desc = cmd_info["desc"]

            lines.append(("fg:#87ceeb bold", "│"))

            # 고정 폭으로 정렬 (명령어 15자, 설명은 나머지)
            cmd_part = f"/{cmd}"
            cmd_padded = cmd_part.ljust(15)

            # 설명 부분 길이 계산 (고정 폭 유지, 한글 폭 고려)
            max_text_width = max_width - 4  # 좌우 테두리 2칸씩
            prefix_text = f"  {cmd_padded} "
            prefix_width = get_display_width(prefix_text)
            desc_max_width = max_text_width - prefix_width

            # 설명 텍스트를 실제 폭 기준으로 자르기
            desc_display = ""
            current_width = 0

            for char in desc:
                char_width = 2 if ord(char) > 127 else 1
                # "..."을 위한 공간 3칸 확보
                if current_width + char_width > desc_max_width - 3:
                    desc_display += "..."
                    current_width += 3  # "..."은 3칸
                    break
                desc_display += char
                current_width += char_width

            # desc가 비어있거나 모두 표시된 경우
            if not desc_display:
                desc_display = desc

            # 전체 텍스트 구성
            full_text = f"  {cmd_padded} {desc_display}"

            # 정확한 패딩 계산 (실제 표시 폭 기준, 항상 정확히 맞춤)
            text_display_width = get_display_width(full_text)
            padding_right = max_text_width - text_display_width

            # 음수가 되면 안되므로 보정 (텍스트가 너무 길면 강제로 자르기)
            if padding_right < 0:
                padding_right = 0

            # 항상 고정 폭으로 맞추기 위해 공백 추가
            # 패딩이 0이어도 항상 추가하여 일관성 유지
            if is_selected:
                # 선택된 항목: 파스텔 cyan 배경, 검은색 글자
                lines.append(("bg:#b0e0e6 fg:#000000 bold", full_text))
                # 오른쪽 테두리까지 배경색으로 채움 (정확한 폭 유지)
                lines.append(("bg:#b0e0e6", " " * padding_right))
            else:
                # 일반 항목: 투명 배경, 파스텔 cyan 글자
                lines.append(("fg:#87ceeb", full_text))
                # 오른쪽 테두리까지 공백으로 채움 (정확한 폭 유지)
                lines.append(("", " " * padding_right))

            lines.append(("fg:#87ceeb bold", "│\n"))

        # 하단 도움말
        lines.append(("fg:#87ceeb bold", "├" + "─" * (max_width - 2) + "┤\n"))
        lines.append(("fg:#87ceeb bold", "│"))
        help_text = " ↑↓:이동 Enter:선택"
        help_width = get_display_width(help_text)
        help_padding = max_width - 2 - help_width
        lines.append(("fg:#87ceeb dim", help_text))
        if help_padding > 0:
            lines.append(("", " " * help_padding))
        lines.append(("fg:#87ceeb bold", "│\n"))

        # 하단 테두리
        lines.append(("fg:#87ceeb bold", "╰" + "─" * (max_width - 2) + "╯\n"))

        return FormattedText(lines)

    kb = KeyBindings()

    @kb.add("c-c")
    def exit_app(event):
        action[0] = "quit"
        event.app.exit()

    @kb.add("enter")
    def submit_input(event):
        text = input_buffer.text
        if text.startswith("/"):
            filtered = get_filtered_commands(text)
            if filtered and selected_completion_index[0] < len(filtered):
                # 선택된 명령어 사용
                selected_cmd = filtered[selected_completion_index[0]]["cmd"]
                input_buffer.text = f"/{selected_cmd}"

        action[0] = "text"
        user_input_text[0] = input_buffer.text
        event.app.exit()

    @kb.add("up")
    def move_up(event):
        text = input_buffer.text
        if text.startswith("/"):
            filtered = get_filtered_commands(text)
            if filtered:
                if selected_completion_index[0] > 0:
                    selected_completion_index[0] -= 1
                else:
                    selected_completion_index[0] = len(filtered) - 1
                event.app.invalidate()

    @kb.add("down")
    def move_down(event):
        text = input_buffer.text
        if text.startswith("/"):
            filtered = get_filtered_commands(text)
            if filtered:
                if selected_completion_index[0] < len(filtered) - 1:
                    selected_completion_index[0] += 1
                else:
                    selected_completion_index[0] = 0
                event.app.invalidate()

    @kb.add("tab")
    def complete_next(event):
        text = input_buffer.text
        if text.startswith("/"):
            filtered = get_filtered_commands(text)
            if filtered:
                if selected_completion_index[0] < len(filtered) - 1:
                    selected_completion_index[0] += 1
                else:
                    selected_completion_index[0] = 0
                event.app.invalidate()

    try:
        from prompt_toolkit.layout import Float, FloatContainer

        main_content = Window(
            content=FormattedTextControl(text=render_banner),
            wrap_lines=False,
        )

        input_window = Window(
            content=BufferControl(buffer=input_buffer),
            height=1,
            style="fg:#87ceeb",  # 파스텔 cyan 색상
            wrap_lines=False,
        )

        # 자동완성 메뉴 Window
        def get_completion_content():
            return render_completions()

        class ConditionalCompletionWindow(Window):
            def get_height(self):
                text = input_buffer.text
                if text.startswith("/"):
                    filtered = get_filtered_commands(text)
                    if filtered:
                        # 헤더 제거로 높이 조정 (헤더 3줄 제거)
                        return len(filtered) + 3
                # 명령어가 없을 때는 높이 0 (공간 차지 안함)
                return 0

        completion_window = ConditionalCompletionWindow(
            content=FormattedTextControl(text=get_completion_content),
            width=52,  # 고정 폭 (max_width + 테두리 2칸)
            style="fg:#000000",
            dont_extend_width=True,
            dont_extend_height=True,
            ignore_content_width=True,  # 내용과 관계없이 고정 폭 유지
        )

        # Float로 입력창 아래에 배치
        completion_float = Float(
            completion_window,
            xcursor=True,
            ycursor=True,
        )

        # Float는 항상 포함하되, Window 높이가 0이면 공간 차지 안함
        container = FloatContainer(
            HSplit(
                [
                    main_content,
                    input_window,
                ]
            ),
            floats=[completion_float],
        )

        layout = Layout(container)

        app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            mouse_support=True,
        )

        def on_text_changed(buffer):
            text = buffer.text
            user_input_text[0] = text
            # / 입력 시 선택 인덱스 초기화
            if text.startswith("/"):
                selected_completion_index[0] = 0
            else:
                selected_completion_index[0] = 0
            app.invalidate()

        input_buffer.on_text_changed += on_text_changed

        app.run()

        if action[0] == "quit":
            return "q"
        elif action[0] == "text":
            text = user_input_text[0].strip()
            if not text:
                return show_menu(commands)

            cmd_map = {c["cmd"]: c["key"] for c in commands}
            cmd_map.update({c["key"]: c["key"] for c in commands})

            if text.startswith("/"):
                cmd = text[1:].strip().lower()
                return cmd_map.get(cmd, text)

            return cmd_map.get(text.lower(), text)
        else:
            return "q"

    except (KeyboardInterrupt, EOFError):
        return "q"
    except Exception as e:
        console.print(f"\n[dim]에러: {e}[/dim]")
        user_input = Prompt.ask("명령어 입력", default="")

        if user_input.startswith("/"):
            cmd = user_input[1:].strip().lower()
            cmd_map = {c["cmd"]: c["key"] for c in commands}
            return cmd_map.get(cmd, "q")
        else:
            return show_menu(commands)


def select_directory(start_path: str | None = None) -> str | None:
    """디렉토리 선택"""
    from prompt_toolkit import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    current_path = Path(start_path).expanduser() if start_path else Path.cwd()
    if not current_path.exists():
        current_path = Path.cwd()
    if not current_path.is_dir():
        current_path = current_path.parent

    selected_index = [0]
    action = [None]

    def get_items():
        items = []
        items.append({"name": "현재 디렉토리 선택", "path": current_path, "is_current": True})
        if current_path.parent != current_path:
            items.append({"name": "..", "path": current_path.parent, "is_parent": True})
        try:
            for item in sorted(current_path.iterdir()):
                if item.is_dir():
                    items.append({"name": item.name, "path": item, "is_dir": True})
        except PermissionError:
            pass
        return items

    def render_content() -> FormattedText:
        items = get_items()
        if selected_index[0] >= len(items):
            selected_index[0] = len(items) - 1
        if selected_index[0] < 0:
            selected_index[0] = 0

        lines = []
        lines.append(("bold cyan", "디렉토리 탐색\n"))
        lines.append(("yellow", f"현재 경로: {current_path}\n\n"))
        lines.append(("bold", "항목:\n"))

        for i, item in enumerate(items):
            is_selected = i == selected_index[0]
            prefix = ("bold green", "▶ ") if is_selected else ("", "  ")

            if item.get("is_parent"):
                name_part = [("bold blue", f"{item['name']}"), ("", " (상위 디렉토리)")]
            elif item.get("is_current"):
                name_part = [("bold green", f"{item['name']}")]
            else:
                name_part = [("blue", f"{item['name']}/")]

            lines.append(prefix)
            lines.extend(name_part)
            lines.append(("", "\n"))

        lines.append(("", "\n"))
        lines.append(("dim", "조작법: ↑/↓:이동 Enter:선택 Backspace:뒤로가기\n"))

        return FormattedText(lines)

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def go_up(event):
        if selected_index[0] > 0:
            selected_index[0] -= 1
        control = FormattedTextControl(text=render_content)
        event.app.layout = Layout(Window(content=control))
        event.app.invalidate()

    @kb.add("down")
    @kb.add("c-n")
    def go_down(event):
        items = get_items()
        if selected_index[0] < len(items) - 1:
            selected_index[0] += 1
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
        if current_path.parent != current_path:
            current_path = current_path.parent
            selected_index[0] = 0
            control = FormattedTextControl(text=render_content)
            event.app.layout = Layout(Window(content=control))
            event.app.invalidate()

    @kb.add("q")
    @kb.add("c-c")
    def cancel(event):
        action[0] = "cancel"
        event.app.exit()

    while True:
        try:
            control = FormattedTextControl(text=render_content)
            layout = Layout(Window(content=control))
            app = Application(layout=layout, key_bindings=kb, full_screen=False)
            app.run()

            if action[0] == "cancel" or action[0] is None:
                return None

            if action[0] == "select":
                items = get_items()
                selected_item = items[selected_index[0]]
                selected_path = selected_item["path"]

                if selected_item.get("is_current", False):
                    return str(current_path.resolve())

                if selected_item.get("is_parent", False):
                    current_path = selected_path
                    selected_index[0] = 0
                    action[0] = None
                    continue

                if selected_item["is_dir"]:
                    current_path = selected_path.resolve()
                    selected_index[0] = 0
                    action[0] = None
                    continue

        except KeyboardInterrupt:
            return None
        except Exception as e:
            console.print(f"\n[red]오류: {e}[/red]")
            user_input = Prompt.ask("경로 입력 (Enter: 현재, q: 취소)", default="")
            if user_input == "q":
                return None
            elif user_input == "":
                return str(current_path.resolve())
            else:
                new_path = Path(user_input).expanduser()
                if new_path.exists() and new_path.is_dir():
                    current_path = new_path.resolve()
                    selected_index = 0
                else:
                    console.print("[red]존재하지 않는 경로입니다.[/red]")


def select_repo_from_list(repos: list[dict]) -> str | None:
    """저장소 목록에서 선택"""
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
        choice = Prompt.ask("\n저장소 번호 선택 (취소: Enter)", default="")
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


def show_search_results(results: list[dict], query: str):
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

    for result in results[:20]:
        file_path = result.get("file_path", "")
        span = result.get("span", [0, 0, 0, 0])
        score = result.get("score", 0.0)
        text = result.get("text", "")[:100]

        table.add_row(
            file_path,
            f"{span[0] + 1}-{span[2] + 1}" if len(span) >= 3 else str(span[0] + 1),
            text,
            f"{score:.3f}",
        )

    console.print(table)

    if len(results) > 20:
        console.print(f"\n[dim]... 외 {len(results) - 20}개 결과[/dim]")
