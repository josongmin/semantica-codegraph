from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Input, Static

# ----------------------------------------------------------------------
# 커스텀 Input: placeholder는 연한 색, 실제 입력은 진한 색
# ----------------------------------------------------------------------


class FadedPlaceholderInput(Input):
    """placeholder는 연하게, 입력 값은 진하게 렌더링하는 Input"""

    # 색 설정
    text_color = "#3a3a3a"  # 실제 입력 텍스트 색
    placeholder_color = "#e0e0e0"  # placeholder 텍스트 색 (더 연하게)

    def render(self) -> Text:
        # 값이 있으면: 실제 입력 텍스트 스타일
        if self.value:
            return Text(self.value, style=self.text_color)
        # 값이 없으면: placeholder 스타일 (연한 + 이탤릭)
        if self.placeholder is not None:
            return Text(self.placeholder, style=f"italic {self.placeholder_color}")
        # placeholder도 없으면 기본
        return Text("", style=self.text_color)


# ----------------------------------------------------------------------
# 중앙에 보여줄 ASCII 아트 + 웰컴 텍스트
# ----------------------------------------------------------------------


ASCII_ART = r"""
    ..        ..
  ......    ......
 ........  ........
 ........  ........
 ........  ........
  ......    ......
    ..        ..
"""

WELCOME_TEXT = """[b]Welcome to Semantica Codegraph CLI[/b]

The AI-first environment for navigating,
indexing, and transforming your codebase.

Use / to run commands, type [b]semantica:help[/b] for more information.
"""


# ----------------------------------------------------------------------
# 하단 영역
# ----------------------------------------------------------------------


class Bottom(Static):
    def compose(self) -> ComposeResult:
        # 인풋 컨테이너 (레이아웃만 담당)
        with Container(id="input_container"):
            yield FadedPlaceholderInput(
                id="input",
                placeholder="Try /help, /open, or just type / to see commands",
            )

        # 우측 정렬 경로
        yield Static(
            "~/Documents/code-jo/semantica-codegraph (main)",
            id="status_row",
        )


# ----------------------------------------------------------------------
# 메인 스크린
# ----------------------------------------------------------------------


class MainScreen(Screen):
    CSS = """
    MainScreen {
        layout: vertical;
        background: #fffef7;
        color: #5a5a5a;
    }

    Static, Container {
        background: #fffef7;
        color: #5a5a5a;
    }

    /* 헤더 (위쪽 여백용) */
    #header {
        height: 2;
        dock: top;
        align: center middle;
    }

    /* 가운데 본문 전체 */
    #body {
        height: 1fr;
        align: center middle;         /* 세로/가로 중앙 정렬 */
    }

    #welcome-art {
        width: auto;
        color: #2f7b4f;
        content-align: center middle;
    }

    #welcome-text {
        width: auto;
        color: #5a5a5a;
        padding-top: 1;
        content-align: center middle;
    }

    /* 하단 영역 */
    #bottom {
        dock: bottom;
        height: 5;                    /* 인풋 3 + 경로 1 + 여유 */
        layout: vertical;
        padding: 0 1;
    }

    #input_container {
        height: 3;
        layout: vertical;
        align: left middle;
        /* border 없음: border는 Input에 직접 적용 */
    }

    /* 인풋 박스: border + 동일 padding */
    #input {
        width: 1fr;
        height: 100%;
        border: round #d4d4d4;
        background: #fffef7;

        padding-left: 1;              /* 커맨드 팔레트와 x좌표 맞춤 */
        padding-right: 0;
        padding-top: 0;
        padding-bottom: 0;
    }

    #status_row {
        height: 1;
        width: 100%;
        content-align: right middle;
        color: #a0a0a0;
        text-style: dim italic;
    }

    /* 커맨드 팔레트 (인풋 위에 떠 있는 오버레이) */
    #cmd_palette {
        layer: overlay;
        dock: bottom;
        width: 100%;
        height: auto;

        margin-bottom: 5;             /* bottom 높이만큼 위로 띄움 */
        margin-left: 1;
        margin-right: 1;

        border: round #d4d4d4;
        background: #fffef7;
        overflow-y: auto;
        color: #5a5a5a;

        padding-left: 1;              /* 인풋과 동일 */
        padding-right: 0;
        padding-top: 0;
        padding-bottom: 0;

        display: none;
    }
    """

    commands = {
        "/help": "Show help",
        "/open": "Open a repository",
        "/git status": "Show git status",
        "/run": "Run codegraph analysis",
        "/clear": "Clear terminal output",
    }

    def compose(self) -> ComposeResult:
        # 헤더 (현재는 빈 영역)
        yield Static("", id="header")

        # 가운데: ASCII + 텍스트
        with Container(id="body"):
            yield Static(ASCII_ART, id="welcome-art")
            yield Static(WELCOME_TEXT, id="welcome-text")

        # 하단 바
        yield Bottom(id="bottom")

        # 커맨드 팔레트 오버레이
        yield Static("", id="cmd_palette")

    # 인풋 변경 시 커맨드 팔레트 표시/숨김
    def on_input_changed(self, event: Input.Changed) -> None:
        palette = self.query_one("#cmd_palette", Static)
        value = event.value

        if value.startswith("/"):
            matched = [(cmd, desc) for cmd, desc in self.commands.items() if cmd.startswith(value)]

            if matched:
                lines = []
                for cmd, desc in matched:
                    line = f"[#7eb8d4]{cmd}[/#7eb8d4]   [#a0a0a0 italic]{desc}[/#a0a0a0 italic]"
                    lines.append(line)

                palette.update("\n".join(lines))
                palette.display = True

            else:
                palette.update("")
                palette.display = False
        else:
            palette.update("")
            palette.display = False


# ----------------------------------------------------------------------
# Textual App
#  - interactive.py 에서 SemanticaApp(Bootstrap(...)) 이렇게 호출해도
#    driver_class 로 Bootstrap 이 안 넘어가도록 방어
# ----------------------------------------------------------------------


class SemanticaApp(App):
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    CSS = """
    App {
        background: #fffef7;
        color: #5a5a5a;
    }
    """

    def __init__(self, *args, **kwargs):
        """
        interactive.py 쪽에서 SemanticaApp(Bootstrap(...)) 처럼
        positional 인자를 넘겨도, Textual App.__init__ 에는
        절대 전달하지 않도록 args 를 모두 무시함.
        """
        # Bootstrap 같은 건 필요하면 따로 보관
        self.bootstrap = args[0] if args else None

        # Textual 쪽에는 driver_class 만 명시적으로 넘김
        super().__init__(driver_class=None, **kwargs)

    def on_mount(self) -> None:
        self.push_screen(MainScreen())
