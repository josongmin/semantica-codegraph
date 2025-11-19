"""Textual 기반 CLI 앱"""

import asyncio
import logging
import subprocess
import traceback
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Static, TextArea

from src.core.bootstrap import Bootstrap

# 로깅 설정
# 프로젝트 루트 찾기 (apps/cli/app.py -> 프로젝트 루트)
project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / "semantica_cli.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class AnimatedArt(Static):
    """애니메이션 ASCII 아트 (파일에서 로드)"""

    @staticmethod
    def load_frames(filepath: str) -> list[str]:
        """파일에서 프레임 로드 (빈 줄로 구분)"""
        from pathlib import Path

        file_path = Path(__file__).parent / "resources" / filepath
        with file_path.open(encoding="utf-8") as f:
            content = f.read()

        # 빈 줄로 프레임 분리
        frames = content.split("\n\n")
        # 빈 프레임 제거
        frames = [f for f in frames if f.strip()]
        return frames

    @staticmethod
    def scale_down_ascii(ascii_text: str, scale: float = 0.5) -> str:
        """ASCII 아트를 스케일 다운 (비율 유지)"""
        lines = ascii_text.split("\n")
        if not lines:
            return ascii_text

        # 스케일링: 세로로는 일정 간격으로 줄 선택, 가로로는 문자 간격 조정
        scaled_lines = []
        step = int(1 / scale)  # 0.5면 2줄마다 1줄 선택

        for i in range(0, len(lines), step):
            line = lines[i]
            # 가로 스케일링: 2칸마다 1칸 선택
            scaled_line = "".join(line[j] for j in range(0, len(line), step))
            scaled_lines.append(scaled_line)

        return "\n".join(scaled_lines)

    def __init__(self, **kwargs):
        # 프레임 로드
        try:
            frames = self.load_frames("ascii-animation.txt")
            # 각 프레임을 절반 크기로 스케일링
            self.frames = [self.scale_down_ascii(frame, 0.5) for frame in frames]
        except Exception:
            # 파일 로드 실패 시 기본 프레임
            self.frames = ["[Loading...]"]

        super().__init__(self.frames[0] if self.frames else "", **kwargs)
        self.frame_index = 0
        self._task = None

    async def on_mount(self) -> None:
        """마운트 시 애니메이션 시작"""
        if self.frames:
            self._task = asyncio.create_task(self._animate_loop())

    async def _animate_loop(self) -> None:
        """애니메이션 루프"""
        while True:
            await asyncio.sleep(0.1)  # 72프레임을 빠르게 재생
            self.frame_index = (self.frame_index + 1) % len(self.frames)
            self.update(self.frames[self.frame_index])

    def on_unmount(self) -> None:
        """언마운트 시 태스크 취소"""
        if self._task:
            self._task.cancel()


class WelcomeView(Container):
    """첫 화면: ASCII 로고 + 안내 텍스트"""

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield AnimatedArt(id="welcome-art")
            yield Static(
                "\n[bold #7eb8d4]Welcome to Semantica Codegraph[/bold #7eb8d4]\n"
                "[#7eb8d4]Ver 0.1\n\n"
                "[#ffb380]/help[/#ffb380] - 도움말 표시\n"
                "[#ffb380]/index[/#ffb380] - 저장소 인덱싱\n"
                "[#ffb380]/search[/#ffb380] - 코드 검색\n"
                "[#ffb380]/repos[/#ffb380] - 저장소 관리\n"
                "[#ffb380]/exit[/#ffb380] - 종료\n\n",
                id="welcome-text",
            )


class ChatView(ScrollableContainer):
    """채팅 로그 스크롤 영역"""

    # 스크롤 속도 설정
    VERTICAL_SCROLL_SPEED = 3  # 기본값보다 빠르게

    def on_mount(self) -> None:
        self.styles.padding = (1, 2)
        self.can_focus = True


class SemanticaApp(App):
    """Semantica Codegraph CLI 앱"""

    CSS = """
    /* 전체 레이아웃 */
    Screen {
        layout: vertical;
        background: #fffef7;
        color: #5a5a5a;
    }

    /* 본문 영역 (가운데 영역) */
    #body {
        height: 1fr;
        background: #fffef7;
        align: center middle;
    }

    /* Bottom 영역 (입력 + 경로) */
    #bottom {
        height: auto;
        dock: bottom;
        layout: vertical;
        background: #fffef7;
        border-top: solid #d4d4d4;
        padding: 1 2;
    }

    #input-box {
        height: 1;
        background: #fffef7;
        color: #5a5a5a;
        border: none;
        margin-bottom: 1;
    }

    #path-info {
        height: 1;
        content-align: right middle;
        color: #a0a0a0;
        text-style: dim italic;
    }

    /* 커맨드 팔레트 (오버레이) */
    #command-suggestions-container {
        layer: overlay;
        width: 100%;
        height: 100%;
        align: center middle;
        display: none;
        background: rgba(0, 0, 0, 0.3);
    }

    #command-suggestions {
        width: 60;
        height: auto;
        max-height: 20;
        background: #fffef7;
        border: round #d4d4d4;
        padding: 1;
    }

    /* Welcome View */
    WelcomeView {
        width: 100%;
        height: 100%;
        align: center middle;
    }

    WelcomeView > Horizontal {
        width: auto;
        height: auto;
        align: center middle;
    }

    #welcome-art {
        width: auto;
        height: auto;
        padding-right: 2;
        margin-right: 10;
        color: #7eb8d4;
    }

    #welcome-text {
        width: auto;
        height: auto;
        padding-left: 2;
        content-align: left middle;
        color: #5a5a5a;
        background: #fffef7;
    }

    /* Chat View */
    ChatView {
        scrollbar-gutter: stable;
        width: 100%;
        height: 100%;
    }

    ScrollableContainer {
        overflow-y: auto;
    }

    .user-msg {
        color: #5a5a5a;
        margin: 1 0;
    }

    .assistant-msg {
        color: #7eb8d4;
        margin: 1 0;
    }

    .assistant-thinking {
        color: #a0a0a0;
        margin: 1 0;
    }

    """

    mode = reactive("welcome")  # "welcome" → "chat" 전환용
    BINDINGS = [
        ("ctrl+c", "quit", "종료"),
    ]

    def __init__(self, bootstrap: Bootstrap):
        super().__init__()
        self.bootstrap = bootstrap

    @staticmethod
    def get_current_path() -> str:
        """현재 작업 디렉토리와 Git 브랜치 정보 반환"""
        try:
            cwd = Path.cwd()
            home = Path.home()

            # 홈 디렉토리 기준으로 경로 축약
            try:
                path_str = str(cwd.relative_to(home))
                display_path = f"~/{path_str}"
            except ValueError:
                display_path = str(cwd)

            # Git 브랜치 정보 가져오기
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                branch = result.stdout.strip() if result.returncode == 0 else None
            except (subprocess.TimeoutExpired, FileNotFoundError):
                branch = None

            if branch:
                return f"{display_path} ({branch})"
            return display_path
        except Exception:
            return str(Path.cwd())

    def compose(self) -> ComposeResult:
        # Body: 중앙 영역 (처음엔 WelcomeView, 나중엔 ChatView로 교체)
        yield Container(WelcomeView(), id="body")

        # Bottom: 입력 영역 (하단 고정)
        with Container(id="bottom"):
            yield TextArea(
                id="input-box", placeholder="명령어를 입력하세요 (예: /help, /index, /search)"
            )
            yield Static(self.get_current_path(), id="path-info")

        # Overlay: 커맨드 팔레트 (/ 입력 시만 표시)
        with Container(id="command-suggestions-container"):
            yield Static("", id="command-suggestions")

    def on_mount(self) -> None:
        """마운트 시 초기화"""
        self.title = "Semantica Codegraph"
        self.query_one("#input-box", TextArea).focus()

        # 경로 정보 주기적 업데이트 (10초마다)
        self.set_interval(10, self.update_path_info)

    def update_path_info(self) -> None:
        """경로 정보 업데이트"""
        try:
            path_widget = self.query_one("#path-info", Static)
            path_widget.update(self.get_current_path())
        except Exception:
            pass

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """TextArea 입력 변경 시 명령어 제안 표시"""
        try:
            text = event.text_area.text
            logger.debug(f"입력 변경: '{text}'")
            suggestions_container = self.query_one("#command-suggestions-container", Container)
            suggestions_widget = self.query_one("#command-suggestions", Static)

            # "/"로 시작하면 명령어 목록 표시
            if text.startswith("/"):
                prefix = text[1:].strip().lower()
                commands = {
                    "help": "도움말 표시",
                    "index": "저장소 인덱싱",
                    "search": "코드 검색",
                    "repos": "저장소 관리",
                    "exit": "종료",
                }

                # prefix에 맞는 명령어 필터링
                filtered = (
                    {k: v for k, v in commands.items() if k.startswith(prefix)}
                    if prefix
                    else commands
                )

                if filtered:
                    # 제안 내용 생성
                    suggestion_lines = ["[bold #7eb8d4]명령어 목록:[/bold #7eb8d4]"]
                    for cmd, desc in filtered.items():
                        suggestion_lines.append(f"  [#ffb380]/{cmd}[/#ffb380] - {desc}")

                    content = "\n".join(suggestion_lines)
                    suggestions_widget.update(content)
                    suggestions_container.styles.display = "block"

                    logger.debug(f"제안 표시: {len(filtered)}개")
                else:
                    suggestions_container.styles.display = "none"
            else:
                # "/"로 시작하지 않으면 숨김
                suggestions_container.styles.display = "none"
        except Exception as e:
            logger.error(f"명령어 제안 표시 중 에러: {e}")
            logger.error(traceback.format_exc())

    def on_key(self, event) -> None:
        """키 이벤트 처리: Enter는 제출, Shift+Enter는 개행"""
        try:
            input_box = self.query_one("#input-box", TextArea)
            if input_box.has_focus and event.key == "enter" and not event.shift:
                # Enter만 누르면 제출
                event.prevent_default()
                self.action_submit_input()
                # Shift+Enter는 기본 동작(개행) 유지
        except Exception as e:
            logger.error(f"키 이벤트 처리 중 에러 발생: {e}")
            logger.error(traceback.format_exc())

    def _handle_command(self, text: str, chat_view: ChatView) -> None:
        """명령어 처리"""
        if text == "help":
            chat_view.mount(
                Static(
                    "[#7eb8d4]사용 가능한 명령어:[/#7eb8d4]\n"
                    "• help - 도움말\n"
                    "• index - 저장소 인덱싱\n"
                    "• search - 코드 검색\n"
                    "• repos - 저장소 목록\n"
                    "• exit - 종료",
                    classes="assistant-msg",
                )
            )
        elif text == "index":
            chat_view.mount(
                Static("[#a0a0a0]인덱싱 기능 준비 중...[/#a0a0a0]", classes="assistant-thinking")
            )
        elif text == "search":
            chat_view.mount(
                Static("[#a0a0a0]검색 기능 준비 중...[/#a0a0a0]", classes="assistant-thinking")
            )
        elif text == "repos":
            chat_view.mount(
                Static(
                    "[#a0a0a0]저장소 목록 기능 준비 중...[/#a0a0a0]", classes="assistant-thinking"
                )
            )
        else:
            chat_view.mount(
                Static(
                    f"[#a0a0a0]알 수 없는 명령어: {text}[/#a0a0a0]\n"
                    "help를 입력하여 사용 가능한 명령어를 확인하세요.",
                    classes="assistant-msg",
                )
            )

    def action_quit(self) -> None:  # type: ignore[override]
        """종료 액션"""
        self.exit()

    def action_submit_input(self) -> None:
        """입력 제출 액션"""
        try:
            input_box = self.query_one("#input-box", TextArea)
            text = input_box.text.strip()
            if not text:
                return

            # 명령어 제안 숨기기
            try:
                suggestions_container = self.query_one("#command-suggestions-container", Container)
                suggestions_container.styles.display = "none"
            except Exception:
                pass

            # 명령어 처리
            if text in ["exit", "quit", "q"]:
                self.exit()
                return

            body = self.query_one("#body", Container)

            # 처음 입력이 들어오면 welcome → chat 레이아웃으로 교체
            if self.mode == "welcome":
                body.remove_children()
                chat_view = ChatView(id="chat-view")
                body.mount(chat_view)
                self.mode = "chat"

            chat_view = self.query_one("#chat-view", ChatView)

            # 유저 메시지 추가
            chat_view.mount(Static(f"[bold]>[/bold] {text}", classes="user-msg"))

            # 명령어 처리
            self._handle_command(text, chat_view)

            # 입력창 초기화 + 스크롤 맨 아래
            input_box.text = ""
            chat_view.scroll_end()
        except Exception as e:
            logger.error(f"입력 제출 중 에러 발생: {e}")
            logger.error(traceback.format_exc())
