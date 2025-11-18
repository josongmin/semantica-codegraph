"""Textual 기반 CLI 앱"""

import asyncio
import logging
import subprocess
import traceback
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Static, Input, TextArea
from textual.containers import Container, ScrollableContainer, Horizontal, Vertical
from textual.reactive import reactive
from textual import events

from src.core.bootstrap import Bootstrap

# 로깅 설정
# 프로젝트 루트 찾기 (apps/cli/app.py -> 프로젝트 루트)
project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / "semantica_cli.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnimatedArt(Static):
    """애니메이션 ASCII 아트 (파일에서 로드)"""
    
    @staticmethod
    def load_frames(filepath: str) -> list[str]:
        """파일에서 프레임 로드 (빈 줄로 구분)"""
        from pathlib import Path
        
        file_path = Path(__file__).parent / "resources" / filepath
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 빈 줄로 프레임 분리
        frames = content.split('\n\n')
        # 빈 프레임 제거
        frames = [f for f in frames if f.strip()]
        return frames
    
    @staticmethod
    def scale_down_ascii(ascii_text: str, scale: float = 0.5) -> str:
        """ASCII 아트를 스케일 다운 (비율 유지)"""
        lines = ascii_text.split('\n')
        if not lines:
            return ascii_text
        
        # 스케일링: 세로로는 일정 간격으로 줄 선택, 가로로는 문자 간격 조정
        scaled_lines = []
        step = int(1 / scale)  # 0.5면 2줄마다 1줄 선택
        
        for i in range(0, len(lines), step):
            line = lines[i]
            # 가로 스케일링: 2칸마다 1칸 선택
            scaled_line = ''.join(line[j] for j in range(0, len(line), step))
            scaled_lines.append(scaled_line)
        
        return '\n'.join(scaled_lines)
    
    def __init__(self, **kwargs):
        # 프레임 로드
        try:
            frames = self.load_frames("ascii-animation.txt")
            # 각 프레임을 절반 크기로 스케일링
            self.frames = [self.scale_down_ascii(frame, 0.5) for frame in frames]
        except Exception as e:
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
                "\n[bold #7eb8d4]Welcome to Semantica Codegraph[/bold #7eb8d4]\n\n"
                
                "[#ffb380]/help[/#ffb380] - 도움말 표시\n"
                "[#ffb380]/index[/#ffb380] - 저장소 인덱싱\n"
                "[#ffb380]/search[/#ffb380] - 코드 검색\n"
                "[#ffb380]/repos[/#ffb380] - 저장소 관리\n"
                "[#ffb380]/exit[/#ffb380] - 종료\n\n"
                "[#a0a0a0]명령어를 입력하세요[/#a0a0a0]",
                id="welcome-text",
            )


class ChatView(ScrollableContainer):
    """채팅 로그 스크롤 영역"""
    
    # 스크롤 속도 설정
    VERTICAL_SCROLL_SPEED = 3  # 기본값보다 빠르게

    def on_mount(self) -> None:
        self.styles.padding = (1, 2)
        self.can_focus = True


class InputPanel(Vertical):
    """하단 입력 패널 (라벨 + 입력 + 경로)"""

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
                    timeout=1
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
        yield TextArea(id="input-box", placeholder="명령어를 입력하세요 (예: help, index, search)")
        yield Static(self.get_current_path(), id="input-path")



class SemanticaApp(App):
    """Semantica Codegraph CLI 앱"""

    CSS = """
    Screen {
        layout: vertical;
        background: #fffef7;
        color: #5a5a5a;
    }

    #body {
        height: 1fr;
        background: #fffef7;
    }

    #command-suggestions {
        layer: overlay;
        height: auto;
        width: auto;
        background: #fffef7;
        border: round #d4d4d4;
        padding: 1;
        display: none;
    }

    .suggestions-visible {
        display: block;
    }

    .footer {
        height: auto;
        dock: bottom;
        layout: vertical;
    }

    .input-box-panel {
        border: round #d4d4d4;
        padding: 1;
        background: #fffef7;
        margin-bottom: 0;
        height: auto;
    }
    
    ChatView {
        scrollbar-gutter: stable;
    }
    
    ScrollableContainer {
        overflow-y: auto;
    }

    WelcomeView {
        width: 100%;
        height: 100%;
        align: center middle;
    }

    WelcomeView > Horizontal {
        width: auto;
        height: 100%;
        align: center middle;
    }

    #welcome-art {
        width: auto;
        padding-right: 2;
        margin-right: 10;
        color: #7eb8d4;
        height: auto;
    }

    #welcome-text {
        width: auto;
        padding-left: 2;
        content-align: left middle;
        color: #5a5a5a;
        background: #fffef7;
        height: 100%;
        align: left middle;
    }

    /* 채팅 뷰 */
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

    /* 아래쪽 입력 패널 전체 박스 */
    InputPanel {
        height: auto;
        border: round #d4d4d4;
        padding: 1 0 0 0;
        
        background: #fffef7;
    }

    #input-box {
        height: 1;
        background: #fffef7;
        color: #5a5a5a;
        border: none;
    }

    #input-path {
        margin-top: 0;
        padding-top: 0;
        content-align: right middle;
        color: #a0a0a0;
        text-style: dim italic;
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
                    timeout=1
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
        # 중앙 영역: 처음엔 WelcomeView, 나중엔 ChatView로 교체
        yield Container(WelcomeView(), id="body")
        # 명령어 제안 오버레이
        yield Static("", id="command-suggestions")
        # 하단 입력 패널은 항상 고정
        with Container(classes="footer", id="footer-container"):
            with Container(classes="input-box-panel"):
                yield TextArea(id="input-box", placeholder="명령어를 입력하세요 (예: help, index, search)")
            yield Static(self.get_current_path(), id="input-path")

    def on_mount(self) -> None:
        """마운트 시 초기화"""
        self.title = "Semantica Codegraph"
        self.query_one("#input-box", TextArea).focus()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """TextArea 입력 변경 시 명령어 제안 표시"""
        try:
            text = event.text_area.text
            logger.debug(f"입력 변경: '{text}'")
            suggestions_widget = self.query_one("#command-suggestions", Static)
            input_box = event.text_area
            
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
                filtered = {k: v for k, v in commands.items() if k.startswith(prefix)} if prefix else commands
                
                if filtered:
                    # 제안 내용 생성
                    suggestion_lines = ["[bold #7eb8d4]명령어 목록:[/bold #7eb8d4]"]
                    for cmd, desc in filtered.items():
                        suggestion_lines.append(f"  [#ffb380]/{cmd}[/#ffb380] - {desc}")
                    
                    content = "\n".join(suggestion_lines)
                    suggestions_widget.update(content)
                    suggestions_widget.styles.display = "block"
                    
                    # 입력창 위치 기준으로 오버레이 위치 설정
                    input_region = input_box.region
                    # 입력창 위에 배치 (y 위치 - 제안 높이)
                    suggestions_widget.styles.offset = (2, input_region.y - len(filtered) - 3)
                    
                    logger.debug(f"제안 표시: {len(filtered)}개, offset=({2}, {input_region.y - len(filtered) - 3}), input_region={input_region}")
                else:
                    suggestions_widget.styles.display = "none"
            else:
                # "/"로 시작하지 않으면 숨김
                suggestions_widget.styles.display = "none"
        except Exception as e:
            logger.error(f"명령어 제안 표시 중 에러: {e}")
            logger.error(traceback.format_exc())



    def on_key(self, event) -> None:
        """키 이벤트 처리: Enter는 제출, Shift+Enter는 개행"""
        try:
            input_box = self.query_one("#input-box", TextArea)
            if input_box.has_focus and event.key == "enter":
                if not event.shift:
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
                    classes="assistant-msg"
                )
            )
        elif text == "index":
            chat_view.mount(Static("[#a0a0a0]인덱싱 기능 준비 중...[/#a0a0a0]", classes="assistant-thinking"))
        elif text == "search":
            chat_view.mount(Static("[#a0a0a0]검색 기능 준비 중...[/#a0a0a0]", classes="assistant-thinking"))
        elif text == "repos":
            chat_view.mount(Static("[#a0a0a0]저장소 목록 기능 준비 중...[/#a0a0a0]", classes="assistant-thinking"))
        else:
            chat_view.mount(
                Static(
                    f"[#a0a0a0]알 수 없는 명령어: {text}[/#a0a0a0]\n"
                    "help를 입력하여 사용 가능한 명령어를 확인하세요.",
                    classes="assistant-msg"
                )
            )

    def action_quit(self) -> None:
        """종료 액션"""
        self.exit()

    def action_submit_input(self) -> None:
        """입력 제출 액션"""
        try:
            input_box = self.query_one("#input-box", TextArea)
            text = input_box.text.strip()
            if not text:
                return

            # 툴팁 제거
            try:
                tooltip = self.query_one("#command-tooltip", Container)
                tooltip.remove()
            except:
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

