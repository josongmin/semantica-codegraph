# CLI 개발 가이드

Semantica Codegraph CLI 개발을 위한 기술 스택 및 가이드라인

## 현재 프로젝트 상태


현재 프로젝트는 다음 기술 스택을 사용 중입니다:
- **CLI 프레임워크**: Click (Typer로 마이그레이션 권장)
- **출력/UX**: Rich + Textual
- **설정**: dataclass + python-dotenv (pydantic-settings로 마이그레이션 권장)
- **로깅**: 표준 logging (structlog 도입 고려)
- **빌드**: setuptools (uv로 마이그레이션 권장)
- **포맷터**: black (Ruff로 통합 고려)

## 1. 러닝타임 / 패키지 / 환경

### Python 버전
- **권장**: Python 3.11+ (가능하면 3.12)
- **현재**: Python 3.10+ (호환성 유지)

### 패키지/환경 관리
**uv** 사용 (권장)

패키지 설치, venv 관리, 빌드까지 uv 하나로 통합

예시:
```bash
uv init mycli
uv add typer rich
uv run mycli ...
```

**현재 프로젝트 마이그레이션**:
```bash
# 기존 setuptools에서 uv로 전환
uv pip compile requirements.txt -o pyproject.toml
uv sync
```

## 2. CLI 프레임워크

### 메인 선택: Typer
- 타입힌트 기반, FastAPI와 동일한 감각
- 멀티 커맨드, 자동 help, 자동 completion 지원

### 참고
- Click은 Typer 내부 기반이니 직접 쓸 일은 거의 없음
- 레거시 프로젝트 아니면 Typer 우선

### 현재 프로젝트 마이그레이션 (Click → Typer)

**현재 상태**: `app/cli/main.py`에서 Click 사용 중

**마이그레이션 예시**:
```python
# Before (Click)
@click.group()
@click.option("--config", type=click.Path(exists=True))
def cli(ctx, config):
    pass

# After (Typer)
import typer
app = typer.Typer()

@app.callback()
def main(config: Path = typer.Option(None, exists=True)):
    pass
```

**점진적 마이그레이션 전략**:
1. 새 명령어는 Typer로 작성
2. 기존 Click 명령어는 유지하면서 점진적 전환
3. Typer와 Click 혼용 가능 (별도 앱으로 분리)

## 3. 출력 / UX 계층

### 기본 출력 라이브러리: Rich
- 컬러, 테이블, 패널, 프로그레스바, 로거 연동
- **현재 프로젝트**: Rich 사용 중 ✅

### TUI(인터랙티브 화면) 필요할 때
- **권장**: Textual (Rich 기반, 더 현대적) ✅
- **현재 프로젝트**: Textual 사용 중
  - `app/cli/ui/`에서 Textual로 대화형 UI 구현
  - Rich 기반으로 일관된 스타일링 가능

**Textual 특징**:
- Rich 기반, 더 간단한 API, 자동 레이아웃
- 반응형 디자인 지원
- CSS 스타일링 지원

### 배너/아트
- pyfiglet, art 정도만 필요 시 추가
- **현재 프로젝트**: 필요 시 pyfiglet 추가 가능

## 4. 설정 / 구성 / 로깅

### 설정 관리: pydantic-settings (권장)
- .env, 환경변수, config 파일 통합 관리
- 타입 검증 자동화
- 설정 문서화 자동 생성

**현재 프로젝트**: `src/core/config.py`에서 dataclass + python-dotenv 사용
```python
# 현재 방식
@dataclass
class Config:
    postgres_host: str = "localhost"

    @classmethod
    def from_env(cls) -> "Config":
        # 수동으로 os.getenv() 호출
```

**마이그레이션 예시**:
```python
# 권장 방식 (pydantic-settings)
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    postgres_host: str = "localhost"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

### 로깅: structlog (권장)
- CLI용 structured logging에 적합
- Rich와 붙여서 pretty 로그도 가능
- JSON 로그 출력 지원 (프로덕션)

**현재 프로젝트**: 표준 logging 사용 중
- `app/cli/main.py`에서 Rich Console로 에러 출력
- structlog 도입 시 더 구조화된 로깅 가능

**마이그레이션 예시**:
```python
import structlog

logger = structlog.get_logger()
logger.info("인덱싱 시작", repo_path=repo_path)
```

## 5. 패키징 / 배포 전략

### 기본 패키징
- pyproject.toml + uv
- 빌드: `uv build`
- publish (필요 시): `uv publish`

### 배포 채널

#### pipx (기본)
파이썬 유저 대상 전역 설치 기본안
```bash
pipx install mycli
```

#### uv tool build (옵션, 미래지향)
단일 실행 파일 만들 때
- 파이썬 없는 환경, 사내용 배포에 유리

#### 옵션: Docker 이미지
인프라/CI 용도로 Docker 이미지로 감싸서 전달

## 6. 품질 / 테스트

### 코드 품질
- **Ruff**: lint + format 통합 (권장)
- **Pyright** 또는 **mypy**: 타입 체크

**현재 프로젝트**:
- Ruff 사용 중 ✅ (`pyproject.toml`에 설정)
- mypy 사용 중 ✅ (`pyproject.toml`에 설정)
- black 사용 중 (Ruff로 통합 가능)

**Ruff로 통합**:
```bash
# black 제거하고 Ruff 포맷터 사용
uv remove black
# Ruff는 이미 포맷터 포함 (ruff format)
```

### 테스트
- **pytest**
- 출력 검증은 snapshot 테스트 패턴 사용 권장

**현재 프로젝트**: pytest 사용 중 ✅

## 7. 프로젝트 구조 권장안

### 간단 CLI 기준 스캐폴드

```
mycli/
  src/
    mycli/
      __init__.py
      cli.py          # Typer app entry
      commands/
        __init__.py
        init.py
        run.py
        info.py
      core/
        config.py      # pydantic-settings
        logger.py      # structlog + rich
      ui/
        style.py       # Rich 스타일, 공통 출력 함수
  tests/
    test_cli_basic.py
  pyproject.toml       # uv 관리
```

### 현재 프로젝트 구조 (Semantica Codegraph)

```
semantica-codegraph/
  app/
    cli/
      __init__.py
      main.py          # Click 기반 CLI 진입점
      interactive.py   # 대화형 모드 로직
      ui/              # Textual 기반 TUI 컴포넌트
        app.py         # 메인 TUI 앱
        menu.py        # 메뉴 화면
        search.py      # 검색 화면
      commands/        # 명령어 모듈 (예정)
        __init__.py
        index.py
        search.py
        list.py
      ui/              # UI 컴포넌트 (예정)
        __init__.py
        style.py
        components.py
      core/            # CLI 전용 설정/로깅 (예정)
        __init__.py
        logger.py
  src/
    core/
      config.py        # dataclass 기반 설정
      bootstrap.py     # DI 컨테이너
    ...
  pyproject.toml       # setuptools 빌드
```

**구조 개선 계획**:
- `app/cli/commands/` 폴더로 명령어 분리
- `app/cli/ui/` 폴더로 UI 컴포넌트 분리
- `app/cli/core/` 폴더로 CLI 전용 설정/로깅 분리

## 마이그레이션 우선순위

### 높은 우선순위
1. **Click → Typer**: 새 명령어부터 Typer로 작성
2. **dataclass Config → pydantic-settings**: 타입 검증 강화
3. **black → Ruff**: 포맷터 통합

### 중간 우선순위
4. **logging → structlog**: 구조화된 로깅 (필요 시)

### 낮은 우선순위
6. **setuptools → uv**: 빌드 시스템 전환 (점진적)

## 요약

### 권장 스택
- **환경/패키지**: uv
- **CLI 프레임워크**: Typer
- **UI**: Rich (+ Textual 옵션)
- **Config/로그**: pydantic-settings + structlog
- **배포**: pipx 기본, uv tool build 옵션
- **품질**: Ruff + Pyright + pytest

### 현재 프로젝트 상태
- **환경/패키지**: setuptools (uv 전환 권장)
- **CLI 프레임워크**: Click (Typer 전환 권장)
- **UI**: Rich + Textual ✅
- **Config**: dataclass + python-dotenv (pydantic-settings 전환 권장)
- **로깅**: 표준 logging (structlog 도입 고려)
- **품질**: Ruff + mypy + pytest ✅
