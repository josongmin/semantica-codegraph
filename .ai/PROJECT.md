# Semantica Codegraph 프로젝트

## 개요
SourceGraph의 Cody처럼 빠르게 코드베이스를 지식화하여 읽을 수 있게 하는 시스템입니다.

코드베이스를 빠르게 인덱싱하고, 코드 그래프를 구축하여 의미론적 검색과 컨텍스트 이해를 제공합니다.

## 프로젝트 목적
- 코드베이스의 빠른 지식화 및 인덱싱
- 코드 그래프 기반 의미론적 검색
- 컨텍스트 기반 코드 이해 및 탐색
- MCP 프로토콜을 통한 AI 도구 연동

## 프로젝트 구조

```
src/                    # 라이브러리 코드
├── core/              # 핵심 모델, 포트, 설정, bootstrap (DI)
├── indexer/           # 저장소 스캔 및 인덱싱 파이프라인
├── parser/             # 코드 파서 (Python, TypeScript)
├── graph/             # IR 빌더, PostgreSQL 저장소, 그래프 확장
├── chunking/          # 코드 청킹 및 저장
├── embedding/         # 임베딩 서비스 및 pgvector 저장소
├── search/            # 검색 도메인
│   ├── ports/         # 검색 포트 (Lexical, Semantic, Graph, Ranker, ContextPacker)
│   ├── lexical/      # 키워드 검색 어댑터 (MeiliSearch, Zoekt)
│   ├── semantic/     # 의미론적 검색 어댑터
│   ├── graph/         # 그래프 검색 어댑터
│   ├── retriever/     # 하이브리드 리트리버
│   └── ranking/       # 랭커 구현
└── context/           # 컨텍스트 패킹

apps/                   # 실행 가능한 애플리케이션들
├── cli/               # CLI 도구
├── mcp_server/        # MCP 서버
└── api/               # HTTP API 서버
```

## 주요 기술 스택
- PostgreSQL + pgvector
- Meilisearch
- BM25
- Python 3.10+

## 개발 규칙
- 타입 힌팅 사용
- 포트/어댑터 패턴 적용
- 모듈 간 의존성 최소화
- 도메인별 패키지 분리 (검색 기능은 `search/` 패키지에서 관리)
- 의존성 주입은 `core/bootstrap.py`의 `Bootstrap` 클래스를 통해 관리
  - 모든 포트 인스턴스는 `Bootstrap`에서 생성 및 제공
  - 설정은 `Config.from_env()`로 환경변수에서 로드
  - 어댑터 네이밍: `{Backend}Adapter` (예: `MeiliSearchAdapter`, `ZoektAdapter`)
- 포트 분리 원칙:
  - `core/ports.py`: 인프라 레이어 포트 (RepoScanner, Parser, GraphStore, ChunkStore, EmbeddingStore)
  - `search/ports/`: 검색 도메인 포트 (LexicalSearch, SemanticSearch, GraphSearch, Ranker)
- 오픈소스 활용 원칙:
  - 자주 사용되는 기능이나 검증된 로직이 오픈소스로 존재하면 최대한 활용
  - 직접 구현하기 전에 관련 오픈소스 라이브러리/프로젝트를 먼저 검토
  - 오픈소스를 활용할 때는 프로젝트 아키텍처(포트/어댑터 패턴)에 맞게 래핑하여 사용
  - 예: 파싱 캐시, 증분 인덱싱, 파일 해시 계산 등은 검증된 오픈소스 활용 우선
- 문서화 규칙:
  - 마크다운 파일은 `.temp` 폴더에 생성
  - 핵심만 간결하게 작성 (필수 내용만 포함)
  - 이모지 사용 금지

