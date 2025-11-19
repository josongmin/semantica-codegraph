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
src/                    # 라이브러리 코드 (코드 RAG 엔진영역)
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
- Tree-sitter (코드 파싱)
- FastAPI (HTTP API)

## 핵심 컴포넌트

### 인덱싱 파이프라인 (IndexingPipeline)
- **동기/비동기 지원**: 
  - `index_repository()`: CLI, 동기 환경용
  - `index_repository_async()`: FastAPI, async 환경용 (asyncio.run 충돌 방지)
- **순차/병렬 파싱**: 파일 수에 따라 자동 선택 (threshold: 5개)
- **DI 패턴 일관성**: IRBuilder는 메인 프로세스에서만 사용 (병렬 worker는 파싱만 수행)
- **파싱 캐시**: 순차/병렬 경로 모두 ParseCache 지원 (파일 해시 기반)
- **임베딩 최적화**: 
  - 비동기 배치 처리 (중복 제거)
  - 모델별 batch_size 최적화 (Mistral:200, OpenAI:150)
  - max_concurrent로 동시성 제어 (기본값:3)

### 검색 시스템
- **Hybrid Retriever**: Lexical + Semantic + Graph 결합
- **Ranker**: 여러 시그널을 통합하여 최종 순위 결정
- **Context Packer**: LLM에 전달할 컨텍스트 최적화

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
  - **핵심만 간결하게 작성** (필수 내용만 포함)
    - 요약, 핵심 코드 예시, 결론만
    - 상세 설명, 배경, 장황한 예시는 최소화
    - 가능한 100-200줄 이내로 제한
    - 코드 블록은 핵심 부분만 발췌
  - 이모지 사용 금지
  - 오래되거나 중복된 문서는 주기적으로 삭제

## 최근 주요 개선사항

### 1. Async 환경 지원
- FastAPI에서 `asyncio.run()` 충돌 방지
- `IndexingPipeline.index_repository_async()` 추가
- 임베딩 생성 시 `await` 사용

### 2. DI 패턴 일관성
- 병렬 파싱 경로에서 IRBuilder DI 유지
- Worker는 파싱만, IR 변환은 메인 프로세스에서 처리
- 향후 IRBuilder 설정 변경시에도 순차/병렬 경로 동작 일치

### 3. 파싱 캐시 개선
- 병렬 경로에도 ParseCache 지원
- 파일 해시 기반 캐시로 대규모 레포 반복 인덱싱 최적화
- Worker별 독립적인 ParseCache 인스턴스

### 4. 성능 튜닝 포인트
- 임베딩: `batch_size` × `max_concurrent` 조합 최적화
- 진행률 업데이트: 20개 파일마다 또는 마지막 파일
- 병렬 파싱 threshold: 5개 파일

## Cody 대비 개선 계획

### High Priority (2-3주)
1. **랭킹 시스템 강화**
   - Recency Score: Git commit 기반 신선도
   - Popularity Score: PageRank, 참조 빈도
   - Code Quality: Complexity, 문서화 정도

2. **Graph 검색 강화** (현재: 기본 BFS만 구현)
   - PageRank 알고리즘 적용 (미구현)
   - 관계 타입별 가중치 (미구현, 현재는 필터만)
   - BFS with scoring (미구현)

3. **증분 인덱싱**
   - Git diff 기반 변경 감지
   - 변경 파일만 재파싱
   - Embedding 캐시

### Medium Priority (4-6주)
4. **언어 지원 확대**: Go, Java, Rust 파서
5. **컨텍스트 최적화**: 스니펫 압축, 동적 역할 우선순위
6. **Learning to Rank**: LightGBM 기반 ML 랭킹

### 참고 문서
- `.temp/cody-comparison-rag-improvements.md`: 상세 비교 분석
- `.temp/rag-improvement-action-plan.md`: 단계별 구현 계획

