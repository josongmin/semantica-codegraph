# ADR 0001 – Codegraph 도입 배경 및 목적

## Status
Accepted

## Context

대규모 코드베이스에서 LLM이 구조적 이해 없이 파일을 검색·탐색·수정하려면 다음과 같은 문제가 발생한다:

- 에러와 hallucination 빈번
- Cursor, GPT 캔버스는 부분 업데이트가 불가능하거나 낮은 품질을 보임
- 구조적 인덱스가 없으면 LLM은 코드베이스의 topology를 추론할 수 없음

## Decision

Codegraph를 코드 이해 엔진으로 정의한다. 역할은 다음 두 가지이다:

1. 전체 코드베이스를 파싱·인덱싱
2. LLM과 Copilot이 코드를 이해하고 수정하게 하는 Knowledge Layer 제공

## Alternatives Considered

### 단순 벡터 RAG
- 구조적 관계 정보 부족
- 정확도 한계

### LSP 기반 의존
- 실시간 파싱 오버헤드
- 크로스 파일 분석 제한

### GPT large-context window 의존
- 비용 문제
- 관련 정보 필터링 불가

## Consequences

### Positive
- LLM 기반 코드 이해 정확도 향상
- 구조적 코드 수정 가능
- 에이전트 기반 개발의 핵심 엔진 역할

### Negative
- 초기 인덱싱 비용
- 코드베이스 변경 시 재인덱싱 필요

## Notes

Codegraph는 모든 에이전트 기반 개발의 핵심 엔진으로 동작하며 필수 계층이 된다.


