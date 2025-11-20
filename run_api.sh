#!/bin/bash
# Semantica Codegraph API 서버 실행 스크립트 (uv)

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"
exec uv run python -m apps.api.main
