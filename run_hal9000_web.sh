#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$ROOT"

# Auto-detect local direct CUDA 13 llama-server (e.g. ./run_glimmer_server.sh) on :8080 if running
if [ -z "${HAL_LLAMA_SERVER_URL:-}" ]; then
  if curl -sf --max-time 1 "http://127.0.0.1:8080/v1/models" >/dev/null 2>&1; then
    export HAL_LLAMA_SERVER_URL="http://127.0.0.1:8080"
  fi
fi

if [ -x "$ROOT/.venv/bin/hal9000-web" ]; then
  exec "$ROOT/.venv/bin/hal9000-web" "$@"
fi
exec uv run hal9000-web "$@"
