#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$ROOT"
if [ -x "$ROOT/.venv/bin/hal9000-web" ]; then
  exec "$ROOT/.venv/bin/hal9000-web" "$@"
fi
exec uv run hal9000-web "$@"
