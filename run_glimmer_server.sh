#!/usr/bin/env bash
# Launch Meta Muse-Glimmer-30B + DFlash-15 via Ollama's bundled CUDA 13 llama-server
# inside heavy.slice (drops 1.40 GB vision mmproj, avoids 16 GiB ollama.service cap,
# uses glimmer_hal_v2.jinja for per-request Zero-Think vs Low-Think with 100% KV reuse).
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
OLLAMA_LIB="${OLLAMA_LIB:-$HOME/.local/opt/ollama/lib/ollama}"
CUDA_DIR="${OLLAMA_CUDA_DIR:-$OLLAMA_LIB/cuda_v13}"
BLOBS_DIR="${OLLAMA_BLOBS_DIR:-$HOME/.local/share/ollama/models/blobs}"

MODEL_BLOB="${HAL_GLIMMER_MODEL:-$BLOBS_DIR/sha256-71b5c9c9abbc2a9e7c13550b6f9fe1f062650117a24490b15a310cbd7de03f8b}"
DRAFT_BLOB="${HAL_GLIMMER_DRAFT:-$BLOBS_DIR/sha256-27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc}"
TEMPLATE_FILE="${HAL_GLIMMER_TEMPLATE:-$ROOT/templates/glimmer_hal_v2.jinja}"
PORT="${HAL_LLAMA_PORT:-8080}"
DRAFT_N_MAX="${HAL_DRAFT_N_MAX:-15}"
REASONING_BUDGET="${HAL_REASONING_BUDGET:-64}"
SLICE="${HAL_SYSTEMD_SLICE-heavy.slice}"

if [ ! -x "${OLLAMA_LIB}/llama-server" ]; then
  echo "Error: llama-server not found at ${OLLAMA_LIB}/llama-server (set OLLAMA_LIB)." >&2
  exit 1
fi
if [ ! -f "${MODEL_BLOB}" ]; then
  echo "Error: Muse-Glimmer model blob not found at ${MODEL_BLOB} (pull muse-glimmer:30b-q4_K_M-dflash or set HAL_GLIMMER_MODEL)." >&2
  exit 1
fi
if [ ! -f "${DRAFT_BLOB}" ]; then
  echo "Error: DFlash drafter blob not found at ${DRAFT_BLOB} (set HAL_GLIMMER_DRAFT)." >&2
  exit 1
fi
if [ ! -f "${TEMPLATE_FILE}" ]; then
  echo "Error: chat template not found at ${TEMPLATE_FILE} (set HAL_GLIMMER_TEMPLATE)." >&2
  exit 1
fi

export CUDA_CACHE_MAXSIZE=4294967296
export LD_LIBRARY_PATH="${OLLAMA_LIB}:${CUDA_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SERVER_ARGS=(
  --host 127.0.0.1 --port "${PORT}"
  --no-webui --offline
  --alias muse-glimmer-30b-dflash
  -m "${MODEL_BLOB}"
  --spec-type draft-dflash
  --spec-draft-model "${DRAFT_BLOB}"
  --spec-draft-ngl 99
  --spec-draft-n-max "${DRAFT_N_MAX}"
  --spec-draft-p-min 0
  -ngl 999 --device CUDA0
  -c 4096 --parallel 1
  -fa on -ctk f16 -ctv f16 -b 512 -ub 512
  --load-mode none --jinja
  --chat-template-file "${TEMPLATE_FILE}"
  --reasoning-format deepseek
  --reasoning-budget "${REASONING_BUDGET}"
  --chat-template-kwargs '{"reasoning_strength":"low"}'
)

if [ -n "${SLICE}" ] && command -v systemd-run >/dev/null 2>&1 && systemctl --user status "${SLICE}" >/dev/null 2>&1; then
  exec systemd-run --user --slice="${SLICE}" --scope \
    --working-directory="${CUDA_DIR}" \
    "${OLLAMA_LIB}/llama-server" "${SERVER_ARGS[@]}" "$@"
fi

cd "${CUDA_DIR}"
exec "${OLLAMA_LIB}/llama-server" "${SERVER_ARGS[@]}" "$@"
