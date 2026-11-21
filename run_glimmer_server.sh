#!/usr/bin/env bash
# Launch Meta Muse-Glimmer-30B + DFlash-15 via native Blackwell sm_121a llama-server
# (falling back to Ollama's bundled CUDA 13 binary) inside heavy.slice: drops 1.40 GB
# vision mmproj, avoids 16 GiB ollama.service cap, and uses glimmer_hal_v2.jinja for
# per-request Zero-Think vs Low-Think with 100% KV reuse.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
NATIVE_BIN="${HAL_LLAMA_NATIVE_BIN:-$HOME/.local/opt/llama-cpp-blackwell/build/bin/llama-server}"
SYSTEM_CUDA_LIB="${HAL_CUDA_LIB:-/opt/cuda/lib64}"
OLLAMA_LIB="${OLLAMA_LIB:-$HOME/.local/opt/ollama/lib/ollama}"
CUDA_DIR="${OLLAMA_CUDA_DIR:-$OLLAMA_LIB/cuda_v13}"
BLOBS_DIR="${OLLAMA_BLOBS_DIR:-$HOME/.local/share/ollama/models/blobs}"

MODEL_BLOB="${HAL_GLIMMER_MODEL:-$BLOBS_DIR/sha256-71b5c9c9abbc2a9e7c13550b6f9fe1f062650117a24490b15a310cbd7de03f8b}"
DRAFT_BLOB="${HAL_GLIMMER_DRAFT:-$BLOBS_DIR/sha256-27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc}"
TEMPLATE_FILE="${HAL_GLIMMER_TEMPLATE:-$ROOT/templates/glimmer_hal_v2.jinja}"
PORT="${HAL_LLAMA_PORT:-8080}"
DRAFT_N_MAX="${HAL_DRAFT_N_MAX:-15}"
DRAFT_N_MIN="${HAL_DRAFT_N_MIN:-}"
DRAFT_P_MIN="${HAL_DRAFT_P_MIN:-0}"
CACHE_TYPE_K="${HAL_CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${HAL_CACHE_TYPE_V:-f16}"
DRAFT_CACHE_TYPE_K="${HAL_DRAFT_CACHE_TYPE_K:-}"
DRAFT_CACHE_TYPE_V="${HAL_DRAFT_CACHE_TYPE_V:-}"
BATCH_SIZE="${HAL_BATCH_SIZE:-512}"
UBATCH_SIZE="${HAL_UBATCH_SIZE:-512}"
REASONING_BUDGET="${HAL_REASONING_BUDGET:-48}"
REASONING_BUDGET_MSG="${HAL_REASONING_BUDGET_MESSAGE:-}"
POLL_LEVEL="${HAL_POLL:-}"
VERBOSITY="${HAL_VERBOSITY:-}"
CACHE_REUSE="${HAL_CACHE_REUSE:-0}"
BACKEND_SAMPLING="${HAL_BACKEND_SAMPLING:-0}"
NO_REASONING_PRESERVE="${HAL_NO_REASONING_PRESERVE:-1}"
SLICE="${HAL_SYSTEMD_SLICE-heavy.slice}"
# Require >= 28 GiB cgroup headroom (model + draft + KV + unified page-cache) before entering slice
MIN_SLICE_HEADROOM_BYTES="${HAL_MIN_SLICE_HEADROOM_BYTES:-30064771072}"

if [ -n "${HAL_LLAMA_SERVER:-}" ]; then
  LLAMA_BIN="${HAL_LLAMA_SERVER}"
  WORK_DIR="$(dirname "${LLAMA_BIN}")"
  export LD_LIBRARY_PATH="${WORK_DIR}:${SYSTEM_CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  ENGINE_LABEL="custom (${LLAMA_BIN})"
elif [ "${HAL_USE_BUNDLED_LLAMA:-0}" != "1" ] && [ -x "${NATIVE_BIN}" ]; then
  LLAMA_BIN="${NATIVE_BIN}"
  WORK_DIR="$(dirname "${LLAMA_BIN}")"
  export LD_LIBRARY_PATH="${WORK_DIR}:${SYSTEM_CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  ENGINE_LABEL="native Blackwell sm_121a (${LLAMA_BIN})"
else
  LLAMA_BIN="${OLLAMA_LIB}/llama-server"
  WORK_DIR="${CUDA_DIR}"
  export LD_LIBRARY_PATH="${OLLAMA_LIB}:${CUDA_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  ENGINE_LABEL="bundled Ollama CUDA 13 (${LLAMA_BIN})"
fi

if [ ! -x "${LLAMA_BIN}" ]; then
  echo "Error: llama-server not found at ${LLAMA_BIN} (build ${NATIVE_BIN} or set OLLAMA_LIB / HAL_LLAMA_SERVER)." >&2
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
export GGML_CUDA_GRAPH_CHUNK_SIZE="${GGML_CUDA_GRAPH_CHUNK_SIZE:-0}"

SERVER_ARGS=(
  --host 127.0.0.1 --port "${PORT}"
  --no-webui --offline
  --alias muse-glimmer-30b-dflash
  -m "${MODEL_BLOB}"
  --spec-type draft-dflash
  --spec-draft-model "${DRAFT_BLOB}"
  --spec-draft-ngl 99
  --spec-draft-n-max "${DRAFT_N_MAX}"
  --spec-draft-p-min "${DRAFT_P_MIN}"
  -ngl 999 --device CUDA0
  -c 4096 --parallel 1
  -fa on -ctk "${CACHE_TYPE_K}" -ctv "${CACHE_TYPE_V}" -b "${BATCH_SIZE}" -ub "${UBATCH_SIZE}"
  --load-mode none --jinja
  --chat-template-file "${TEMPLATE_FILE}"
  --reasoning-format deepseek
  --reasoning-budget "${REASONING_BUDGET}"
  --chat-template-kwargs '{"reasoning_strength":"low"}'
)

if [ -n "${DRAFT_N_MIN}" ]; then
  SERVER_ARGS+=(--spec-draft-n-min "${DRAFT_N_MIN}")
fi
if [ -n "${DRAFT_CACHE_TYPE_K}" ]; then
  SERVER_ARGS+=(-ctkd "${DRAFT_CACHE_TYPE_K}")
fi
if [ -n "${DRAFT_CACHE_TYPE_V}" ]; then
  SERVER_ARGS+=(-ctvd "${DRAFT_CACHE_TYPE_V}")
fi
if [ -n "${REASONING_BUDGET_MSG}" ]; then
  SERVER_ARGS+=(--reasoning-budget-message "${REASONING_BUDGET_MSG}")
fi
if [ -n "${POLL_LEVEL}" ]; then
  SERVER_ARGS+=(--poll "${POLL_LEVEL}")
fi
if [ -n "${VERBOSITY}" ]; then
  SERVER_ARGS+=(-lv "${VERBOSITY}")
fi
if [ "${CACHE_REUSE}" != "0" ]; then
  SERVER_ARGS+=(--cache-reuse "${CACHE_REUSE}")
fi
if [ "${BACKEND_SAMPLING}" = "1" ]; then
  SERVER_ARGS+=(-bs)
fi
if [ "${NO_REASONING_PRESERVE}" = "1" ]; then
  SERVER_ARGS+=(--no-reasoning-preserve)
fi

echo "Launching Muse-Glimmer-30B + DFlash-${DRAFT_N_MAX} via ${ENGINE_LABEL}" >&2

if [ -n "${SLICE}" ] && command -v systemd-run >/dev/null 2>&1 && systemctl --user status "${SLICE}" >/dev/null 2>&1; then
  USE_SLICE=1
  if [[ "${MIN_SLICE_HEADROOM_BYTES}" =~ ^[0-9]+$ ]] && [ "${MIN_SLICE_HEADROOM_BYTES}" -gt 0 ]; then
    SLICE_MEM_CUR="$(systemctl --user show "${SLICE}" -p MemoryCurrent --value 2>/dev/null || true)"
    SLICE_MEM_HIGH="$(systemctl --user show "${SLICE}" -p MemoryHigh --value 2>/dev/null || true)"
    SLICE_MEM_MAX="$(systemctl --user show "${SLICE}" -p MemoryMax --value 2>/dev/null || true)"
    SLICE_LIMIT=""
    for cand in "${SLICE_MEM_HIGH}" "${SLICE_MEM_MAX}"; do
      if [[ "${cand}" =~ ^[0-9]+$ ]]; then
        if [ -z "${SLICE_LIMIT}" ] || [ "${cand}" -lt "${SLICE_LIMIT}" ]; then
          SLICE_LIMIT="${cand}"
        fi
      fi
    done
    if [[ "${SLICE_MEM_CUR}" =~ ^[0-9]+$ ]] && [ -n "${SLICE_LIMIT}" ]; then
      SLICE_HEADROOM=$(( SLICE_LIMIT - SLICE_MEM_CUR ))
      if [ "${SLICE_HEADROOM}" -lt "${MIN_SLICE_HEADROOM_BYTES}" ]; then
        echo "Warning: ${SLICE} headroom ($(( SLICE_HEADROOM / 1073741824 )) GiB; cur=$(( SLICE_MEM_CUR / 1073741824 )) GiB, limit=$(( SLICE_LIMIT / 1073741824 )) GiB) < $(( MIN_SLICE_HEADROOM_BYTES / 1073741824 )) GiB — bypassing ${SLICE} to avoid unified-memory CUDA OOM." >&2
        USE_SLICE=0
      fi
    fi
  fi
  if [ "${USE_SLICE}" = "1" ]; then
    exec systemd-run --user --slice="${SLICE}" --scope \
      --working-directory="${WORK_DIR}" \
      "${LLAMA_BIN}" "${SERVER_ARGS[@]}" "$@"
  fi
fi

cd "${WORK_DIR}"
exec "${LLAMA_BIN}" "${SERVER_ARGS[@]}" "$@"
