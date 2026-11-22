#!/usr/bin/env bash
# Launch Thinking Machines Inkling-2-Mini (Looped MoE 26.4B / 3.4B active) via native
# Blackwell sm_121a llama-server inside heavy.slice: uses f16 KV cache (required for
# Inkling), inkling_hal_v1.jinja for per-request Zero-Think vs Low-Think with static
# HAL system-prompt KV reuse, and optional DSpark speculative decoding.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
NATIVE_BIN="${HAL_LLAMA_NATIVE_BIN:-$HOME/.local/opt/llama-cpp-blackwell/build/bin/llama-server}"
SYSTEM_CUDA_LIB="${HAL_CUDA_LIB:-/opt/cuda/lib64}"

MODEL_DIR="${HAL_INKLING_DIR:-$HOME/.local/share/ioghal/models/inkling-2-mini}"
MODEL_FILE="${HAL_INKLING_MODEL:-$MODEL_DIR/Inkling-2-Mini-UD-Q4_K_XL.gguf}"
DRAFT_FILE="${HAL_INKLING_DRAFT:-$MODEL_DIR/dspark-Inkling-2-Mini-Q8_0.gguf}"
USE_DSPARK="${HAL_INKLING_USE_DSPARK:-0}"
TEMPLATE_FILE="${HAL_INKLING_TEMPLATE:-$ROOT/templates/inkling_hal_v1.jinja}"

PORT="${HAL_LLAMA_PORT:-8080}"
CTX_SIZE="${HAL_CTX_SIZE:-4096}"
CACHE_TYPE_K="${HAL_CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${HAL_CACHE_TYPE_V:-f16}"
BATCH_SIZE="${HAL_BATCH_SIZE:-512}"
UBATCH_SIZE="${HAL_UBATCH_SIZE:-512}"
REASONING_BUDGET="${HAL_REASONING_BUDGET:-48}"
REASONING_BUDGET_MSG="${HAL_REASONING_BUDGET_MESSAGE:-}"
TEMP="${HAL_TEMP:-1.0}"
TOP_P="${HAL_TOP_P:-1.0}"
TOP_K="${HAL_TOP_K:-0}"
MIN_P="${HAL_MIN_P:-0.0}"
DRAFT_N_MAX="${HAL_DRAFT_N_MAX:-3}"
DRAFT_P_MIN="${HAL_DRAFT_P_MIN:-0}"
POLL_LEVEL="${HAL_POLL:-}"
VERBOSITY="${HAL_VERBOSITY:-}"
CACHE_REUSE="${HAL_CACHE_REUSE:-0}"
BACKEND_SAMPLING="${HAL_BACKEND_SAMPLING:-0}"
NO_REASONING_PRESERVE="${HAL_NO_REASONING_PRESERVE:-1}"
SLICE="${HAL_SYSTEMD_SLICE-heavy.slice}"
# Require >= 28 GiB cgroup headroom (model + KV + unified page-cache) before entering slice
MIN_SLICE_HEADROOM_BYTES="${HAL_MIN_SLICE_HEADROOM_BYTES:-30064771072}"

if [ -n "${HAL_LLAMA_SERVER:-}" ]; then
  LLAMA_BIN="${HAL_LLAMA_SERVER}"
else
  LLAMA_BIN="${NATIVE_BIN}"
fi
WORK_DIR="$(dirname "${LLAMA_BIN}")"
export LD_LIBRARY_PATH="${WORK_DIR}:${SYSTEM_CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ENGINE_LABEL="native Blackwell sm_121a (${LLAMA_BIN})"

if [ ! -x "${LLAMA_BIN}" ]; then
  echo "Error: llama-server not found at ${LLAMA_BIN} (build ${NATIVE_BIN} or set HAL_LLAMA_SERVER)." >&2
  exit 1
fi
if [ ! -f "${MODEL_FILE}" ]; then
  echo "Error: Inkling-2-Mini GGUF not found at ${MODEL_FILE} (set HAL_INKLING_MODEL)." >&2
  exit 1
fi
if [ ! -f "${TEMPLATE_FILE}" ]; then
  echo "Error: chat template not found at ${TEMPLATE_FILE} (set HAL_INKLING_TEMPLATE)." >&2
  exit 1
fi

export CUDA_CACHE_MAXSIZE=4294967296
export GGML_CUDA_GRAPH_CHUNK_SIZE="${GGML_CUDA_GRAPH_CHUNK_SIZE:-0}"

SERVER_ARGS=(
  --host 127.0.0.1 --port "${PORT}"
  --no-webui --offline
  --alias inkling-2-mini
  -m "${MODEL_FILE}"
  -ngl 999 --device CUDA0
  -c "${CTX_SIZE}" --parallel 1
  -fa on -ctk "${CACHE_TYPE_K}" -ctv "${CACHE_TYPE_V}" -b "${BATCH_SIZE}" -ub "${UBATCH_SIZE}"
  --load-mode none --jinja
  --chat-template-file "${TEMPLATE_FILE}"
  --reasoning-budget "${REASONING_BUDGET}"
  --chat-template-kwargs '{"reasoning_effort":"low"}'
  --temp "${TEMP}" --top-p "${TOP_P}" --top-k "${TOP_K}" --min-p "${MIN_P}"
)

if [ "${USE_DSPARK}" = "1" ]; then
  if [ ! -f "${DRAFT_FILE}" ]; then
    echo "Error: DSpark drafter not found at ${DRAFT_FILE} (set HAL_INKLING_DRAFT)." >&2
    exit 1
  fi
  SERVER_ARGS+=(
    --spec-type draft-dspark
    --spec-draft-model "${DRAFT_FILE}"
    --spec-draft-ngl 99
    --spec-draft-n-max "${DRAFT_N_MAX}"
    --spec-draft-p-min "${DRAFT_P_MIN}"
  )
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

if [ "${USE_DSPARK}" = "1" ]; then
  echo "Launching Inkling-2-Mini + DSpark-${DRAFT_N_MAX} via ${ENGINE_LABEL}" >&2
else
  echo "Launching Inkling-2-Mini via ${ENGINE_LABEL}" >&2
fi

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
