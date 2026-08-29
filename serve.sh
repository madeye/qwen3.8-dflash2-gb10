#!/bin/bash
# Launch vLLM for Qwen3.8-27B (NVFP4), optionally with a speculative decoder.
#   ./serve.sh base    -> autoregressive baseline
#   ./serve.sh mtp     -> Qwen3.8's built-in multi-token-prediction head
#   ./serve.sh dflash  -> DFlash2 block-diffusion draft model (z-lab)
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$(pwd)/.venv/bin:$PATH"   # vLLM shells out to ninja/gcc for JIT kernels
# FlashInfer JIT-builds its NVFP4 CUTLASS GEMM on first run. Unbounded, ninja fans out
# to nproc(20) jobs and each nvcc `cicc` peaks near 7 GB RSS -- on this unified-memory
# box that races the resident weights and the OOM killer takes ninja (exit 137).
export MAX_JOBS="${MAX_JOBS:-4}"       # honored by flashinfer/jit/cpp_ext.py:_get_num_workers
MODE="${1:-base}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-32768}"
UTIL="${UTIL:-0.60}"
NSPEC="${NSPEC:-7}"
SCRATCH=${SCRATCH:-/tmp/claude-1000/-home-mlv-workspace-qwen3-8/a90c22ea-64ba-4da0-b5e2-503d64c0f49a/scratchpad}
LOG="${LOG:-$SCRATCH/vllm_${MODE}.log}"

TARGET=./models/Qwen3.8-27B-NVFP4
ARGS=(
  "$TARGET"
  --port "$PORT"
  --served-model-name qwen3.8-27b
  --max-model-len "$MAXLEN"
  --gpu-memory-utilization "$UTIL"
)

case "$MODE" in
  base)   ;;
  mtp)    ARGS+=( --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NSPEC}}" ) ;;
  dflash) ARGS+=( --speculative-config "{\"method\":\"dflash\",\"model\":\"z-lab/Qwen3.8-27B-DFlash2\",\"num_speculative_tokens\":${NSPEC}}" ) ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

echo $$ > "$SCRATCH/vllm_${MODE}.pgid"   # own PID == PGID under setsid; survives exec
echo "launching: vllm serve ${ARGS[*]}" > "$LOG"
exec .venv/bin/vllm serve "${ARGS[@]}" >> "$LOG" 2>&1
