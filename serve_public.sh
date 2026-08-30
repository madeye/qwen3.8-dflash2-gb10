#!/bin/bash
# Serve the model on a public address, behind an authenticating gateway.
#
#   ./serve_public.sh            # dflash (fast) + gateway on 0.0.0.0:8080
#   ./serve_public.sh base       # autoregressive baseline instead
#   GW_PORT=9000 ./serve_public.sh
#
# vLLM itself is pinned to loopback -- only gateway.py faces the network, so
# every request needs a bearer key from the dashboard. Ctrl-C stops both.
set -uo pipefail
cd "$(dirname "$0")"

MODE="${1:-dflash}"
PORT="${PORT:-8000}"              # vLLM, loopback only
# Agent clients need real context: Hermes refuses anything under 64K, and a
# coding agent burns 32K on file reads alone. The benchmark keeps serve.sh's
# 32768 so its published numbers stay comparable; serving is a different job.
MAXLEN="${MAXLEN:-131072}"
GW_PORT="${GW_PORT:-8080}"        # gateway, public
GW_HOST="${GW_HOST:-0.0.0.0}"
SCRATCH="${SCRATCH:-/tmp}"
LOG="$SCRATCH/vllm_${MODE}.log"
PIDF="$SCRATCH/vllm_${MODE}.pgid"

stop_server() {
  [ -f "$PIDF" ] || return 0
  local pgid; pgid=$(cat "$PIDF")
  echo "stopping vLLM (pgid $pgid)"
  kill -TERM -- "-$pgid" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$pgid" 2>/dev/null
  rm -f "$PIDF"
}
trap 'stop_server; exit 0' INT TERM EXIT

if curl -sf --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "reusing the vLLM server already on :${PORT}"
  trap - INT TERM EXIT                       # not ours to kill
else
  rm -f "$PIDF"
  echo "starting vLLM ($MODE) on 127.0.0.1:${PORT}, log $LOG"
  # TOOLS=1: serving is for real clients, which need tool calling and the <think> split.
  HOST=127.0.0.1 PORT="$PORT" SCRATCH="$SCRATCH" TOOLS="${TOOLS:-1}" MAXLEN="$MAXLEN" \
    setsid ./serve.sh "$MODE" </dev/null >/dev/null 2>&1 &
  for _ in $(seq 1 30); do [ -s "$PIDF" ] && break; sleep 1; done
  ./wait_ready.sh "$LOG" "$PORT" "${READY_TIMEOUT:-2400}" || { echo "vLLM failed to start"; exit 1; }
fi

# Not exec'd: the EXIT trap has to survive the gateway to take vLLM down with it.
uv run gateway.py --upstream "http://127.0.0.1:${PORT}" --host "$GW_HOST" --port "$GW_PORT"
