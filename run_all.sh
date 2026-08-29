#!/bin/bash
# End-to-end benchmark: baseline vs MTP vs DFlash2 on Qwen3.8-27B NVFP4.
# Starts each server in turn, benchmarks it, shuts it down cleanly (PID-file based,
# so nothing here can pattern-match and kill its own shell).
set -uo pipefail
cd "$(dirname "$0")"
SCRATCH=${SCRATCH:-/tmp/claude-1000/-home-mlv-workspace-qwen3-8/a90c22ea-64ba-4da0-b5e2-503d64c0f49a/scratchpad}
RESULTS=./results; mkdir -p "$RESULTS"
PORT=8000
CONC="${CONC:-1,4,8}"
MAXTOK="${MAXTOK:-1024}"
TEMP="${TEMP:-0.0}"
MODES="${MODES:-base dflash}"

stop_server() {
  local pid_file="$1"
  [ -f "$pid_file" ] || return 0
  local pgid; pgid=$(cat "$pid_file")
  kill -TERM -- "-$pgid" 2>/dev/null
  for _ in $(seq 1 60); do
    kill -0 -- "-$pgid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$pgid" 2>/dev/null
  rm -f "$pid_file"
  sleep 5
}

for MODE in $MODES; do
  echo "================ MODE: $MODE ================"
  LOG="$SCRATCH/vllm_${MODE}.log"
  PIDF="$SCRATCH/vllm_${MODE}.pgid"
  rm -f "$PIDF"
  setsid ./serve.sh "$MODE" < /dev/null > /dev/null 2>&1 &
  for _ in $(seq 1 30); do [ -s "$PIDF" ] && break; sleep 1; done
  echo "server pgid $(cat "$PIDF" 2>/dev/null), log $LOG"

  if ! ./wait_ready.sh "$LOG" "$PORT" 2400; then
    echo "!!! $MODE failed to come up; skipping"
    stop_server "$PIDF"
    continue
  fi
  echo "--- $MODE ready, benchmarking ---"
  .venv/bin/python bench.py --model qwen3.8-27b --label "$MODE" \
      --concurrency "$CONC" --max-tokens "$MAXTOK" --temperature "$TEMP" \
      --out "$RESULTS/${MODE}.json" --save-text "$RESULTS/${MODE}_text.json"
  stop_server "$PIDF"
done

echo "================ COMPARISON ================"
ARGS=""
for MODE in $MODES; do [ -f "$RESULTS/${MODE}.json" ] && ARGS="$ARGS $RESULTS/${MODE}.json"; done
.venv/bin/python compare.py $ARGS | tee "$RESULTS/summary.txt"
