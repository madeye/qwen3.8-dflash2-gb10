#!/bin/bash
# Poll until the vLLM server answers /health, or its log shows a fatal error.
LOG="$1"; PORT="${2:-8000}"; DEADLINE=$(( $(date +%s) + ${3:-1800} ))
while true; do
  if curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then echo "SERVER_READY"; exit 0; fi
  if grep -qE "Traceback|ERROR|Error:|CUDA out of memory|raise |Killed|Aborted" "$LOG" 2>/dev/null; then
    echo "SERVER_ERROR"; grep -nE "Traceback|Error|error" "$LOG" | tail -20; exit 1
  fi
  if ! pgrep -f "vllm serve" >/dev/null 2>&1; then echo "SERVER_DIED"; tail -30 "$LOG"; exit 1; fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo "SERVER_TIMEOUT"; tail -20 "$LOG"; exit 1; fi
  sleep 5
done
