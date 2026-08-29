#!/bin/bash
# Resume-driven fetch of the NVFP4 weights through the flaky local proxy.
# curl aborts as soon as throughput collapses and immediately resumes from the
# byte it reached, so a proxy stall costs seconds instead of minutes.
set -uo pipefail
DEST=/home/mlv/workspace/qwen3.8/models/Qwen3.8-27B-NVFP4
REPO=unsloth/Qwen3.8-27B-NVFP4
FILE=model.safetensors
URL="https://huggingface.co/$REPO/resolve/main/$FILE"
OUT="$DEST/$FILE"

EXPECT=$(curl -sIL "$URL" | awk 'BEGIN{IGNORECASE=1} /^x-linked-size|^content-length/{v=$2} END{gsub(/\r/,"",v); print v}')
echo "expected size: $EXPECT bytes"
[ -z "$EXPECT" ] && { echo "could not determine size"; exit 1; }

for attempt in $(seq 1 500); do
  HAVE=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  if [ "$HAVE" -ge "$EXPECT" ]; then echo "WEIGHTS_COMPLETE $HAVE"; exit 0; fi
  PCT=$(awk -v h="$HAVE" -v e="$EXPECT" 'BEGIN{printf "%.1f", 100*h/e}')
  echo "attempt $attempt: have $((HAVE/1048576)) MiB (${PCT}%), resuming"
  curl -sL -C - -o "$OUT" \
       --speed-limit 200000 --speed-time 15 \
       --connect-timeout 20 --max-time 1800 \
       "$URL"
done
echo "WEIGHTS_GAVE_UP"; exit 1
