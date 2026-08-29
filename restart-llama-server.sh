#!/bin/bash
# Restores the llama.cpp server exactly as it was running before the vLLM benchmark (2026-08-29).
exec \
  '/home/mlv/.unsloth/llama.cpp/llama-server' \
  '-m' \
  '/home/mlv/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf' \
  '--port' \
  '33415' \
  '--parallel' \
  '4' \
  '--flash-attn' \
  'on' \
  '--no-context-shift' \
  '-c' \
  '187136' \
  '--alias' \
  'unsloth/Qwen3.8-27B-GGUF' \
  '-ngl' \
  '-1' \
  '--fit' \
  'off' \
  '--metrics' \
  '--slot-save-path' \
  '/home/mlv/.unsloth/studio/cache/llama-slots' \
  '--kv-unified' \
  '--jinja' \
  '--spec-type' \
  'ngram-mod' \
  '--spec-ngram-mod-n-match' \
  '24' \
  '--spec-ngram-mod-n-min' \
  '48' \
  '--spec-ngram-mod-n-max' \
  '64' \
  '--chat-template-kwargs' \
  '{"enable_thinking": true, "preserve_thinking": true}' \
  '--mmproj' \
  '/home/mlv/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/mmproj-F16.gguf' \
  '--load-mode' \
  'none'
