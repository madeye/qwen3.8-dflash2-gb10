# AGENTS.md

Benchmark harness comparing autoregressive decoding vs DFlash2 speculative
decoding (and optionally Qwen3.8's built-in MTP head) on Qwen3.8-27B-NVFP4,
served by vLLM 0.28.0 on one NVIDIA GB10 (DGX Spark, aarch64, 121 GB unified
memory).

Read `README.md` first — it documents the hardware quirks, the results, and
why expected "bugs" are not bugs. This file only adds operational notes.

## Commands

```bash
uv sync                          # sync .venv from uv.lock (python ==3.12)
./fetch_weights.sh               # resume-driven fetch of the ~22 GB NVFP4 weights
./serve.sh base|dflash|mtp       # launch vLLM (mode = arg 1)
./serve_public.sh [mode]         # vLLM on 127.0.0.1:8000 + gateway on 0.0.0.0:8080
./run_all.sh                     # base vs dflash end-to-end, conc 1/4/8
MODES="base mtp dflash" ./run_all.sh
.venv/bin/python bench.py --model qwen3.8-27b --label dflash   # against a running server
.venv/bin/python compare.py results/base.json results/dflash.json
.venv/bin/python check_lossless.py results/base_text.json results/dflash_text.json
```

`uv run <script>` also works (it syncs first). Scripts are standalone — no
package to build or install.

## Server lifecycle (do not improvise here)

- Servers are started with `setsid` and recorded via a PID file
  (`$SCRATCH/vllm_${MODE}.pgid`). `serve.sh` writes its own PID as the PGID
  *before* `exec vllm`. Stop them with `kill -TERM -- -<pgid>` (process group),
  never by pattern-matching `pkill` — the harness scripts deliberately avoid
  that so a stop function can't kill its own shell.
- `run_all.sh` and `serve_public.sh` own their servers via traps and PID files.
  If the server on `:8000` isn't yours, `serve_public.sh` reuses it and will
  not kill it on exit.
- Readiness: `wait_ready.sh <log> <port> [timeout]` polls `/health` and fails
  fast if the log shows a traceback / OOM / the process died.
- A full model load + JIT takes ~15–40 min on first start; the JIT kernel
  (`fp4_gemm_cutlass_sm120`) is built once, then cached.

## Hardware gotchas (GB10)

- **Cap `MAX_JOBS` (serve.sh sets 4).** FlashInfer JIT-compiles its NVFP4
  GEMM on first run; unbounded ninja fans out to 20 jobs × ~7 GB RSS and the
  OOM killer takes ninja on this unified-memory box. The engine then dies with
  `CalledProcessError ... exit status 137` at `determine_available_memory` —
  that reads like a GPU memory error but is host RAM during compilation. If
  you see exit 137 in a fresh log, check host RAM / MAX_JOBS before assuming a
  GPU OOM.
- `--gpu-memory-utilization 0.60` is intentional: the denominator is the whole
  121 GB that the desktop session also lives in.
- `serve.sh` passes `--kv-cache-dtype fp8_e4m3` (halves KV memory, buys room for
  the 256K public context) and caps the KV cache at 24 GiB via
  `--kv-cache-memory-bytes` (`KV_BYTES` env to override; when set, vLLM ignores
  `gpu-memory-utilization` for KV sizing). The published `results/` numbers
  predate this — rerun `run_all.sh` before comparing new benchmarks against
  them.
- **256K is the context ceiling.** `config.json` sets
  `text_config.max_position_embeddings: 262144` with plain mrope
  (`rope_type: "default"`, no YaRN), so vLLM derives 262144 as the max
  `max_model_len` and rejects anything larger. `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
  only silences the check and extrapolates RoPE past training -- not a way to
  get 512K. Budget ~52 KiB/token when sizing `KV_BYTES`: the 16 full-attention
  layers cost 32 KiB/token at fp8 (4 KV heads x 256 head_dim), and the hybrid
  allocator adds the rest as padding (block size forced to 1648 to match the
  mamba page, plus the padding-layer waste the startup log warns about).
- vLLM's cubins are `sm_120`; they run on `sm_121` by CUDA minor-revision
  compatibility. Fine, don't "fix" it.
- NVFP4 is mandatory here (bf16 weights ≈ 56 GB won't fit).
- `uv sync` re-links `nvidia-cusparselt-cu13` every time (~2 ms). It's a uv
  wheel-tag quirk (manylinux2014_aarch64 filename vs sbsa internal tag), not a
  bug.

## Expected behavior that looks like a bug

- **DFlash2 output is not bit-identical to baseline.** The engine is not
  bitwise-reproducible across batch shapes (different GEMM tilings / reduction
  orders + coarse NVFP4 quantization). Speculation changes batch shape by
  construction. `check_lossless.py` showing a divergence *inside the thinking
  block* with an identical final answer is correct behavior — verified by
  `determinism_test.py` without any drafter involved. The losslessness
  guarantee is distributional, not bitwise.
- Greedy (`temperature=0`) is the benchmark default and matters: it's what
  makes the losslessness comparison meaningful.
- TTFT is expected to be *worse* under speculation (drafting can't help
  prefill).

## Conventions

- Scripts call `.venv/bin/python` / `.venv/bin/vllm` directly. Keep it that
  way; the venv is created by `uv sync` from `uv.lock` (203-package closure,
  torch 2.13.0+cu130).
- Harness scripts (bench, compare, determinism_test, check_lossless) are
  deliberately stdlib + aiohttp only, no vLLM imports — they talk to the server
  over HTTP so they work against any OpenAI-compatible endpoint.
- `SCRATCH` holds server logs and PID files. **Note:** `serve.sh` /
  `run_all.sh` default it to a Claude session scratchpad path under
  `/tmp/claude-1000/...` that may not exist outside that session — override
  with `SCRATCH=/tmp/qwen38` (mkdir it first) or ensure the directory exists,
  or the PID-file write fails immediately.
- `results/` holds per-mode JSON reports, `*_text.json` generations, and
  `summary.txt` (regenerated by `run_all.sh`).
- `gateway.json` (API keys, admin token) is mode 0600 and gitignored —
  never commit or print its contents. `gateway.py` is the only public-facing
  process; vLLM must stay pinned to `127.0.0.1`. Bearer auth over plain HTTP
  is only acceptable on a trusted LAN (see README).
- `restart-llama-server.sh` restores the pre-benchmark llama.cpp server
  verbatim (fixed port 33415). It's a snapshot script, not a general one.
- Git: 2 commits, MIT-licensed harness. `.claude/worktrees/` holds worktrees
  (e.g. the `gh-pages-site` branch) — leave them alone.
