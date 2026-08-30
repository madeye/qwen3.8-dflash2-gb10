# Qwen3.8-27B + DFlash2 speculative decoding on vLLM (DGX Spark / GB10)

Benchmark harness comparing autoregressive decoding against the DFlash2
block-diffusion drafter on one NVIDIA GB10 (Grace-Blackwell, 121 GB unified
memory, aarch64). Qwen3.8's built-in MTP head is supported as an optional third
configuration (`mtp`) but is off by default.

**[Read the write-up →](https://madeye.github.io/qwen3.8-dflash2-gb10/)** — the
results, the roofline argument behind them, and the losslessness caveat, on one
page. Source for that page lives in [`docs/`](docs/).

## Stack

| Component | Version / source |
| --- | --- |
| vLLM | `0.28.0` (PyPI `manylinux_2_28_aarch64` wheel) |
| PyTorch | `2.13.0+cu130` |
| Target model | `unsloth/Qwen3.8-27B-NVFP4` (~22 GB, NVFP4 weights, FP8 `lm_head`) |
| Draft model | `z-lab/Qwen3.8-27B-DFlash2` (3.85 GB, bf16) |
| GPU | NVIDIA GB10, compute capability 12.1 (`sm_121`) |

### Notes on this hardware

* vLLM's bundled cubins target `sm_120`; CUDA guarantees binary compatibility
  across minor revisions within a major architecture, so they run on `sm_121`.
* Memory is *unified* — `--gpu-memory-utilization` is a fraction of the whole
  121 GB the CPU also lives in, so it is set conservatively (0.60) to leave the
  desktop session headroom.
* NVFP4 was required: bf16 weights (56 GB) are not an option on this box.
* **`MAX_JOBS` must be capped** (`serve.sh` sets 4). FlashInfer JIT-builds its
  NVFP4 CUTLASS GEMM (`fp4_gemm_cutlass_sm120`) during the first `profile_run`.
  Unbounded, ninja fans out to `nproc` (20) jobs, each nvcc `cicc` peaks near
  7 GB RSS, and on unified memory that races the 22 GB of resident weights --
  the OOM killer takes ninja and the engine dies with a bare
  `CalledProcessError ... returned non-zero exit status 137`. The traceback
  points at `determine_available_memory`, which reads like a GPU-memory problem
  but is not: it is host RAM, during kernel compilation.

### Why DFlash2 works against a quantized target here

The DFlash2 model card and the `dfischermittwald/Qwen3.8-27B-NVFP4-DFlash2`
repo both warn that DFlash2 rejects a quantized target LM head:

```
ValueError: DFlash2 requires an unquantized target LM head for candidate TopK
```

That guard lived in the **unmerged** PR (vllm-project/vllm#52816). In the
released 0.28.0 it is gone — `LogitsProcessor._apply_head` dispatches through
`lm_head.quant_method.apply(...)`, which handles a quantized head fine. So the
mainstream `unsloth` NVFP4 checkpoint is usable and the specially-requantized
fork is not needed. If a future version reinstates the guard, that repo is a
drop-in replacement for `TARGET` in `serve.sh`.

## Layout

| File | Purpose |
| --- | --- |
| `pyproject.toml` / `uv.lock` | Dependency set, pinned and locked (uv) |
| `serve.sh` | Launch vLLM in `base` / `mtp` / `dflash` mode |
| `serve_public.sh` | Serve on a public address behind the auth gateway |
| `gateway.py` | Auth proxy + dashboard for API URL and API keys |
| `wait_ready.sh` | Poll `/health`, fail fast on a crashed server |
| `bench.py` | Streaming client: TTFT, TPOT, throughput, acceptance length |
| `compare.py` | Side-by-side table from the JSON reports |
| `run_all.sh` | Runs all three modes end-to-end and prints the comparison |
| `check_lossless.py` | Diffs baseline vs speculative generations |
| `determinism_test.py` | Shows the baseline alone is batch-shape sensitive |
| `restart-llama-server.sh` | Restores the pre-existing llama.cpp server verbatim |
| `results/` | Per-mode JSON, generated text, and `summary.txt` |

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/); `pyproject.toml`
declares them and `uv.lock` pins the full 203-package closure (torch 2.13.0+cu130
and flashinfer come in transitively through vLLM).

```bash
uv sync              # create/refresh .venv from the lock
./fetch_weights.sh   # ~22 GB of NVFP4 weights
```

`uv run <script>` syncs first, so `uv run bench.py …` always runs against the
locked environment. The scripts call `.venv/bin/python` directly, which is the
same interpreter.

One cosmetic quirk: every `uv sync` re-links `nvidia-cusparselt-cu13`. NVIDIA
ships that wheel with a filename tag of `manylinux2014_aarch64` but an internal
`WHEEL` tag of `manylinux2014_sbsa`, so uv's installed-state check never
matches. It is the wheel the lock names, and the re-link takes ~2 ms.

## Running

```bash
./run_all.sh                    # base vs dflash, concurrency 1/4/8
MODES="base mtp dflash" ./run_all.sh   # add Qwen3.8's built-in MTP head
```

Defaults: greedy (`temperature=0`), `max_tokens=1024`, `max-model-len=32768`.

Greedy matters for more than reproducibility: DFlash2 is **lossless**, so its
output should match the baseline. `results/*_text.json` holds the generations
and `check_lossless.py` diffs them.

That diff is expected to show a *small* divergence, and it does -- see
"Losslessness" below. Read it before treating a mismatch as a bug.

## Serving it publicly

```bash
./serve_public.sh            # dflash + gateway on 0.0.0.0:8080
./serve_public.sh base       # baseline instead
GW_PORT=9000 ./serve_public.sh
```

`serve_public.sh` serves a 262144-token window (`MAXLEN`), not `serve.sh`'s
benchmark default of 32768: agent clients need real context — Hermes refuses
any model under 64K outright, and a coding agent spends 32K on file reads
alone. 262144 is also the ceiling: it is the checkpoint's
`max_position_embeddings`, and the rope config is plain mrope with no YaRN
section, so vLLM refuses a larger `max_model_len` outright.

This one does cost something. `serve.sh` caps the KV pool at 24 GiB
(`KV_BYTES`), which measures 544,617 KV tokens — 2.08× concurrency at full
length, down from the 6.33× the uncapped pool gave at 128K. Budget ~46 KiB per
token (47,317 B measured): the 16 full-attention layers are 32 KiB/token at
fp8 and the hybrid allocator's page alignment accounts for the rest. Raise
`KV_BYTES` if you want more concurrency, but deliberately: setting it makes
vLLM skip memory profiling altogether and ignore `gpu-memory-utilization` for
KV, so the only ceiling is real free memory (111.83 GiB at startup here) — the
`UTIL` guard that keeps the desktop session alive does not apply to it. The
exact pool is printed on startup (`GPU KV cache size: ... Maximum
concurrency`).

vLLM is pinned to `127.0.0.1`; only `gateway.py` faces the network. It proxies
`/v1/*` and `/metrics`, requires `Authorization: Bearer <key>` on every one of
them, and 404s everything else — vLLM's other routes (`/tokenize`, `/sleep`,
the shutdown endpoints) never reach the public interface. Streaming passes
through chunk-by-chunk, so SSE latency is unaffected.

Keys live in the gateway rather than in vLLM's argv, which is the point:
rotating one takes effect on the next request instead of reloading 22 GB of
weights. Startup prints the dashboard URL with its admin token:

```
gateway    0.0.0.0:8080  ->  http://127.0.0.1:8000
api base   http://192.168.0.4:8080/v1
dashboard  http://192.168.0.4:8080/?token=<admin token>
```

The dashboard shows upstream health and the served model, lets you edit the
advertised API base URL (override it if you front the gateway with a tunnel or
domain), and manages keys — create, label, reveal, copy, rotate, revoke, with
per-key request counts and last-used times. It also renders a ready-to-paste
curl and OpenAI-SDK snippet. State lives in `gateway.json` (mode 0600,
gitignored); the admin token is generated on first run and persists there.

If vLLM itself is started with `--api-key`, pass the same value as
`--upstream-key` — the gateway terminates the client's credential at the
boundary and presents that one upstream instead of forwarding it.

#### Getting and setting keys

Three equivalent routes, in rough order of convenience:

```bash
# the dashboard: show / copy / rotate / revoke, per key
python3 -c "import json;d=json.load(open('gateway.json'));\
print(f\"http://127.0.0.1:8080/?token={d['admin_token']}\")"

# the admin API
curl -s -H "X-Admin-Token: $TOK" localhost:8080/admin/state          # list
curl -s -H "X-Admin-Token: $TOK" -d '{"label":"laptop"}' \
     -H 'Content-Type: application/json' localhost:8080/admin/keys   # create
curl -s -H "X-Admin-Token: $TOK" -d '{}' \
     localhost:8080/admin/keys/<id>/rotate                           # rotate

# or just edit gateway.json -- the only way to set a *chosen* value,
# since the dashboard and API only generate random ones
```

`gateway.json` is re-read within a second of changing, so a hand-edited key,
`public_url`, or `admin_token` takes effect on the next request with no restart.
Per-key request counters are preserved across a reload, an unparseable file is
ignored (and warned about once) rather than crashing the gateway, and emptying
the `keys` list regenerates one instead of locking everyone out.

Rotating still means updating whichever clients hold the old key —
`~/.pi/agent/models.json` below keeps its own copy.

**This is bearer auth over plain HTTP.** It is enough for a trusted LAN. Before
exposing it further, put TLS in front of it — a Cloudflare tunnel, Tailscale, or
a reverse proxy — and set the dashboard's API base URL to that public address.

### Driving it with the pi coding agent

[pi](https://github.com/earendil-works/pi) (`@earendil-works/pi-coding-agent`)
talks to the gateway as an ordinary OpenAI-compatible provider. It is a Node CLI,
so it is outside uv's scope; `~/.pi/agent/models.json` holds the wiring:

```json
{ "providers": { "local-vllm": {
  "baseUrl": "http://127.0.0.1:8080/v1",
  "api": "openai-completions",
  "apiKey": "<a key from the dashboard>",
  "compat": { "supportsDeveloperRole": false, "supportsReasoningEffort": false },
  "models": [ { "id": "qwen3.8-27b", "reasoning": true,
                "contextWindow": 262144, "maxTokens": 8192 } ] } } }
```

```bash
pi --provider local-vllm --model qwen3.8-27b
```

The `compat` pair matters: vLLM's OpenAI layer rejects the `developer` role and
`reasoning_effort` that pi sends to reasoning-capable models by default. Rotating
the key in the dashboard means updating `apiKey` here to match.

## Metrics

* **TTFT** — time to first token (prefill; speculation should not help here).
* **TPOT** — mean inter-token latency; the number speculative decoding targets.
* **decode tok/s per request** — `1 / TPOT`, the single-stream speed a user feels.
* **acceptance length** — mean tokens committed per verification step
  (`1 + accepted / drafts`), scraped from vLLM's Prometheus `/metrics`. With 7
  draft tokens the ceiling is 8.0; the z-lab card reports 4.4–5.5 on an H200.

## Results (2026-08-29, GB10, greedy, max_tokens=1024)

| conc | config | decode tok/s per req | sys tok/s | TTFT s | TPOT ms | accept len | speedup |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | base | 11.55 | 11.50 | 0.131 | 86.61 | – | 1.00× |
| 1 | **dflash** | **54.26** | 49.14 | 0.275 | 18.43 | **6.38** | **4.70×** |
| 4 | base | 10.95 | 29.09 | 0.358 | 91.34 | – | 1.00× |
| 4 | **dflash** | **32.20** | 54.13 | 0.492 | 31.06 | 3.79 | **2.94×** |
| 8 | base | 10.48 | 68.11 | 0.404 | 95.40 | – | 1.00× |
| 8 | **dflash** | **27.34** | 108.32 | 0.656 | 36.57 | 3.86 | **2.61×** |

### Why the single-stream gain is so large here

GB10 has roughly 273 GB/s of memory bandwidth. Streaming 22 GB of NVFP4 weights
once per token puts a hard ceiling near 12 tok/s, and the baseline measures
11.55 — about 94% of roofline. Decoding is almost purely bandwidth-bound, and
the GPU's math units sit idle. That is the ideal case for speculation: DFlash2
verifies ~6.4 tokens per weight-streaming pass, converting idle FLOPs into
tokens. The 4.70× at concurrency 1 exceeds the 3.43× the z-lab card reports on
an H200 precisely *because* this box is more bandwidth-starved than an H200,
so there is more headroom to reclaim.

Acceptance length also beats the card's 4.4–5.5: those numbers are at
temperature 1.0, this run is greedy, which is easier to draft for.

The gain narrows as concurrency rises (4.70× → 2.94× → 2.61×) for the usual
reason — batching already amortizes weight streaming across requests, so the
spare bandwidth speculation was exploiting is progressively consumed. Note the
system-wide throughput still climbs (49 → 108 tok/s), and TTFT degrades
slightly under speculation (0.131 → 0.275 s), as expected: drafting cannot help
prefill and adds a little per-step overhead.

### Losslessness

`check_lossless.py results/base_text.json results/dflash_text.json` reports a
divergence at char 110 of 389 — inside the model's thinking block ("Need
concise but show reasoning." vs "Provide reasoning."). Everything after
`</think>`, the entire user-visible answer, is character-for-character
identical.

This is **not** a verifier bug. `determinism_test.py` demonstrates the cause
with no drafter involved anywhere: against the plain baseline server, the same
prompt at temperature 0 run alone twice is identical, but run inside a batch of
8 it diverges — **at the same char 110**. The engine is not bitwise-reproducible
across batch shapes, because a different batch shape selects different GEMM
tilings and reduction orders, and NVFP4's coarse quantization makes near-ties
common. Speculative decoding necessarily changes batch shape (it verifies N
positions per forward pass instead of 1), so it inherits that same sensitivity.

DFlash2's losslessness guarantee is distributional and assumes exact
arithmetic; it does not promise bitwise-identical output under a quantized
target with shape-dependent kernels. The correct check is that the final answer
is unchanged, which it is.

## License

MIT — see [LICENSE](LICENSE). The benchmark harness is original work; the
models it exercises carry their own upstream licenses.
