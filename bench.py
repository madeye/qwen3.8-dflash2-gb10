#!/usr/bin/env python3
"""Benchmark a vLLM OpenAI-compatible server: TTFT, TPOT, throughput, spec-decode acceptance."""
import argparse, asyncio, json, time, statistics, sys, urllib.request

PROMPTS = [
    "Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Show your reasoning step by step.",
    "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals and returns the merged list sorted by start. Include docstring and a few test cases.",
    "A train travels 120 km at 60 km/h, then 180 km at 90 km/h. What is the average speed for the whole journey? Explain carefully.",
    "Explain how a B-tree index speeds up range queries in a relational database, and when it performs worse than a hash index.",
    "Implement binary search over a rotated sorted array in Python. Explain the invariant that makes it correct.",
    "What is the sum of all integers from 1 to 200 that are divisible by 3 but not by 5? Work through it methodically.",
    "Describe the difference between optimistic and pessimistic concurrency control, with a concrete example of a workload favoring each.",
    "Write a SQL query to find the second-highest salary per department, and explain why a naive MAX() approach fails.",
]

async def one_request(session_url, model, prompt, max_tokens, temperature, results, idx):
    import aiohttp
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature > 0:
        body["top_p"] = 0.95
        body["top_k"] = 20
    ttft = None
    ntok = 0
    text_parts = []
    t0 = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.post(f"{session_url}/v1/chat/completions", json=body) as r:
            if r.status != 200:
                results[idx] = {"error": f"HTTP {r.status}: {(await r.text())[:400]}"}
                return
            async for raw in r.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    ntok = chunk["usage"]["completion_tokens"]
                for ch in chunk.get("choices", []):
                    delta = ch.get("delta", {}) or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or ""
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        text_parts.append(piece)
    total = time.perf_counter() - t0
    results[idx] = {
        "ttft": ttft, "total": total, "ntok": ntok,
        "tpot": (total - ttft) / max(ntok - 1, 1) if ttft is not None and ntok > 1 else None,
        "text": "".join(text_parts),
    }

async def run_level(url, model, concurrency, max_tokens, temperature, repeats):
    tasks_input = [(PROMPTS[i % len(PROMPTS)], i) for i in range(concurrency * repeats)]
    results = {}
    t0 = time.perf_counter()
    sem = asyncio.Semaphore(concurrency)
    async def guarded(p, i):
        async with sem:
            await one_request(url, model, p, max_tokens, temperature, results, i)
    await asyncio.gather(*[guarded(p, i) for p, i in tasks_input])
    wall = time.perf_counter() - t0
    return results, wall

def scrape_metrics(url):
    try:
        raw = urllib.request.urlopen(f"{url}/metrics", timeout=15).read().decode()
    except Exception as e:
        return {}
    out = {}
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for key in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
                    "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_emitted_tokens_total"):
            if line.startswith(key):
                try:
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        if line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
            try:
                out.setdefault("per_pos", []).append(float(line.rsplit(" ", 1)[1]))
            except ValueError:
                pass
    return out

def summarize(tag, results, wall, concurrency):
    ok = [r for r in results.values() if "error" not in r and r.get("ttft") is not None]
    errs = [r for r in results.values() if "error" in r]
    if not ok:
        print(f"  [{tag}] ALL FAILED: {errs[:1]}")
        return None
    ttfts = sorted(r["ttft"] for r in ok)
    tpots = [r["tpot"] for r in ok if r["tpot"]]
    toks = sum(r["ntok"] for r in ok)
    per_req_tps = [r["ntok"] / r["total"] for r in ok if r["total"] > 0]
    s = {
        "concurrency": concurrency, "requests": len(ok), "errors": len(errs),
        "output_tokens": toks,
        "wall_s": round(wall, 2),
        "system_output_tps": round(toks / wall, 2),
        "per_request_output_tps_mean": round(statistics.mean(per_req_tps), 2),
        "ttft_mean_s": round(statistics.mean(ttfts), 3),
        "ttft_p50_s": round(statistics.median(ttfts), 3),
        "tpot_mean_ms": round(statistics.mean(tpots) * 1000, 2) if tpots else None,
        "decode_tps_per_request": round(1 / statistics.mean(tpots), 2) if tpots else None,
    }
    print(f"  [{tag}] c={concurrency:<3} reqs={s['requests']:<3} out_tok={toks:<6} "
          f"sys={s['system_output_tps']:>8.2f} tok/s  per-req decode={s['decode_tps_per_request']} tok/s  "
          f"TTFT={s['ttft_mean_s']}s  TPOT={s['tpot_mean_ms']}ms")
    return s

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-text", default=None)
    a = ap.parse_args()

    print(f"=== {a.label} (max_tokens={a.max_tokens}, temp={a.temperature}) ===")
    report = {"label": a.label, "model": a.model, "max_tokens": a.max_tokens,
              "temperature": a.temperature, "levels": []}
    texts = {}
    # warmup
    warm = {}
    await one_request(a.url, a.model, "Say hello.", 16, 0.0, warm, 0)
    for c in [int(x) for x in a.concurrency.split(",")]:
        before = scrape_metrics(a.url)
        results, wall = await run_level(a.url, a.model, c, a.max_tokens, a.temperature, a.repeats)
        after = scrape_metrics(a.url)
        s = summarize(a.label, results, wall, c)
        if s:
            d_drafts = after.get("vllm:spec_decode_num_drafts_total", 0) - before.get("vllm:spec_decode_num_drafts_total", 0)
            d_acc = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
            d_draft_tok = after.get("vllm:spec_decode_num_draft_tokens_total", 0) - before.get("vllm:spec_decode_num_draft_tokens_total", 0)
            if d_drafts > 0:
                s["spec_drafts"] = d_drafts
                s["spec_accepted_tokens"] = d_acc
                s["spec_draft_tokens"] = d_draft_tok
                s["spec_acceptance_rate"] = round(d_acc / d_draft_tok, 4) if d_draft_tok else None
                s["acceptance_length"] = round(1 + d_acc / d_drafts, 3)
                print(f"        spec: acceptance_length={s['acceptance_length']} "
                      f"per-token accept rate={s['spec_acceptance_rate']}")
            report["levels"].append(s)
        if c == 1:
            texts = {str(k): v.get("text", "") for k, v in results.items() if "error" not in v}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
    if a.save_text:
        with open(a.save_text, "w") as f:
            json.dump(texts, f, indent=2)
    print()

if __name__ == "__main__":
    asyncio.run(main())
