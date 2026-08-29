#!/usr/bin/env python3
"""Is the *baseline* engine bitwise-reproducible when batch shape changes?

check_lossless.py compares base-vs-dflash generations. If that comparison
diverges, there are two possible causes:

  (a) the speculative verifier is committing tokens the target would not have
      chosen -- a real correctness bug, which would invalidate the speedups;
  (b) greedy argmax flipping at a near-tie because the forward pass reduced
      its floats in a different order.

Speculation necessarily changes batch shape (it verifies N positions per step
instead of 1), so (b) is expected. This isolates it: same server, same prompt,
same temperature 0 -- only the *batch shape* differs. Any divergence here is
proof of (b), because no drafter is involved at all.
"""
import asyncio, json, sys, aiohttp

URL = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b"
PROBE = ("Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. "
         "How many clips did Natalia sell altogether in April and May? Show your reasoning step by step.")
FILLER = [
    "Explain how a B-tree index speeds up range queries in a relational database, and when it performs worse than a hash index.",
    "Implement binary search over a rotated sorted array in Python. Explain the invariant that makes it correct.",
    "What is the sum of all integers from 1 to 200 that are divisible by 3 but not by 5? Work through it methodically.",
    "Describe the difference between optimistic and pessimistic concurrency control, with a concrete example of a workload favoring each.",
    "Write a SQL query to find the second-highest salary per department, and explain why a naive MAX() approach fails.",
    "A train travels 120 km at 60 km/h, then 180 km at 90 km/h. What is the average speed for the whole journey? Explain carefully.",
    "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals and returns the merged list sorted by start.",
]

async def gen(sess, prompt, max_tokens=400):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0}
    async with sess.post(f"{URL}/v1/chat/completions", json=body) as r:
        j = await r.json()
    m = j["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + (m.get("content") or "")

def diff_at(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i if (i < n or len(a) != len(b)) else -1

async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
        alone1 = await gen(s, PROBE)
        alone2 = await gen(s, PROBE)
        # same probe, but sharing a decode batch with 7 concurrent requests
        outs = await asyncio.gather(gen(s, PROBE), *[gen(s, f) for f in FILLER])
        batched = outs[0]

    print("=== baseline determinism, temperature=0, no drafter anywhere ===\n")
    d12 = diff_at(alone1, alone2)
    db = diff_at(alone1, batched)
    print(f"alone   vs alone   (identical batch shape) : "
          f"{'IDENTICAL' if d12 < 0 else f'DIVERGED at char {d12}'}")
    print(f"alone   vs batch-of-8 (different shape)    : "
          f"{'IDENTICAL' if db < 0 else f'DIVERGED at char {db}'}")
    if db >= 0:
        print(f"\n  alone: ...{alone1[max(0,db-50):db]}>>>{alone1[db:db+50]!r}")
        print(f"  batch: ...{batched[max(0,db-50):db]}>>>{batched[db:db+50]!r}")
        print("\nCONCLUSION: the baseline itself is not bitwise-reproducible across batch\n"
              "shapes, so base-vs-dflash text divergence is NOT evidence of a bad verifier.")
    else:
        print("\nCONCLUSION: baseline is stable across batch shapes here; a base-vs-dflash\n"
              "divergence would then warrant a closer look at the verifier.")
    json.dump({"alone1": alone1, "alone2": alone2, "batched": batched},
              open("results/determinism_probe.json", "w"), indent=2)

asyncio.run(main())
