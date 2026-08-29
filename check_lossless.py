#!/usr/bin/env python3
"""Verify a speculative run reproduced the baseline generations exactly.

DFlash2 advertises lossless decoding: at temperature 0 the drafter only changes
*how fast* tokens are produced, never which tokens. Any divergence here means
the speculative path is not verifying correctly, and the speedup numbers next
to it describe a different computation than the baseline.
"""
import json, sys, os

def load(p):
    with open(p) as f: return json.load(f)

def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def main(base_path, spec_path):
    base, spec = load(base_path), load(spec_path)
    keys = sorted(set(base) & set(spec), key=int)
    if not keys:
        print("no overlapping prompts to compare"); return 1
    bad = 0
    for k in keys:
        b, s = base[k], spec[k]
        if b == s:
            print(f"  prompt {k}: IDENTICAL ({len(b)} chars)")
        else:
            bad += 1
            i = common_prefix(b, s)
            print(f"  prompt {k}: DIVERGED at char {i} of {len(b)}/{len(s)}")
            print(f"      base: ...{b[max(0,i-60):i]}>>>{b[i:i+60]!r}")
            print(f"      spec: ...{s[max(0,i-60):i]}>>>{s[i:i+60]!r}")
    print(f"\n{len(keys)-bad}/{len(keys)} generations identical "
          f"({os.path.basename(base_path)} vs {os.path.basename(spec_path)})")
    return 1 if bad else 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: check_lossless.py <base_text.json> <spec_text.json>"); sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
