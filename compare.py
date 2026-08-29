#!/usr/bin/env python3
"""Render a side-by-side comparison of baseline vs speculative benchmark reports."""
import json, sys

def load(p):
    with open(p) as f: return json.load(f)

def main(paths):
    reports = [load(p) for p in paths]
    base = reports[0]
    by_c = {}
    for r in reports:
        for lv in r["levels"]:
            by_c.setdefault(lv["concurrency"], {})[r["label"]] = lv
    labels = [r["label"] for r in reports]

    print(f"\nModel: {base['model']}   max_tokens={base['max_tokens']}  temperature={base['temperature']}\n")
    hdr = f"{'conc':>5} {'config':<24} {'out tok':>8} {'sys tok/s':>10} {'decode tok/s':>13} {'TTFT s':>8} {'TPOT ms':>9} {'accept len':>11} {'speedup':>8}"
    print(hdr); print("-" * len(hdr))
    for c in sorted(by_c):
        ref = by_c[c].get(labels[0], {}).get("decode_tps_per_request")
        for lab in labels:
            lv = by_c[c].get(lab)
            if not lv: continue
            sp = (lv["decode_tps_per_request"] / ref) if ref and lv.get("decode_tps_per_request") else None
            print(f"{c:>5} {lab:<24} {lv['output_tokens']:>8} {lv['system_output_tps']:>10.2f} "
                  f"{lv['decode_tps_per_request'] or 0:>13.2f} {lv['ttft_mean_s']:>8.3f} "
                  f"{lv['tpot_mean_ms'] or 0:>9.2f} {str(lv.get('acceptance_length','-')):>11} "
                  f"{(f'{sp:.2f}x' if sp else '-'):>8}")
        print()

if __name__ == "__main__":
    main(sys.argv[1:])
