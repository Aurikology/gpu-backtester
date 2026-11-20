#!/usr/bin/env python3
"""Auto-tuner for the GPU sweep kernel.

A small search loop with a correctness oracle:
  1. run the CPU reference once and record its checksum
  2. for every candidate launch configuration, run the GPU kernel
  3. reject any candidate whose checksum differs from the reference (wrong is never fast)
  4. keep the fastest survivor and write it to results/tune.json

This is the shape of the agentic optimisation loops GPU library teams use: an agent (or
a script) may propose any change, and the test oracle decides whether it is allowed to
count. Extending it means adding candidates; the oracle stays the same.

usage: python3 scripts/tune.py [--gbt build/gbt] [--data bars.gbt] [--variants 16384]
"""
import argparse
import json
import os
import subprocess
import sys


def bench(gbt, data, variants, backends, block):
    cmd = [gbt, "bench", "--json", "--variants", str(variants), "--backends", backends,
           "--block", str(block)]
    if data:
        cmd += ["--data", data]
    out = subprocess.run(cmd, capture_output=True, text=True)
    rows = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
    if not rows:
        sys.exit(f"gbt bench failed:\n{out.stderr}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbt", default="build/gbt")
    ap.add_argument("--data", default="")
    ap.add_argument("--variants", type=int, default=16384)
    ap.add_argument("--out", default="results/tune.json")
    args = ap.parse_args()

    ref = bench(args.gbt, args.data, args.variants, "cpu", 256)[0]
    print(f"reference (cpu, 16 threads): {ref['seconds']:.3f}s  checksum {ref['checksum']}")

    candidates = [(kernel, block) for kernel in ("gpu-naive", "gpu")
                  for block in (32, 64, 128, 256, 512, 1024)]
    results = []
    for kernel, block in candidates:
        row = bench(args.gbt, args.data, args.variants, kernel, block)[0]
        ok = row["checksum"] == ref["checksum"]
        results.append({"kernel": kernel, "block": block, "seconds": row["seconds"],
                        "sims_per_s": row["sims_per_s"], "correct": ok})
        print(f"{kernel:10s} block {block:5d}: {row['seconds']:.4f}s "
              f"{row['sims_per_s']:.3g} sims/s  {'ok' if ok else 'REJECTED (wrong result)'}")

    valid = [r for r in results if r["correct"]]
    if not valid:
        sys.exit("no candidate matched the reference")
    best = min(valid, key=lambda r: r["seconds"])
    print(f"\nbest: {best['kernel']} block {best['block']} "
          f"({ref['seconds'] / best['seconds']:.1f}x the 16-thread CPU)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reference": ref, "candidates": results, "best": best}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
