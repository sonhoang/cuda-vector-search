"""Tuning sweep for the fused kernel: queries per block (bq) and corpus chunks.

    python bench/sweep.py --out results/sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_gpu import make_corpus, make_queries  # noqa: E402
from cuda_vector_search import _ext  # noqa: E402
from timing import time_call  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--out", default="results/sweep.json")
    args = ap.parse_args()
    mod = _ext.load()
    x = make_corpus(args.n, args.d, 0)
    qall = make_queries(256, args.d, 1)
    rows = []
    for b in (1, 32, 256):
        q = qall[:b].contiguous()
        for k in (10, 100, 256):
            kpl = 1 if k <= 32 else 2 if k <= 64 else 4 if k <= 128 else 8
            _, _, plan = mod.fused_topk(q, x, k)
            auto_bq, auto_chunks = plan.tolist()
            for bq in (16, 32, 64):
                if bq * kpl > 256 or (bq > 16 and b < bq // 2):
                    continue
                for f in (0.25, 0.5, 1, 2, 4):
                    chunks = max(1, int(auto_chunks * f))
                    try:
                        ms, _ = time_call(lambda: mod.fused_topk(q, x, k, bq, chunks), torch.cuda.synchronize,
                                          target_s=0.15)
                    except RuntimeError as e:
                        print("fail", b, k, bq, chunks, e)
                        continue
                    r = {"n": args.n, "d": args.d, "batch": b, "k": k, "bq": bq, "chunks": chunks,
                         "auto": [auto_bq, auto_chunks], "ms": round(ms, 4)}
                    print(json.dumps(r), flush=True)
                    rows.append(r)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    best = {}
    for r in rows:
        key = (r["batch"], r["k"])
        if key not in best or r["ms"] < best[key]["ms"]:
            best[key] = r
    for (b, k), r in sorted(best.items()):
        auto = [x for x in rows if x["batch"] == b and x["k"] == k and x["bq"] == r["auto"][0]
                and x["chunks"] == r["auto"][1]]
        print(f"b={b:4d} k={k:4d} best bq={r['bq']} chunks={r['chunks']} {r['ms']:.3f} ms;"
              f" auto {r['auto']} {auto[0]['ms'] if auto else 'n/a'} ms")


if __name__ == "__main__":
    main()
