"""Baselines that do not use this project's kernels: FAISS (GPU and CPU) and numpy.

Runs in its own process so a FAISS install can never disturb the PyTorch run.
The data has the same distribution as bench_gpu.py (random unit vectors,
rounded to fp16 so every method searches the same stored values).

    python bench/bench_baselines.py --out results/baselines.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing import recall, time_call  # noqa: E402

CHUNK = 1 << 18
N_REF = 8


def ints(s):
    return [int(float(v)) for v in s.split(",") if v]


def unit_fp16(rng, m, d):
    a = rng.standard_normal((m, d), dtype=np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a.astype(np.float16).astype(np.float32)


def merge_topk(best_v, best_i, v, i, k):
    if best_v is not None:
        v = np.concatenate([best_v, v], axis=1)
        i = np.concatenate([best_i, i], axis=1)
    sel = np.argsort(-v, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(v, sel, 1), np.take_along_axis(i, sel, 1)


def numpy_search(q, x, k):
    s = q @ x.T
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    pv = np.take_along_axis(s, part, 1)
    order = np.argsort(-pv, axis=1)
    return np.take_along_axis(pv, order, 1), np.take_along_axis(part, order, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100000,1000000,4000000")
    ap.add_argument("--dims", default="384,768")
    ap.add_argument("--batches", default="1,32,256")
    ap.add_argument("--ks", default="10,100")
    ap.add_argument("--cpu-max-n", type=int, default=1000000)
    ap.add_argument("--gpu-fp32-max-n", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/baselines.json")
    args = ap.parse_args()
    sizes, dims, batches, ks = ints(args.sizes), ints(args.dims), ints(args.batches), ints(args.ks)

    try:
        import faiss
        faiss_version = faiss.__version__
        ngpu = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    except Exception as e:  # faiss not installed
        print("faiss unavailable:", e)
        faiss, faiss_version, ngpu = None, None, 0
    env = {"faiss": faiss_version, "faiss_gpus": ngpu, "numpy": np.__version__,
           "cpu_count": os.cpu_count(), "omp_threads": faiss.omp_get_max_threads() if faiss else None}
    print(json.dumps(env), flush=True)
    rows = []
    res = faiss.StandardGpuResources() if ngpu else None

    for d in dims:
        for n in sizes:
            rng = np.random.default_rng(args.seed + 7 * d + n)
            qall = unit_fp16(np.random.default_rng(args.seed + 1), max(batches), d)
            kmax = max(ks)
            want_cpu = n <= args.cpu_max_n
            indexes = {}
            if ngpu:
                cfg = faiss.GpuIndexFlatConfig()
                cfg.device = 0
                cfg.useFloat16 = True
                indexes["faiss_gpu_fp16"] = faiss.GpuIndexFlatIP(res, d, cfg)
                if n <= args.gpu_fp32_max_n:
                    cfg32 = faiss.GpuIndexFlatConfig()
                    cfg32.device = 0
                    indexes["faiss_gpu_fp32"] = faiss.GpuIndexFlatIP(res, d, cfg32)
            if faiss is not None and want_cpu:
                indexes["faiss_cpu"] = faiss.IndexFlatIP(d)
            xs = [] if want_cpu else None
            ref_v = ref_i = None
            for start in range(0, n, CHUNK):
                blk = unit_fp16(rng, min(CHUNK, n - start), d)
                for ix in indexes.values():
                    ix.add(blk)
                if xs is not None:
                    xs.append(blk)
                s = qall[:N_REF] @ blk.T
                top = np.argsort(-s, axis=1)[:, :kmax]
                ref_v, ref_i = merge_topk(ref_v, ref_i, np.take_along_axis(s, top, 1), top + start, kmax)
            x = np.concatenate(xs) if xs is not None else None
            del xs
            print(f"== n={n} d={d} built {list(indexes)} cpu={want_cpu}", flush=True)

            searchers = {name: (lambda q, k, ix=ix: ix.search(q, k)[::-1]) for name, ix in indexes.items()}
            if x is not None:
                searchers["numpy"] = lambda q, k: numpy_search(q, x, k)[::-1]
            for b in batches:
                q = np.ascontiguousarray(qall[:b])
                for k in ks:
                    for name, fn in searchers.items():
                        row = {"n": n, "d": d, "batch": b, "k": k, "method": name}
                        try:
                            ids, _ = fn(q, k)
                            nr = min(N_REF, b)
                            row["recall_vs_fp32"] = recall(ids[:nr].tolist(), ref_i[:nr, :k].tolist())
                            cpu = name in ("numpy", "faiss_cpu")
                            ms, reps = time_call(lambda: fn(q, k), None,
                                                 target_s=1.0 if cpu else 0.3,
                                                 max_reps=20 if cpu else 100)
                            row.update(ms=round(ms, 4), qps=round(b / (ms / 1000), 1), reps=reps)
                        except Exception as e:
                            row["error"] = repr(e)[:200]
                        print(json.dumps(row), flush=True)
                        rows.append(row)
            for ix in indexes.values():
                ix.reset()
            del indexes, x, searchers

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"env": env, "rows": rows}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
