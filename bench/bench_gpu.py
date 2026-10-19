"""Benchmark the custom kernels against torch.matmul + torch.topk on one GPU.

Writes a JSON file with one row per (n, d, batch, k, method).

    python bench/bench_gpu.py --out results/gpu.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cuda_vector_search import Index, exact_topk_fp32  # noqa: E402
from cuda_vector_search import _ext  # noqa: E402
from timing import recall, time_call  # noqa: E402

CHUNK = 1 << 18
N_REF = 8  # queries checked against the fp32 ground truth per config


def ints(s):
    return [int(float(v)) for v in s.split(",") if v]


def make_corpus(n, d, seed):
    """Random unit vectors, generated in chunks straight into an fp16 GPU buffer."""
    x = torch.empty(n, d, dtype=torch.float16, device="cuda")
    for c, start in enumerate(range(0, n, CHUNK)):
        g = torch.Generator(device="cuda").manual_seed(seed * 100003 + c)
        blk = torch.randn(min(CHUNK, n - start), d, device="cuda", generator=g)
        x[start : start + blk.shape[0]] = (blk / blk.norm(dim=1, keepdim=True)).half()
    return x


def make_queries(b, d, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, d, device="cuda", generator=g)
    return (q / q.norm(dim=1, keepdim=True)).half()


def env_info():
    def cmd(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # pragma: no cover
            return f"n/a ({e})"

    p = torch.cuda.get_device_properties(0)
    return {
        "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "gpu": p.name,
        "gpu_mem_gb": round(p.total_memory / 2**30, 1),
        "sm_count": p.multi_processor_count,
        "compute_capability": f"{p.major}.{p.minor}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvcc": cmd("nvcc --version | tail -n 2 | head -n 1"),
        "driver": cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1"),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100000,1000000,4000000")
    ap.add_argument("--dims", default="384,768")
    ap.add_argument("--batches", default="1,32,256")
    ap.add_argument("--ks", default="10,100")
    ap.add_argument("--methods", default="fused,twopass,torch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-s", type=float, default=0.3)
    ap.add_argument("--out", default="results/gpu.json")
    args = ap.parse_args()

    sizes, dims, batches, ks = ints(args.sizes), ints(args.dims), ints(args.batches), ints(args.ks)
    methods = args.methods.split(",")
    sync = torch.cuda.synchronize
    _ext.load(verbose=False)
    env = env_info()
    print(json.dumps(env, indent=1), flush=True)
    rows, gemm_rows = [], []
    total_mem = torch.cuda.get_device_properties(0).total_memory

    for d in dims:
        for n in sizes:
            need = n * d * 2 + max(batches) * n * 2 * 2 + (1 << 30)
            if need > total_mem:
                print(f"skip n={n} d={d}: needs ~{need / 2**30:.1f} GB", flush=True)
                rows.append({"n": n, "d": d, "skipped": "memory"})
                continue
            torch.cuda.empty_cache()
            x = make_corpus(n, d, args.seed + d)
            index = Index(x, metric="ip", copy=False)
            qall = make_queries(max(batches), d, args.seed + 1)
            ref_v, ref_i = exact_topk_fp32(qall[:N_REF], x, max(ks))
            ref_i = ref_i.cpu()
            print(f"== n={n} d={d} corpus {n * d * 2 / 2**30:.2f} GB", flush=True)
            for b in batches:
                q = qall[:b].contiguous()
                for k in ks:
                    for m in methods:
                        row = {"n": n, "d": d, "batch": b, "k": k, "method": m}
                        try:
                            sync()
                            torch.cuda.reset_peak_memory_stats()
                            base = torch.cuda.memory_allocated()
                            v, i = index.search(q, k, method=m)
                            sync()
                            row["extra_mem_mb"] = round((torch.cuda.max_memory_allocated() - base) / 2**20, 1)
                            nr = min(N_REF, b)
                            row["recall_vs_fp32"] = recall(i[:nr].cpu().tolist(), ref_i[:nr, :k].tolist())
                            ms, reps = time_call(lambda: index.search(q, k, method=m), sync, target_s=args.target_s)
                            row.update(ms=round(ms, 4), qps=round(b / (ms / 1000), 1), reps=reps)
                            del v, i
                        except torch.cuda.OutOfMemoryError:
                            row["error"] = "OOM"
                            torch.cuda.empty_cache()
                        print(json.dumps(row), flush=True)
                        rows.append(row)
            # raw score-matrix speed: custom WMMA kernel vs cuBLAS, largest batch
            b = max(batches)
            q = qall[:b].contiguous()
            qp = index._prep_queries(q)
            try:
                mod = _ext.load()
                ms_c, _ = time_call(lambda: mod.scores(qp, index._x, False), sync, target_s=args.target_s)
                ms_t, _ = time_call(lambda: qp @ index._x.T, sync, target_s=args.target_s)
                flops = 2.0 * b * n * d
                g = {"n": n, "d": d, "batch": b,
                     "custom_ms": round(ms_c, 3), "cublas_ms": round(ms_t, 3),
                     "custom_tflops": round(flops / ms_c / 1e9, 2), "cublas_tflops": round(flops / ms_t / 1e9, 2)}
            except torch.cuda.OutOfMemoryError:
                g = {"n": n, "d": d, "batch": b, "error": "OOM"}
                torch.cuda.empty_cache()
            print(json.dumps(g), flush=True)
            gemm_rows.append(g)
            del x, index, qall, q, qp
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"env": env, "rows": rows, "gemm": gemm_rows}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
