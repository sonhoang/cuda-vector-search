"""Merge benchmark JSON files, draw the charts and print README tables.

    python bench/report.py results/gpu.json results/baselines.json --out results/t4.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METHODS = [  # fixed order = fixed colour, marker and table column
    ("fused", "fused kernel (this repo)"),
    ("twopass", "score kernel + torch.topk"),
    ("torch", "torch.matmul + topk"),
    ("faiss_gpu_fp16", "FAISS GPU flat fp16"),
    ("faiss_gpu_fp32", "FAISS GPU flat fp32"),
    ("faiss_cpu", "FAISS CPU flat"),
    ("numpy", "numpy CPU"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
STYLES = ["-", "--", "-.", "-", "--", ":", ":"]


def fmt_n(n):
    return f"{n // 1000}k" if n < 1_000_000 else f"{n / 1e6:g}M"


def load(paths):
    env, rows, gemm = {}, [], []
    for p in paths:
        data = json.loads(Path(p).read_text())
        for k, v in data.get("env", {}).items():
            env.setdefault(k, v)
        rows += [r for r in data.get("rows", []) if "method" in r]
        gemm += data.get("gemm", [])
    return env, rows, gemm


def lookup(rows):
    return {(r["n"], r["d"], r["batch"], r["k"], r["method"]): r for r in rows}


def tables(env, rows, gemm):
    t = lookup(rows)
    present = [m for m, _ in METHODS if any(r["method"] == m for r in rows)]
    out = []
    for d in sorted({r["d"] for r in rows}):
        out.append(f"\n**dim {d}**, median latency per batch in ms (lower is better). "
                   "Bold = fastest in the row.\n")
        out.append("| corpus | k | batch | " + " | ".join(m for m in present) + " | fused vs torch |")
        out.append("|---|---|---|" + "---|" * len(present) + "---|")
        for n in sorted({r["n"] for r in rows if r["d"] == d}):
            for k in sorted({r["k"] for r in rows}):
                for b in sorted({r["batch"] for r in rows}):
                    cells = {}
                    for m in present:
                        r = t.get((n, d, b, k, m))
                        cells[m] = r.get("ms") if r else None
                        if r and "error" in r:
                            cells[m] = r["error"]
                    nums = [v for v in cells.values() if isinstance(v, (int, float))]
                    if not nums:
                        continue
                    best = min(nums)
                    line = []
                    for m in present:
                        v = cells[m]
                        if v is None:
                            line.append("")
                        elif isinstance(v, str):
                            line.append(v)
                        else:
                            s = f"{v:.3g}" if v < 100 else f"{v:.0f}"
                            line.append(f"**{s}**" if v == best else s)
                    f, tt = cells.get("fused"), cells.get("torch")
                    ratio = f"{tt / f:.2f}x" if isinstance(f, (int, float)) and isinstance(tt, (int, float)) else ""
                    out.append(f"| {fmt_n(n)} | {k} | {b} | " + " | ".join(line) + f" | {ratio} |")
    if gemm:
        out.append("\n**Score matrix only** (no top-k), fp16 output:\n")
        out.append("| corpus | dim | batch | custom WMMA ms | cuBLAS ms | custom TFLOP/s | cuBLAS TFLOP/s |")
        out.append("|---|---|---|---|---|---|---|")
        for g in gemm:
            if "error" in g:
                continue
            out.append(f"| {fmt_n(g['n'])} | {g['d']} | {g['batch']} | {g['custom_ms']} | {g['cublas_ms']} | "
                       f"{g['custom_tflops']} | {g['cublas_tflops']} |")
    rec = {}
    for r in rows:
        if "recall_vs_fp32" in r:
            rec.setdefault(r["method"], []).append(r["recall_vs_fp32"])
    out.append("\n**Recall@k vs fp32 exact search** (first 8 queries of every config; min / mean):\n")
    out.append("| method | min | mean |")
    out.append("|---|---|---|")
    for m in present:
        if m in rec:
            v = rec[m]
            out.append(f"| {m} | {min(v):.4f} | {sum(v) / len(v):.4f} |")
    return "\n".join(out)


def charts(rows, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = lookup(rows)
    ns = sorted({r["n"] for r in rows})
    ks = sorted({r["k"] for r in rows})
    files = []
    for d in sorted({r["d"] for r in rows}):
        fig, axes = plt.subplots(len(ks), len(ns), figsize=(4.2 * len(ns), 3.4 * len(ks)),
                                 squeeze=False, sharex=True)
        handles = {}
        for i, k in enumerate(ks):
            for j, n in enumerate(ns):
                ax = axes[i][j]
                for c, (m, label) in enumerate(METHODS):
                    pts = [(b, t[(n, d, b, k, m)]["qps"]) for b in sorted({r["batch"] for r in rows})
                           if (n, d, b, k, m) in t and "qps" in t[(n, d, b, k, m)]]
                    if not pts:
                        continue
                    xs, ys = zip(*pts)
                    (h,) = ax.plot(xs, ys, color=COLORS[c], marker=MARKERS[c], linestyle=STYLES[c],
                                   linewidth=2.6 if m == "fused" else 1.6, markersize=6, label=label)
                    handles[label] = h
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                ax.set_title(f"{fmt_n(n)} vectors, k={k}", fontsize=10, color="#0b0b0b")
                ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
                ax.grid(False, which="minor")
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                for s in ("left", "bottom"):
                    ax.spines[s].set_color("#9a9994")
                ax.tick_params(colors="#52514e", labelsize=8)
                if i == len(ks) - 1:
                    ax.set_xlabel("query batch size", color="#52514e", fontsize=9)
                if j == 0:
                    ax.set_ylabel("queries / second", color="#52514e", fontsize=9)
                bs = sorted({r["batch"] for r in rows})
                ax.set_xticks(bs)
                ax.set_xticklabels([str(b) for b in bs])
        fig.suptitle(f"Exact top-k search throughput, dim {d} (higher is better)", fontsize=12, color="#0b0b0b")
        fig.legend(list(handles.values()), list(handles.keys()), loc="lower center",
                   ncol=min(4, len(handles)), frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.08 if len(handles) > 4 else 0.05, 1, 0.96))
        p = Path(outdir) / f"t4_qps_d{d}.png"
        fig.savefig(p, dpi=130, facecolor="#fcfcfb")
        plt.close(fig)
        files.append(str(p))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="results/t4.json")
    args = ap.parse_args()
    env, rows, gemm = load(args.inputs)
    Path(args.out).write_text(json.dumps({"env": env, "rows": rows, "gemm": gemm}, indent=1))
    md = tables(env, rows, gemm)
    Path(args.out).with_suffix(".md").write_text(md + "\n")
    print(md)
    try:
        print("charts:", charts(rows, Path(args.out).parent))
    except ImportError as e:
        print("matplotlib missing, no charts:", e)


if __name__ == "__main__":
    main()
