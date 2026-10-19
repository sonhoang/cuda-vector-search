"""Bundle the repo into a single Kaggle script kernel.

    export KAGGLE_USER=<your kaggle username>
    python kaggle/make_kernel.py --mode full     # or: smoke
    kaggle kernels push -p build/kaggle

The kernel runs on one T4 ("NvidiaTeslaT4"), is private, and needs internet
for pip (ninja, faiss) and the sentence-transformers model.
"""

import argparse
import base64
import io
import json
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["cuda_vector_search", "tests", "bench", "examples", "pyproject.toml"]


def bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name in INCLUDE:
            for p in sorted([ROOT / name] if (ROOT / name).is_file() else (ROOT / name).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    tf.add(p, arcname=str(p.relative_to(ROOT)))
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "smoke", "nofaiss"])
    ap.add_argument("--out", default=str(ROOT / "build" / "kaggle"))
    ap.add_argument("--slug", default="cuda-vector-search-bench")
    args = ap.parse_args()
    user = os.environ.get("KAGGLE_USER") or os.environ.get("KAGGLE_USERNAME")
    if not user:
        raise SystemExit("set KAGGLE_USER")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "kaggle" / "run_template.py").read_text()
    src = src.replace('"__MODE__"', repr(args.mode)).replace('"__BUNDLE__"', repr(bundle()))
    (out / "run.py").write_text(src)
    meta = {
        "id": f"{user}/{args.slug}",
        "title": args.slug,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}/run.py ({(out / 'run.py').stat().st_size // 1024} KB), mode={args.mode}")


if __name__ == "__main__":
    main()
