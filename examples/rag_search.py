"""Tiny retrieval demo: embed a folder of text files, then answer questions
with the passages that match best.

    pip install sentence-transformers
    python examples/rag_search.py build examples/sample_docs --out rag_index
    python examples/rag_search.py ask rag_index "how do I keep bread fresh?" -k 3

The retrieved passages are what you would paste into an LLM prompt in a RAG
pipeline. Works on CPU too (it then uses the plain torch path).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuda_vector_search import Index  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def passages(folder: Path, max_chars: int = 300):
    """Split each .txt/.md file on blank lines and pack paragraphs up to max_chars."""
    out = []
    for f in sorted(folder.rglob("*")):
        if f.suffix.lower() not in (".txt", ".md") or not f.is_file():
            continue
        buf = ""
        for para in f.read_text(encoding="utf-8", errors="ignore").split("\n\n"):
            para = " ".join(para.split())
            if not para:
                continue
            if buf and len(buf) + len(para) > max_chars:
                out.append({"file": str(f.relative_to(folder)), "text": buf})
                buf = ""
            buf = f"{buf} {para}".strip()
        if buf:
            out.append({"file": str(f.relative_to(folder)), "text": buf})
    return out


def build(args):
    from sentence_transformers import SentenceTransformer

    docs = passages(Path(args.folder))
    if not docs:
        sys.exit(f"no .txt or .md files under {args.folder}")
    model = SentenceTransformer(args.model)
    emb = model.encode([d["text"] for d in docs], batch_size=64, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=len(docs) > 256)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Index(emb, metric="cosine").save(out / "embeddings.npy")
    (out / "passages.json").write_text(json.dumps({"model": args.model, "passages": docs}, indent=1))
    print(f"indexed {len(docs)} passages from {args.folder} -> {out} (dim {emb.shape[1]})")


def ask(args):
    from sentence_transformers import SentenceTransformer

    root = Path(args.index)
    meta = json.loads((root / "passages.json").read_text())
    index = Index.load(root / "embeddings.npy")
    model = SentenceTransformer(meta["model"])
    q = model.encode(args.question, convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(q, args.k)
    print(f"Q: {args.question}")
    for rank, (s, i) in enumerate(zip(scores[0].tolist(), ids[0].tolist()), 1):
        p = meta["passages"][i]
        print(f"\n[{rank}] {s:.3f}  {p['file']}")
        print(textwrap.indent(textwrap.fill(p["text"], 88), "    "))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="embed a folder of .txt/.md files")
    b.add_argument("folder")
    b.add_argument("--out", default="rag_index")
    b.add_argument("--model", default=DEFAULT_MODEL)
    a = sub.add_parser("ask", help="retrieve passages for a question")
    a.add_argument("index")
    a.add_argument("question", nargs="+")
    a.add_argument("-k", type=int, default=3)
    args = ap.parse_args()
    if args.cmd == "ask":
        args.question = " ".join(args.question)
    build(args) if args.cmd == "build" else ask(args)


if __name__ == "__main__":
    main()
