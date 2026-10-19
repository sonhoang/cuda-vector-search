"""Exact top-k search over an in-GPU-memory fp16 embedding matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import _ext
from .reference import torch_topk

PAD = 64          # kernel depth step; the vector width is zero-padded to a multiple of this
MAX_FUSED_K = 256
_NORM_CHUNK = 1 << 18

METHODS = ("auto", "fused", "twopass", "torch")


def _to_tensor(a) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a
    return torch.from_numpy(np.ascontiguousarray(a))


def _pad_width(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[1]
    padded = -(-d // PAD) * PAD
    if padded == d:
        return x
    return torch.nn.functional.pad(x, (0, padded - d))


def _normalize_(x: torch.Tensor) -> torch.Tensor:
    """L2-normalise rows in place, in fp32 chunks so a big fp16 matrix fits."""
    for start in range(0, x.shape[0], _NORM_CHUNK):
        blk = x[start : start + _NORM_CHUNK].float()
        blk = blk / blk.norm(dim=1, keepdim=True).clamp_min(1e-12)
        x[start : start + _NORM_CHUNK] = blk.to(x.dtype)
    return x


class Index:
    """Brute-force (exact) top-k index.

    Args:
        embeddings: [n, d] array or tensor. Stored as fp16 on the GPU
            (fp32 on the CPU fallback).
        metric: "cosine" (rows and queries are L2-normalised) or "ip" (raw inner product).
        device: "cuda", "cuda:1", "cpu"...; defaults to CUDA when available.
        copy: if False and `embeddings` is already a contiguous fp16 CUDA tensor with a
            width that is a multiple of 64, it is used as-is (and normalised in place for cosine).
    """

    def __init__(self, embeddings, metric: str = "cosine", device=None, copy: bool = True):
        if metric not in ("cosine", "ip"):
            raise ValueError("metric must be 'cosine' or 'ip'")
        x = _to_tensor(embeddings)
        if x.dim() != 2:
            raise ValueError("embeddings must be 2-D [n, d]")
        if x.shape[0] == 0:
            raise ValueError("embeddings is empty")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.metric = metric
        self.dim = int(x.shape[1])
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        x = x.to(self.device, dtype)  # no-op when it already matches
        if copy and x.data_ptr() == _to_tensor(embeddings).data_ptr():
            x = x.clone()
        x = x.contiguous()
        if metric == "cosine":
            _normalize_(x)
        self._x = _pad_width(x) if self.device.type == "cuda" else x
        self._use_kernels = self.device.type == "cuda" and _ext.available()

    # -- basics -------------------------------------------------------------
    def __len__(self) -> int:
        return int(self._x.shape[0])

    @property
    def embeddings(self) -> torch.Tensor:
        """The stored matrix (normalised for cosine), without padding."""
        return self._x[:, : self.dim]

    def _prep_queries(self, queries) -> torch.Tensor:
        q = _to_tensor(queries)
        if q.dim() == 1:
            q = q[None]
        if q.dim() != 2 or q.shape[1] != self.dim:
            raise ValueError(f"queries must be [b, {self.dim}]")
        q = q.to(self.device, self._x.dtype).contiguous()
        if self.metric == "cosine":
            q = _normalize_(q.clone())
        return _pad_width(q) if self.device.type == "cuda" else q

    # -- search -------------------------------------------------------------
    def search(self, queries, k: int = 10, method: str = "auto"):
        """Return (scores [b, k] float32, ids [b, k] int64), best first.

        method:
            "fused"   - custom kernel: tensor-core scores + top-k in one pass (k <= 256)
            "twopass" - custom score kernel writes the full matrix, then torch.topk
            "torch"   - torch.matmul (cuBLAS) + torch.topk
            "auto"    - fused when possible, else torch
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        k = min(k, len(self))
        q = self._prep_queries(queries)
        if method == "auto":
            method = "fused" if (self._use_kernels and k <= MAX_FUSED_K) else "torch"
        if method in ("fused", "twopass") and not self._use_kernels:
            raise RuntimeError(f"method={method!r} needs a CUDA GPU with compute capability 7.0+")

        if method == "fused":
            if k > MAX_FUSED_K:
                raise ValueError(f"fused search supports k <= {MAX_FUSED_K}")
            v, i, _ = _ext.load().fused_topk(q, self._x, k)
            return v, i.long()
        if method == "twopass":
            s = _ext.load().scores(q, self._x, False)
            v, i = torch.topk(s, k, dim=1)
            return v.float(), i
        return torch_topk(q, self._x, k)

    def scores(self, queries, fp32: bool = True) -> torch.Tensor:
        """Full [b, n] score matrix from the custom tensor-core kernel."""
        q = self._prep_queries(queries)
        if not self._use_kernels:
            return (q @ self._x.T).float()
        return _ext.load().scores(q, self._x, fp32)

    # -- persistence --------------------------------------------------------
    def save(self, path) -> None:
        """Write the stored vectors to `path` (.npy, fp16) plus a small `<path>.json` sidecar."""
        path = Path(path)
        np.save(path, self.embeddings.to(torch.float16).cpu().numpy())
        meta = {"metric": self.metric, "dim": self.dim, "n": len(self)}
        Path(str(path) + ".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path, metric: str | None = None, device=None) -> "Index":
        path = Path(path)
        if metric is None:
            side = Path(str(path) + ".json")
            metric = json.loads(side.read_text())["metric"] if side.exists() else "cosine"
        x = torch.from_numpy(np.load(path))
        return cls(x, metric=metric, device=device, copy=False)
