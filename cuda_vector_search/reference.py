"""Plain PyTorch reference paths (no custom kernels)."""

from __future__ import annotations

import torch


def torch_topk(queries: torch.Tensor, corpus: torch.Tensor, k: int):
    """What most people write: one matmul for the full score matrix, then topk.

    On the GPU with fp16 inputs this runs cuBLAS and stores fp16 scores.
    """
    s = queries @ corpus.T
    v, i = torch.topk(s, k, dim=1)
    return v.float(), i


def exact_topk_fp32(queries: torch.Tensor, corpus: torch.Tensor, k: int, chunk: int = 1 << 18):
    """Ground truth: fp32 math on the stored vectors, chunked to bound memory."""
    q = queries.float()
    best_v = best_i = None
    for start in range(0, corpus.shape[0], chunk):
        s = q @ corpus[start : start + chunk].float().T
        kk = min(k, s.shape[1])
        v, i = torch.topk(s, kk, dim=1)
        i = i + start
        if best_v is None:
            best_v, best_i = v, i
        else:
            v = torch.cat([best_v, v], dim=1)
            i = torch.cat([best_i, i], dim=1)
            best_v, sel = torch.topk(v, min(k, v.shape[1]), dim=1)
            best_i = torch.gather(i, 1, sel)
    return best_v, best_i
