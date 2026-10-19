"""Builds the CUDA extension on first use (JIT, cached by torch)."""

from __future__ import annotations

import os
import threading
from pathlib import Path

_lock = threading.Lock()
_mod = None

SOURCE = Path(__file__).with_name("csrc") / "kernels.cu"


def available() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 7


def load(verbose: bool = False):
    """Compile (once) and return the extension module."""
    global _mod
    if _mod is not None:
        return _mod
    with _lock:
        if _mod is not None:
            return _mod
        import torch
        from torch.utils.cpp_extension import load as _load

        if not available():
            raise RuntimeError("cuda-vector-search kernels need a CUDA GPU with compute capability 7.0+")
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        _mod = _load(
            name="cuda_vector_search_ext",
            sources=[str(SOURCE)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=verbose,
        )
        return _mod
