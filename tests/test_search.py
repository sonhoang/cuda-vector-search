import numpy as np
import pytest
import torch

from cuda_vector_search import Index, exact_topk_fp32
from cuda_vector_search import _ext

cuda = pytest.mark.skipif(not _ext.available(), reason="needs a CUDA GPU (sm_70+)")


def make(n, d, b, seed=0, normal=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    q = torch.randn(b, d, generator=g)
    if normal:
        x = x / x.norm(dim=1, keepdim=True)
        q = q / q.norm(dim=1, keepdim=True)
    return x.half(), q.half()


def check_exact(index, q, k, scores, ids, tol=1e-3):
    """Recall@k == 1.0 against fp32 math on the stored vectors, ties aside.

    Every returned id must score at least the true k-th best (minus a small
    tolerance for summation order), ids must be unique, and the returned
    scores must match the fp32 scores of those ids.
    """
    x = index.embeddings
    qq = index._prep_queries(q)[:, : index.dim]
    full = qq.float() @ x.float().T
    ref_v, ref_i = torch.topk(full, k, dim=1)
    kth = ref_v[:, -1:]
    got_true = torch.gather(full, 1, ids)
    scale = max(1.0, float(full.abs().max()))
    assert ids.shape == (q.shape[0], k)
    assert bool((ids >= 0).all()) and bool((ids < len(index)).all())
    assert bool((got_true >= kth - tol * scale).all()), "returned an id outside the true top-k"
    assert bool((scores - got_true).abs().max() <= tol * scale), "scores do not match their ids"
    assert bool((scores[:, :-1] >= scores[:, 1:] - 1e-6).all()), "results not sorted"
    for row in ids.cpu().numpy():
        assert len(set(row.tolist())) == k, "duplicate ids"
    # exact set recall, reported for information (ties can make it < 1)
    hit = sum(len(set(a) & set(b)) for a, b in zip(ids.cpu().tolist(), ref_i.cpu().tolist()))
    return hit / ids.numel()


@cuda
@pytest.mark.parametrize(
    "n,d,b,k",
    [
        (1000, 64, 1, 1),
        (1000, 64, 3, 10),
        (4099, 128, 17, 10),       # n not a multiple of the 128-row tile
        (20000, 384, 1, 10),
        (20000, 384, 32, 100),
        (20000, 384, 100, 33),     # k not a power of two, b not a multiple of 16
        (20000, 768, 256, 64),
        (50000, 384, 70, 128),
        (30000, 384, 40, 256),     # largest fused k
        (300, 96, 5, 300 // 2),    # width padded 96 -> 128, tiny corpus
        (100, 64, 2, 100),         # k == n
    ],
)
def test_fused_matches_exact(n, d, b, k):
    x, q = make(n, d, b, seed=n + d + b + k)
    index = Index(x.cuda(), metric="ip")
    v, i = index.search(q.cuda(), k, method="fused")
    recall = check_exact(index, q.cuda(), k, v, i)
    assert recall >= 0.999


@cuda
@pytest.mark.parametrize("n,d,b,k", [(5000, 384, 8, 10), (20000, 768, 64, 100)])
def test_twopass_and_torch_paths(n, d, b, k):
    x, q = make(n, d, b, seed=3)
    index = Index(x.cuda(), metric="ip")
    for method in ("twopass", "torch"):
        v, i = index.search(q.cuda(), k, method=method)
        # these paths keep fp16 scores, so allow fp16 rounding at the boundary
        check_exact(index, q.cuda(), k, v, i, tol=2e-3)


@cuda
def test_cosine_metric_unnormalised_input():
    x, q = make(8000, 384, 16, seed=7, normal=False)
    x = x * torch.rand(x.shape[0], 1).half() * 4 + 0  # vary the norms
    index = Index(x.cuda(), metric="cosine")
    v, i = index.search(q.cuda(), 10)
    xn = x.float() / x.float().norm(dim=1, keepdim=True)
    qn = q.float() / q.float().norm(dim=1, keepdim=True)
    ref_v, _ = torch.topk(qn @ xn.T, 10, dim=1)
    assert torch.allclose(v.cpu(), ref_v, atol=3e-3)


@cuda
def test_raw_ip_large_values():
    x, q = make(10000, 256, 9, seed=11, normal=False)  # |scores| ~ 16-60
    index = Index(x.cuda(), metric="ip")
    v, i = index.search(q.cuda(), 50, method="fused")
    check_exact(index, q.cuda(), 50, v, i)


@cuda
def test_scores_kernel_matches_fp32():
    x, q = make(3001, 192, 37, seed=5)
    index = Index(x.cuda(), metric="ip")
    s = index.scores(q.cuda())
    ref = q.float() @ x.float().T
    assert s.shape == (37, 3001)
    assert torch.allclose(s.cpu(), ref, atol=1e-3)
    s16 = index.scores(q.cuda(), fp32=False)
    assert s16.dtype == torch.float16
    assert torch.allclose(s16.float().cpu(), ref, atol=3e-3)


@cuda
def test_exact_reference_helper():
    x, q = make(70000, 128, 4, seed=9)
    v, i = exact_topk_fp32(q.cuda(), x.cuda(), 20, chunk=8192)
    ref_v, ref_i = torch.topk(q.float() @ x.float().T, 20, dim=1)
    assert torch.allclose(v.cpu(), ref_v, atol=1e-5)


@cuda
def test_one_dim_query_and_large_k_fallback():
    x, q = make(5000, 128, 1, seed=2)
    index = Index(x.cuda(), metric="ip")
    v, i = index.search(q[0].cuda(), 5)
    assert v.shape == (1, 5)
    v, i = index.search(q.cuda(), 400)  # beyond the fused limit -> torch path
    assert i.shape == (1, 400)


@cuda
def test_fused_plan_overrides():
    mod = _ext.load()
    x, q = make(40000, 384, 50, seed=4)
    index = Index(x.cuda(), metric="ip")
    qq = index._prep_queries(q.cuda())
    base_v, base_i = index.search(q.cuda(), 10, method="fused")
    for bq, chunks in [(16, 1), (32, 7), (64, 1000)]:
        v, i, plan = mod.fused_topk(qq, index._x, 10, bq, chunks)
        assert torch.allclose(v, base_v, atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    x, q = make(500, 64, 3, seed=1, normal=False)
    index = Index(x.float().numpy(), metric="cosine")
    path = tmp_path / "emb.npy"
    index.save(path)
    again = Index.load(path)
    assert again.metric == "cosine" and len(again) == 500 and again.dim == 64
    a = index.search(q.float().numpy(), 5, method="torch")
    b = again.search(q.float().numpy(), 5, method="torch")
    assert torch.equal(a[1].cpu(), b[1].cpu())
    assert np.load(path).dtype == np.float16


@cuda
@pytest.mark.parametrize("k,bq", [(100, 64), (256, 32), (200, 16), (40, 64)])
def test_fused_large_state_configs(k, bq):
    mod = _ext.load()
    x, q = make(30000, 384, 70, seed=k + bq)
    index = Index(x.cuda(), metric="ip")
    qq = index._prep_queries(q.cuda())
    for chunks in (1, 5, 0):
        v, i, plan = mod.fused_topk(qq, index._x, k, bq, chunks)
        assert plan.tolist()[0] == bq
        check_exact(index, q.cuda(), k, v, i.long())
