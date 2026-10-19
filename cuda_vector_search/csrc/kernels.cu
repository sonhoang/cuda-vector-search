// Exact top-k inner-product search kernels.
//
// Two entry points:
//   scores(q, x)          -> full score matrix q @ x^T (fp16 in, fp32 accumulate)
//   fused_topk(q, x, k)   -> top-k (scores, ids) per query; the score matrix
//                            never leaves shared memory
//
// Both use the same tile routine: a block of 256 threads (8 warps) computes a
// BQ x 128 tile of scores with WMMA tensor-core instructions (16x16x16 fp16
// fragments, fp32 accumulators). Each warp owns one 16-column slab of the tile.
//
// The fused kernel splits the corpus into chunks (grid.y). Each block walks
// its chunk tile by tile and keeps a running top-k per query in registers,
// spread across the 32 lanes of a warp. New scores are filtered against the
// current k-th best with one ballot, so after the first few tiles almost every
// candidate is rejected in a couple of instructions. A small merge kernel then
// reduces the per-chunk lists to the final sorted top-k.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <vector>

namespace wmma = nvcuda::wmma;

namespace cvs {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int BN = 128;        // corpus rows per tile (one 16-wide slab per warp)
constexpr int DK = 64;         // depth step loaded into shared memory
constexpr int LDS = DK + 8;    // padded smem row (halves) to spread banks
constexpr int LDC = BN + 4;    // padded smem row (floats) for the score tile
constexpr unsigned kFull = 0xffffffffu;
constexpr int kMergeWarps = 4;
static_assert(BN == 16 * kWarps, "each warp owns one 16-column slab");

template <int BQ>
struct TileCfg {
  static constexpr int FR = BQ / 16;           // accumulator fragments per warp
  static constexpr int VPR = DK / 8;           // uint4 vectors per smem row
  static constexpr int QV = BQ * VPR;
  static constexpr int XV = BN * VPR;
  static constexpr int QPT = (QV + kThreads - 1) / kThreads;
  static constexpr int XPT = XV / kThreads;
  static constexpr int in_bytes = (BQ + BN) * LDS * 2;
  static constexpr int out_bytes = BQ * LDC * 4;
  static constexpr int smem_bytes = in_bytes > out_bytes ? in_bytes : out_bytes;
  static_assert(XV % kThreads == 0, "X tile must split evenly over threads");
};

// Load one DK-wide slice of the Q and X tiles from global memory into registers.
template <int BQ>
__device__ __forceinline__ void tile_gload(const half* __restrict__ Q, const half* __restrict__ X,
                                           int nq, long long row_end, int D, int q0, long long n0,
                                           int kk, uint4 (&qreg)[TileCfg<BQ>::QPT],
                                           uint4 (&xreg)[TileCfg<BQ>::XPT]) {
  using C = TileCfg<BQ>;
  const int tid = threadIdx.x;
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
  for (int i = 0; i < C::QPT; ++i) {
    const int v = tid + i * kThreads;
    const int r = v / C::VPR;
    const int c = (v % C::VPR) * 8;
    const int q = q0 + r;
    qreg[i] = (v < C::QV && q < nq)
                  ? __ldg(reinterpret_cast<const uint4*>(Q + (size_t)q * D + kk + c))
                  : zero;
  }
#pragma unroll
  for (int i = 0; i < C::XPT; ++i) {
    const int v = tid + i * kThreads;
    const int r = v / C::VPR;
    const int c = (v % C::VPR) * 8;
    const long long n = n0 + r;
    xreg[i] = (n < row_end)
                  ? __ldg(reinterpret_cast<const uint4*>(X + (size_t)n * D + kk + c))
                  : zero;
  }
}

template <int BQ>
__device__ __forceinline__ void tile_sstore(half* Qs, half* Xs,
                                            const uint4 (&qreg)[TileCfg<BQ>::QPT],
                                            const uint4 (&xreg)[TileCfg<BQ>::XPT]) {
  using C = TileCfg<BQ>;
  const int tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < C::QPT; ++i) {
    const int v = tid + i * kThreads;
    if (v < C::QV) {
      const int r = v / C::VPR;
      const int c = (v % C::VPR) * 8;
      *reinterpret_cast<uint4*>(Qs + r * LDS + c) = qreg[i];
    }
  }
#pragma unroll
  for (int i = 0; i < C::XPT; ++i) {
    const int v = tid + i * kThreads;
    const int r = v / C::VPR;
    const int c = (v % C::VPR) * 8;
    *reinterpret_cast<uint4*>(Xs + r * LDS + c) = xreg[i];
  }
}

// Computes S = Q[q0:q0+BQ] . X[n0:n0+BN]^T into shared memory as floats
// (row stride LDC). Rows past nq / row_end read as zero. Ends with a barrier;
// the caller must barrier again before calling this a second time.
template <int BQ>
__device__ __forceinline__ void score_tile(const half* __restrict__ Q, const half* __restrict__ X,
                                           int nq, long long row_end, int D, int q0, long long n0,
                                           unsigned char* smem) {
  using C = TileCfg<BQ>;
  half* Qs = reinterpret_cast<half*>(smem);
  half* Xs = Qs + BQ * LDS;
  float* Cs = reinterpret_cast<float*>(smem);
  const int warp = threadIdx.x >> 5;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[C::FR];
#pragma unroll
  for (int r = 0; r < C::FR; ++r) wmma::fill_fragment(acc[r], 0.0f);

  uint4 qreg[C::QPT];
  uint4 xreg[C::XPT];
  tile_gload<BQ>(Q, X, nq, row_end, D, q0, n0, 0, qreg, xreg);

  for (int kk = 0; kk < D; kk += DK) {
    tile_sstore<BQ>(Qs, Xs, qreg, xreg);
    __syncthreads();
    // Prefetch the next slice into registers while the tensor cores work.
    if (kk + DK < D) tile_gload<BQ>(Q, X, nq, row_end, D, q0, n0, kk + DK, qreg, xreg);
#pragma unroll
    for (int k16 = 0; k16 < DK; k16 += 16) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
      wmma::load_matrix_sync(b, Xs + warp * 16 * LDS + k16, LDS);
#pragma unroll
      for (int r = 0; r < C::FR; ++r) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::load_matrix_sync(a, Qs + r * 16 * LDS + k16, LDS);
        wmma::mma_sync(acc[r], a, b, acc[r]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < C::FR; ++r)
    wmma::store_matrix_sync(Cs + r * 16 * LDC + warp * 16, acc[r], LDC, wmma::mem_row_major);
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Warp-level top-k state. A warp holds 32 * KPL slots, slot s of lane l is
// position p = s * 32 + l. Positions p < k are real slots (start at -inf);
// the rest are parked at +inf so they never become the minimum. `thr` is the
// smallest real value (the current k-th best) and `mpos` its position; both
// are identical on every lane.
// ---------------------------------------------------------------------------

template <int KPL>
__device__ __forceinline__ void warp_min(const float (&tv)[KPL], float& thr, int& mpos, int lane) {
  float mv = tv[0];
  int mp = lane;
#pragma unroll
  for (int s = 1; s < KPL; ++s) {
    if (tv[s] < mv) {
      mv = tv[s];
      mp = s * 32 + lane;
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_xor_sync(kFull, mv, off);
    const int op = __shfl_xor_sync(kFull, mp, off);
    if (ov < mv || (ov == mv && op < mp)) {
      mv = ov;
      mp = op;
    }
  }
  thr = mv;
  mpos = mp;
}

template <int KPL>
__device__ __forceinline__ void warp_topk_init(float (&tv)[KPL], int (&ti)[KPL], float& thr,
                                               int& mpos, int k, int lane) {
#pragma unroll
  for (int s = 0; s < KPL; ++s) {
    tv[s] = (s * 32 + lane < k) ? -INFINITY : INFINITY;
    ti[s] = -1;
  }
  warp_min(tv, thr, mpos, lane);
}

// Offer one candidate per lane. Must be called by all 32 lanes together.
template <int KPL>
__device__ __forceinline__ void warp_topk_offer(float (&tv)[KPL], int (&ti)[KPL], float& thr,
                                                int& mpos, float v, int id, int lane) {
  unsigned m = __ballot_sync(kFull, v > thr);
  while (m) {
    const int src = __ffs(m) - 1;
    m &= m - 1;
    const float cv = __shfl_sync(kFull, v, src);
    const int cid = __shfl_sync(kFull, id, src);
    if (cv > thr) {  // re-check: thr may have risen since the ballot
      if (lane == (mpos & 31)) {
        const int slot = mpos >> 5;
#pragma unroll
        for (int s = 0; s < KPL; ++s) {
          if (s == slot) {
            tv[s] = cv;
            ti[s] = cid;
          }
        }
      }
      warp_min(tv, thr, mpos, lane);
    }
  }
}

__device__ __forceinline__ bool ranks_before(float v, int id, int p, float bv, int bid, int bp) {
  return v > bv || (v == bv && (id < bid || (id == bid && p < bp)));
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

template <int BQ, bool F32>
__global__ void __launch_bounds__(kThreads)
score_kernel(const half* __restrict__ Q, const half* __restrict__ X, void* __restrict__ out,
             int nq, long long n, int D, long long ntiles) {
  __shared__ __align__(128) unsigned char smem[TileCfg<BQ>::smem_bytes];
  const float* Cs = reinterpret_cast<const float*>(smem);
  const int q0 = blockIdx.x * BQ;
  for (long long t = blockIdx.y; t < ntiles; t += gridDim.y) {
    const long long n0 = t * BN;
    score_tile<BQ>(Q, X, nq, n, D, q0, n0, smem);
    for (int i = threadIdx.x; i < BQ * BN; i += kThreads) {
      const int r = i / BN;
      const int c = i % BN;
      const int q = q0 + r;
      const long long col = n0 + c;
      if (q < nq && col < n) {
        const size_t o = (size_t)q * n + col;
        if constexpr (F32) {
          reinterpret_cast<float*>(out)[o] = Cs[r * LDC + c];
        } else {
          reinterpret_cast<half*>(out)[o] = __float2half(Cs[r * LDC + c]);
        }
      }
    }
    __syncthreads();
  }
}

// grid = (query blocks, corpus chunks). Writes k unsorted candidates per
// (query, chunk) to part_v / part_i.
template <int BQ, int KPL>
__global__ void __launch_bounds__(kThreads)
fused_topk_kernel(const half* __restrict__ Q, const half* __restrict__ X, int nq, long long n,
                  int D, int k, long long cols_per_chunk, int nchunks,
                  float* __restrict__ part_v, int* __restrict__ part_i) {
  constexpr int QPW = BQ / kWarps;  // queries owned by each warp in the select phase
  __shared__ __align__(128) unsigned char smem[TileCfg<BQ>::smem_bytes];
  const float* Cs = reinterpret_cast<const float*>(smem);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int q0 = blockIdx.x * BQ;
  const int chunk = blockIdx.y;
  const long long c_begin = (long long)chunk * cols_per_chunk;
  const long long c_end = min(n, c_begin + cols_per_chunk);

  float tv[QPW][KPL];
  int ti[QPW][KPL];
  float thr[QPW];
  int mpos[QPW];
#pragma unroll
  for (int j = 0; j < QPW; ++j) warp_topk_init(tv[j], ti[j], thr[j], mpos[j], k, lane);

  for (long long n0 = c_begin; n0 < c_end; n0 += BN) {
    score_tile<BQ>(Q, X, nq, c_end, D, q0, n0, smem);
#pragma unroll
    for (int j = 0; j < QPW; ++j) {
      const int r = warp * QPW + j;
      if (q0 + r < nq) {  // warp-uniform
#pragma unroll
        for (int t = 0; t < BN / 32; ++t) {
          const int c = t * 32 + lane;
          const long long col = n0 + c;
          const float v = (col < c_end) ? Cs[r * LDC + c] : -INFINITY;
          warp_topk_offer(tv[j], ti[j], thr[j], mpos[j], v, (int)col, lane);
        }
      }
    }
    __syncthreads();  // score tile is overwritten by the next tile's inputs
  }

#pragma unroll
  for (int j = 0; j < QPW; ++j) {
    const int q = q0 + warp * QPW + j;
    if (q < nq) {
      const size_t base = ((size_t)q * nchunks + chunk) * k;
#pragma unroll
      for (int s = 0; s < KPL; ++s) {
        const int p = s * 32 + lane;
        if (p < k) {
          part_v[base + p] = tv[j][s];
          part_i[base + p] = ti[j][s];
        }
      }
    }
  }
}

// One warp per query: select the top-k of the per-chunk candidates, then emit
// them sorted by descending score (ties: smaller id first).
template <int KPL>
__global__ void __launch_bounds__(kMergeWarps * 32)
merge_topk_kernel(const float* __restrict__ part_v, const int* __restrict__ part_i, int nq,
                  long long per_query, int k, float* __restrict__ out_v, int* __restrict__ out_i) {
  const int lane = threadIdx.x & 31;
  const int q = blockIdx.x * kMergeWarps + (threadIdx.x >> 5);
  if (q >= nq) return;  // whole warp exits together

  float tv[KPL];
  int ti[KPL];
  float thr;
  int mpos;
  warp_topk_init(tv, ti, thr, mpos, k, lane);

  const float* pv = part_v + (size_t)q * per_query;
  const int* pi = part_i + (size_t)q * per_query;
  for (long long b = 0; b < per_query; b += 32) {
    const long long i = b + lane;
    const float v = (i < per_query) ? pv[i] : -INFINITY;
    const int id = (i < per_query) ? pi[i] : -1;
    warp_topk_offer(tv, ti, thr, mpos, v, id, lane);
  }

  // Parked slots become empty entries; taken entries get id INT_MAX so an
  // untaken slot always wins a tie at -inf.
#pragma unroll
  for (int s = 0; s < KPL; ++s) {
    if (s * 32 + lane >= k) {
      tv[s] = -INFINITY;
      ti[s] = -1;
    }
  }
  for (int r = 0; r < k; ++r) {
    float bv = tv[0];
    int bid = ti[0];
    int bp = lane;
#pragma unroll
    for (int s = 1; s < KPL; ++s) {
      if (ranks_before(tv[s], ti[s], s * 32 + lane, bv, bid, bp)) {
        bv = tv[s];
        bid = ti[s];
        bp = s * 32 + lane;
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_xor_sync(kFull, bv, off);
      const int oid = __shfl_xor_sync(kFull, bid, off);
      const int op = __shfl_xor_sync(kFull, bp, off);
      if (ranks_before(ov, oid, op, bv, bid, bp)) {
        bv = ov;
        bid = oid;
        bp = op;
      }
    }
    if (lane == 0) {
      out_v[(size_t)q * k + r] = bv;
      out_i[(size_t)q * k + r] = bid;
    }
    if (lane == (bp & 31)) {
      const int slot = bp >> 5;
#pragma unroll
      for (int s = 0; s < KPL; ++s) {
        if (s == slot) {
          tv[s] = -INFINITY;
          ti[s] = INT_MAX;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

static void check_matrix(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == torch::kHalf, name, " must be float16");
  TORCH_CHECK(t.dim() == 2, name, " must be 2-D");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.size(1) % DK == 0, name, " width must be a multiple of 64 (zero-pad it)");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16 == 0, name,
              " must be 16-byte aligned");
}

static void check_pair(const torch::Tensor& q, const torch::Tensor& x) {
  check_matrix(q, "queries");
  check_matrix(x, "corpus");
  TORCH_CHECK(q.device() == x.device(), "queries and corpus must be on the same device");
  TORCH_CHECK(q.size(1) == x.size(1), "queries and corpus must have the same width");
  TORCH_CHECK(x.size(0) < INT_MAX, "corpus must have fewer than 2^31 rows");
  TORCH_CHECK(q.size(0) < INT_MAX, "too many queries");
  const auto* prop = at::cuda::getDeviceProperties(q.device().index());
  TORCH_CHECK(prop->major >= 7, "tensor-core kernels need compute capability 7.0+");
}

torch::Tensor scores(torch::Tensor q, torch::Tensor x, bool out_fp32) {
  check_pair(q, x);
  const c10::cuda::CUDAGuard guard(q.device());
  const int nq = (int)q.size(0);
  const long long n = x.size(0);
  const int D = (int)x.size(1);
  auto out = torch::empty({nq, n}, q.options().dtype(out_fp32 ? torch::kFloat : torch::kHalf));
  if (nq == 0 || n == 0) return out;

  const long long ntiles = (n + BN - 1) / BN;
  const int bq = nq <= 16 ? 16 : 64;
  const dim3 grid((nq + bq - 1) / bq, (unsigned)std::min<long long>(ntiles, 65535));
  auto stream = at::cuda::getCurrentCUDAStream();
  const half* Qp = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
  const half* Xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  void* op = out.data_ptr();
  if (bq == 16) {
    if (out_fp32) score_kernel<16, true><<<grid, kThreads, 0, stream>>>(Qp, Xp, op, nq, n, D, ntiles);
    else score_kernel<16, false><<<grid, kThreads, 0, stream>>>(Qp, Xp, op, nq, n, D, ntiles);
  } else {
    if (out_fp32) score_kernel<64, true><<<grid, kThreads, 0, stream>>>(Qp, Xp, op, nq, n, D, ntiles);
    else score_kernel<64, false><<<grid, kThreads, 0, stream>>>(Qp, Xp, op, nq, n, D, ntiles);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

static int slots_per_lane(int64_t k) { return k <= 32 ? 1 : k <= 64 ? 2 : k <= 128 ? 4 : 8; }

// Returns {scores [nq, k] float32, ids [nq, k] int32, plan [bq, chunks]}.
std::vector<torch::Tensor> fused_topk(torch::Tensor q, torch::Tensor x, int64_t k, int64_t bq,
                                      int64_t chunks) {
  check_pair(q, x);
  const c10::cuda::CUDAGuard guard(q.device());
  const int nq = (int)q.size(0);
  const long long n = x.size(0);
  const int D = (int)x.size(1);
  TORCH_CHECK(k >= 1 && k <= 256, "fused_topk supports 1 <= k <= 256");
  TORCH_CHECK(k <= n, "k must not exceed the corpus size");

  const int kpl = slots_per_lane(k);
  if (bq <= 0) bq = nq <= 16 ? 16 : nq <= 32 ? 32 : 64;
  TORCH_CHECK(bq == 16 || bq == 32 || bq == 64, "bq must be 16, 32 or 64");
  while (bq * kpl > 128) bq /= 2;  // keep the per-thread top-k state in registers

  const long long ntiles = (n + BN - 1) / BN;
  const long long qblocks = (nq + bq - 1) / bq;
  if (chunks <= 0) {
    const int sms = at::cuda::getDeviceProperties(q.device().index())->multiProcessorCount;
    const long long min_tiles = std::max<long long>(2, (8 * k + BN - 1) / BN);
    const long long max_chunks = std::max<long long>(1, ntiles / min_tiles);
    const long long target = std::max<long long>(1, (8LL * sms + qblocks - 1) / std::max<long long>(qblocks, 1));
    chunks = std::min(target, max_chunks);
  }
  chunks = std::max<long long>(1, std::min<long long>({(long long)chunks, ntiles, 65535LL}));
  const long long tiles_per_chunk = (ntiles + chunks - 1) / chunks;
  chunks = (ntiles + tiles_per_chunk - 1) / tiles_per_chunk;
  const long long cols = tiles_per_chunk * BN;

  auto fopt = q.options().dtype(torch::kFloat);
  auto iopt = q.options().dtype(torch::kInt);
  auto out_v = torch::empty({nq, k}, fopt);
  auto out_i = torch::empty({nq, k}, iopt);
  auto plan = torch::tensor({(int64_t)bq, (int64_t)chunks}, torch::kLong);
  if (nq == 0) return {out_v, out_i, plan};

  auto part_v = torch::empty({nq, chunks, k}, fopt);
  auto part_i = torch::empty({nq, chunks, k}, iopt);
  auto stream = at::cuda::getCurrentCUDAStream();
  const half* Qp = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
  const half* Xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  const dim3 grid((unsigned)qblocks, (unsigned)chunks);
  const int ki = (int)k;
  const int nc = (int)chunks;
  float* pv = part_v.data_ptr<float>();
  int* pi = part_i.data_ptr<int>();

  bool launched = false;
#define CVS_FUSED(BQV, KPLV)                                                                    \
  if (!launched && bq == BQV && kpl == KPLV) {                                                  \
    fused_topk_kernel<BQV, KPLV><<<grid, kThreads, 0, stream>>>(Qp, Xp, nq, n, D, ki, cols, nc, \
                                                                pv, pi);                        \
    launched = true;                                                                            \
  }
  CVS_FUSED(16, 1) CVS_FUSED(16, 2) CVS_FUSED(16, 4) CVS_FUSED(16, 8)
  CVS_FUSED(32, 1) CVS_FUSED(32, 2) CVS_FUSED(32, 4)
  CVS_FUSED(64, 1) CVS_FUSED(64, 2)
#undef CVS_FUSED
  TORCH_CHECK(launched, "no fused kernel for bq=", bq, " kpl=", kpl);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 mgrid((unsigned)((nq + kMergeWarps - 1) / kMergeWarps));
  const long long per_query = chunks * k;
  float* ov = out_v.data_ptr<float>();
  int* oi = out_i.data_ptr<int>();
  switch (kpl) {
    case 1: merge_topk_kernel<1><<<mgrid, kMergeWarps * 32, 0, stream>>>(pv, pi, nq, per_query, ki, ov, oi); break;
    case 2: merge_topk_kernel<2><<<mgrid, kMergeWarps * 32, 0, stream>>>(pv, pi, nq, per_query, ki, ov, oi); break;
    case 4: merge_topk_kernel<4><<<mgrid, kMergeWarps * 32, 0, stream>>>(pv, pi, nq, per_query, ki, ov, oi); break;
    default: merge_topk_kernel<8><<<mgrid, kMergeWarps * 32, 0, stream>>>(pv, pi, nq, per_query, ki, ov, oi); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_v, out_i, plan};
}

}  // namespace cvs

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Exact top-k inner-product search kernels (fp16 WMMA + fused warp top-k)";
  m.def("scores", &cvs::scores, "Full score matrix q @ x^T", py::arg("q"), py::arg("x"),
        py::arg("out_fp32") = true);
  m.def("fused_topk", &cvs::fused_topk, "Fused score + top-k", py::arg("q"), py::arg("x"),
        py::arg("k"), py::arg("bq") = 0, py::arg("chunks") = 0);
}
