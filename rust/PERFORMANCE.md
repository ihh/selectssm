# Performance & scaling (Rust / burn)

All numbers below are on one **NVIDIA RTX A6000** (wgpu/Vulkan backend), f32, at
`B=4, D=128, N=8` unless noted. Latencies are per-iteration averages; "fwd+bwd" is a full
training step (loss = `sum(y²)`, gradients to inputs + all parameters). Peak VRAM is sampled
from `nvidia-smi` (whole-process, so it includes a fixed ~300 MB driver/runtime baseline).

Reproduce:

```bash
# one configuration (device mode B L D N chunk iters [remat|ref] [matrix|hillis|cubecl])
cargo run --release --features wgpu        --example scaling -- gpu real 4 4096 128 8 64 12 remat hillis
cargo run --release --features "wgpu cubecl" --example scaling -- gpu real 4 4096 128 8 64 12 remat cubecl

# full cross-implementation sweep (writes ../fixtures/benchmark_scaling.csv)
python3 ../scripts/benchmark_scaling.py
```

## 1. Linear-time in sequence length

Every variant is `O(L)` in time and (with rematerialization) ~flat in memory — the selective
SSM's defining property. Doubling `L` roughly doubles the step time and barely moves peak VRAM.

## 2. Rematerialization (`use_remat`, default on)

The reference scan keeps every chunk's intermediates on the autodiff tape; the rematerializing
scan recomputes them in a hand-derived backward (see [`src/remat.rs`](src/remat.rs)). Forward
output is identical; gradients match the JAX oracle. At `chunk=64`, real, matrix algorithm:

| L | fwd+bwd remat | fwd+bwd reference | peak VRAM remat | peak VRAM reference |
|---|---|---|---|---|
| 1024 | 56 ms | 66 ms | 0.68 GB | 4.8 GB |
| 4096 | 211 ms | 261 ms | 0.94 GB | 17.6 GB |
| 8192 | 420 ms | 533 ms | **1.06 GB** | **34.5 GB** |

→ remat is **~32× less training VRAM** at `L=8192` (and ~flat in `L`), and ~20% faster (no
giant graph to build). The reference scan is kept as `use_remat=false`.

## 3. Within-chunk scan algorithm (`scan_algo`)

Three interchangeable algorithms compute the same linear recurrence (parity within fp
tolerance), trading work, memory, and parallelism:

- **`Matrix`** — materialise the `cs×cs` decay matrix and contract it: `O(L·cs)` work, an
  `O(cs²·B·D·N)` transient that also hits wgpu's max-buffer-size limit at large `cs`.
- **`Hillis`** (default) — Hillis–Steele parallel prefix scan: `O(L·log cs)`, no `cs×cs`
  tensor, pure burn ops (runs on every backend). The portable analogue of JAX's
  `jax.lax.associative_scan`.
- **`Cubecl`** — a custom GPU prefix-scan kernel (one thread per `(b,d,n)` channel, sequential
  over the chunk; [`selectssm-cubecl`](selectssm-cubecl)). Needs `--features cubecl` + wgpu in
  the remat path; falls back to `Hillis` elsewhere.

`L=8192, chunk=64, real, remat`:

| algorithm | forward | fwd+bwd |
|---|---|---|
| Matrix | 177 ms | 548 ms |
| Hillis | 46 ms | 167 ms |
| **Cubecl** | **8.8 ms** | **53 ms** |

Hillis is ~3–4× faster than Matrix; Cubecl is a further ~5× on the forward. Both the forward
state recurrence and the backward adjoint scan are GPU kernels (`chunk_scan_kernel` /
`chunk_rev_scan_kernel`); the remaining backward cost is the elementwise gradient reductions,
which are still burn ops.

### Chunk-size sensitivity (`L=4096`, real, remat — forward ms)

| chunk | Matrix | Hillis | Cubecl |
|---|---|---|---|
| 16  | 33  | 65 | 13.0 |
| 32  | 53  | 37 | 7.6 |
| 64  | 90  | 24 | 4.7 |
| 128 | 167 | 15 | **4.4** |
| 256 | 326 | 15 | 4.6 |

Matrix degrades ∝ `cs` (and large `cs` exceeds GPU buffer limits — those points are
unreliable); Hillis and Cubecl are best around `cs≈128` and robust. Default `chunk_size`
stays as configured by the fixtures; `cs≈128` is a good choice for the scan-heavy regime.

## 4. Versus JAX / XLA

| | Rust Matrix | Rust Hillis | Rust Cubecl | JAX chunked (XLA) |
|---|---|---|---|---|
| forward (L=8192) | 177 ms | 46 ms | **8.8 ms** | 6.0 ms |
| fwd+bwd (L=8192) | 548 ms | 167 ms | **53 ms** | 29 ms |

The progression closes most of the original ~30× gap. The algorithmic change (Hillis) and then
the GPU kernels (Cubecl, forward **and** backward) bring the **forward to ~1.5× of XLA** and
**forward+backward to ~1.8×**. The residual gap is the elementwise gradient reductions in the
backward (still burn ops) plus the small per-chunk forward ops that XLA fuses.

JAX's own three scan strategies were measured too: **chunked is both its fastest and its
default** (~10× faster forward and ~25–30× faster fwd+bwd than `recursive_scan` /
`custom_vjp_scan`, at equal memory). burn's kernel-fusion backend (`--features fusion`) made no
difference to this workload — it is dominated by large matmul/reduction ops, not the
element-wise chains fusion targets; the lever was the scan algorithm, not fusion.

## Recommendations

- **Training, any backend:** defaults (`use_remat=true`, `scan_algo=hillis`) — low memory,
  good speed, fully portable.
- **Training on GPU, max speed:** build `--features cubecl` and set `scan_algo=cubecl`.
- **`scan_algo=matrix`** is retained as the simple reference; prefer small `chunk_size` if you
  use it, and avoid large `chunk_size` on wgpu (buffer-size limits).

Known issue: the `cubecl` path occasionally segfaults *on process exit* (a wgpu/cubecl
teardown race) — results and the parity tests are unaffected (the crash is after the work and
the printed output).
