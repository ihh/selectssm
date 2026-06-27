# selectssm (Rust / burn)

A Rust port of the JAX `selectssm` package (see [`../jax`](../jax)), built on the
[burn](https://burn.dev) framework. It implements the **chunked-scan** strategy —
matmul-form computation within each chunk plus a plain sequential loop across chunks — and
the Mamba-3 complex-SSM **RoPE trick** (`use_complex_ssm`), with config field names and
semantics matching the JAX implementation.

The chunked scan has two orthogonal knobs (both validated to match the JAX oracle on the same
parity fixtures):

- **`use_remat`** (default on) — a **rematerializing** scan that recomputes each chunk's
  intermediates in a hand-derived backward (low training memory) vs. the **reference** scan
  that retains them on the autodiff tape. See [Rematerialization](#rematerialization-use_remat).
- **`scan_algo`** — the within-chunk algorithm: `matrix` (the `cs×cs` decay matrix), `hillis`
  (Hillis–Steele parallel prefix scan, the **default**), or `cubecl` (custom parallel-scan GPU
  kernels, `--features cubecl`). See [`PERFORMANCE.md`](PERFORMANCE.md) — `hillis` is ~3–4×
  faster than `matrix`, and `cubecl` brings the full training step to ~1.25× of JAX/XLA.

## Modules

| Module | Description |
|---|---|
| `config` | `SelectiveSsmConfig`, `BidirectionalMambaConfig` (deserialized from the fixture manifest) |
| `chunked_scan` | `ssm_chunked_scan` (reference) + `scan_chunk_body`: within-chunk decay-matrix matmul + cross-chunk sequential loop carrying the SSM state **and** the cumulative rotation-angle accumulator |
| `remat` | `ScanBackend::chunked_scan`: backend dispatch for the scan, including the rematerializing custom autodiff op (hand-derived VJP) |
| `selective_ssm` | `SelectiveSsm` (causal + `reverse`) |
| `bidirectional` | `BidirectionalMamba` (forward + reverse, gated) |
| `rcps` | `RcpsWrapper` — reverse-complement parameter sharing (feature `rcps`) |
| `loader` | load JAX-exported safetensors parameters into burn tensors |

## Backends

- **`ndarray`** (default) — CPU, deterministic; the parity-learning and fixture-parity tests run here.
- **`wgpu`** (feature `wgpu`) — WebGPU. The crate sets `#![recursion_limit = "256"]` for the
  wgpu backend's deep type nesting.
- **`fusion`** (feature `fusion`, combine with `wgpu`) — wraps the wgpu backend in burn's
  kernel-fusion JIT decorator (`burn_fusion::Fusion<…>`), the analogue of XLA op fusion. It is
  correct but did **not** speed up the chunked scan in benchmarking (the cost is large matmuls
  and reductions, not the elementwise chains fusion targets); offered for experimentation.
- **`cubecl`** (feature `cubecl`, implies `wgpu`) — enables `scan_algo=cubecl`, a custom GPU
  prefix-scan kernel in the [`selectssm-cubecl`](selectssm-cubecl) member crate. Isolated there
  so its `unsafe` kernel launches stay out of this crate (`#![forbid(unsafe_code)]`). Not
  compatible with `fusion` (which changes the backend type); with `fusion` on, `cubecl` falls
  back to `hillis`.

The CPU/GPU backends are wrapped in `burn::backend::Autodiff` so gradients are available and
tested. This crate is a **Cargo workspace**: the root package `selectssm` plus the optional
`selectssm-cubecl` member.

## What is ported (and what is not)

Only the chunked-scan strategy is ported — **not** JAX's recursive-scan or custom-VJP-scan
strategies (which exist to trade compute for memory; here the rematerializing variant plays
that role). The cross-chunk scan is a plain sequential loop; no associative-scan / prefix-sum
primitive is assumed. When a chunk size does not divide the sequence length, the sequence is
end-padded and the padded outputs are dropped, exactly as in the JAX code.

`RcpsWrapper` is severable behind the `rcps` feature: the build and tests are green with or
without it.

## Rematerialization (`use_remat`)

Under an autodiff backend the **reference** scan builds the forward out of stock burn ops, so
every chunk's `(cs, cs, B, D, N)` decay matrix (and the rest) stays live on the tape until the
backward pass — peak memory grows like `O(L · cs · B · D · N)`. The **rematerializing** scan
(`use_remat=true`, the default) registers the whole scan as a single custom autodiff op:

- The **forward** runs the chunk loop on the *inner* (non-autodiff) backend, keeping only the
  inputs and the small per-chunk boundary carries `(hstate, angle_acc)` — no `cs²` term.
- The **backward** is a hand-derived VJP (the matrix-form linear SSM plus the RoPE rotary,
  including `theta` and the cross-chunk carry), recomputing each chunk's intermediates one at a
  time. It does **not** use nested autodiff — burn's autodiff runtime holds a global lock across
  the backward pass, so a reentrant `.backward()` deadlocks — so the per-chunk gradients are
  written out analytically (the Rust analogue of JAX's `custom_vjp`).

Both variants are bitwise-equivalent in the forward and match the JAX reference gradients on
every parity fixture (`cargo test` runs the suite both ways). On an RTX A6000 at `L=8192`,
remat cuts training peak VRAM ~**32×** (≈1 GB vs ≈34 GB) and is ~20% faster. Set
`use_remat=false` to fall back to the reference scan.

## Performance

See **[`PERFORMANCE.md`](PERFORMANCE.md)** for the full story (linear-time scaling,
rematerialization, the three scan algorithms, chunk-size sensitivity, and the comparison to
JAX/XLA). In short: `use_remat` cuts training VRAM ~32×; `hillis` is ~3–4× faster than
`matrix`; `cubecl` brings the forward to ~1.5× of XLA.

`examples/scaling.rs` benchmarks one configuration; the repo-level
`scripts/benchmark_scaling.py` drives the full cross-implementation sweep and samples GPU VRAM.
Quick single runs (`… remat|ref  matrix|hillis|cubecl`):

```bash
cargo run --release --features wgpu          --example scaling -- gpu real 4 4096 128 8 64 12 remat hillis
cargo run --release --features "wgpu cubecl" --example scaling -- gpu real 4 4096 128 8 64 12 remat cubecl
```

## Build & test

```bash
cargo test                          # ndarray backend (fixture parity + parity learning)
cargo test --features rcps          # include the RCPS wrapper + its parity fixtures
cargo test --features wgpu          # also run fixture parity on the wgpu backend
cargo test --features "wgpu cubecl" # also validate the cubecl GPU scan kernel vs the oracle
cargo test --features "wgpu rcps"   # everything
```

The fixture-parity suite runs **both** the rematerializing and the reference scan and asserts
both match the JAX oracle. The optional `fusion` feature (combine with `wgpu`) enables burn's
kernel-fusion backend.

### Fixture parity (`tests/parity.rs`)

Loads the golden vectors in [`../fixtures`](../fixtures) (generated by the JAX reference) and
asserts forward parity (≤ 1e-4 relative) and gradient parity (≤ 1e-3 relative) for
`{causal, reverse, bidirectional, rcps} × {complex off, on}`, including a sequence length that
is not a multiple of the chunk size. On wgpu the gradient tolerance is loosened (f32 reduction
order on the GPU differs from XLA).

### Parity learning (`tests/parity_learning.rs`)

Trains a tiny one-layer model on short binary sequences and evaluates on longer, held-out
ones: with `use_complex_ssm` on it learns parity and length-generalizes; with it off it stays
near chance. The solves-vs-chance gap is the gate.

## Loading JAX parameters

`loader::Store` parses a safetensors file (tensor names are the flattened Flax parameter
paths). The typed `SsmWeights` / `BidirWeights` structs load by name, so identical weights can
be exercised across both languages — see `tests/parity.rs`.
