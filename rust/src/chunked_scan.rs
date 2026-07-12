//! The chunked-scan strategy, ported from `ssm_chunked_scan` in the JAX implementation.
//!
//! Within each chunk the recurrence is materialised in matmul form (a per-pair decay
//! matrix); across chunks a plain sequential loop carries the SSM state `h` and, for the
//! complex-SSM RoPE trick, the cumulative rotation angle accumulator.  No associative-scan
//! or prefix-sum primitive is assumed.  When a chunk size does not divide the sequence
//! length, the sequence is end-padded with zeros and the padded outputs are dropped (the
//! scan is causal, so this does not affect any real position) --- mirroring the JAX code.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::config::ScanAlgo;
use crate::remat::ScanBackend;

/// Lower-triangular (inclusive) ones matrix: `L[i,j] = 1` if `i >= j` else `0`.
/// Used both as an inclusive-cumsum operator (`cumsum = L @ v`) and as the causal mask
/// for the within-chunk decay matrix.
pub(crate) fn tril_ones<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut v = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            v[i * n + j] = 1.0;
        }
    }
    Tensor::from_data(TensorData::new(v, [n, n]), device)
}

/// Inclusive cumulative sum along dim 0 of a (cs, B, H) tensor, via `tril @ v`.
pub(crate) fn cumsum0_3<B: Backend>(tril: &Tensor<B, 2>, v: Tensor<B, 3>) -> Tensor<B, 3> {
    let [cs, b, h] = v.dims();
    tril.clone().matmul(v.reshape([cs, b * h])).reshape([cs, b, h])
}

/// Inclusive cumulative sum along dim 0 of a (cs, B, D, N) tensor, via `tril @ v`.
pub(crate) fn cumsum0_4<B: Backend>(tril: &Tensor<B, 2>, v: Tensor<B, 4>) -> Tensor<B, 4> {
    let [cs, b, d, n] = v.dims();
    tril.clone()
        .matmul(v.reshape([cs, b * d * n]))
        .reshape([cs, b, d, n])
}

/// Apply the data-dependent rotary embedding `R(-Phi)` to a (cs, B, N) projection,
/// pairing the first N/2 ("real") entries with the last N/2 ("imag") entries.
pub(crate) fn apply_rotary<B: Backend>(v: Tensor<B, 3>, phi: &Tensor<B, 3>) -> Tensor<B, 3> {
    let [cs, b, n] = v.dims();
    let h = n / 2;
    let v_re = v.clone().slice([0..cs, 0..b, 0..h]);
    let v_im = v.slice([0..cs, 0..b, h..n]);
    let cos = phi.clone().cos();
    let sin = phi.clone().sin();
    let new_re = v_re.clone() * cos.clone() + v_im.clone() * sin.clone();
    let new_im = v_re * sin.neg() + v_im * cos;
    Tensor::cat(vec![new_re, new_im], 2)
}

/// Pad a (B, L, C) tensor with `pad` zero rows at the end of the L axis.
pub(crate) fn pad_l<B: Backend>(x: Tensor<B, 3>, pad: usize) -> Tensor<B, 3> {
    if pad == 0 {
        return x;
    }
    let [b, _l, c] = x.dims();
    let zeros = Tensor::zeros([b, pad, c], &x.device());
    Tensor::cat(vec![x, zeros], 1)
}

/// Hillis–Steele inclusive scan over dim 0 of `(cs,B,D,N)` tensors, with the linear-recurrence
/// operator `(a₁,b₁)⊕(a₂,b₂) = (a₂·a₁, a₂·b₁+b₂)`.  Returns `(a_cum, b_scan)` where
/// `a_cum[t] = ∏_{k≤t} a[k]` and `b_scan[t] = Σ_{j≤t} (∏_{k=j+1..t} a[k]) b[j]` — i.e. the
/// closed form of the recurrence `h[t] = a[t]·h[t-1] + b[t]`.  `O(cs·log cs)` work, no `cs×cs`
/// tensor; the portable analogue of `jax.lax.associative_scan`.
pub(crate) fn assoc_scan0<B: Backend>(
    a: Tensor<B, 4>,
    b: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [cs, bb, d, n] = a.dims();
    let dev = a.device();
    let (mut a_s, mut b_s) = (a, b);
    let mut step = 1;
    while step < cs {
        let keep = cs - step;
        // prev[t] = x[t-step] for t≥step, else the operator identity (a=1, b=0).
        let a_prev = Tensor::cat(
            vec![
                Tensor::ones([step, bb, d, n], &dev),
                a_s.clone().slice([0..keep, 0..bb, 0..d, 0..n]),
            ],
            0,
        );
        let b_prev = Tensor::cat(
            vec![
                Tensor::zeros([step, bb, d, n], &dev),
                b_s.clone().slice([0..keep, 0..bb, 0..d, 0..n]),
            ],
            0,
        );
        b_s = a_s.clone() * b_prev + b_s;
        a_s = a_s * a_prev;
        step *= 2;
    }
    (a_s, b_s)
}

/// Reverse linear-recurrence scan `r[t] = src[t] + al[t+1]·r[t+1]` (with `r[cs]=0`), i.e.
/// `r[t] = Σ_{p≥t} (∏_{k=t+1..p} al[k]) src[p]` — the adjoint of the within-chunk state
/// recurrence.  Realised by flipping into a forward [`assoc_scan0`].
pub(crate) fn rev_assoc_scan0<B: Backend>(al: Tensor<B, 4>, src: Tensor<B, 4>) -> Tensor<B, 4> {
    let [cs, bb, d, n] = al.dims();
    let dev = al.device();
    // a_shift[t] = al[t+1]; the last (unused) multiplier is 0.
    let a_shift = Tensor::cat(
        vec![
            al.slice([1..cs, 0..bb, 0..d, 0..n]),
            Tensor::zeros([1, bb, d, n], &dev),
        ],
        0,
    );
    let (_a_cum, g) = assoc_scan0(a_shift.flip([0]), src.flip([0]));
    g.flip([0])
}

/// Build the chunk-shaped constants `(tril, mask)` for the scan, allocating the `cs×cs` matrix
/// only when an algorithm actually needs it.  The `Matrix` algorithm uses both; the rotary
/// (complex mode) uses `tril` for the cumulative angle; the `Hillis`/`Cubecl` real-mode paths
/// use neither, so we return 1-element placeholders (important when `cs` is the whole sequence,
/// where a real `cs×cs` tril would be huge and built on the CPU).
pub(crate) fn scan_consts<B: Backend>(
    cs: usize,
    device: &B::Device,
    algo: ScanAlgo,
    complex: bool,
) -> (Tensor<B, 2>, Tensor<B, 5>) {
    let need_tril = algo == ScanAlgo::Matrix || complex;
    let tril = tril_ones::<B>(if need_tril { cs } else { 1 }, device);
    let mask = if algo == ScanAlgo::Matrix {
        tril.clone().reshape([cs, cs, 1, 1, 1])
    } else {
        Tensor::zeros([1, 1, 1, 1, 1], device)
    };
    (tril, mask)
}

/// One chunk of the selective scan, written so it can be reused by both the reference
/// (non-remat) loop and the rematerializing autodiff op (which re-runs it in the backward
/// pass).  All inputs are length-major and already padded/sliced for this chunk.
///
/// Given the carried state `hstate` (1,B,D,N) and rotation accumulator `angle_acc` (1,B,H)
/// at the chunk's left edge, returns `(y_chunk, hstate_next, angle_next)`:
/// * `y_chunk`     (cs,B,D)
/// * `hstate_next` (1,B,D,N)  state carried to the next chunk
/// * `angle_next`  (1,B,H)    rotation accumulator carried to the next chunk
///
/// `b_c` / `c_c` are the *raw* (pre-rotary) projections; the rotary is applied inside.
/// `tril`, `mask`, and `a4` are the chunk-shaped constants (`tril_ones(cs)`, the causal
/// decay mask, and `a` reshaped to (1,1,D,N)); the caller precomputes them once.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_chunk_body<B: ScanBackend>(
    tril: &Tensor<B, 2>,
    mask: &Tensor<B, 5>,
    a4: &Tensor<B, 4>,
    u_c: Tensor<B, 3>,
    b_c: Tensor<B, 3>,
    c_c: Tensor<B, 3>,
    dt_c: Tensor<B, 3>,
    theta_c: Option<Tensor<B, 3>>,
    hstate: Tensor<B, 4>,
    angle_acc: Tensor<B, 3>,
    algo: ScanAlgo,
) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 3>) {
    let [cs, bb, d] = u_c.dims();
    let n = a4.dims()[3];

    // Complex-SSM RoPE trick: cumulative angle = carried offset + within-chunk cumsum.
    let (b_c, c_c, angle_next) = match theta_c {
        Some(theta_c) => {
            let phi = cumsum0_3(tril, theta_c.clone()) + angle_acc.clone(); // (cs,B,H)
            let total = theta_c.sum_dim(0); // (1,B,H)
            (apply_rotary(b_c, &phi), apply_rotary(c_c, &phi), angle_acc + total)
        }
        None => (b_c, c_c, angle_acc),
    };

    // Per-step log-decay a*dt; dB = (rotated B) * u * dt.
    let a_dt = a4.clone() * dt_c.clone().reshape([cs, bb, d, 1]); // (cs,B,D,N)
    let d_b = b_c.reshape([cs, bb, 1, n])
        * u_c.reshape([cs, bb, d, 1])
        * dt_c.reshape([cs, bb, d, 1]); // (cs,B,D,N)

    // hs[t] = (∏_{k≤t} dA) · h_init + Σ_{j≤t} (∏_{k=j+1..t} dA) dB[j], computed either by the
    // matrix form (a `cs×cs` decay matrix) or the Hillis–Steele scan.
    let hs = within_chunk_hs(tril, mask, a_dt, d_b, hstate, algo);

    let y_c = (c_c.reshape([cs, bb, 1, n]) * hs.clone()).sum_dim(3); // (cs,B,D,1)
    let hstate_next = hs.slice([cs - 1..cs, 0..bb, 0..d, 0..n]); // (1,B,D,N)
    (y_c.reshape([cs, bb, d]), hstate_next, angle_next)
}

/// Within-chunk inclusive scan producing `hs` (the carried state at every step).  Shared by
/// the forward body and the rematerializing VJP recompute; selects matrix / Hillis–Steele /
/// cubecl.
pub(crate) fn within_chunk_hs<B: ScanBackend>(
    tril: &Tensor<B, 2>,
    mask: &Tensor<B, 5>,
    a_dt: Tensor<B, 4>,
    d_b: Tensor<B, 4>,
    hstate: Tensor<B, 4>,
    algo: ScanAlgo,
) -> Tensor<B, 4> {
    let [cs, bb, d, n] = a_dt.dims();
    match algo {
        ScanAlgo::Cubecl => B::within_chunk_scan_cubecl(a_dt.exp(), d_b, hstate),
        ScanAlgo::Matrix => {
            let a_cumsum = cumsum0_4(tril, a_dt); // (cs,B,D,N)
            let gs = a_cumsum.clone().exp(); // inclusive product of dA
            // Within-chunk decay matrix L[p,j] = exp(A_cumsum[p] - A_cumsum[j]) * 1[p>=j].
            // A_cumsum is monotonically non-increasing (it is the inclusive cumsum of a*dt
            // <= 0), so for the kept entries (p>=j) the exponent is <= 0.  We clamp the
            // exponent to <= 0 before exp() purely as a numerical guard: for the masked-out
            // entries (p<j) the exponent is positive and large decay magnitudes would
            // overflow exp() to +inf, and inf * 0 (the mask) = NaN.  clamp_max(0) is a no-op
            // on the kept entries (and on every fixture), so parity with the JAX reference is
            // preserved; it only tames the entries that the mask discards anyway.
            let cp = a_cumsum.clone().reshape([cs, 1, bb, d, n]);
            let cj = a_cumsum.reshape([1, cs, bb, d, n]);
            let lmat = (cp - cj).clamp_max(0.0).exp() * mask.clone(); // (cs,cs,B,D,N)
            let intra = (lmat * d_b.reshape([1, cs, bb, d, n])).sum_dim(1).reshape([cs, bb, d, n]);
            gs * hstate + intra
        }
        ScanAlgo::Hillis => {
            // dA = exp(a*dt) ∈ (0,1]; products stay bounded so no overflow guard is needed.
            let al = a_dt.exp();
            let (a_cum, b_scan) = assoc_scan0(al, d_b);
            a_cum * hstate + b_scan
        }
    }
}

/// Branchless minimax `f32` exp for the CPU/host hot loops (Cephes `expf`, ~1 ULP).  The stock
/// `f32::exp` on wasm is software libm — scalar, branchy, with an f64 core — and the SSM decay
/// `exp(a·dt)` calls it ~`2·B·L·ED·N` times per forward (~5.67M at the deploy shape), dominating
/// the CPU cost.  This inlinable, branch-free form avoids that call/branch/f64 overhead AND
/// autovectorises under `simd128`.  Deterministic (fixed polynomial + a bit-field `ldexp`), so the
/// cross-engine bit-stability the deploy path relies on is preserved.  Accuracy (~1e-7 rel) is far
/// inside both the internal `<1e-4` fused-parity gate and the `~1e-3` JAX-golden gate.
#[inline(always)]
pub(crate) fn fast_exp(x: f32) -> f32 {
    // Clamp to the f32 exp domain (saturate like libm: ~0 below −87, ~max above +88).  The SSM
    // hot path has x ≤ 0, but keep it general/safe.
    let x = x.clamp(-87.336_54, 88.722_84);
    const LOG2E: f32 = 1.442_695_04;
    const C1: f32 = 0.693_359_375; // ln2 hi
    const C2: f32 = -2.121_944_4e-4; // ln2 lo (C1 − C2·… splits ln2 for extra precision)
    // Range reduce: exp(x) = 2^k · exp(r),  k = round(x·log2e),  r = x − k·ln2 ∈ [−ln2/2, ln2/2].
    let kf = (x * LOG2E + 0.5).floor();
    let r = x - kf * C1 - kf * C2;
    // Minimax polynomial for exp(r) (Cephes expf coefficients).
    let mut p = 1.987_569_15e-4_f32;
    p = p * r + 1.398_199_95e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 5.000_000_1e-1;
    let er = p * (r * r) + r + 1.0;
    // Multiply by 2^k via the exponent bit-field: (k + 127) << 23.  After the clamp, k ∈ [−126,128].
    let pow2 = f32::from_bits(((kf as i32 + 127) as u32) << 23);
    er * pow2
}

/// Direct sequential selective-scan for host (CPU) backends, REAL mode (no rotary).  Extracts the
/// inputs to host `f32` once and runs the exact recurrence
///   `h[t] = exp(a·dt[t])·h[t-1] + b[t]·u[t]·dt[t]`,   `y[t] = Σ_n c[t,n]·h[t,·,n]`
/// in tight scalar loops with a SINGLE output allocation.  The generic [`assoc_scan0`] Hillis scan
/// instead emits ~`O(n_chunks·log cs)` full-tensor `cat`/mul allocations per forward, and on the
/// ndarray backend that allocation/dispatch overhead — not the (~5 ms) arithmetic — dominates cost
/// (~340 ms at L=369).  Algebraically this IS the reference recurrence, so it matches
/// [`ssm_chunked_scan`] to `f32` tolerance; used only for the plain-`NdArray` inference path
/// (training's autodiff forward goes through `scan_collect`/`chunk_vjp`, untouched).  Complex/RoPE
/// mode is left to the reference scan.
pub(crate) fn sequential_scan_host<B: Backend>(
    u: Tensor<B, 3>,  // (B,L,D)
    a: Tensor<B, 2>,  // (D,N)
    b: Tensor<B, 3>,  // (B,L,N)
    c: Tensor<B, 3>,  // (B,L,N)
    dt: Tensor<B, 3>, // (B,L,D)
) -> Tensor<B, 3> {
    let dev = u.device();
    let [bb, l, d] = u.dims();
    let n = a.dims()[1];
    let uv = u.into_data().to_vec::<f32>().unwrap();
    let av = a.into_data().to_vec::<f32>().unwrap();
    let bv = b.into_data().to_vec::<f32>().unwrap();
    let cv = c.into_data().to_vec::<f32>().unwrap();
    let dv = dt.into_data().to_vec::<f32>().unwrap();
    let mut y = vec![0f32; bb * l * d];
    let mut hstate = vec![0f32; d * n]; // carried SSM state (D,N), reset per batch element
    for bi in 0..bb {
        hstate.iter_mut().for_each(|x| *x = 0.0);
        for t in 0..l {
            let base_ld = (bi * l + t) * d;
            let base_ln = (bi * l + t) * n;
            for di in 0..d {
                let u_v = uv[base_ld + di];
                let dt_v = dv[base_ld + di];
                let hoff = di * n;
                let aoff = di * n;
                let mut acc = 0f32;
                for ni in 0..n {
                    let da = fast_exp(av[aoff + ni] * dt_v);
                    let db = bv[base_ln + ni] * u_v * dt_v;
                    let hh = da * hstate[hoff + ni] + db;
                    hstate[hoff + ni] = hh;
                    acc += cv[base_ln + ni] * hh;
                }
                y[base_ld + di] = acc;
            }
        }
    }
    Tensor::from_data(TensorData::new(y, [bb, l, d]), &dev)
}

/// Chunked selective-scan (reference / non-rematerializing).
///
/// * `u`      (B, L, D)   --- SSM input (post conv + activation)
/// * `a`      (D, N)      --- transition coefficients (already `-exp(A_log)`, possibly tied)
/// * `b_proj` (B, L, N)   --- B projection
/// * `c_proj` (B, L, N)   --- C projection
/// * `dt`     (B, L, D)   --- (already softplus'd) step sizes
/// * `theta`  Option<(B, L, N/2)> --- per-step rotation angle for the complex-SSM RoPE trick
///
/// Returns the SSM output (B, L, D) (the `D`-skip term is added by the caller).  Under an
/// autodiff backend this retains every chunk's intermediates on the tape; see
/// [`crate::remat`] for the rematerializing variant that recomputes them in the backward pass.
pub fn ssm_chunked_scan<B: ScanBackend>(
    u: Tensor<B, 3>,
    a: Tensor<B, 2>,
    b_proj: Tensor<B, 3>,
    c_proj: Tensor<B, 3>,
    dt: Tensor<B, 3>,
    theta: Option<Tensor<B, 3>>,
    chunk_size: usize,
    algo: ScanAlgo,
) -> Tensor<B, 3> {
    let device = u.device();
    let [bb, l, d] = u.dims();
    let n = a.dims()[1];
    let h = n / 2;
    let cs = chunk_size;
    let pad = (cs - l % cs) % cs;
    let lp = l + pad;
    let n_chunks = lp / cs;

    // Pad and move to length-major layout.
    let u = pad_l(u, pad).swap_dims(0, 1); // (Lp, B, D)
    let b_proj = pad_l(b_proj, pad).swap_dims(0, 1); // (Lp, B, N)
    let c_proj = pad_l(c_proj, pad).swap_dims(0, 1); // (Lp, B, N)
    let dt = pad_l(dt, pad).swap_dims(0, 1); // (Lp, B, D)
    let theta = theta.map(|t| pad_l(t, pad).swap_dims(0, 1)); // (Lp, B, H)

    let tril = tril_ones::<B>(cs, &device);
    let mask = tril.clone().reshape([cs, cs, 1, 1, 1]); // causal mask for the decay matrix
    let a4 = a.reshape([1, 1, d, n]);

    let mut hstate = Tensor::<B, 4>::zeros([1, bb, d, n], &device); // carried state (1,B,D,N)
    let mut angle_acc = Tensor::<B, 3>::zeros([1, bb, h], &device); // carried angle (1,B,H)
    let mut y_chunks: Vec<Tensor<B, 3>> = Vec::with_capacity(n_chunks);

    for c in 0..n_chunks {
        let s = c * cs;
        let u_c = u.clone().slice([s..s + cs, 0..bb, 0..d]); // (cs,B,D)
        let b_c = b_proj.clone().slice([s..s + cs, 0..bb, 0..n]); // (cs,B,N)
        let c_c = c_proj.clone().slice([s..s + cs, 0..bb, 0..n]); // (cs,B,N)
        let dt_c = dt.clone().slice([s..s + cs, 0..bb, 0..d]); // (cs,B,D)
        let theta_c = theta.as_ref().map(|t| t.clone().slice([s..s + cs, 0..bb, 0..h]));

        let (y_c, hstate_next, angle_next) = scan_chunk_body(
            &tril, &mask, &a4, u_c, b_c, c_c, dt_c, theta_c, hstate, angle_acc, algo,
        );
        y_chunks.push(y_c);
        hstate = hstate_next;
        angle_acc = angle_next;
    }

    let y = Tensor::cat(y_chunks, 0); // (Lp, B, D)
    let y = y.swap_dims(0, 1); // (B, Lp, D)
    y.slice([0..bb, 0..l, 0..d])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::{Tensor, TensorData};

    type B = NdArray;

    // Regression guard for the within-chunk decay matrix: with large decay magnitudes the
    // masked (upper-triangle) exponent A_cumsum[p]-A_cumsum[j] (p<j) is large and positive.
    // Without clamping the exponent to <= 0 before exp(), it overflowed to +inf and the mask
    // multiply produced inf*0 = NaN.  This must stay finite.
    #[test]
    fn decay_matrix_does_not_overflow() {
        let dev = Default::default();
        let (bb, l, d, n) = (1usize, 6usize, 2usize, 4usize);
        let ones = |shape: [usize; 3]| {
            let k: usize = shape.iter().product();
            Tensor::<B, 3>::from_data(TensorData::new(vec![1f32; k], shape), &dev)
        };
        // A = -100 (very negative), dt = 5 (large) => A_cumsum reaches ~ -1500 within a chunk.
        let a = Tensor::<B, 2>::from_data(TensorData::new(vec![-100f32; d * n], [d, n]), &dev);
        let dt = Tensor::<B, 3>::from_data(TensorData::new(vec![5f32; bb * l * d], [bb, l, d]), &dev);
        for algo in [ScanAlgo::Matrix, ScanAlgo::Hillis] {
            let y = ssm_chunked_scan(
                ones([bb, l, d]), a.clone(), ones([bb, l, n]), ones([bb, l, n]), dt.clone(), None, 3,
                algo,
            );
            let v: Vec<f32> = y.into_data().to_vec().unwrap();
            assert!(v.iter().all(|x| x.is_finite()), "{algo:?}: scan produced non-finite");
        }
    }

    // The two within-chunk algorithms must agree to floating-point tolerance on a generic
    // (non-degenerate) input, including a length that is not a multiple of the chunk size.
    #[test]
    fn matrix_and_hillis_agree() {
        let dev = Default::default();
        let (bb, l, d, n, cs) = (2usize, 10usize, 3usize, 4usize, 4usize);
        let rnd = |shape: [usize; 3], seed: u64| {
            let k: usize = shape.iter().product();
            // cheap deterministic pseudo-random in [-0.5, 0.5)
            let v: Vec<f32> = (0..k)
                .map(|i| (((i as u64 * 2654435761 + seed) % 1000) as f32) / 1000.0 - 0.5)
                .collect();
            Tensor::<B, 3>::from_data(TensorData::new(v, shape), &dev)
        };
        let a = Tensor::<B, 2>::from_data(
            TensorData::new((0..d * n).map(|i| -((i % n + 1) as f32)).collect(), [d, n]),
            &dev,
        );
        let dt = rnd([bb, l, d], 7).abs() + 0.1; // positive step sizes
        let theta = rnd([bb, l, n / 2], 9);
        for th in [None, Some(theta)] {
            let run = |algo| {
                ssm_chunked_scan(
                    rnd([bb, l, d], 1), a.clone(), rnd([bb, l, n], 2), rnd([bb, l, n], 3),
                    dt.clone(), th.clone(), cs, algo,
                )
                .into_data()
                .to_vec::<f32>()
                .unwrap()
            };
            let m = run(ScanAlgo::Matrix);
            let hh = run(ScanAlgo::Hillis);
            let max_abs = m.iter().fold(0f32, |a, &x| a.max(x.abs())).max(1e-6);
            let err = m.iter().zip(&hh).fold(0f32, |a, (&x, &y)| a.max((x - y).abs())) / max_abs;
            assert!(err < 1e-5, "matrix vs hillis rel err {err:.2e} (complex={})", th.is_some());
        }
    }

    // The fast host recurrence (real mode) must agree with the Hillis scan to f32 tolerance,
    // including a non-chunk-multiple length and multiple batch elements.  This is the parity
    // guard for the NdArray inference fast path.
    #[test]
    fn sequential_host_matches_hillis() {
        let dev = Default::default();
        let (bb, l, d, n, cs) = (2usize, 10usize, 3usize, 4usize, 4usize);
        let rnd = |shape: [usize; 3], seed: u64| {
            let k: usize = shape.iter().product();
            let v: Vec<f32> = (0..k)
                .map(|i| (((i as u64 * 2654435761 + seed) % 1000) as f32) / 1000.0 - 0.5)
                .collect();
            Tensor::<B, 3>::from_data(TensorData::new(v, shape), &dev)
        };
        let a = Tensor::<B, 2>::from_data(
            TensorData::new((0..d * n).map(|i| -((i % n + 1) as f32)).collect(), [d, n]),
            &dev,
        );
        let dt = rnd([bb, l, d], 7).abs() + 0.1;
        let (u, bp, cp) = (rnd([bb, l, d], 1), rnd([bb, l, n], 2), rnd([bb, l, n], 3));
        let hillis = ssm_chunked_scan(
            u.clone(), a.clone(), bp.clone(), cp.clone(), dt.clone(), None, cs, ScanAlgo::Hillis,
        )
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        let seq = sequential_scan_host(u, a, bp, cp, dt).into_data().to_vec::<f32>().unwrap();
        let max_abs = hillis.iter().fold(0f32, |a, &x| a.max(x.abs())).max(1e-6);
        let err = hillis.iter().zip(&seq).fold(0f32, |a, (&x, &y)| a.max((x - y).abs())) / max_abs;
        assert!(err < 1e-5, "sequential-host vs hillis rel err {err:.2e}");
    }
}
