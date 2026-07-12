//! Fused host (CPU) forward for [`crate::BidirectionalMamba`], specialized for the concrete
//! `burn::backend::NdArray<f32>` backend.
//!
//! selectssm is GPU-first: the generic [`crate::BidirectionalMamba::forward`] is built out of
//! stock `burn` tensor ops, which is right for the GPU/autodiff tiers but on the `ndarray` CPU
//! backend runs the block as ~35 separate allocating ops.  Profiling at (nodes=15, cols=369,
//! d_model=32) shows the forward is OVERHEAD-bound, not FLOP-bound (wasm SIMD moved it only
//! ~1.15×).  This module extracts every weight to `Vec<f32>` ONCE and runs the whole block in
//! tight scalar loops with a SINGLE output allocation — the CPU analogue of the already-fused
//! [`crate::chunked_scan::sequential_scan_host`], extended from the scan to the entire block.
//!
//! SCOPE: real mode only (`use_complex_ssm=false`), `norm_type="rms"`, `activation="silu"`,
//! `mlp_layer=false`, `dt_proj=true`, `concatenate_fwd_rev=true`, no tied proj/gate — i.e. the
//! deployed move-type config.  [`bidirectional_forward_fused_ndarray`] asserts this envelope;
//! callers outside it must use the generic `forward`.  The math is algebraically identical to
//! the generic forward, so it matches to `f32` tolerance (see the parity test in
//! `bidirectional.rs`).  Autodiff/backward and the Wgpu tier are untouched.

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

use crate::bidirectional::BidirectionalMamba;
use crate::chunked_scan::fast_exp;
use crate::loader::{Linear, NormKind, SsmWeights};

type Nd = NdArray<f32>;

/// Extract a rank-`D` NdArray<f32> tensor to a row-major `Vec<f32>`.
fn vec_of<const D: usize>(t: &Tensor<Nd, D>) -> Vec<f32> {
    t.clone().into_data().to_vec::<f32>().unwrap()
}

/// A `Linear` extracted to host arrays: `weight` (in,out) row-major + optional `bias` (out).
struct HostLinear {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    din: usize,
    dout: usize,
}

impl HostLinear {
    fn from(l: &Linear<Nd>) -> Self {
        let [din, dout] = l.weight.dims();
        HostLinear {
            w: vec_of(&l.weight),
            b: l.bias.as_ref().map(vec_of),
            din,
            dout,
        }
    }

    /// `y[o] = bias[o] + Σ_i x[i]·W[i,o]` for one length-`din` input row → length-`dout` output.
    /// Written as a `din` sequence of AXPY updates over the contiguous `(dout)` weight rows so the
    /// inner `o`-loop is a clean fma the compiler auto-vectorizes (simd128 on wasm); no per-`i`
    /// branch (a `xi==0` early-out kills vectorization and dense f32 activations are ~never 0).
    #[inline]
    fn apply_row(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.din);
        debug_assert_eq!(y.len(), self.dout);
        let dout = self.dout;
        match &self.b {
            Some(b) => y.copy_from_slice(b),
            None => y.iter_mut().for_each(|v| *v = 0.0),
        }
        for i in 0..self.din {
            let xi = x[i];
            let row = &self.w[i * dout..i * dout + dout];
            for o in 0..dout {
                y[o] += xi * row[o];
            }
        }
    }
}

/// One inner SSM's weights, extracted to host arrays.
struct HostSsm {
    conv: Vec<f32>, // (k,1,D) row-major  → conv[i*D + d]
    k: usize,
    bc: HostLinear,      // D → 2N
    dt: HostLinear,      // D → R
    dt_proj: HostLinear, // R → D
    a: Vec<f32>,         // (D,N) = -exp(A_log)
    d_skip: Vec<f32>,    // (D,)
    n: usize,
    reverse: bool,
}

impl HostSsm {
    fn from(w: &SsmWeights<Nd>, n: usize, reverse: bool) -> Self {
        // a = -exp(clamp(a_log,-20,20)) ; a_log is (D,N).
        let a: Vec<f32> = vec_of(&w.a_log)
            .into_iter()
            .map(|v| -(v.clamp(-20.0, 20.0).exp()))
            .collect();
        let [k, _one, _d] = w.conv.dims();
        HostSsm {
            conv: vec_of(&w.conv),
            k,
            bc: HostLinear::from(&w.bc),
            dt: HostLinear::from(&w.dt),
            dt_proj: HostLinear::from(w.dt_proj.as_ref().expect("dt_proj required in fused path")),
            a,
            d_skip: vec_of(&w.d),
            n,
            reverse,
        }
    }
}

/// Numerically-stable softplus `max(x,0)+log(1+exp(-|x|))`, matching `selective_ssm::softplus`.
#[inline]
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (1.0 + fast_exp(-x.abs())).ln()
}

/// SiLU `x·sigmoid(x)`, matching `selective_ssm::activation("silu", ·)`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + fast_exp(-x))
}

/// Run one inner selective SSM over a `(B,L,D)` host buffer `x` (row-major, `x[(b*L+t)*D+d]`),
/// producing `y` in the same layout.  REAL mode (no rotary).  Mirrors [`crate::selective_ssm::
/// forward_ssm`] exactly for the scoped config: (reverse-flip) → causal_conv1d(k) → silu →
/// bc/dt projections → softplus dt → sequential scan → (flip back) → `y + u·d` (with the
/// flipped-order `u`, matching the JAX/generic code).
fn run_ssm(ssm: &HostSsm, x: &[f32], bb: usize, l: usize, d: usize) -> Vec<f32> {
    let n = ssm.n;
    let k = ssm.k;

    // Anti-causal mode: process the sequence back-to-front.  We map an internal step `t`
    // (0..L) to the source position `sp(t)`.  In forward mode sp(t)=t; in reverse mode the
    // sequence is flipped up front (sp(t)=L-1-t), the whole SSM runs on the flipped order, and
    // the output (plus the D-skip's `u`) is flipped back at the end.  Emulating the flip by an
    // index map avoids materialising the flipped copy.
    let src = |t: usize| if ssm.reverse { l - 1 - t } else { t };

    // Per-(b) working buffers reused across the length loop.
    let mut u_row = vec![0f32; d]; // post conv+silu, at internal step t
    let mut bc_out = vec![0f32; 2 * n];
    let mut dt_lr = vec![0f32; ssm.dt.dout]; // low-rank dt (width R)
    let mut dt_row = vec![0f32; d]; // dt after dt_proj + softplus
    let mut hstate = vec![0f32; d * n]; // carried SSM state (D,N)
    // The conv needs the last (k-1) conv-input rows; we recompute the conv input on the fly by
    // walking source positions, so keep a small ring of the last k activated? No — the conv is
    // over the RAW (post-flip) input x, activation is applied AFTER the conv.  So we need the
    // last k raw input rows (in internal order).  Store them in a ring of k rows of width d.
    let mut ring = vec![0f32; k * d]; // ring[(t mod k)*d + di] = raw x at internal step t

    // We also need `u` (post conv+silu) at every internal step to (a) feed the scan and (b) add
    // the D-skip in flipped order.  The D-skip in reverse mode uses the flipped-order u and is
    // NOT flipped back (matching the generic code), so we accumulate y in INTERNAL (flipped)
    // order and only remap to source order via `src` when writing the scan's y — but the skip
    // `u·d` is added in internal order too.  Net: write everything at index src(t).
    let mut y = vec![0f32; bb * l * d];

    for bi in 0..bb {
        hstate.iter_mut().for_each(|v| *v = 0.0);
        ring.iter_mut().for_each(|v| *v = 0.0);
        for t in 0..l {
            // Load the raw input row at internal step t (source position src(t)) into the ring.
            let sp = src(t);
            let in_off = (bi * l + sp) * d;
            let rslot = (t % k) * d;
            ring[rslot..rslot + d].copy_from_slice(&x[in_off..in_off + d]);

            // causal_conv1d: u_raw[t,di] = Σ_{i=0..k-1} conv[i,di]·xin[t-(k-1)+i, di]
            //             = Σ_{j=0..k-1} conv[k-1-j, di]·xin[t-j, di]  (j = k-1-i)
            // xin[t-j] is the ring row written j steps ago; xin[<0]=0.
            for di in 0..d {
                let mut acc = 0f32;
                for j in 0..k {
                    if j > t {
                        break; // xin[t-j] is before the sequence start → 0
                    }
                    let conv_i = k - 1 - j; // conv kernel index
                    let cval = ssm.conv[conv_i * d + di];
                    let rr = ((t - j) % k) * d + di;
                    acc += cval * ring[rr];
                }
                // silu activation on the conv output.
                u_row[di] = silu(acc);
            }

            // Projections from u.
            ssm.bc.apply_row(&u_row, &mut bc_out); // (2N,)
            ssm.dt.apply_row(&u_row, &mut dt_lr); // (R,)
            ssm.dt_proj.apply_row(&dt_lr, &mut dt_row); // (D,)
            for v in dt_row.iter_mut() {
                *v = softplus(*v);
            }

            // Sequential scan step + D-skip.  b_proj = bc_out[0..n], c_proj = bc_out[n..2n].
            // Ordering subtlety (matches the generic forward / JAX exactly): the SCAN runs in
            // internal (post-flip) order and its output is flipped BACK, so the scan term for
            // internal step t lands at source position sp=src(t).  But the D-skip `u·d` uses the
            // flipped-order `u` and is NOT flipped back (JAX adds it after the flip-back), so the
            // skip term for internal step t lands at output position t (NOT sp).  Forward mode:
            // sp==t, so the two coincide.  Reverse mode: they differ (the deliberate asymmetry).
            let scan_off = (bi * l + sp) * d; // where the scan term is written
            let skip_off = (bi * l + t) * d; // where the D-skip term is written
            for di in 0..d {
                let u_v = u_row[di];
                let dt_v = dt_row[di];
                let hoff = di * n;
                let aoff = di * n;
                let mut acc = 0f32;
                for ni in 0..n {
                    let da = fast_exp(ssm.a[aoff + ni] * dt_v);
                    let db = bc_out[ni] * u_v * dt_v;
                    let hh = da * hstate[hoff + ni] + db;
                    hstate[hoff + ni] = hh;
                    acc += bc_out[n + ni] * hh;
                }
                y[scan_off + di] += acc;
                y[skip_off + di] += u_v * ssm.d_skip[di];
            }
        }
    }
    y
}

/// Fused host forward of the whole [`BidirectionalMamba`] block on the `NdArray<f32>` backend,
/// for the scoped config (real / rms / silu / no-mlp / dt_proj / concatenate / untied).  Falls
/// back to the generic [`BidirectionalMamba::forward`] for any config outside that envelope, so
/// it is always a safe drop-in.  `x`/output are `(B,L,D_in)`.
pub fn bidirectional_forward_fused_ndarray(
    block: &BidirectionalMamba<Nd>,
    x: Tensor<Nd, 3>,
) -> Tensor<Nd, 3> {
    let cfg = &block.cfg;
    // Fast-path eligibility (matches the deployed move-type config).  Anything else → generic.
    let eligible = !cfg.use_complex_ssm
        && !cfg.mlp_layer
        && cfg.dt_proj
        && cfg.concatenate_fwd_rev
        && !cfg.tie_in_proj
        && !cfg.tie_gate
        && !cfg.complement
        && cfg.activation == "silu"
        && block.weights.norm.kind == NormKind::Rms;
    if !eligible {
        return block.forward(x);
    }

    let dev = x.device();
    let [bb, l, d_in] = x.dims();
    let ed = (cfg.expansion_factor * d_in as f64).ceil() as usize;
    let n = cfg.hidden_features;
    let w = &block.weights;

    let xv = vec_of(&x);

    // --- rms_norm over the last axis: x·rsqrt(mean(x²)+eps)·scale ---
    let scale = vec_of(w.norm.scale.as_ref().expect("rms scale"));
    const NORM_EPS: f32 = 1e-6;
    let mut xn = vec![0f32; bb * l * d_in];
    for row in 0..bb * l {
        let off = row * d_in;
        let mut ms = 0f32;
        for di in 0..d_in {
            let v = xv[off + di];
            ms += v * v;
        }
        ms /= d_in as f32;
        let inv = 1.0 / (ms + NORM_EPS).sqrt();
        for di in 0..d_in {
            xn[off + di] = xv[off + di] * inv * scale[di];
        }
    }

    // --- in_proj: (B,L,d_in) → (B,L, 4·ed), split [xf, xr, zf, zr] each width ed ---
    let in_proj = HostLinear::from(&w.in_proj);
    debug_assert_eq!(in_proj.dout, 4 * ed);
    let mut proj_row = vec![0f32; in_proj.dout];
    let mut xf = vec![0f32; bb * l * ed];
    let mut xr = vec![0f32; bb * l * ed];
    let mut zf = vec![0f32; bb * l * ed];
    let mut zr = vec![0f32; bb * l * ed];
    for row in 0..bb * l {
        in_proj.apply_row(&xn[row * d_in..(row + 1) * d_in], &mut proj_row);
        let o = row * ed;
        xf[o..o + ed].copy_from_slice(&proj_row[0..ed]);
        xr[o..o + ed].copy_from_slice(&proj_row[ed..2 * ed]);
        zf[o..o + ed].copy_from_slice(&proj_row[2 * ed..3 * ed]);
        zr[o..o + ed].copy_from_slice(&proj_row[3 * ed..4 * ed]);
    }

    // --- forward + reverse selective SSMs at width ed ---
    let ssm_fwd = HostSsm::from(&w.ssm_fwd, n, false);
    let ssm_rev = HostSsm::from(&w.ssm_rev, n, true);
    let yf = run_ssm(&ssm_fwd, &xf, bb, l, ed);
    let yr = run_ssm(&ssm_rev, &xr, bb, l, ed);

    // --- gated silu + concat → (B,L, 2·ed) ---
    let two_ed = 2 * ed;
    let mut combined = vec![0f32; bb * l * two_ed];
    for row in 0..bb * l {
        let so = row * ed;
        let co = row * two_ed;
        for e in 0..ed {
            combined[co + e] = yf[so + e] * silu(zf[so + e]);
            combined[co + ed + e] = yr[so + e] * silu(zr[so + e]);
        }
    }

    // --- out_proj: (B,L, 2·ed) → (B,L, d_in), then residual (skip = original x) ---
    let out_proj = HostLinear::from(&w.out_proj);
    debug_assert_eq!(out_proj.din, two_ed);
    debug_assert_eq!(out_proj.dout, d_in);
    let mut out = vec![0f32; bb * l * d_in];
    let mut out_row = vec![0f32; d_in];
    for row in 0..bb * l {
        out_proj.apply_row(&combined[row * two_ed..(row + 1) * two_ed], &mut out_row);
        let o = row * d_in;
        for di in 0..d_in {
            out[o + di] = xv[o + di] + out_row[di];
        }
    }

    Tensor::from_data(TensorData::new(out, [bb, l, d_in]), &dev)
}
