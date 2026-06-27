//! Rematerializing chunked scan: a custom `burn` autodiff op that recomputes each chunk's
//! intermediates in the backward pass instead of retaining them on the tape.
//!
//! The reference [`crate::chunked_scan::ssm_chunked_scan`] builds the forward out of stock
//! burn ops, so every chunk's `(cs, cs, B, D, N)` decay matrix (and the rest) stays live on
//! the autodiff graph until backward — peak memory grows like `O(L · cs · B · D · N)`.  This
//! is the Rust analogue of the JAX side's `@jax.remat` / `custom_vjp` strategies.
//!
//! Here the whole scan is registered as a single custom op:
//!
//! * **Forward** runs the chunk loop on the *inner* (non-autodiff) backend, so nothing is
//!   recorded.  It keeps only the inputs and the small per-chunk boundary carries
//!   `(hstate, angle_acc)` — `O(L · (D+N)) + O(n_chunks · B · D · N)`, with no `cs²` term.
//! * **Backward** walks the chunks in reverse and, for each one, rebuilds *just that chunk's*
//!   forward under a fresh nested `Autodiff` graph, backpropagates the incoming output- and
//!   carry-gradients through it, and stitches the carry gradient to the previous chunk.  Burn
//!   computes the per-chunk VJP, so there is no hand-derived gradient math; peak extra memory
//!   is one chunk's graph at a time.
//!
//! Numerically this is identical to the reference scan (same per-chunk `scan_chunk_body`), so
//! it passes the same cross-language parity fixtures.  Dispatch goes through [`ScanBackend`]:
//! plain backends and `use_remat=false` use the reference; an autodiff backend with
//! `use_remat=true` uses this op.

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorPrimitive};

use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::checkpoint::strategy::CheckpointStrategy;
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};

use crate::chunked_scan::{
    apply_rotary, cumsum0_3, cumsum0_4, pad_l, rev_assoc_scan0, scan_chunk_body, ssm_chunked_scan,
    tril_ones, within_chunk_hs,
};
use crate::config::ScanAlgo;

/// Options controlling the chunked scan, bundled to keep [`ScanBackend::chunked_scan`] tidy as
/// more backends/algorithms are added.
#[derive(Debug, Clone, Copy)]
pub struct ScanOpts {
    pub chunk_size: usize,
    /// Rematerialize the backward pass (autodiff backends only).
    pub use_remat: bool,
    /// Within-chunk algorithm.
    pub algo: ScanAlgo,
}

/// Backend extension selecting the chunked-scan implementation.  Plain backends (and any
/// `use_remat=false` call) run the reference scan; the `Autodiff` impl additionally offers
/// the rematerializing custom op.
pub trait ScanBackend: Backend {
    /// `(B,L,D) → (B,L,D)` selective scan.  `b`/`c` are the raw (pre-rotary) projections;
    /// `theta` enables the complex-SSM RoPE trick.  `opts.use_remat` is honoured only under an
    /// autodiff backend (it is a no-op for inference, where there is no tape to shrink).
    #[allow(clippy::too_many_arguments)]
    fn chunked_scan(
        opts: ScanOpts,
        u: Tensor<Self, 3>,
        a: Tensor<Self, 2>,
        b: Tensor<Self, 3>,
        c: Tensor<Self, 3>,
        dt: Tensor<Self, 3>,
        theta: Option<Tensor<Self, 3>>,
    ) -> Tensor<Self, 3>;

    /// Within-chunk inclusive scan `hs = scan(da, db, hstate)` for [`ScanAlgo::Cubecl`], where
    /// `da` is the per-step decay `exp(a·dt)`.  The default is the portable Hillis–Steele scan;
    /// the wgpu backend overrides it with a custom GPU kernel (when built `--features cubecl`).
    fn within_chunk_scan_cubecl(
        da: Tensor<Self, 4>,
        db: Tensor<Self, 4>,
        hstate: Tensor<Self, 4>,
    ) -> Tensor<Self, 4> {
        let (a_cum, b_scan) = crate::chunked_scan::assoc_scan0(da, db);
        a_cum * hstate + b_scan
    }

    /// Reverse adjoint scan `bar_h[t] = Σ_{p≥t} (∏_{k=t+1..p} da[k]) src[p]` for the backward
    /// pass of [`ScanAlgo::Cubecl`].  Default is the portable reverse Hillis–Steele scan; the
    /// wgpu backend overrides it with the reverse GPU kernel.
    fn within_chunk_rev_scan_cubecl(da: Tensor<Self, 4>, src: Tensor<Self, 4>) -> Tensor<Self, 4> {
        crate::chunked_scan::rev_assoc_scan0(da, src)
    }
}

/// Reference implementation for the concrete (non-autodiff) backends used by this crate.
macro_rules! impl_scan_backend_reference {
    ($backend:ty) => {
        impl ScanBackend for $backend {
            fn chunked_scan(
                opts: ScanOpts,
                u: Tensor<Self, 3>,
                a: Tensor<Self, 2>,
                b: Tensor<Self, 3>,
                c: Tensor<Self, 3>,
                dt: Tensor<Self, 3>,
                theta: Option<Tensor<Self, 3>>,
            ) -> Tensor<Self, 3> {
                ssm_chunked_scan(u, a, b, c, dt, theta, opts.chunk_size, opts.algo)
            }
        }
    };
}

impl_scan_backend_reference!(burn::backend::NdArray);

// The wgpu backend gets the reference `chunked_scan` plus, when built `--features cubecl`
// (and not `fusion`, which would change the backend type), the custom GPU scan kernel.
#[cfg(feature = "wgpu")]
impl ScanBackend for burn::backend::Wgpu {
    fn chunked_scan(
        opts: ScanOpts,
        u: Tensor<Self, 3>,
        a: Tensor<Self, 2>,
        b: Tensor<Self, 3>,
        c: Tensor<Self, 3>,
        dt: Tensor<Self, 3>,
        theta: Option<Tensor<Self, 3>>,
    ) -> Tensor<Self, 3> {
        ssm_chunked_scan(u, a, b, c, dt, theta, opts.chunk_size, opts.algo)
    }

    #[cfg(all(feature = "cubecl", not(feature = "fusion")))]
    fn within_chunk_scan_cubecl(
        da: Tensor<Self, 4>,
        db: Tensor<Self, 4>,
        hstate: Tensor<Self, 4>,
    ) -> Tensor<Self, 4> {
        selectssm_cubecl::within_chunk_scan(da, db, hstate)
    }

    #[cfg(all(feature = "cubecl", not(feature = "fusion")))]
    fn within_chunk_rev_scan_cubecl(da: Tensor<Self, 4>, src: Tensor<Self, 4>) -> Tensor<Self, 4> {
        selectssm_cubecl::within_chunk_rev_scan(da, src)
    }
}

/// Per-chunk boundary carry saved by the forward pass: the SSM state and rotation
/// accumulator entering chunk `c` (both small).
type Carry<B> = (Tensor<B, 4>, Tensor<B, 3>);

/// State stashed by the forward op for the backward pass.  Holds the (padded, length-major)
/// inputs and the per-chunk incoming carries — but none of the heavy `cs²` intermediates.
#[derive(Clone, Debug)]
struct RematState<B: Backend> {
    u: Tensor<B, 3>,             // (Lp, B, D)
    a: Tensor<B, 2>,             // (D, N)
    b: Tensor<B, 3>,             // (Lp, B, N) raw
    c: Tensor<B, 3>,             // (Lp, B, N) raw
    dt: Tensor<B, 3>,            // (Lp, B, D)
    theta: Option<Tensor<B, 3>>, // (Lp, B, H) when complex
    carries: Vec<Carry<B>>,      // incoming (hstate, angle) per chunk
    cs: usize,
    l: usize,
    pad: usize,
    n_chunks: usize,
    bb: usize,
    d: usize,
    n: usize,
    h: usize,
    use_complex: bool,
    algo: ScanAlgo,
}

/// Run the chunk loop on a plain backend, returning the (length-major) output and the
/// incoming boundary carry for every chunk.  Mirrors `ssm_chunked_scan` but keeps the
/// carries so the backward pass can replay each chunk independently.
#[allow(clippy::too_many_arguments)]
fn scan_collect<B: ScanBackend>(
    u_lm: &Tensor<B, 3>,
    a: &Tensor<B, 2>,
    b_lm: &Tensor<B, 3>,
    c_lm: &Tensor<B, 3>,
    dt_lm: &Tensor<B, 3>,
    theta_lm: &Option<Tensor<B, 3>>,
    cs: usize,
    n_chunks: usize,
    algo: ScanAlgo,
) -> (Tensor<B, 3>, Vec<Carry<B>>) {
    let device = u_lm.device();
    let [_lp, bb, d] = u_lm.dims();
    let n = a.dims()[1];
    let h = n / 2;

    let tril = tril_ones::<B>(cs, &device);
    let mask = tril.clone().reshape([cs, cs, 1, 1, 1]);
    let a4 = a.clone().reshape([1, 1, d, n]);

    let mut hstate = Tensor::<B, 4>::zeros([1, bb, d, n], &device);
    let mut angle = Tensor::<B, 3>::zeros([1, bb, h], &device);
    let mut carries = Vec::with_capacity(n_chunks);
    let mut y_chunks = Vec::with_capacity(n_chunks);

    for c in 0..n_chunks {
        let s = c * cs;
        carries.push((hstate.clone(), angle.clone()));
        let u_c = u_lm.clone().slice([s..s + cs, 0..bb, 0..d]);
        let b_c = b_lm.clone().slice([s..s + cs, 0..bb, 0..n]);
        let c_c = c_lm.clone().slice([s..s + cs, 0..bb, 0..n]);
        let dt_c = dt_lm.clone().slice([s..s + cs, 0..bb, 0..d]);
        let theta_c = theta_lm.as_ref().map(|t| t.clone().slice([s..s + cs, 0..bb, 0..h]));
        let (y_c, hn, an) = scan_chunk_body(
            &tril, &mask, &a4, u_c, b_c, c_c, dt_c, theta_c, hstate, angle, algo,
        );
        y_chunks.push(y_c);
        hstate = hn;
        angle = an;
    }
    (Tensor::cat(y_chunks, 0), carries)
}

/// Reverse inclusive cumulative sum along dim 0 of a (cs,B,H) tensor: `r[t] = sum_{s>=t} v[s]`.
/// Realised as `triuᵀ @ v`, the transpose of the forward inclusive cumsum.
fn rcumsum0_3<B: Backend>(tril: &Tensor<B, 2>, v: Tensor<B, 3>) -> Tensor<B, 3> {
    let [cs, b, h] = v.dims();
    tril.clone().transpose().matmul(v.reshape([cs, b * h])).reshape([cs, b, h])
}

/// Vector-Jacobian product of [`apply_rotary`] for one (cs,B,N) projection.  Given the raw
/// (pre-rotary) `v_raw`, the angle `phi` (cs,B,H), and the upstream grad `bar_out` of the
/// rotated result, returns `(grad_v_raw (cs,B,N), grad_phi (cs,B,H))`.
fn rotary_vjp<B: Backend>(
    v_raw: &Tensor<B, 3>,
    phi: &Tensor<B, 3>,
    bar_out: Tensor<B, 3>,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let [cs, b, nn] = v_raw.dims();
    let h = nn / 2;
    let cos = phi.clone().cos();
    let sin = phi.clone().sin();
    let bar_re = bar_out.clone().slice([0..cs, 0..b, 0..h]);
    let bar_im = bar_out.slice([0..cs, 0..b, h..nn]);
    // R(-phi) is orthogonal: grad wrt the raw v is R(+phi) applied to bar_out.
    let g_re = bar_re.clone() * cos.clone() - bar_im.clone() * sin.clone();
    let g_im = bar_re.clone() * sin.clone() + bar_im.clone() * cos.clone();
    let grad_v = Tensor::cat(vec![g_re, g_im], 2);
    // d out_re/d phi = out_im,  d out_im/d phi = -out_re, with out = R(-phi) v_raw.
    let v_re = v_raw.clone().slice([0..cs, 0..b, 0..h]);
    let v_im = v_raw.clone().slice([0..cs, 0..b, h..nn]);
    let out_re = v_re.clone() * cos.clone() + v_im.clone() * sin.clone();
    let out_im = v_re * sin.neg() + v_im * cos;
    let grad_phi = bar_re * out_im - bar_im * out_re;
    (grad_v, grad_phi)
}

/// Grads w.r.t. one chunk's inputs and incoming carry, given the upstream output grad
/// `gy` (cs,B,D) and the incoming carry grads `gh` (1,B,D,N) / `ga` (1,B,H).  Hand-derived
/// VJP of [`scan_chunk_body`] (the matrix-form linear SSM + RoPE rotary), recomputing the
/// forward intermediates — no nested autodiff (burn's runtime forbids reentrant backward).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn chunk_vjp<B: ScanBackend>(
    tril: &Tensor<B, 2>,
    mask: &Tensor<B, 5>,
    a4: &Tensor<B, 4>,
    u_c: &Tensor<B, 3>,
    b_raw: &Tensor<B, 3>,
    c_raw: &Tensor<B, 3>,
    dt_c: &Tensor<B, 3>,
    theta_c: &Option<Tensor<B, 3>>,
    hstate: &Tensor<B, 4>,
    angle_acc: &Tensor<B, 3>,
    gy: Tensor<B, 3>,
    gh: Tensor<B, 4>,
    ga: Tensor<B, 3>,
    algo: ScanAlgo,
) -> ChunkGrads<B> {
    let [cs, bb, d] = u_c.dims();
    let n = a4.dims()[3];
    let h = n / 2;

    // --- recompute forward intermediates (matches scan_chunk_body) ---
    let (b_c, c_c, phi) = match theta_c {
        Some(theta_c) => {
            let phi = cumsum0_3(tril, theta_c.clone()) + angle_acc.clone(); // (cs,B,H)
            (apply_rotary(b_raw.clone(), &phi), apply_rotary(c_raw.clone(), &phi), Some(phi))
        }
        None => (b_raw.clone(), c_raw.clone(), None),
    };
    let a_dt = a4.clone() * dt_c.clone().reshape([cs, bb, d, 1]);
    let al = a_dt.clone().exp(); // per-step decay dA = exp(a*dt)
    let d_b = b_c.clone().reshape([cs, bb, 1, n])
        * u_c.clone().reshape([cs, bb, d, 1])
        * dt_c.clone().reshape([cs, bb, d, 1]);
    let hs = within_chunk_hs(tril, mask, a_dt.clone(), d_b, hstate.clone(), algo); // (cs,B,D,N)
    // h_prev[t] = hs[t-1], with h_prev[0] = hstate.
    let h_prev = Tensor::cat(
        vec![hstate.clone(), hs.clone().slice([0..cs - 1, 0..bb, 0..d, 0..n])],
        0,
    );

    // --- adjoint ---
    // src[p] = gy[p] ⊗ c[p]; the outgoing-state grad gh adds into the last step.
    let mut src = gy.clone().reshape([cs, bb, d, 1]) * c_c.clone().reshape([cs, bb, 1, n]);
    let last = src.clone().slice([cs - 1..cs, 0..bb, 0..d, 0..n]) + gh;
    src = src.slice_assign([cs - 1..cs, 0..bb, 0..d, 0..n], last);
    // bar_h[t] = Σ_{p≥t} (∏_{k=t+1..p} dA[k]) src[p]: the adjoint of the state recurrence,
    // computed by the same algorithm as the forward (transpose of the decay matrix, or a
    // reverse Hillis–Steele scan).
    let bar_h = match algo {
        ScanAlgo::Matrix => {
            let a_cumsum = cumsum0_4(tril, a_dt);
            let cp = a_cumsum.clone().reshape([cs, 1, bb, d, n]);
            let cj = a_cumsum.reshape([1, cs, bb, d, n]);
            let lmat = (cp - cj).clamp_max(0.0).exp() * mask.clone(); // (cs,cs,B,D,N), L[p,j]
            (lmat * src.reshape([cs, 1, bb, d, n])).sum_dim(0).reshape([cs, bb, d, n])
        }
        ScanAlgo::Hillis => rev_assoc_scan0(al.clone(), src),
        // Cubecl uses the reverse GPU kernel for the adjoint scan (falls back to the reverse
        // Hillis scan off-GPU / without the feature).
        ScanAlgo::Cubecl => B::within_chunk_rev_scan_cubecl(al.clone(), src),
    };

    let bar_db = bar_h.clone(); // grad wrt d_b (h[t] = ... + d_b[t], coefficient 1)
    let bar_a_dt = bar_h.clone() * h_prev * al.clone(); // bar_al * al, bar_al = bar_h * h_prev
    let bar_hstate = (al.slice([0..1, 0..bb, 0..d, 0..n])
        * bar_h.slice([0..1, 0..bb, 0..d, 0..n]))
    .reshape([1, bb, d, n]); // grad wrt incoming hstate (= al[0] * bar_h[0])

    let u4 = u_c.clone().reshape([cs, bb, d, 1]);
    let dt4 = dt_c.clone().reshape([cs, bb, d, 1]);
    let bc4 = b_c.reshape([cs, bb, 1, n]); // rotated B

    // c (rotated) grad from y = sum_n c * hs.
    let bar_c_rot = (gy.reshape([cs, bb, d, 1]) * hs).sum_dim(2).reshape([cs, bb, n]);
    // d_b = b_rot * u * dt: grads wrt b_rot (sum over D), u (sum over N), dt (sum over N).
    let bar_b_rot =
        (bar_db.clone() * u4.clone() * dt4.clone()).sum_dim(2).reshape([cs, bb, n]);
    let bar_u = (bar_db.clone() * bc4.clone() * dt4.clone()).sum_dim(3).reshape([cs, bb, d]);
    let bar_dt_from_db = (bar_db * bc4 * u4).sum_dim(3).reshape([cs, bb, d]);
    // dt also enters through a_dt = a * dt (the decay): bar_dt_al = sum_n bar_a_dt * a.
    let bar_dt_from_al = (bar_a_dt.clone() * a4.clone()).sum_dim(3).reshape([cs, bb, d]);
    let bar_dt = bar_dt_from_db + bar_dt_from_al;
    // a[d,n] += sum_{t,b} bar_a_dt[t,b,d,n] * dt[t,b,d].
    let bar_a = (bar_a_dt * dt4).sum_dim(0).sum_dim(1).reshape([d, n]);

    // Rotary / theta path.
    let (bar_b_raw, bar_c_raw, bar_theta, bar_angle) = match (&phi, theta_c) {
        (Some(phi), Some(_)) => {
            let (bb_raw, phi_b) = rotary_vjp(b_raw, phi, bar_b_rot);
            let (bc_raw, phi_c) = rotary_vjp(c_raw, phi, bar_c_rot);
            let bar_phi = phi_b + phi_c; // (cs,B,H)
            // phi[s] = angle_acc + sum_{k<=s} theta[k];  angle_next = angle_acc + sum theta.
            let bar_theta = rcumsum0_3(tril, bar_phi.clone()) + ga.clone(); // +ga from angle_next
            let bar_angle = bar_phi.sum_dim(0).reshape([1, bb, h]) + ga; // grad wrt angle_acc
            (bb_raw, bc_raw, Some(bar_theta), bar_angle)
        }
        _ => (bar_b_rot, bar_c_rot, None, ga),
    };

    ChunkGrads {
        u: bar_u, a: bar_a, b: bar_b_raw, c: bar_c_raw, dt: bar_dt,
        theta: bar_theta, hstate: bar_hstate, angle: bar_angle,
    }
}

/// Per-chunk gradients returned by [`chunk_vjp`].
struct ChunkGrads<B: Backend> {
    u: Tensor<B, 3>,             // (cs,B,D)
    a: Tensor<B, 2>,             // (D,N)
    b: Tensor<B, 3>,             // (cs,B,N)
    c: Tensor<B, 3>,             // (cs,B,N)
    dt: Tensor<B, 3>,            // (cs,B,D)
    theta: Option<Tensor<B, 3>>, // (cs,B,H)
    hstate: Tensor<B, 4>,        // (1,B,D,N) grad wrt incoming state
    angle: Tensor<B, 3>,         // (1,B,H)   grad wrt incoming angle
}

/// The custom backward op for the rematerializing scan.  `B` is the *inner* backend.
#[derive(Debug)]
struct ChunkedScanRemat;

impl<B: ScanBackend> Backward<B, 6> for ChunkedScanRemat {
    type State = RematState<B>;

    fn backward(self, ops: Ops<Self::State, 6>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        let st = ops.state;
        let (cs, bb, d, n, h) = (st.cs, st.bb, st.d, st.n, st.h);
        let device = st.u.device();

        // Upstream gradient of the (B,L,D) output, lifted to length-major padded form.
        let grad_out = grads.consume::<B>(&ops.node);
        let grad_y = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(grad_out));
        let grad_y_lm = pad_l(grad_y, st.pad).swap_dims(0, 1); // (Lp, B, D)

        let tril = tril_ones::<B>(cs, &device);
        let mask = tril.clone().reshape([cs, cs, 1, 1, 1]);
        let a4 = st.a.clone().reshape([1, 1, d, n]);

        // Per-chunk input-gradient slots (filled in reverse, concatenated in order).
        let mut du: Vec<Option<Tensor<B, 3>>> = (0..st.n_chunks).map(|_| None).collect();
        let mut db = du.clone();
        let mut dc = du.clone();
        let mut ddt = du.clone();
        let mut dtheta = du.clone();
        let mut da = Tensor::<B, 2>::zeros([d, n], &device); // a is shared: accumulate

        // Carry gradients flowing right-to-left (grad wrt each chunk's outgoing state/angle).
        let mut gh = Tensor::<B, 4>::zeros([1, bb, d, n], &device);
        let mut ga = Tensor::<B, 3>::zeros([1, bb, h], &device);

        for c in (0..st.n_chunks).rev() {
            let s = c * cs;
            let (h_in, ang_in) = st.carries[c].clone();
            let u_c = st.u.clone().slice([s..s + cs, 0..bb, 0..d]);
            let b_c = st.b.clone().slice([s..s + cs, 0..bb, 0..n]);
            let c_c = st.c.clone().slice([s..s + cs, 0..bb, 0..n]);
            let dt_c = st.dt.clone().slice([s..s + cs, 0..bb, 0..d]);
            let theta_c = st.theta.as_ref().map(|t| t.clone().slice([s..s + cs, 0..bb, 0..h]));
            let gy = grad_y_lm.clone().slice([s..s + cs, 0..bb, 0..d]);

            let g = chunk_vjp(
                &tril, &mask, &a4, &u_c, &b_c, &c_c, &dt_c, &theta_c, &h_in, &ang_in, gy,
                gh, ga, st.algo,
            );
            du[c] = Some(g.u);
            db[c] = Some(g.b);
            dc[c] = Some(g.c);
            ddt[c] = Some(g.dt);
            dtheta[c] = g.theta;
            da = da + g.a;
            // The incoming-carry grad of chunk c is the outgoing-carry grad of chunk c-1.
            gh = g.hstate;
            ga = g.angle;
        }

        // Reassemble (Lp,B,*) → original (B,L,*) layout and register to the parents.
        let to_orig3 = |chunks: Vec<Option<Tensor<B, 3>>>, width: usize| {
            let cat = Tensor::cat(chunks.into_iter().map(|x| x.unwrap()).collect(), 0); // (Lp,B,*)
            cat.swap_dims(0, 1).slice([0..st.bb, 0..st.l, 0..width]) // (B,L,*)
        };
        let du = to_orig3(du, d);
        let db = to_orig3(db, n);
        let dc = to_orig3(dc, n);
        let ddt = to_orig3(ddt, d);

        let [u_p, a_p, b_p, c_p, dt_p, theta_p] = ops.parents;
        let reg3 = |grads: &mut Gradients, p: Option<_>, t: Tensor<B, 3>| {
            if let Some(node) = p {
                grads.register::<B>(node, t.into_primitive().tensor());
            }
        };
        reg3(grads, u_p.map(|p| p.id), du);
        if let Some(node) = a_p {
            grads.register::<B>(node.id, da.into_primitive().tensor());
        }
        reg3(grads, b_p.map(|p| p.id), db);
        reg3(grads, c_p.map(|p| p.id), dc);
        reg3(grads, dt_p.map(|p| p.id), ddt);
        if st.use_complex {
            reg3(grads, theta_p.map(|p| p.id), to_orig3(dtheta, h));
        }
    }
}

impl<B: ScanBackend, C: CheckpointStrategy> ScanBackend for Autodiff<B, C> {
    fn chunked_scan(
        opts: ScanOpts,
        u: Tensor<Self, 3>,
        a: Tensor<Self, 2>,
        b: Tensor<Self, 3>,
        c: Tensor<Self, 3>,
        dt: Tensor<Self, 3>,
        theta: Option<Tensor<Self, 3>>,
    ) -> Tensor<Self, 3> {
        if !opts.use_remat {
            return ssm_chunked_scan(u, a, b, c, dt, theta, opts.chunk_size, opts.algo);
        }

        let [bb, l, d] = u.dims();
        let n = a.dims()[1];
        let h = n / 2;
        let cs = opts.chunk_size;
        let pad = (cs - l % cs) % cs;
        let n_chunks = (l + pad) / cs;
        let use_complex = theta.is_some();

        // Synthesize a constant theta when real so the op always has 6 parents; a non-grad
        // tensor contributes a `None` parent slot (its gradient is never registered).
        let theta_in =
            theta.unwrap_or_else(|| Tensor::zeros([bb, l, h.max(1)], &u.device()));

        // Parent nodes (in the order the backward registers gradients).
        let nodes = [
            u.clone().into_primitive().tensor().node,
            a.clone().into_primitive().tensor().node,
            b.clone().into_primitive().tensor().node,
            c.clone().into_primitive().tensor().node,
            dt.clone().into_primitive().tensor().node,
            theta_in.clone().into_primitive().tensor().node,
        ];

        // Inner (padded, length-major) inputs — the forward runs without a tape.
        let u_lm = pad_l(u.inner(), pad).swap_dims(0, 1);
        let a_in = a.inner();
        let b_lm = pad_l(b.inner(), pad).swap_dims(0, 1);
        let c_lm = pad_l(c.inner(), pad).swap_dims(0, 1);
        let dt_lm = pad_l(dt.inner(), pad).swap_dims(0, 1);
        let theta_lm = use_complex.then(|| pad_l(theta_in.inner(), pad).swap_dims(0, 1));

        let (y_lm, carries) =
            scan_collect(&u_lm, &a_in, &b_lm, &c_lm, &dt_lm, &theta_lm, cs, n_chunks, opts.algo);
        let y_inner = y_lm.swap_dims(0, 1).slice([0..bb, 0..l, 0..d]); // (B,L,D)

        let state = RematState {
            u: u_lm, a: a_in, b: b_lm, c: c_lm, dt: dt_lm, theta: theta_lm,
            carries, cs, l, pad, n_chunks, bb, d, n, h, use_complex, algo: opts.algo,
        };

        let out_prim = y_inner.into_primitive().tensor();
        let out = match ChunkedScanRemat.prepare::<C>(nodes).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish(state, out_prim),
            OpsKind::UnTracked(prep) => prep.finish(out_prim),
        };
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }
}
