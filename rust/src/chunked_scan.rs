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

/// Lower-triangular (inclusive) ones matrix: `L[i,j] = 1` if `i >= j` else `0`.
/// Used both as an inclusive-cumsum operator (`cumsum = L @ v`) and as the causal mask
/// for the within-chunk decay matrix.
fn tril_ones<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut v = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            v[i * n + j] = 1.0;
        }
    }
    Tensor::from_data(TensorData::new(v, [n, n]), device)
}

/// Inclusive cumulative sum along dim 0 of a (cs, B, H) tensor, via `tril @ v`.
fn cumsum0_3<B: Backend>(tril: &Tensor<B, 2>, v: Tensor<B, 3>) -> Tensor<B, 3> {
    let [cs, b, h] = v.dims();
    tril.clone().matmul(v.reshape([cs, b * h])).reshape([cs, b, h])
}

/// Inclusive cumulative sum along dim 0 of a (cs, B, D, N) tensor, via `tril @ v`.
fn cumsum0_4<B: Backend>(tril: &Tensor<B, 2>, v: Tensor<B, 4>) -> Tensor<B, 4> {
    let [cs, b, d, n] = v.dims();
    tril.clone()
        .matmul(v.reshape([cs, b * d * n]))
        .reshape([cs, b, d, n])
}

/// Apply the data-dependent rotary embedding `R(-Phi)` to a (cs, B, N) projection,
/// pairing the first N/2 ("real") entries with the last N/2 ("imag") entries.
fn apply_rotary<B: Backend>(v: Tensor<B, 3>, phi: &Tensor<B, 3>) -> Tensor<B, 3> {
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
fn pad_l<B: Backend>(x: Tensor<B, 3>, pad: usize) -> Tensor<B, 3> {
    if pad == 0 {
        return x;
    }
    let [b, _l, c] = x.dims();
    let zeros = Tensor::zeros([b, pad, c], &x.device());
    Tensor::cat(vec![x, zeros], 1)
}

/// Chunked selective-scan.
///
/// * `u`      (B, L, D)   --- SSM input (post conv + activation)
/// * `a`      (D, N)      --- transition coefficients (already `-exp(A_log)`, possibly tied)
/// * `b_proj` (B, L, N)   --- B projection
/// * `c_proj` (B, L, N)   --- C projection
/// * `dt`     (B, L, D)   --- (already softplus'd) step sizes
/// * `theta`  Option<(B, L, N/2)> --- per-step rotation angle for the complex-SSM RoPE trick
///
/// Returns the SSM output (B, L, D) (the `D`-skip term is added by the caller).
pub fn ssm_chunked_scan<B: Backend>(
    u: Tensor<B, 3>,
    a: Tensor<B, 2>,
    b_proj: Tensor<B, 3>,
    c_proj: Tensor<B, 3>,
    dt: Tensor<B, 3>,
    theta: Option<Tensor<B, 3>>,
    chunk_size: usize,
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
        let mut b_c = b_proj.clone().slice([s..s + cs, 0..bb, 0..n]); // (cs,B,N)
        let mut c_c = c_proj.clone().slice([s..s + cs, 0..bb, 0..n]); // (cs,B,N)
        let dt_c = dt.clone().slice([s..s + cs, 0..bb, 0..d]); // (cs,B,D)

        // Complex-SSM RoPE trick: cumulative angle = carried offset + within-chunk cumsum.
        if let Some(theta_l) = &theta {
            let theta_c = theta_l.clone().slice([s..s + cs, 0..bb, 0..h]); // (cs,B,H)
            let phi = cumsum0_3(&tril, theta_c.clone()) + angle_acc.clone(); // (cs,B,H)
            b_c = apply_rotary(b_c, &phi);
            c_c = apply_rotary(c_c, &phi);
            // carry the running total to the next chunk
            let total = theta_c.sum_dim(0); // (1,B,H)
            angle_acc = angle_acc + total;
        }

        // Per-step log-decay a*dt and its inclusive cumulative sum within the chunk.
        let a_dt = a4.clone() * dt_c.clone().reshape([cs, bb, d, 1]); // (cs,B,D,N)
        let a_cumsum = cumsum0_4(&tril, a_dt); // (cs,B,D,N)
        let gs = a_cumsum.clone().exp(); // inclusive product of dA

        // dB = (rotated B) * u * dt
        let d_b = b_c.reshape([cs, bb, 1, n])
            * u_c.reshape([cs, bb, d, 1])
            * dt_c.reshape([cs, bb, d, 1]); // (cs,B,D,N)

        // Within-chunk decay matrix L[p,j] = exp(A_cumsum[p] - A_cumsum[j]) * 1[p>=j].
        let cp = a_cumsum.clone().reshape([cs, 1, bb, d, n]);
        let cj = a_cumsum.reshape([1, cs, bb, d, n]);
        let lmat = (cp - cj).exp() * mask.clone(); // (cs,cs,B,D,N)
        let intra = (lmat * d_b.reshape([1, cs, bb, d, n])).sum_dim(1); // (cs,1,B,D,N)
        let intra = intra.reshape([cs, bb, d, n]);

        // hs = gs * h_init + intra ; y = sum_n C * hs
        let hs = gs * hstate.clone() + intra; // broadcasts h_init (1,B,D,N)
        let y_c = (c_c.reshape([cs, bb, 1, n]) * hs.clone()).sum_dim(3); // (cs,B,D,1)
        y_chunks.push(y_c.reshape([cs, bb, d]));

        hstate = hs.slice([cs - 1..cs, 0..bb, 0..d, 0..n]); // (1,B,D,N)
    }

    let y = Tensor::cat(y_chunks, 0); // (Lp, B, D)
    let y = y.swap_dims(0, 1); // (B, Lp, D)
    y.slice([0..bb, 0..l, 0..d])
}
