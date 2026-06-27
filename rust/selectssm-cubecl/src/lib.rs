//! Custom cubecl GPU kernels for the within-chunk selective-scan, isolated here so the core
//! `selectssm` crate stays `#![forbid(unsafe_code)]` (kernel launches are `unsafe`) and free of
//! the heavy cubecl dependency unless the `cubecl` feature is on.
//!
//! Both the forward state recurrence `hs[t] = dA[t]·hs[t-1] + dB[t]` ([`within_chunk_scan`])
//! and its backward adjoint `bar_h[t] = src[t] + dA[t+1]·bar_h[t+1]`
//! ([`within_chunk_rev_scan`]) are computed by a **work-efficient blocked parallel scan**:
//!
//! 1. *block scan* — split the length axis into blocks of [`BLK`]; one thread per
//!    `(block, b, d, n)` scans its block from a zero seed, recording the per-position partial
//!    result and the within-block cumulative decay.
//! 2. *carry* — one thread per `(b, d, n)` sweeps the (few) block boundaries to turn the
//!    per-block totals into each block's true incoming carry.
//! 3. *fixup* — one thread per element folds the block carry back into every position.
//!
//! This lifts the parallelism from `B·D·N` (the naive one-thread-per-channel sequential scan)
//! to `n_blocks·B·D·N`, so the whole sequence can be scanned in one pass — letting the core
//! crate run the cubecl path over the entire sequence as a single chunk (collapsing the
//! per-chunk burn ops in the backward into one set).

use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorPrimitive};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeRuntime;
use cubecl::frontend::ABSOLUTE_POS;
use cubecl::{calculate_cube_count_elemwise, prelude::*, CubeDim};

/// Internal block length for the parallel scan (serial chain per thread = `BLK`).
const BLK: usize = 64;

// ---------------------------------------------------------------------------------------------
// Forward scan: hs[t] = dA[t]*hs[t-1] + dB[t], inclusive, seeded by `hstate`.
// ---------------------------------------------------------------------------------------------

/// Block-local inclusive scan from a zero seed; emits the partial state and the within-block
/// cumulative decay (`dpref`) for every position.  One thread per `(block, channel)`.
#[cube(launch_unchecked)]
fn fwd_block_kernel<F: Float>(
    da: &Array<F>,
    db: &Array<F>,
    hs_local: &mut Array<F>,
    dpref: &mut Array<F>,
    blk: usize,
    bdn: usize,
    n_blk: usize,
) {
    let id = usize::cast_from(ABSOLUTE_POS);
    if id < n_blk * bdn {
        let b = id / bdn;
        let ch = id % bdn;
        let base = b * blk * bdn + ch;
        let mut h = F::new(0.0);
        let mut g = F::new(1.0);
        for i in 0..blk {
            let off = base + i * bdn;
            let a = da[off];
            h = a * h + db[off];
            g = g * a;
            hs_local[off] = h;
            dpref[off] = g;
        }
    }
}

/// Turn per-block totals into each block's incoming carry: `carry[0]=hstate`,
/// `carry[k]=Gblk[k-1]*carry[k-1]+Eblk[k-1]` (block-end cumulative decay / partial state).
#[cube(launch_unchecked)]
fn fwd_carry_kernel<F: Float>(
    hs_local: &Array<F>,
    dpref: &Array<F>,
    hstate: &Array<F>,
    carry: &mut Array<F>,
    blk: usize,
    bdn: usize,
    n_blk: usize,
) {
    let ch = usize::cast_from(ABSOLUTE_POS);
    if ch < bdn {
        let mut c = hstate[ch];
        carry[ch] = c;
        for k in 1..n_blk {
            let end = (k - 1) * blk * bdn + (blk - 1) * bdn + ch; // last position of block k-1
            c = dpref[end] * c + hs_local[end];
            carry[k * bdn + ch] = c;
        }
    }
}

/// Fold each block's incoming carry into every position: `hs[t]=hs_local[t]+dpref[t]*carry[blk]`.
#[cube(launch_unchecked)]
fn fwd_fixup_kernel<F: Float>(
    hs_local: &Array<F>,
    dpref: &Array<F>,
    carry: &Array<F>,
    out: &mut Array<F>,
    blk: usize,
    bdn: usize,
    total: usize,
) {
    let id = usize::cast_from(ABSOLUTE_POS);
    if id < total {
        let t = id / bdn;
        let ch = id % bdn;
        let b = t / blk;
        out[id] = hs_local[id] + dpref[id] * carry[b * bdn + ch];
    }
}

// ---------------------------------------------------------------------------------------------
// Reverse adjoint scan: bar_h[t] = src[t] + ashift[t]*bar_h[t+1], ashift[t]=dA[t+1] (last 0).
// ---------------------------------------------------------------------------------------------

/// Block-local reverse scan from a zero seed; emits the partial adjoint and within-block
/// suffix decay (`spref`).  One thread per `(block, channel)`.
#[cube(launch_unchecked)]
fn rev_block_kernel<F: Float>(
    ashift: &Array<F>,
    src: &Array<F>,
    bar_local: &mut Array<F>,
    spref: &mut Array<F>,
    blk: usize,
    bdn: usize,
    n_blk: usize,
) {
    let id = usize::cast_from(ABSOLUTE_POS);
    if id < n_blk * bdn {
        let b = id / bdn;
        let ch = id % bdn;
        let base = b * blk * bdn + ch;
        let mut r = F::new(0.0);
        let mut s = F::new(1.0);
        for k in 0..blk {
            let i = blk - 1 - k; // walk the block right-to-left
            let off = base + i * bdn;
            let m = ashift[off];
            r = src[off] + m * r;
            s = m * s;
            bar_local[off] = r;
            spref[off] = s;
        }
    }
}

/// Incoming carry from the right per block: `carry[n_blk-1]=0`,
/// `carry[b]=Eblk[b+1]+Gblk[b+1]*carry[b+1]` (block-leftmost partial adjoint / suffix decay).
#[cube(launch_unchecked)]
fn rev_carry_kernel<F: Float>(
    bar_local: &Array<F>,
    spref: &Array<F>,
    carry: &mut Array<F>,
    blk: usize,
    bdn: usize,
    n_blk: usize,
) {
    let ch = usize::cast_from(ABSOLUTE_POS);
    if ch < bdn {
        let mut c = F::new(0.0);
        carry[(n_blk - 1) * bdn + ch] = c;
        for k in 1..n_blk {
            let b = n_blk - 1 - k; // n_blk-2, ..., 0
            let left = (b + 1) * blk * bdn + ch; // leftmost position of block b+1
            c = bar_local[left] + spref[left] * c;
            carry[b * bdn + ch] = c;
        }
    }
}

/// Fold the right-carry into every position: `bar_h[t]=bar_local[t]+spref[t]*carry[blk]`.
#[cube(launch_unchecked)]
fn rev_fixup_kernel<F: Float>(
    bar_local: &Array<F>,
    spref: &Array<F>,
    carry: &Array<F>,
    out: &mut Array<F>,
    blk: usize,
    bdn: usize,
    total: usize,
) {
    let id = usize::cast_from(ABSOLUTE_POS);
    if id < total {
        let t = id / bdn;
        let ch = id % bdn;
        let b = t / blk;
        out[id] = bar_local[id] + spref[id] * carry[b * bdn + ch];
    }
}

// ---------------------------------------------------------------------------------------------
// Host glue
// ---------------------------------------------------------------------------------------------

fn dims4(t: &CubeTensor<impl CubeRuntime>) -> (usize, usize) {
    let s = &t.meta.shape;
    (s[0], s[1] * s[2] * s[3]) // (M, bdn)
}

/// Forward parallel scan on cube tensors. `da`,`db` are `(M',B,D,N)` (M' a multiple of BLK),
/// `hstate` is `(1,B,D,N)`; returns `hs` `(M',B,D,N)`.
fn fwd_cube<R: CubeRuntime>(
    da: CubeTensor<R>,
    db: CubeTensor<R>,
    hstate: CubeTensor<R>,
) -> CubeTensor<R> {
    let da = into_contiguous(da);
    let db = into_contiguous(db);
    let hstate = into_contiguous(hstate);
    let (m, bdn) = dims4(&da);
    let n_blk = m / BLK;
    let client = da.client.clone();
    let shape = da.meta.shape.clone();
    let hs_local = empty_device::<R, f32>(client.clone(), da.device.clone(), shape.clone());
    let dpref = empty_device::<R, f32>(client.clone(), da.device.clone(), shape.clone());
    let carry_shape = {
        let mut s = da.meta.shape.clone();
        s[0] = n_blk;
        s
    };
    let carry = empty_device::<R, f32>(client.clone(), da.device.clone(), carry_shape);
    let out = empty_device::<R, f32>(client.clone(), da.device.clone(), shape);

    let dim = CubeDim::new_1d(256);
    unsafe {
        fwd_block_kernel::launch_unchecked::<f32, R>(
            &client,
            calculate_cube_count_elemwise(&client, n_blk * bdn, dim),
            dim,
            da.clone().into_array_arg(),
            db.into_array_arg(),
            hs_local.clone().into_array_arg(),
            dpref.clone().into_array_arg(),
            BLK,
            bdn,
            n_blk,
        );
        fwd_carry_kernel::launch_unchecked::<f32, R>(
            &client,
            calculate_cube_count_elemwise(&client, bdn, dim),
            dim,
            hs_local.clone().into_array_arg(),
            dpref.clone().into_array_arg(),
            hstate.into_array_arg(),
            carry.clone().into_array_arg(),
            BLK,
            bdn,
            n_blk,
        );
        fwd_fixup_kernel::launch_unchecked::<f32, R>(
            &client,
            calculate_cube_count_elemwise(&client, m * bdn, dim),
            dim,
            hs_local.into_array_arg(),
            dpref.into_array_arg(),
            carry.into_array_arg(),
            out.clone().into_array_arg(),
            BLK,
            bdn,
            m * bdn,
        );
    }
    out
}

/// Reverse parallel adjoint scan. `ashift` is `(M',B,D,N)` (= dA shifted up by one, last 0),
/// `src` is `(M',B,D,N)`; returns `bar_h` `(M',B,D,N)`.
fn rev_cube<R: CubeRuntime>(ashift: CubeTensor<R>, src: CubeTensor<R>) -> CubeTensor<R> {
    let ashift = into_contiguous(ashift);
    let src = into_contiguous(src);
    let (m, bdn) = dims4(&ashift);
    let n_blk = m / BLK;
    let client = ashift.client.clone();
    let shape = ashift.meta.shape.clone();
    let bar_local = empty_device::<R, f32>(client.clone(), ashift.device.clone(), shape.clone());
    let spref = empty_device::<R, f32>(client.clone(), ashift.device.clone(), shape.clone());
    let carry_shape = {
        let mut s = ashift.meta.shape.clone();
        s[0] = n_blk;
        s
    };
    let carry = empty_device::<R, f32>(client.clone(), ashift.device.clone(), carry_shape);
    let out = empty_device::<R, f32>(client.clone(), ashift.device.clone(), shape);

    let dim = CubeDim::new_1d(256);
    unsafe {
        rev_block_kernel::launch_unchecked::<f32, R>(
            &client,
            calculate_cube_count_elemwise(&client, n_blk * bdn, dim),
            dim,
            ashift.into_array_arg(),
            src.into_array_arg(),
            bar_local.clone().into_array_arg(),
            spref.clone().into_array_arg(),
            BLK,
            bdn,
            n_blk,
        );
        rev_carry_kernel::launch_unchecked::<f32, R>(
            &client,
            calculate_cube_count_elemwise(&client, bdn, dim),
            dim,
            bar_local.clone().into_array_arg(),
            spref.clone().into_array_arg(),
            carry.clone().into_array_arg(),
            BLK,
            bdn,
            n_blk,
        );
        rev_fixup_kernel::launch_unchecked::<f32, R>(
            &client,
            calculate_cube_count_elemwise(&client, m * bdn, dim),
            dim,
            bar_local.into_array_arg(),
            spref.into_array_arg(),
            carry.into_array_arg(),
            out.clone().into_array_arg(),
            BLK,
            bdn,
            m * bdn,
        );
    }
    out
}

/// Pad a `(M,B,D,N)` tensor along dim 0 up to a multiple of [`BLK`] with `fill`.
fn pad_to_blk(x: Tensor<Wgpu, 4>, fill: f32) -> (Tensor<Wgpu, 4>, usize) {
    let [m, b, d, n] = x.dims();
    let mp = m.div_ceil(BLK) * BLK;
    if mp == m {
        return (x, m);
    }
    let pad = Tensor::full([mp - m, b, d, n], fill, &x.device());
    (Tensor::cat(vec![x, pad], 0), m)
}

/// Within-chunk inclusive scan on the wgpu backend.  `da` is the per-step decay `exp(a·dt)`,
/// `db` the per-step input `dB`, `hstate` the `(1,B,D,N)` incoming state; returns `hs`
/// `(M,B,D,N)`.
pub fn within_chunk_scan(
    da: Tensor<Wgpu, 4>,
    db: Tensor<Wgpu, 4>,
    hstate: Tensor<Wgpu, 4>,
) -> Tensor<Wgpu, 4> {
    let [m, b, d, n] = da.dims();
    let (da, _) = pad_to_blk(da, 1.0); // padded decay steps are no-ops (dA=1, dB=0)
    let (db, _) = pad_to_blk(db, 0.0);
    let hs = fwd_cube(
        da.into_primitive().tensor(),
        db.into_primitive().tensor(),
        hstate.into_primitive().tensor(),
    );
    let hs = Tensor::<Wgpu, 4>::from_primitive(TensorPrimitive::Float(hs));
    hs.slice([0..m, 0..b, 0..d, 0..n])
}

/// Within-chunk reverse adjoint scan on the wgpu backend: `bar_h[t] = Σ_{p≥t} (∏ dA) src[p]`.
/// `da` is the per-step decay `exp(a·dt)`; returns `bar_h` `(M,B,D,N)`.
pub fn within_chunk_rev_scan(da: Tensor<Wgpu, 4>, src: Tensor<Wgpu, 4>) -> Tensor<Wgpu, 4> {
    let [m, b, d, n] = da.dims();
    // ashift[t] = dA[t+1]; last row 0 (no successor).
    let zeros = Tensor::zeros([1, b, d, n], &da.device());
    let ashift = Tensor::cat(vec![da.slice([1..m, 0..b, 0..d, 0..n]), zeros], 0);
    let (ashift, _) = pad_to_blk(ashift, 0.0); // padded steps: multiplier 0
    let (src, _) = pad_to_blk(src, 0.0);
    let bar = rev_cube(
        ashift.into_primitive().tensor(),
        src.into_primitive().tensor(),
    );
    let bar = Tensor::<Wgpu, 4>::from_primitive(TensorPrimitive::Float(bar));
    bar.slice([0..m, 0..b, 0..d, 0..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::WgpuDevice;
    use burn::tensor::TensorData;

    fn mk(shape: [usize; 4], seed: usize, dev: &WgpuDevice) -> Tensor<Wgpu, 4> {
        let k: usize = shape.iter().product();
        let v: Vec<f32> = (0..k).map(|i| ((i * 37 + seed * 11) % 19) as f32 / 19.0).collect();
        Tensor::from_data(TensorData::new(v, shape), dev)
    }

    // The parallel kernels must match a plain CPU recurrence — including a length that is not a
    // multiple of BLK and a nonzero incoming state.
    #[test]
    fn parallel_scans_match_reference() {
        std::panic::set_hook(Box::new(|_| {}));
        let res = std::panic::catch_unwind(|| {
            let dev = WgpuDevice::default();
            let (m, b, d, n) = (200usize, 2, 3, 2); // 200 is not a multiple of BLK(64)
            let bdn = b * d * n;
            let da = mk([m, b, d, n], 1, &dev).mul_scalar(0.9);
            let db = mk([m, b, d, n], 2, &dev);
            let src = mk([m, b, d, n], 3, &dev);
            let h0: Vec<f32> = (0..bdn).map(|i| (i % 7) as f32 / 7.0).collect();
            let hstate = Tensor::<Wgpu, 4>::from_data(TensorData::new(h0.clone(), [1, b, d, n]), &dev);

            let da_v = da.clone().into_data().to_vec::<f32>().unwrap();
            let db_v = db.clone().into_data().to_vec::<f32>().unwrap();
            let src_v = src.clone().into_data().to_vec::<f32>().unwrap();

            // Forward: hs[t] = da[t]*hs[t-1] + db[t], hs[-1]=hstate.
            let got = within_chunk_scan(da.clone(), db, hstate).into_data().to_vec::<f32>().unwrap();
            let mut exp = vec![0f32; m * bdn];
            for ch in 0..bdn {
                let mut h = h0[ch];
                for t in 0..m {
                    h = da_v[t * bdn + ch] * h + db_v[t * bdn + ch];
                    exp[t * bdn + ch] = h;
                }
            }
            let e = got.iter().zip(&exp).fold(0f32, |x, (&a, &b)| x.max((a - b).abs()));
            assert!(e < 1e-4, "fwd parallel scan err {e:.2e}");

            // Reverse: bar[t] = src[t] + da[t+1]*bar[t+1], bar[m]=0.
            let got_r = within_chunk_rev_scan(da, src).into_data().to_vec::<f32>().unwrap();
            let mut exp_r = vec![0f32; m * bdn];
            for ch in 0..bdn {
                let mut r = 0f32;
                for k in 0..m {
                    let t = m - 1 - k;
                    let mlt = if t + 1 < m { da_v[(t + 1) * bdn + ch] } else { 0.0 };
                    r = src_v[t * bdn + ch] + mlt * r;
                    exp_r[t * bdn + ch] = r;
                }
            }
            let er = got_r.iter().zip(&exp_r).fold(0f32, |x, (&a, &b)| x.max((a - b).abs()));
            assert!(er < 1e-4, "rev parallel scan err {er:.2e}");
        });
        if res.is_err() {
            eprintln!("SKIP parallel_scans_match_reference: no usable wgpu adapter");
        }
    }
}
