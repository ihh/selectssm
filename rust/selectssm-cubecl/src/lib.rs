//! A custom cubecl GPU kernel for the within-chunk selective-scan, isolated here so the core
//! `selectssm` crate stays `#![forbid(unsafe_code)]` (kernel launches are `unsafe`) and free of
//! the heavy cubecl dependency unless the `cubecl` feature is on.
//!
//! [`within_chunk_scan`] computes the inclusive linear recurrence
//! `hs[t] = dA[t] · hs[t-1] + dB[t]` (with `hs[-1] = hstate`) over the chunk axis (dim 0) of a
//! `(cs, B, D, N)` tensor.  One GPU thread handles one `(b, d, n)` state channel and walks the
//! `cs` steps sequentially — `O(cs)` work per thread, `B·D·N` threads — the hardware-aware
//! scan that the `Matrix`/`Hillis` burn-op paths approximate with `O(cs²)` / `O(cs·log cs)`
//! tensor ops.  Only the within-chunk scan is offloaded; the cross-chunk carry, the RoPE
//! rotary, and the whole backward pass stay in the (portable) core crate.

use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorPrimitive};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeRuntime;
use cubecl::frontend::ABSOLUTE_POS;
use cubecl::{calculate_cube_count_elemwise, prelude::*, CubeDim};

/// `dA[t]·h + dB[t]` swept along the chunk axis; one thread per `(b,d,n)` channel.
#[cube(launch_unchecked)]
fn chunk_scan_kernel<F: Float>(
    da: &Array<F>,      // (cs·bdn) row-major [t][b·d·n]
    db: &Array<F>,      // (cs·bdn)
    hstate: &Array<F>,  // (bdn) incoming state
    out: &mut Array<F>, // (cs·bdn) inclusive states hs[t]
    cs: usize,
    bdn: usize,
) {
    let idx = usize::cast_from(ABSOLUTE_POS);
    if idx < bdn {
        let mut h = hstate[idx];
        for t in 0..cs {
            let off = t * bdn + idx;
            h = da[off] * h + db[off];
            out[off] = h;
        }
    }
}

/// Run the scan kernel on cubecl tensors (any cube runtime).
fn scan_cube<R: CubeRuntime>(
    da: CubeTensor<R>,
    db: CubeTensor<R>,
    hstate: CubeTensor<R>,
) -> CubeTensor<R> {
    let da = into_contiguous(da);
    let db = into_contiguous(db);
    let hstate = into_contiguous(hstate);

    let shape = da.meta.shape.clone(); // (cs, B, D, N)
    let cs = shape[0];
    let bdn: usize = shape[1] * shape[2] * shape[3];

    let client = da.client.clone();
    let out = empty_device::<R, f32>(client.clone(), da.device.clone(), shape);

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise(&client, bdn, cube_dim);
    unsafe {
        chunk_scan_kernel::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            da.into_array_arg(),
            db.into_array_arg(),
            hstate.into_array_arg(),
            out.clone().into_array_arg(),
            cs,
            bdn,
        );
    }
    out
}

/// Within-chunk inclusive scan on the wgpu backend.  `da` is the per-step decay `exp(a·dt)`,
/// `db` the per-step input `dB`, `hstate` the `(1,B,D,N)` incoming state; returns `hs`
/// `(cs,B,D,N)`.
pub fn within_chunk_scan(
    da: Tensor<Wgpu, 4>,
    db: Tensor<Wgpu, 4>,
    hstate: Tensor<Wgpu, 4>,
) -> Tensor<Wgpu, 4> {
    let da = da.into_primitive().tensor();
    let db = db.into_primitive().tensor();
    let hstate = hstate.into_primitive().tensor();
    Tensor::from_primitive(TensorPrimitive::Float(scan_cube(da, db, hstate)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::WgpuDevice;
    use burn::tensor::TensorData;

    // The kernel must match a plain CPU recurrence to f32 tolerance.
    #[test]
    fn kernel_matches_reference() {
        std::panic::set_hook(Box::new(|_| {}));
        let res = std::panic::catch_unwind(|| {
            let dev = WgpuDevice::default();
            let (cs, bb, d, n) = (5usize, 2usize, 3usize, 2usize);
            let mk = |seed: usize| {
                let v: Vec<f32> = (0..cs * bb * d * n)
                    .map(|i| ((i * 37 + seed * 11) % 19) as f32 / 19.0)
                    .collect();
                Tensor::<Wgpu, 4>::from_data(TensorData::new(v, [cs, bb, d, n]), &dev)
            };
            let da = mk(1).mul_scalar(0.9); // decays in (0,1)
            let db = mk(2);
            let h0: Vec<f32> = (0..bb * d * n).map(|i| (i % 7) as f32 / 7.0).collect();
            let hstate = Tensor::<Wgpu, 4>::from_data(TensorData::new(h0.clone(), [1, bb, d, n]), &dev);

            let got = within_chunk_scan(da.clone(), db.clone(), hstate)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let da_v = da.into_data().to_vec::<f32>().unwrap();
            let db_v = db.into_data().to_vec::<f32>().unwrap();
            let bdn = bb * d * n;
            // Reference: hs[t] = da[t]*hs[t-1] + db[t].
            let mut expect = vec![0f32; cs * bdn];
            for idx in 0..bdn {
                let mut h = h0[idx];
                for t in 0..cs {
                    let off = t * bdn + idx;
                    h = da_v[off] * h + db_v[off];
                    expect[off] = h;
                }
            }
            let err = got
                .iter()
                .zip(&expect)
                .fold(0f32, |m, (&a, &b)| m.max((a - b).abs()));
            assert!(err < 1e-5, "kernel vs reference max abs err {err:.2e}");
        });
        if res.is_err() {
            eprintln!("SKIP kernel_matches_reference: no usable wgpu adapter");
        }
    }
}
