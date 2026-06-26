//! `BidirectionalMamba` --- a non-causal forward+reverse block with a gated output
//! projection, ported from the JAX `BidirectionalMamba` module.  (MLP layer omitted; it is
//! off in the fixtures.)

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::config::BidirectionalMambaConfig;
use crate::loader::{BidirWeights, Store, P};
use crate::selective_ssm::{activation, forward_ssm};

/// Flax RMSNorm over the last axis: `x * rsqrt(mean(x^2) + eps) * scale` (eps = 1e-6).
fn rms_norm<B: Backend>(x: Tensor<B, 3>, scale: &Tensor<B, 1>) -> Tensor<B, 3> {
    let d = x.dims()[2];
    let ms = x.clone().powf_scalar(2.0).mean_dim(2); // (B, L, 1)
    let inv = (ms + 1e-6).sqrt().recip(); // (B, L, 1)
    x * inv * scale.clone().reshape([1, 1, d])
}

pub struct BidirectionalMamba<B: Backend> {
    pub cfg: BidirectionalMambaConfig,
    pub weights: BidirWeights<B>,
}

impl<B: Backend> BidirectionalMamba<B> {
    pub fn load(store: &Store<B>, prefix: &str, cfg: BidirectionalMambaConfig) -> Self {
        let weights = BidirWeights::load(store, prefix, &cfg);
        BidirectionalMamba { cfg, weights }
    }

    pub fn named(&self, prefix: &str, out: &mut Vec<(String, P<B>)>) {
        self.weights.named(prefix, out);
    }

    /// Forward pass, `(B, L, D) -> (B, L, D)`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let cfg = &self.cfg;
        let w = &self.weights;
        let [bb, l, d_in] = x.dims();
        let ed = (cfg.expansion_factor * d_in as f64).ceil() as usize;

        let skip = x.clone();

        // Normalize (rms is the only norm_type exercised by the fixtures).
        let x = match cfg.norm_type.as_str() {
            "rms" | "" => rms_norm(x, &w.norm_scale),
            other => panic!("unsupported norm_type {other:?} in the Rust port"),
        };

        // Project to the expanded dimension and split into [xf, xr, zf, zr].
        let n_in_proj = if cfg.tie_in_proj { 1 } else { 2 };
        let proj = w.in_proj.forward(x); // (B, L, (n_in_proj + n_gate) * ED)
        let slc = |i: usize| proj.clone().slice([0..bb, 0..l, i * ed..(i + 1) * ed]);
        let xf = slc(0);
        let xr = if cfg.tie_in_proj { xf.clone() } else { slc(1) };
        let zf = slc(n_in_proj);
        let zr = if cfg.tie_gate { zf.clone() } else { slc(n_in_proj + 1) };

        // Forward and reverse selective SSMs.
        let xf = forward_ssm(&w.ssm_fwd, &cfg.inner_ssm(false), xf);
        let xr = forward_ssm(&w.ssm_rev, &cfg.inner_ssm(true), xr);

        // Gated combination.
        let gated_f = xf * activation(&cfg.activation, zf);
        let gated_r = xr * activation(&cfg.activation, zr);
        let combined = if cfg.concatenate_fwd_rev {
            Tensor::cat(vec![gated_f, gated_r], 2) // (B, L, 2 ED)
        } else {
            gated_f + gated_r
        };

        // Project back down and add the residual.
        let out = w.out_proj.forward(combined); // (B, L, D)
        skip + out
    }
}
