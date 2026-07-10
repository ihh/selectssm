//! `BidirectionalMamba` --- a non-causal forward+reverse block with a gated output
//! projection, ported from the JAX `BidirectionalMamba` module.  (MLP layer omitted; it is
//! off in the fixtures.)

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::config::BidirectionalMambaConfig;
use crate::loader::{BidirWeights, NormKind, NormWeights, Store, P};
use crate::selective_ssm::{activation, forward_ssm};

/// Flax normalization epsilon shared by RMS/Layer/Group (BatchNorm uses 1e-5).
const NORM_EPS: f64 = 1e-6;
const BN_EPS: f64 = 1e-5;

/// Flax RMSNorm over the last axis: `x * rsqrt(mean(x^2) + eps) * scale` (eps = 1e-6).
fn rms_norm<B: Backend>(x: Tensor<B, 3>, scale: &Tensor<B, 1>) -> Tensor<B, 3> {
    let d = x.dims()[2];
    let ms = x.clone().powf_scalar(2.0).mean_dim(2); // (B, L, 1)
    let inv = (ms + NORM_EPS).sqrt().recip(); // (B, L, 1)
    x * inv * scale.clone().reshape([1, 1, d])
}

/// Flax LayerNorm over the last axis (per (B,L)): `(x - mean) * rsqrt(var + eps) * scale + bias`,
/// with `var = mean(x^2) - mean(x)^2` (Flax `use_fast_variance=True`), eps = 1e-6.
fn layer_norm<B: Backend>(
    x: Tensor<B, 3>,
    scale: &Tensor<B, 1>,
    bias: &Tensor<B, 1>,
) -> Tensor<B, 3> {
    let d = x.dims()[2];
    let mean = x.clone().mean_dim(2); // (B,L,1)
    let mean_sq = x.clone().powf_scalar(2.0).mean_dim(2); // (B,L,1)
    let var = mean_sq - mean.clone().powf_scalar(2.0);
    let inv = (var + NORM_EPS).sqrt().recip();
    (x - mean) * inv * scale.clone().reshape([1, 1, d]) + bias.clone().reshape([1, 1, d])
}

/// Flax GroupNorm for a (B, L, C) input with `num_groups=32` (the module default): reshape the
/// channel axis into (G, C/G), compute mean/var per (B, group) over the length axis AND the
/// within-group channel axis, broadcast back, then apply the per-channel scale/bias.
/// `var = mean(x^2) - mean(x)^2` (fast variance), eps = 1e-6.
fn group_norm<B: Backend>(
    x: Tensor<B, 3>,
    scale: &Tensor<B, 1>,
    bias: &Tensor<B, 1>,
    num_groups: usize,
) -> Tensor<B, 3> {
    let [bb, l, c] = x.dims();
    assert!(c % num_groups == 0, "GroupNorm: {num_groups} groups must divide {c} channels");
    let gs = c / num_groups;
    // (B, L, G, gs) -> stats over (L, gs) leaving (B, 1, G, 1).
    let xg = x.clone().reshape([bb, l, num_groups, gs]);
    let mean = xg.clone().mean_dim(1).mean_dim(3); // (B,1,G,1)
    let mean_sq = xg.powf_scalar(2.0).mean_dim(1).mean_dim(3); // (B,1,G,1)
    let var = mean_sq - mean.clone().powf_scalar(2.0);
    let inv = (var + NORM_EPS).sqrt().recip(); // (B,1,G,1)
    // Broadcast (B,1,G,1) over (B,L,G,gs), normalize, back to (B,L,C).
    let xn = (x.reshape([bb, l, num_groups, gs]) - mean) * inv;
    let xn = xn.reshape([bb, l, c]);
    xn * scale.clone().reshape([1, 1, c]) + bias.clone().reshape([1, 1, c])
}

/// Flax BatchNorm at inference (`use_running_average=True`): normalize by the stored running
/// `mean`/`var` (per channel), then apply scale/bias.  eps = 1e-5.
fn batch_norm_infer<B: Backend>(
    x: Tensor<B, 3>,
    scale: &Tensor<B, 1>,
    bias: &Tensor<B, 1>,
    mean: &Tensor<B, 1>,
    var: &Tensor<B, 1>,
) -> Tensor<B, 3> {
    let c = x.dims()[2];
    let r = |t: &Tensor<B, 1>| t.clone().reshape([1, 1, c]);
    let inv = (var.clone() + BN_EPS).sqrt().recip().reshape([1, 1, c]);
    (x - r(mean)) * inv * r(scale) + r(bias)
}

/// Apply the configured normalization (identity if `norm_type` is `""`/`"none"`/unknown).
fn apply_norm<B: Backend>(
    norm: &NormWeights<B>,
    num_groups: usize,
    x: Tensor<B, 3>,
) -> Tensor<B, 3> {
    match norm.kind {
        NormKind::Identity => x,
        NormKind::Rms => rms_norm(x, norm.scale.as_ref().unwrap()),
        NormKind::Layer => {
            layer_norm(x, norm.scale.as_ref().unwrap(), norm.bias.as_ref().unwrap())
        }
        NormKind::Group => group_norm(
            x,
            norm.scale.as_ref().unwrap(),
            norm.bias.as_ref().unwrap(),
            num_groups,
        ),
        NormKind::Batch => batch_norm_infer(
            x,
            norm.scale.as_ref().unwrap(),
            norm.bias.as_ref().unwrap(),
            norm.mean.as_ref().unwrap(),
            norm.var.as_ref().unwrap(),
        ),
    }
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

    /// From-scratch init with fresh seeded weights for a block whose input feature width is `d`
    /// (the inner SSMs run at the expanded width `ceil(expansion_factor * d)`).  See
    /// [`BidirWeights::init`] for the per-parameter init distributions.
    pub fn init(d: usize, cfg: BidirectionalMambaConfig, rng: &mut crate::rng::Rng, dev: &B::Device) -> Self {
        let weights = BidirWeights::init(d, &cfg, rng, dev);
        BidirectionalMamba { cfg, weights }
    }

    pub fn named(&self, prefix: &str, out: &mut Vec<(String, P<B>)>) {
        self.weights.named(prefix, out);
    }

    /// Forward pass, `(B, L, D) -> (B, L, D)`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3>
    where
        B: crate::remat::ScanBackend,
    {
        let cfg = &self.cfg;
        let w = &self.weights;
        let [bb, l, d_in] = x.dims();
        let ed = (cfg.expansion_factor * d_in as f64).ceil() as usize;

        let skip = x.clone();

        // Normalize.  `GroupNorm()` in JAX uses its default num_groups=32.
        let x = apply_norm(&w.norm, 32, x);

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
        let mut x = skip + out;

        // Optional MLP sub-layer: Dense(dense_expansion*D) -> dropout(identity in eval) ->
        // activation -> Dense(D) -> dropout -> residual.  At inference dropout is a no-op.
        if let Some(mlp) = &w.mlp {
            let skip = x.clone();
            let h = mlp.mlp.forward(x); // (B, L, dense_expansion*D)
            let h = activation(&cfg.activation, h);
            let h = mlp.mlp_proj.forward(h); // (B, L, D)
            x = skip + h;
        }

        x
    }
}
