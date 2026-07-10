//! `SelectiveSsm` --- the causal (and `reverse`) selective state space model, ported from
//! the JAX `SelectiveSSM` module.  Uses the chunked scan, with the optional complex-SSM
//! RoPE trick.

use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::config::SelectiveSsmConfig;
use crate::loader::{SsmWeights, Store, P};
use crate::remat::{ScanBackend, ScanOpts};

/// Causal depthwise 1-D convolution matching Flax `nn.Conv(feature_group_count=D,
/// padding=[(k-1, 0)])`: `u[t] = sum_i kernel[i] * x[t-(k-1)+i]` with left zero-padding.
pub(crate) fn causal_conv1d<B: Backend>(
    x: Tensor<B, 3>,
    kernel: &Tensor<B, 3>,
    k: usize,
) -> Tensor<B, 3> {
    let [bb, l, d] = x.dims();
    let zeros = Tensor::zeros([bb, k - 1, d], &x.device());
    let xpad = Tensor::cat(vec![zeros, x], 1); // (B, L+k-1, D)
    let mut acc: Option<Tensor<B, 3>> = None;
    for i in 0..k {
        let ki = kernel.clone().slice([i..i + 1, 0..1, 0..d]).reshape([1, 1, d]);
        let seg = xpad.clone().slice([0..bb, i..i + l, 0..d]); // (B, L, D)
        let term = seg * ki;
        acc = Some(match acc {
            Some(a) => a + term,
            None => term,
        });
    }
    acc.unwrap()
}

/// Numerically-stable softplus: `max(x,0) + log(1 + exp(-|x|))` (matches `jax.nn.softplus`).
pub(crate) fn softplus<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let stable = (x.clone().abs().neg().exp() + 1.0).log();
    x.clamp_min(0.0) + stable
}

/// Apply the configured pointwise activation (only `silu` is exercised by the fixtures).
pub(crate) fn activation<B: Backend>(name: &str, u: Tensor<B, 3>) -> Tensor<B, 3> {
    match name {
        "silu" => u.clone() * sigmoid(u),
        "relu" => u.clamp_min(0.0),
        "gelu" => {
            // Flax `nn.gelu` (approximate=True): 0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))
            let c = (2.0f64 / std::f64::consts::PI).sqrt();
            let inner = (u.clone() + u.clone().powf_scalar(3.0) * 0.044715) * c;
            u * (inner.tanh() + 1.0) * 0.5
        }
        other => panic!("unknown activation {other:?}"),
    }
}

/// One selective SSM layer with loaded weights.
pub struct SelectiveSsm<B: Backend> {
    pub cfg: SelectiveSsmConfig,
    pub weights: SsmWeights<B>,
}

impl<B: Backend> SelectiveSsm<B> {
    pub fn load(store: &Store<B>, prefix: &str, cfg: SelectiveSsmConfig) -> Self {
        let weights = SsmWeights::load(store, prefix, &cfg);
        SelectiveSsm { cfg, weights }
    }

    /// From-scratch init at channel width `d` with fresh seeded weights (see
    /// [`SsmWeights::init`] for the per-parameter init distributions).
    pub fn init(d: usize, cfg: SelectiveSsmConfig, rng: &mut crate::rng::Rng, dev: &B::Device) -> Self {
        let weights = SsmWeights::init(d, &cfg, rng, dev);
        SelectiveSsm { cfg, weights }
    }

    pub fn named(&self, prefix: &str, out: &mut Vec<(String, P<B>)>) {
        self.weights.named(prefix, out);
    }

    /// Forward pass, `(B, L, D) -> (B, L, D)`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3>
    where
        B: ScanBackend,
    {
        forward_ssm(&self.weights, &self.cfg, x)
    }
}

/// Free-function forward, so `BidirectionalMamba` can reuse it on its inner SSM weights.
pub fn forward_ssm<B: ScanBackend>(
    w: &SsmWeights<B>,
    cfg: &SelectiveSsmConfig,
    x: Tensor<B, 3>,
) -> Tensor<B, 3> {
    {
        let [bb, l, d] = x.dims();
        let n = cfg.hidden_features;

        // Anti-causal mode: flip the sequence (and channels, if complement) up front.
        let x = if cfg.reverse {
            if cfg.complement { x.flip([1, 2]) } else { x.flip([1]) }
        } else {
            x
        };

        let u = causal_conv1d(x, &w.conv, cfg.shift_conv_size);
        let u = activation(&cfg.activation, u);

        // The complex-SSM RoPE trick pairs (re, im) state entries, so it requires even N.
        if cfg.use_complex_ssm && n % 2 != 0 {
            panic!("use_complex_ssm requires an even hidden_features (N); got N={n}");
        }

        // Transition coefficients A = -exp(A_log); complex mode ties each (re, im) pair.
        // A_log is clamped to [-20, 20] purely as a numerical guard: exp() is the only
        // unbounded function of a learnable parameter here, and an exploding A_log would make
        // the decay matrix evaluate -inf - (-inf) = NaN during aggressive training.  Every
        // shipped fixture has A_log in [0, 2.77], so the clamp is inactive and parity with the
        // JAX reference (which does not clamp) is preserved exactly.
        let a_log = w.a_log.clone().clamp(-20.0, 20.0);
        let a = if cfg.use_complex_ssm {
            Tensor::cat(vec![a_log.clone(), a_log], 1).exp().neg()
        } else {
            a_log.exp().neg()
        };

        let bc = w.bc.forward(u.clone()); // (B, L, 2N)
        let b_proj = bc.clone().slice([0..bb, 0..l, 0..n]);
        let c_proj = bc.slice([0..bb, 0..l, n..2 * n]);

        let mut dt = w.dt.forward(u.clone()); // (B, L, R)
        if let Some(dp) = &w.dt_proj {
            dt = dp.forward(dt); // (B, L, D)
        } else if cfg.dt_rank > 1 {
            // dt_proj disabled: map the low-rank dt (width R) to D by block-repeat, matching
            // JAX `jnp.repeat(dt, D // dt_rank, axis=-1)` — each of the R entries repeated D/R
            // times (a,a,...,b,b,...).  NB `repeat_dim` in burn TILES (a,b,a,b,...), so we build
            // the block-repeat via reshape+tile+reshape.  Requires dt_rank | D.
            if d % cfg.dt_rank != 0 {
                panic!("dt_rank={} must divide D={d}", cfg.dt_rank);
            }
            let rep = d / cfg.dt_rank;
            dt = dt
                .reshape([bb, l, cfg.dt_rank, 1])
                .repeat_dim(3, rep)
                .reshape([bb, l, d]); // (a,a,...,b,b,...) block-repeat R -> D
        } else {
            // dt_rank == 1: JAX broadcasts width-1 dt against D; the scan needs an explicit
            // width-D tensor, so broadcast-repeat (every channel gets the same dt) — identical.
            dt = dt.repeat_dim(2, d);
        }
        let dt = softplus(dt);

        let theta = w.theta.as_ref().map(|t| t.forward(u.clone())); // (B, L, N/2)

        // n_channel_groups (K) must divide D.  The Rust scan shares B/C across groups, which is
        // numerically identical to the JAX grouped tiling (B/C do not vary across groups), so
        // the tiling itself is a no-op — only the divisibility validation is ported here.
        if d % cfg.n_channel_groups != 0 {
            panic!(
                "n_channel_groups={} must divide D={d}",
                cfg.n_channel_groups
            );
        }

        let opts = ScanOpts {
            chunk_size: crate::config::resolve_chunk_size(cfg.chunk_size, l, cfg.n_channel_groups),
            use_remat: cfg.use_remat,
            algo: cfg.scan_algo,
        };
        let y = B::chunked_scan(opts, u.clone(), a, b_proj, c_proj, dt, theta);

        // Flip the scan output back to the original order.
        let y = if cfg.reverse {
            if cfg.complement { y.flip([1, 2]) } else { y.flip([1]) }
        } else {
            y
        };

        // D-skip term.  NB: matching the JAX code, `u` is in flipped order in reverse mode
        // (it is never flipped back), so the skip is added with the flipped-order `u`.
        y + u * w.d.clone().reshape([1, 1, d])
    }
}
