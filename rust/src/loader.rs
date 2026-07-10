//! Loading JAX-exported parameters (safetensors) into burn tensors, plus the typed
//! parameter structs used by the forward passes.  Tensor names match the flattened
//! Flax parameter paths emitted by `jax/scripts/generate_fixtures.py`.

use std::collections::HashMap;

use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

use crate::config::SelectiveSsmConfig;
use crate::rng::Rng;

/// dt-bias uniform range (mirrors the JAX `SelectiveSSM.dt_min`/`dt_max`).
const DT_MIN: f32 = 0.001;
const DT_MAX: f32 = 0.1;

/// The standard-deviation of a standard normal truncated to `±2` (i.e. the factor Flax's
/// `variance_scaling` divides out with `1 / .8796256...` so that the *post*-truncation stddev
/// equals `sqrt(scale / fan_in)`).  See `jax.nn.initializers.variance_scaling`.
const TRUNC2_STD: f32 = 0.879_925_66;

/// `inverse_softplus(x) = x + ln(1 - exp(-x))` (matches `selectssm.py:16`).
fn inverse_softplus(x: f32) -> f32 {
    x + (1.0 - (-x).exp()).ln()
}

/// A single draw from a standard normal truncated to `±2` (rejection sampling).
fn trunc_normal2(rng: &mut Rng) -> f32 {
    loop {
        let z = rng.normal();
        if z.abs() <= 2.0 {
            return z;
        }
    }
}

/// A Vec of `n` lecun-normal draws with the given `fan_in`.
///
/// This is Flax's `lecun_normal` = `variance_scaling(1.0, "fan_in", "truncated_normal")`
/// implemented faithfully: standard-normal samples truncated to `±2` then scaled so the
/// post-truncation stddev is exactly `sqrt(1 / fan_in)` (the `1/0.8796256` correction).
fn lecun_normal_vec(n: usize, fan_in: usize, rng: &mut Rng) -> Vec<f32> {
    let std = (1.0 / fan_in as f32).sqrt() / TRUNC2_STD;
    (0..n).map(|_| trunc_normal2(rng) * std).collect()
}

/// A rank-2 lecun-normal kernel of shape `(fan_in, fan_out)`.
fn lecun_kernel2<B: Backend>(fan_in: usize, fan_out: usize, rng: &mut Rng, dev: &B::Device) -> Tensor<B, 2> {
    let v = lecun_normal_vec(fan_in * fan_out, fan_in, rng);
    Tensor::from_data(TensorData::new(v, [fan_in, fan_out]), dev)
}

fn zeros1<B: Backend>(n: usize, dev: &B::Device) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(vec![0.0f32; n], [n]), dev)
}

fn ones1<B: Backend>(n: usize, dev: &B::Device) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(vec![1.0f32; n], [n]), dev)
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A parsed safetensors file: name -> (row-major f32 values, shape).
pub struct Store<B: Backend> {
    data: HashMap<String, (Vec<f32>, Vec<usize>)>,
    device: B::Device,
}

impl<B: Backend> Store<B> {
    pub fn from_bytes(buffer: &[u8], device: B::Device) -> Self {
        let st = safetensors::SafeTensors::deserialize(buffer).expect("parse safetensors");
        let mut data = HashMap::new();
        for name in st.names() {
            let view = st.tensor(name).unwrap();
            data.insert(
                name.to_string(),
                (bytes_to_f32(view.data()), view.shape().to_vec()),
            );
        }
        Store { data, device }
    }

    pub fn from_file(path: &std::path::Path, device: B::Device) -> Self {
        let buffer = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        Self::from_bytes(&buffer, device)
    }

    pub fn has(&self, name: &str) -> bool {
        self.data.contains_key(name)
    }

    fn raw(&self, name: &str) -> &(Vec<f32>, Vec<usize>) {
        self.data
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor {name:?}"))
    }

    /// Golden (values, shape) for comparison tensors such as "output" / "grad_input".
    pub fn values(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        self.raw(name).clone()
    }

    pub fn tensor<const D: usize>(&self, name: &str) -> Tensor<B, D> {
        let (v, shape) = self.raw(name);
        let s: [usize; D] = shape
            .clone()
            .try_into()
            .unwrap_or_else(|_| panic!("tensor {name:?} has rank {} != {D}", shape.len()));
        Tensor::from_data(TensorData::new(v.clone(), s), &self.device)
    }
}

/// A rank-erased parameter handle, used to extract gradients uniformly for parity checks.
#[derive(Clone)]
pub enum P<B: Backend> {
    R1(Tensor<B, 1>),
    R2(Tensor<B, 2>),
    R3(Tensor<B, 3>),
}

impl<B: AutodiffBackend> P<B> {
    /// Flatten this parameter's gradient (row-major) for comparison with the fixture.
    pub fn grad_flat(&self, grads: &B::Gradients) -> Vec<f32> {
        fn flat<BB: Backend, const D: usize>(t: Tensor<BB, D>) -> Vec<f32> {
            t.into_data().to_vec::<f32>().unwrap()
        }
        match self {
            P::R1(t) => flat(t.grad(grads).expect("grad R1")),
            P::R2(t) => flat(t.grad(grads).expect("grad R2")),
            P::R3(t) => flat(t.grad(grads).expect("grad R3")),
        }
    }
}

/// A dense layer: `y = x @ weight + bias`, with `weight` in (in, out) layout (as in Flax).
#[derive(Clone)]
pub struct Linear<B: Backend> {
    pub weight: Tensor<B, 2>,
    pub bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> Linear<B> {
    /// From-scratch init: lecun-normal `(in, out)` kernel; bias per `bias` (`Some(vec)` sets an
    /// explicit bias, `None` omits it).  Mirrors Flax `nn.Dense(kernel_init=lecun_normal(), ...)`.
    fn init_with_bias(
        din: usize,
        dout: usize,
        bias: Option<Vec<f32>>,
        rng: &mut Rng,
        dev: &B::Device,
    ) -> Self {
        let weight = lecun_kernel2::<B>(din, dout, rng, dev).require_grad();
        let bias = bias.map(|v| {
            debug_assert_eq!(v.len(), dout);
            Tensor::from_data(TensorData::new(v, [dout]), dev).require_grad()
        });
        Linear { weight, bias }
    }

    /// Lecun-normal kernel + zero bias (Flax `nn.Dense` default: `use_bias=True`, zero bias).
    pub fn init(din: usize, dout: usize, rng: &mut Rng, dev: &B::Device) -> Self {
        Self::init_with_bias(din, dout, Some(vec![0.0; dout]), rng, dev)
    }

    /// Lecun-normal kernel with an explicit bias vector (e.g. the dt inverse-softplus bias).
    pub fn init_bias(din: usize, dout: usize, bias: Vec<f32>, rng: &mut Rng, dev: &B::Device) -> Self {
        Self::init_with_bias(din, dout, Some(bias), rng, dev)
    }

    pub fn load(store: &Store<B>, prefix: &str, bias: bool) -> Self {
        let weight = store
            .tensor::<2>(&format!("param/{prefix}/kernel"))
            .require_grad();
        let bias = if bias {
            Some(store.tensor::<1>(&format!("param/{prefix}/bias")).require_grad())
        } else {
            None
        };
        Linear { weight, bias }
    }

    /// Apply to a (B, L, in) tensor, returning (B, L, out).
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [din, out] = self.weight.dims();
        let w = self.weight.clone().reshape([1, din, out]);
        let mut y = x.matmul(w);
        if let Some(b) = &self.bias {
            y = y + b.clone().reshape([1, 1, out]);
        }
        y
    }

    pub fn named(&self, prefix: &str, out: &mut Vec<(String, P<B>)>) {
        out.push((format!("{prefix}/kernel"), P::R2(self.weight.clone())));
        if let Some(b) = &self.bias {
            out.push((format!("{prefix}/bias"), P::R1(b.clone())));
        }
    }
}

/// Parameters of one [`crate::SelectiveSsm`].
pub struct SsmWeights<B: Backend> {
    pub conv: Tensor<B, 3>, // shift_conv/kernel: (k, 1, D)
    pub bc: Linear<B>,
    pub a_log: Tensor<B, 2>, // (D, N) real, or (D, N/2) complex
    pub d: Tensor<B, 1>,
    pub dt: Linear<B>,
    pub dt_proj: Option<Linear<B>>,
    pub theta: Option<Linear<B>>,
}

impl<B: Backend> SsmWeights<B> {
    /// From-scratch init of one selective SSM's weights, mirroring the JAX `SelectiveSSM`
    /// `@nn.compact __call__`.  `d` is the (expanded) channel width the SSM runs at; `cfg`
    /// supplies `hidden_features` (N), `dt_rank` (R), `dt_proj`, `use_complex_ssm`, and
    /// `shift_conv_size` (k).
    ///
    /// Init map (see `jax/src/selectssm/selectssm.py`):
    /// * `shift_conv` — depthwise kernel `(k, 1, d)`, lecun-normal (fan_in = k), **no bias**.
    /// * `BC` Dense (d → 2N) — lecun-normal kernel, zero bias.
    /// * `A_log` — `log(repeat(arange(1..=w), d))` → `(d, w)`, `w = N` (real) or `N/2` (complex).
    /// * `D` (skip) — ones `(d,)`.
    /// * `dt` Dense (d → R) — lecun-normal kernel; bias = zeros if `dt_proj`, else
    ///   `inverse_softplus(uniform(dt_min, dt_max))` `(R,)`.
    /// * `dt_proj` Dense (R → d) — lecun-normal kernel; bias = `inverse_softplus(uniform)` `(d,)`.
    /// * `theta` Dense (d → N/2) — lecun-normal kernel, zero bias (only if `use_complex_ssm`).
    pub fn init(d: usize, cfg: &SelectiveSsmConfig, rng: &mut Rng, dev: &B::Device) -> Self {
        let n = cfg.hidden_features;
        let r = cfg.dt_rank;
        let k = cfg.shift_conv_size;

        // shift_conv depthwise kernel (k, 1, d), lecun-normal with fan_in = k, no bias.
        let conv_v = lecun_normal_vec(k * d, k, rng);
        let conv = Tensor::<B, 3>::from_data(TensorData::new(conv_v, [k, 1, d]), dev).require_grad();

        // BC Dense (d -> 2N), lecun kernel + zero bias.
        let bc = Linear::init(d, 2 * n, rng, dev);

        // A_log = log(repeat(arange(1..=w)[None,:], d, axis=0)) -> (d, w).
        let w = if cfg.use_complex_ssm { n / 2 } else { n };
        let a_row: Vec<f32> = (1..=w).map(|kk| (kk as f32).ln()).collect();
        let a_vals: Vec<f32> = (0..d).flat_map(|_| a_row.clone()).collect();
        let a_log = Tensor::<B, 2>::from_data(TensorData::new(a_vals, [d, w]), dev).require_grad();

        // D skip = ones(d).
        let d_skip = ones1::<B>(d, dev).require_grad();

        // dt Dense (d -> R): zero bias if dt_proj, else inverse_softplus(uniform(dt_min, dt_max)).
        let dt = if cfg.dt_proj {
            Linear::init(d, r, rng, dev)
        } else {
            let bias: Vec<f32> = (0..r)
                .map(|_| inverse_softplus(rng.uniform_range(DT_MIN, DT_MAX)))
                .collect();
            Linear::init_bias(d, r, bias, rng, dev)
        };

        // dt_proj Dense (R -> d): lecun kernel + inverse_softplus(uniform) bias (d,).
        let dt_proj = if cfg.dt_proj {
            let bias: Vec<f32> = (0..d)
                .map(|_| inverse_softplus(rng.uniform_range(DT_MIN, DT_MAX)))
                .collect();
            Some(Linear::init_bias(r, d, bias, rng, dev))
        } else {
            None
        };

        // theta Dense (d -> N/2): lecun kernel + zero bias (complex-SSM only).
        let theta = if cfg.use_complex_ssm {
            Some(Linear::init(d, n / 2, rng, dev))
        } else {
            None
        };

        SsmWeights { conv, bc, a_log, d: d_skip, dt, dt_proj, theta }
    }

    pub fn load(store: &Store<B>, prefix: &str, cfg: &SelectiveSsmConfig) -> Self {
        let p = |n: &str| format!("{prefix}{n}");
        SsmWeights {
            conv: store
                .tensor::<3>(&format!("param/{}", p("shift_conv/kernel")))
                .require_grad(),
            bc: Linear::load(store, &p("BC"), true),
            a_log: store.tensor::<2>(&format!("param/{}", p("A_log"))).require_grad(),
            d: store.tensor::<1>(&format!("param/{}", p("D"))).require_grad(),
            dt: Linear::load(store, &p("dt"), true),
            dt_proj: if cfg.dt_proj {
                Some(Linear::load(store, &p("dt_proj"), true))
            } else {
                None
            },
            theta: if cfg.use_complex_ssm {
                Some(Linear::load(store, &p("theta"), true))
            } else {
                None
            },
        }
    }

    pub fn named(&self, prefix: &str, out: &mut Vec<(String, P<B>)>) {
        out.push((format!("{prefix}shift_conv/kernel"), P::R3(self.conv.clone())));
        self.bc.named(&format!("{prefix}BC"), out);
        out.push((format!("{prefix}A_log"), P::R2(self.a_log.clone())));
        out.push((format!("{prefix}D"), P::R1(self.d.clone())));
        self.dt.named(&format!("{prefix}dt"), out);
        if let Some(p) = &self.dt_proj {
            p.named(&format!("{prefix}dt_proj"), out);
        }
        if let Some(t) = &self.theta {
            t.named(&format!("{prefix}theta"), out);
        }
    }
}

/// Which normalization the block applies (parsed from `norm_type`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NormKind {
    /// No normalization: `""`, `"none"`, or any unrecognized value (matches the JAX fall-through).
    Identity,
    Rms,
    Layer,
    Group,
    Batch,
}

impl NormKind {
    pub fn parse(norm_type: &str) -> Self {
        match norm_type {
            "rms" => NormKind::Rms,
            "layer" => NormKind::Layer,
            "group" => NormKind::Group,
            "batch" => NormKind::Batch,
            _ => NormKind::Identity,
        }
    }

    /// The Flax autogenerated module name (and thus exported param path prefix) for this norm.
    fn module_name(self) -> &'static str {
        match self {
            NormKind::Identity => "",
            NormKind::Rms => "RMSNorm_0",
            NormKind::Layer => "LayerNorm_0",
            NormKind::Group => "GroupNorm_0",
            NormKind::Batch => "BatchNorm_0",
        }
    }
}

/// Loaded normalization parameters, keyed by [`NormKind`].
#[derive(Clone)]
pub struct NormWeights<B: Backend> {
    pub kind: NormKind,
    /// `scale` for rms/layer/group/batch (None for identity).
    pub scale: Option<Tensor<B, 1>>,
    /// `bias` for layer/group/batch (None for rms/identity).
    pub bias: Option<Tensor<B, 1>>,
    /// Running `mean`/`var` for batch (inference `use_running_average=True`).
    pub mean: Option<Tensor<B, 1>>,
    pub var: Option<Tensor<B, 1>>,
}

impl<B: Backend> NormWeights<B> {
    /// From-scratch init over `d` channels (Flax norm defaults): scale = ones for all norms;
    /// bias = zeros for layer/group/batch; BatchNorm running mean = zeros, var = ones.
    pub fn init(d: usize, norm_type: &str, dev: &B::Device) -> Self {
        let kind = NormKind::parse(norm_type);
        let (scale, bias, mean, var) = match kind {
            NormKind::Identity => (None, None, None, None),
            NormKind::Rms => (Some(ones1::<B>(d, dev).require_grad()), None, None, None),
            NormKind::Layer | NormKind::Group => (
                Some(ones1::<B>(d, dev).require_grad()),
                Some(zeros1::<B>(d, dev).require_grad()),
                None,
                None,
            ),
            NormKind::Batch => (
                Some(ones1::<B>(d, dev).require_grad()),
                Some(zeros1::<B>(d, dev).require_grad()),
                // Running stats: mean = zeros, var = ones (Flax BatchNorm defaults).
                Some(zeros1::<B>(d, dev)),
                Some(ones1::<B>(d, dev)),
            ),
        };
        NormWeights { kind, scale, bias, mean, var }
    }

    fn load(store: &Store<B>, prefix: &str, norm_type: &str) -> Self {
        let kind = NormKind::parse(norm_type);
        let m = kind.module_name();
        let p1 = |n: &str| store.tensor::<1>(&format!("param/{prefix}{m}/{n}")).require_grad();
        let (scale, bias, mean, var) = match kind {
            NormKind::Identity => (None, None, None, None),
            NormKind::Rms => (Some(p1("scale")), None, None, None),
            NormKind::Layer | NormKind::Group => {
                (Some(p1("scale")), Some(p1("bias")), None, None)
            }
            NormKind::Batch => (
                Some(p1("scale")),
                Some(p1("bias")),
                // Running stats live in Flax's `batch_stats` collection; the fixture generator
                // exports them under `param/<prefix>BatchNorm_0/{mean,var}`.
                Some(p1("mean")),
                Some(p1("var")),
            ),
        };
        NormWeights { kind, scale, bias, mean, var }
    }

    fn named(&self, prefix: &str, out: &mut Vec<(String, P<B>)>) {
        if self.kind == NormKind::Identity {
            return;
        }
        let m = self.kind.module_name();
        if let Some(s) = &self.scale {
            out.push((format!("{prefix}{m}/scale"), P::R1(s.clone())));
        }
        if let Some(b) = &self.bias {
            out.push((format!("{prefix}{m}/bias"), P::R1(b.clone())));
        }
        // NB: running mean/var are batch_stats, not trainable params; the fixture stores no
        // grad for them (the loss does not depend on them at inference), so they are excluded
        // from `named` (which drives the gradient-parity comparison).
    }
}

/// The MLP sub-layer parameters (present only when `mlp_layer`).
#[derive(Clone)]
pub struct MlpWeights<B: Backend> {
    pub mlp: Linear<B>,
    pub mlp_proj: Linear<B>,
}

/// Parameters of one [`crate::BidirectionalMamba`].
pub struct BidirWeights<B: Backend> {
    pub norm: NormWeights<B>,
    pub in_proj: Linear<B>,
    pub ssm_fwd: SsmWeights<B>,
    pub ssm_rev: SsmWeights<B>,
    pub out_proj: Linear<B>,
    pub mlp: Option<MlpWeights<B>>,
}

impl<B: Backend> BidirWeights<B> {
    /// From-scratch init of a whole bidirectional block, mirroring the JAX `BidirectionalMamba`
    /// `@nn.compact __call__`.  `d` is the block's input feature width; the inner SSMs run at the
    /// expanded width `ED = ceil(expansion_factor * d)`.
    ///
    /// * `norm` — per `norm_type` (see [`NormWeights::init`]).
    /// * `in_proj` Dense (d → (n_in_proj + n_gate)·ED) — lecun kernel, zero bias.
    /// * `ssm_fwd` / `ssm_rev` — [`SsmWeights::init`] at width ED (the reverse half's config
    ///   carries `reverse=true` / `complement`).
    /// * `out_proj` Dense ((concatenate ? 2·ED : ED) → d) — lecun kernel, zero bias.
    /// * `mlp` / `mlp_proj` (when `mlp_layer`) — Dense (d → dense_expansion·d) and back, lecun + zero bias.
    pub fn init(
        d: usize,
        cfg: &crate::config::BidirectionalMambaConfig,
        rng: &mut Rng,
        dev: &B::Device,
    ) -> Self {
        let ed = (cfg.expansion_factor * d as f64).ceil() as usize;
        let n_in_proj = if cfg.tie_in_proj { 1 } else { 2 };
        let n_gate = if cfg.tie_gate { 1 } else { 2 };

        let norm = NormWeights::init(d, &cfg.norm_type, dev);
        let in_proj = Linear::init(d, (n_in_proj + n_gate) * ed, rng, dev);
        let ssm_fwd = SsmWeights::init(ed, &cfg.inner_ssm(false), rng, dev);
        let ssm_rev = SsmWeights::init(ed, &cfg.inner_ssm(true), rng, dev);
        let out_proj_in = if cfg.concatenate_fwd_rev { 2 * ed } else { ed };
        let out_proj = Linear::init(out_proj_in, d, rng, dev);
        let mlp = if cfg.mlp_layer {
            Some(MlpWeights {
                mlp: Linear::init(d, cfg.dense_expansion * d, rng, dev),
                mlp_proj: Linear::init(cfg.dense_expansion * d, d, rng, dev),
            })
        } else {
            None
        };

        BidirWeights { norm, in_proj, ssm_fwd, ssm_rev, out_proj, mlp }
    }

    pub fn load(
        store: &Store<B>,
        prefix: &str,
        cfg: &crate::config::BidirectionalMambaConfig,
    ) -> Self {
        let mlp = if cfg.mlp_layer {
            Some(MlpWeights {
                mlp: Linear::load(store, &format!("{prefix}mlp"), true),
                mlp_proj: Linear::load(store, &format!("{prefix}mlp_proj"), true),
            })
        } else {
            None
        };
        BidirWeights {
            norm: NormWeights::load(store, prefix, &cfg.norm_type),
            in_proj: Linear::load(store, &format!("{prefix}in_proj"), true),
            ssm_fwd: SsmWeights::load(store, &format!("{prefix}ssm_fwd/"), &cfg.inner_ssm(false)),
            ssm_rev: SsmWeights::load(store, &format!("{prefix}ssm_rev/"), &cfg.inner_ssm(true)),
            out_proj: Linear::load(store, &format!("{prefix}out_proj"), true),
            mlp,
        }
    }

    pub fn named(&self, prefix: &str, out: &mut Vec<(String, P<B>)>) {
        self.norm.named(prefix, out);
        self.in_proj.named(&format!("{prefix}in_proj"), out);
        self.ssm_fwd.named(&format!("{prefix}ssm_fwd/"), out);
        self.ssm_rev.named(&format!("{prefix}ssm_rev/"), out);
        self.out_proj.named(&format!("{prefix}out_proj"), out);
        if let Some(m) = &self.mlp {
            m.mlp.named(&format!("{prefix}mlp"), out);
            m.mlp_proj.named(&format!("{prefix}mlp_proj"), out);
        }
    }
}
