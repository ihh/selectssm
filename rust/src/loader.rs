//! Loading JAX-exported parameters (safetensors) into burn tensors, plus the typed
//! parameter structs used by the forward passes.  Tensor names match the flattened
//! Flax parameter paths emitted by `jax/scripts/generate_fixtures.py`.

use std::collections::HashMap;

use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

use crate::config::SelectiveSsmConfig;

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
