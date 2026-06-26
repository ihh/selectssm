//! Functional state-tracking test (Rust ndarray): PARITY with length generalization.
//!
//! Mirrors `jax/tests/test_parity.py`.  A tiny one-layer model is trained on short binary
//! sequences and evaluated on longer, held-out ones.  With the complex-SSM RoPE trick on it
//! learns parity and length-generalizes; with it off (a purely real SSM) it stays near
//! chance.  The solves-vs-chance gap is the gate.

use burn::backend::{Autodiff, NdArray};
use burn::module::{Module, Param};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use selectssm::config::SelectiveSsmConfig;
use selectssm::loader::{Linear, SsmWeights};
use selectssm::selective_ssm::forward_ssm;

type B = Autodiff<NdArray>;
type Dev = burn::backend::ndarray::NdArrayDevice;

const DM: usize = 32; // d_model
const N: usize = 16; // state width
const R: usize = 2; // dt rank
const CHUNK: usize = 4;

// ---- a small, fully seeded RNG so the test is deterministic ----
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    fn bit(&mut self) -> f32 {
        (self.next_u64() & 1) as f32
    }
}

fn randn<const D: usize>(shape: [usize; D], std: f32, rng: &mut Rng, dev: &Dev) -> Param<Tensor<B, D>> {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|_| rng.normal() * std).collect();
    Param::from_tensor(Tensor::from_data(TensorData::new(v, shape), dev))
}

fn constant<const D: usize>(shape: [usize; D], val: f32, dev: &Dev) -> Param<Tensor<B, D>> {
    let n: usize = shape.iter().product();
    Param::from_tensor(Tensor::from_data(TensorData::new(vec![val; n], shape), dev))
}

#[derive(Module, Debug)]
struct ParityModel<B: Backend> {
    embed: Param<Tensor<B, 2>>,  // (2, DM)
    conv: Param<Tensor<B, 3>>,   // (3, 1, DM)
    bc_k: Param<Tensor<B, 2>>,   // (DM, 2N)
    bc_b: Param<Tensor<B, 1>>,   // (2N)
    a_log: Param<Tensor<B, 2>>,  // (DM, N) or (DM, N/2)
    d_skip: Param<Tensor<B, 1>>, // (DM)
    dt_k: Param<Tensor<B, 2>>,   // (DM, R)
    dt_b: Param<Tensor<B, 1>>,   // (R)
    dtp_k: Param<Tensor<B, 2>>,  // (R, DM)
    dtp_b: Param<Tensor<B, 1>>,  // (DM)
    theta_k: Param<Tensor<B, 2>>, // (DM, N/2)
    theta_b: Param<Tensor<B, 1>>, // (N/2)
    norm: Param<Tensor<B, 1>>,   // (DM)
    head_k: Param<Tensor<B, 2>>, // (DM, 2)
    head_b: Param<Tensor<B, 1>>, // (2)
    #[module(skip)]
    complex: bool,
}

fn init_model(complex: bool, rng: &mut Rng, dev: &Dev) -> ParityModel<B> {
    let h = N / 2;
    let a_w = if complex { h } else { N };
    // A_log initialised so that A = -(1..=a_w), tiled across channels (matches JAX).
    let a_row: Vec<f32> = (1..=a_w).map(|k| (k as f32).ln()).collect();
    let a_vals: Vec<f32> = (0..DM).flat_map(|_| a_row.clone()).collect();
    let a_log = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(a_vals, [DM, a_w]), dev));
    ParityModel {
        embed: randn([2, DM], 1.0, rng, dev),
        conv: randn([3, 1, DM], 0.3, rng, dev),
        bc_k: randn([DM, 2 * N], (1.0 / DM as f32).sqrt(), rng, dev),
        bc_b: constant([2 * N], 0.0, dev),
        a_log,
        d_skip: constant([DM], 1.0, dev),
        dt_k: randn([DM, R], (1.0 / DM as f32).sqrt(), rng, dev),
        dt_b: constant([R], 0.0, dev),
        dtp_k: randn([R, DM], (1.0 / R as f32).sqrt(), rng, dev),
        // dt bias = inverse_softplus(uniform(dt_min, dt_max)) per channel, matching JAX.
        // This spreads dt across [0.001, 0.1], giving some near-unit-decay channels that can
        // persist state over long sequences (essential for length-generalizing parity).
        dtp_b: {
            let vals: Vec<f32> = (0..DM)
                .map(|_| {
                    let y = 0.001 + (0.1 - 0.001) * rng.uniform();
                    (y.exp() - 1.0).ln()
                })
                .collect();
            Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(vals, [DM]), dev))
        },
        theta_k: randn([DM, h], 0.5, rng, dev),
        theta_b: constant([h], 0.0, dev),
        norm: constant([DM], 1.0, dev),
        head_k: randn([DM, 2], (1.0 / DM as f32).sqrt(), rng, dev),
        head_b: constant([2], 0.0, dev),
        complex,
    }
}

impl<B: Backend> ParityModel<B> {
    fn ssm_cfg(&self) -> SelectiveSsmConfig {
        SelectiveSsmConfig {
            hidden_features: N,
            reverse: false,
            complement: false,
            use_complex_ssm: self.complex,
            chunk_size: CHUNK,
            n_channel_groups: 1,
            dt_rank: R,
            dt_proj: true,
            shift_conv_size: 3,
            activation: "silu".to_string(),
        }
    }

    fn ssm_weights(&self) -> SsmWeights<B> {
        SsmWeights {
            conv: self.conv.val(),
            bc: Linear { weight: self.bc_k.val(), bias: Some(self.bc_b.val()) },
            a_log: self.a_log.val(),
            d: self.d_skip.val(),
            dt: Linear { weight: self.dt_k.val(), bias: Some(self.dt_b.val()) },
            dt_proj: Some(Linear { weight: self.dtp_k.val(), bias: Some(self.dtp_b.val()) }),
            theta: if self.complex {
                Some(Linear { weight: self.theta_k.val(), bias: Some(self.theta_b.val()) })
            } else {
                None
            },
        }
    }

    /// x_onehot: (Bb, L, 2) -> logits (Bb, L, 2)
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [bb, l, _] = x.dims();
        let emb = x.matmul(self.embed.val().reshape([1, 2, DM])); // (Bb, L, DM)
        let y = forward_ssm(&self.ssm_weights(), &self.ssm_cfg(), emb);
        // RMS norm
        let ms = y.clone().powf_scalar(2.0).mean_dim(2);
        let y = y * (ms + 1e-6).sqrt().recip() * self.norm.val().reshape([1, 1, DM]);
        // head
        let logits = y.matmul(self.head_k.val().reshape([1, DM, 2])) + self.head_b.val().reshape([1, 1, 2]);
        logits.reshape([bb, l, 2])
    }
}

/// Cross-entropy against one-hot targets, averaged over (B, L).
fn cross_entropy(logits: Tensor<B, 3>, target_oh: Tensor<B, 3>) -> Tensor<B, 1> {
    let [bb, l, _] = logits.dims();
    let m = logits.clone().max_dim(2); // (B,L,1)
    let lse = m.clone() + (logits.clone() - m).exp().sum_dim(2).log(); // (B,L,1)
    let logp = logits - lse; // (B,L,2)
    (target_oh * logp).sum().neg() / (bb * l) as f32
}

/// Build a parity batch: input one-hot (Bb,L,2) and target one-hot (running parity) (Bb,L,2).
fn batch(rng: &mut Rng, bb: usize, l: usize, dev: &Dev) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let mut xin = vec![0f32; bb * l * 2];
    let mut tgt = vec![0f32; bb * l * 2];
    for b in 0..bb {
        let mut par = 0u32;
        for t in 0..l {
            let bit = rng.bit() as u32;
            par ^= bit;
            xin[(b * l + t) * 2 + bit as usize] = 1.0;
            tgt[(b * l + t) * 2 + par as usize] = 1.0;
        }
    }
    (
        Tensor::from_data(TensorData::new(xin, [bb, l, 2]), dev),
        Tensor::from_data(TensorData::new(tgt, [bb, l, 2]), dev),
    )
}

fn accuracy(model: &ParityModel<B>, rng: &mut Rng, l: usize, dev: &Dev) -> f32 {
    let (x, t) = batch(rng, 256, l, dev);
    let pred = model.forward(x).argmax(2);
    let tgt = t.argmax(2);
    let eq = pred.equal(tgt).int().sum().into_scalar();
    eq as f32 / (256 * l) as f32
}

const L_TR: usize = 16; // max training length (even lengths in [6, L_TR])
const L_EVAL: usize = 24; // held-out evaluation length (longer than training)

/// Returns (accuracy at training length, accuracy at the longer eval length).
fn train(complex: bool, steps: usize, lr: f64, dev: &Dev) -> (f32, f32) {
    let mut rng = Rng(0xC0FFEE);
    let mut model = init_model(complex, &mut rng, dev);
    // epsilon = 1e-8 to match optax.adam (burn's default is larger); gradient-norm clipping
    // keeps A_log from exploding (which would make the decay matrix produce NaNs) and lets a
    // larger learning rate converge stably.
    let mut optim = AdamConfig::new()
        .with_epsilon(1e-7)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let l_min = 6;
    for s in 0..steps {
        // Cosine decay to zero: high LR converges fast; the vanishing tail keeps Adam stable
        // once the loss is near zero (a constant LR explodes after the model has solved parity).
        let lr_t = lr * 0.5 * (1.0 + (std::f64::consts::PI * s as f64 / steps as f64).cos());
        let l = (l_min / 2 + (rng.next_u64() as usize % (L_TR / 2 - l_min / 2 + 1))) * 2;
        let (x, t) = batch(&mut rng, 64, l, dev);
        let loss = cross_entropy(model.forward(x), t);
        if std::env::var("PARITY_VERBOSE").is_ok() && complex && s % 100 == 0 {
            println!("  step {s:4}  loss {:.4}", loss.clone().into_scalar());
        }
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(lr_t, model, grads);
    }
    let mut eval_rng = Rng(0x1234_5678);
    let at_train = accuracy(&model, &mut eval_rng, L_TR, dev);
    let at_eval = accuracy(&model, &mut eval_rng, L_EVAL, dev);
    (at_train, at_eval)
}

#[test]
fn parity_complex_solves_real_fails() {
    let dev = Default::default();
    let (on_tr, on_ev) = train(true, 1000, 4e-3, &dev);
    let (off_tr, off_ev) = train(false, 1000, 4e-3, &dev);
    println!(
        "\nparity: complex-on [train_len={on_tr:.3}, eval_len={on_ev:.3}]  \
         real-off [train_len={off_tr:.3}, eval_len={off_ev:.3}]"
    );
    // The qualitative solves-vs-chance gap is the gate (the complex SSM length-generalizes;
    // the real SSM stays at chance), not a precise number.
    assert!(on_ev > 0.85, "complex SSM failed to length-generalize on parity: {on_ev:.3}");
    assert!(off_ev < 0.65, "real SSM unexpectedly solved parity: {off_ev:.3}");
    assert!(on_ev - off_ev > 0.3, "solves-vs-chance gap too small: {on_ev:.3} vs {off_ev:.3}");
}
