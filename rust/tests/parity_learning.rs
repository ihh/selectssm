//! Functional state-tracking test (Rust ndarray): PARITY with length generalization.
//!
//! Mirrors `jax/tests/test_parity.py`.  A tiny one-layer model is trained on short binary
//! sequences and evaluated on longer, held-out ones.  With the complex-SSM RoPE trick on it
//! learns parity and length-generalizes; with it off (a purely real SSM) it stays near
//! chance.  The solves-vs-chance gap is the gate.

use burn::backend::{Autodiff, NdArray};
use burn::module::{Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use selectssm::config::SelectiveSsmConfig;
use selectssm::loader::{Linear, SsmWeights};
use selectssm::rng::Rng;
use selectssm::selective_ssm::forward_ssm;

type B = Autodiff<NdArray>;
type Dev = burn::backend::ndarray::NdArrayDevice;

const DM: usize = 32; // d_model
const N: usize = 16; // state width
const R: usize = 2; // dt rank
const CHUNK: usize = 4;

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

/// Wrap a plain (require_grad) tensor as a trainable `Param`.
fn as_param<const D: usize>(t: Tensor<B, D>) -> Param<Tensor<B, D>> {
    Param::from_tensor(t)
}

fn init_model(complex: bool, rng: &mut Rng, dev: &Dev) -> ParityModel<B> {
    let h = N / 2;
    // Build the selective-SSM weights via the shared library init (the ONE init path), then
    // adopt its tensors as trainable Params.  This exercises `SsmWeights::init` end-to-end as a
    // training initializer — A_log/D/dt-bias constants and the lecun-normal kernels all come from
    // the library, so the test doubles as the proof the init is training-usable.
    let cfg = ssm_cfg_for(complex);
    let ssm = SsmWeights::<B>::init(DM, &cfg, rng, dev);
    let dt_proj = ssm.dt_proj.expect("dt_proj enabled in ssm_cfg_for");
    // The parity head/embedding/norm are outside the SSM module, so they stay hand-initialized.
    let (theta_k, theta_b) = match ssm.theta {
        Some(t) => (as_param(t.weight), as_param(t.bias.expect("theta bias"))),
        // Non-complex: theta unused, but the field must exist; init it lecun-like for shape.
        None => (
            randn([DM, h], (1.0 / DM as f32).sqrt(), rng, dev),
            constant([h], 0.0, dev),
        ),
    };
    ParityModel {
        embed: randn([2, DM], 1.0, rng, dev),
        conv: as_param(ssm.conv),
        bc_k: as_param(ssm.bc.weight),
        bc_b: as_param(ssm.bc.bias.expect("BC bias")),
        a_log: as_param(ssm.a_log),
        d_skip: as_param(ssm.d),
        dt_k: as_param(ssm.dt.weight),
        dt_b: as_param(ssm.dt.bias.expect("dt bias")),
        dtp_k: as_param(dt_proj.weight),
        dtp_b: as_param(dt_proj.bias.expect("dt_proj bias")),
        theta_k,
        theta_b,
        norm: constant([DM], 1.0, dev),
        head_k: randn([DM, 2], (1.0 / DM as f32).sqrt(), rng, dev),
        head_b: constant([2], 0.0, dev),
        complex,
    }
}

/// The inner SSM config used both to init the weights and to run the forward pass.
fn ssm_cfg_for(complex: bool) -> SelectiveSsmConfig {
    SelectiveSsmConfig {
        hidden_features: N,
        reverse: false,
        complement: false,
        use_complex_ssm: complex,
        chunk_size: Some(CHUNK),
        n_channel_groups: 1,
        use_remat: true,
        scan_algo: selectssm::config::ScanAlgo::Hillis,
        dt_rank: R,
        dt_proj: true,
        shift_conv_size: 3,
        activation: "silu".to_string(),
    }
}

impl<B: Backend + selectssm::remat::ScanBackend> ParityModel<B> {
    fn ssm_cfg(&self) -> SelectiveSsmConfig {
        ssm_cfg_for(self.complex)
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
const L_TRAIN_EVAL: usize = 16; // accuracy at the training length (did it learn at all?)
const L_GEN: usize = 32; // held-out length, 2x training (does it length-generalize?)

/// Train and return (accuracy at the training length, accuracy at the 2x held-out length).
fn train(complex: bool, steps: usize, lr: f64, dev: &Dev) -> (f32, f32) {
    let mut rng = Rng(0xC0FFEE);
    let mut model = init_model(complex, &mut rng, dev);
    // Plain constant-LR Adam (epsilon 1e-8 to match optax.adam), no schedule or gradient
    // clipping -- the training is numerically stable now that the within-chunk decay matrix
    // clamps its exponent (see chunked_scan::ssm_chunked_scan).
    let mut optim = AdamConfig::new().with_epsilon(1e-8).init();
    let l_min = 6;
    for s in 0..steps {
        let l = (l_min / 2 + (rng.next_u64() as usize % (L_TR / 2 - l_min / 2 + 1))) * 2;
        let (x, t) = batch(&mut rng, 64, l, dev);
        let loss = cross_entropy(model.forward(x), t);
        if std::env::var("PARITY_VERBOSE").is_ok() && complex && s % 100 == 0 {
            println!("  step {s:4}  loss {:.4}", loss.clone().into_scalar());
        }
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(lr, model, grads);
    }
    let mut eval_rng = Rng(0x1234_5678);
    let at_train = accuracy(&model, &mut eval_rng, L_TRAIN_EVAL, dev);
    let at_gen = accuracy(&model, &mut eval_rng, L_GEN, dev);
    (at_train, at_gen)
}

#[test]
fn parity_complex_solves_real_fails() {
    let dev = Default::default();
    let (on_tr, on_gen) = train(true, 700, 4e-3, &dev);
    let (off_tr, off_gen) = train(false, 700, 4e-3, &dev);
    println!(
        "\nparity: complex-on [train_len={on_tr:.3}, gen_len(2x)={on_gen:.3}]  \
         real-off [train_len={off_tr:.3}, gen_len(2x)={off_gen:.3}]"
    );
    // The discriminator is length generalization (as in the Mamba-3 paper): the complex SSM
    // learns parity and holds up at 2x the training length; the real SSM cannot represent the
    // rotation, so even where it partially fits the training length its accuracy collapses
    // toward chance at the longer length.  The qualitative gap is the gate, not a precise number.
    assert!(on_tr > 0.95, "complex SSM did not learn parity at the training length: {on_tr:.3}");
    assert!(on_gen > 0.85, "complex SSM did not length-generalize on parity: {on_gen:.3}");
    assert!(off_gen < 0.7, "real SSM unexpectedly generalized parity: {off_gen:.3}");
    assert!(on_gen - off_gen > 0.25, "solves-vs-fails gap too small: {on_gen:.3} vs {off_gen:.3}");
}
