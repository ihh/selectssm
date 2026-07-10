//! From-scratch initializer sanity: init a `BidirectionalMamba` with fresh seeded weights,
//! run a forward pass, and check the init distributions/constants match the JAX/Flax spec.

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

use selectssm::config::{BidirectionalMambaConfig, ScanAlgo};
use selectssm::rng::Rng;
use selectssm::BidirectionalMamba;

type B = NdArray;
type Dev = burn::backend::ndarray::NdArrayDevice;

/// Small config: D=8, N=8, dt_rank=2, expansion=2, rms norm, mlp on, complex SSM on.
fn small_cfg() -> BidirectionalMambaConfig {
    BidirectionalMambaConfig {
        hidden_features: 8, // N
        expansion_factor: 2.0,
        dt_rank: 2,
        complement: false,
        tie_in_proj: false,
        tie_gate: false,
        concatenate_fwd_rev: true,
        activation: "silu".to_string(),
        norm_type: "rms".to_string(),
        bn_momentum: 0.9,
        mlp_layer: true,
        dense_expansion: 2,
        mlp_dropout_rate: 0.1,
        use_complex_ssm: true,
        chunk_size: Some(4),
        n_channel_groups: 1,
        use_remat: true,
        scan_algo: ScanAlgo::Hillis,
        dt_proj: true,
        shift_conv_size: 3,
    }
}

/// Sample mean and (population) stddev of a flattened tensor's values.
fn mean_std<const D: usize>(t: &Tensor<B, D>) -> (f32, f32) {
    let v: Vec<f32> = t.clone().into_data().to_vec::<f32>().unwrap();
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

#[test]
fn bidirectional_init_forward_and_stats() {
    let dev: Dev = Default::default();
    let mut rng = Rng::new(0xA11CE);
    let d_in = 8usize;
    let cfg = small_cfg();
    let model = BidirectionalMamba::<B>::init(d_in, cfg.clone(), &mut rng, &dev);

    // ---- forward on a random (2, 16, 8) input: finite + correctly shaped ----
    let (bb, l) = (2usize, 16usize);
    let xn = bb * l * d_in;
    let xv: Vec<f32> = (0..xn).map(|_| rng.normal()).collect();
    let x = Tensor::<B, 3>::from_data(TensorData::new(xv, [bb, l, d_in]), &dev);
    let y = model.forward(x);
    assert_eq!(y.dims(), [bb, l, d_in], "output shape");
    let yv: Vec<f32> = y.into_data().to_vec::<f32>().unwrap();
    assert!(yv.iter().all(|v| v.is_finite()), "output must be finite");

    // ---- param-stat sanity ----
    let w = &model.weights;
    let ed = (cfg.expansion_factor * d_in as f64).ceil() as usize; // 16

    // A Dense kernel's stddev ~ 1/sqrt(fan_in) (lecun-normal), within 30%.
    let check_kernel = |name: &str, k: &Tensor<B, 2>| {
        let fan_in = k.dims()[0];
        let (_m, sd) = mean_std(k);
        let target = 1.0 / (fan_in as f32).sqrt();
        let ratio = sd / target;
        assert!(
            (0.7..=1.3).contains(&ratio),
            "{name}: kernel stddev {sd:.4} vs target {target:.4} (fan_in={fan_in}) ratio {ratio:.3}"
        );
    };
    check_kernel("in_proj", &w.in_proj.weight);
    check_kernel("out_proj", &w.out_proj.weight);
    check_kernel("ssm_fwd.BC", &w.ssm_fwd.bc.weight);
    check_kernel("ssm_fwd.dt", &w.ssm_fwd.dt.weight);
    check_kernel(
        "ssm_fwd.dt_proj",
        &w.ssm_fwd.dt_proj.as_ref().unwrap().weight,
    );
    check_kernel(
        "ssm_fwd.theta",
        &w.ssm_fwd.theta.as_ref().unwrap().weight,
    );
    let mlp = w.mlp.as_ref().expect("mlp_layer on");
    check_kernel("mlp", &mlp.mlp.weight);
    check_kernel("mlp_proj", &mlp.mlp_proj.weight);

    // in_proj kernel fan_in = d_in; out width = (n_in_proj + n_gate)*ED = 4*ED.
    assert_eq!(w.in_proj.weight.dims(), [d_in, 4 * ed]);
    // out_proj input = 2*ED (concatenate_fwd_rev), output = d_in.
    assert_eq!(w.out_proj.weight.dims(), [2 * ed, d_in]);

    // A_log matches log(arange(1..=w)) tiled across channels; complex => w = N/2 = 4.
    let aw = cfg.hidden_features / 2;
    assert_eq!(w.ssm_fwd.a_log.dims(), [ed, aw]);
    let a_vals: Vec<f32> = w.ssm_fwd.a_log.clone().into_data().to_vec::<f32>().unwrap();
    for (i, v) in a_vals.iter().enumerate() {
        let col = i % aw;
        let expected = ((col + 1) as f32).ln();
        assert!(
            (v - expected).abs() < 1e-5,
            "A_log[{i}] = {v} != log({}) = {expected}",
            col + 1
        );
    }

    // D (skip) is all ones, shape (ED,).
    assert_eq!(w.ssm_fwd.d.dims(), [ed]);
    let d_vals: Vec<f32> = w.ssm_fwd.d.clone().into_data().to_vec::<f32>().unwrap();
    assert!(d_vals.iter().all(|v| (v - 1.0).abs() < 1e-6), "D must be all ones");

    // Biases: BC / in_proj / out_proj / theta / dt (dt_proj on) are zero.
    let is_zero = |t: &Tensor<B, 1>| {
        t.clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|v| *v == 0.0)
    };
    assert!(is_zero(w.in_proj.bias.as_ref().unwrap()), "in_proj bias zero");
    assert!(is_zero(w.out_proj.bias.as_ref().unwrap()), "out_proj bias zero");
    assert!(is_zero(w.ssm_fwd.bc.bias.as_ref().unwrap()), "BC bias zero");
    assert!(is_zero(w.ssm_fwd.dt.bias.as_ref().unwrap()), "dt bias zero (dt_proj on)");
    assert!(
        is_zero(w.ssm_fwd.theta.as_ref().unwrap().bias.as_ref().unwrap()),
        "theta bias zero"
    );

    // dt_proj bias = inverse_softplus(uniform(dt_min,dt_max)); softplus(bias) in [dt_min,dt_max].
    let dtp_b: Vec<f32> = w
        .ssm_fwd
        .dt_proj
        .as_ref()
        .unwrap()
        .bias
        .as_ref()
        .unwrap()
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for b in &dtp_b {
        let sp = b.max(0.0) + (1.0 + (-b.abs()).exp()).ln(); // softplus(b)
        assert!(
            (0.0009..=0.1001).contains(&sp),
            "softplus(dt_proj bias) = {sp} not in [dt_min, dt_max]"
        );
    }

    // RMS norm scale is all ones.
    let scale = w.norm.scale.as_ref().expect("rms scale");
    assert_eq!(scale.dims(), [d_in]);
    let sv: Vec<f32> = scale.clone().into_data().to_vec::<f32>().unwrap();
    assert!(sv.iter().all(|v| (v - 1.0).abs() < 1e-6), "rms scale all ones");
    assert!(w.norm.bias.is_none(), "rms has no bias");
}
