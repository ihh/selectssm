//! Performance-scaling micro-benchmark for the Rust/burn `SelectiveSsm`.
//!
//! Runs ONE configuration (parsed from argv) and prints a single `RESULT ...` line of
//! `key=value` pairs: average forward latency, average forward+backward latency, and the
//! process peak resident set size (host RSS, from `/proc/self/status` `VmHWM`).  GPU VRAM
//! is not visible from inside the process; the orchestrator samples `nvidia-smi` for that.
//!
//! Run one config directly, e.g.:
//!   cargo run --release --example scaling -- cpu real 8 1024 256 16 128 20
//!   cargo run --release --features wgpu --example scaling -- gpu real 8 1024 256 16 128 20
//!
//! argv: <device cpu|gpu> <mode real|complex> <B> <L> <D> <N> <chunk> <iters>
//! The whole sweep is driven by ../scripts/benchmark_scaling.py.

use std::time::Instant;

use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Distribution, Tensor, TensorData};

use selectssm::config::{ScanAlgo, SelectiveSsmConfig};
use selectssm::loader::{Linear, SsmWeights};
use selectssm::remat::ScanBackend;
use selectssm::SelectiveSsm;

/// Peak resident set size of this process, in MiB (Linux `/proc/self/status` `VmHWM`).
fn peak_rss_mib() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

/// A dense layer with random kernel and zero bias (values are irrelevant to timing).
fn lin<B: Backend>(din: usize, dout: usize, dev: &B::Device) -> Linear<B> {
    Linear {
        weight: Tensor::random([din, dout], Distribution::Normal(0.0, 0.05), dev).require_grad(),
        bias: Some(Tensor::zeros([dout], dev).require_grad()),
    }
}

/// Build a `SelectiveSsm` with random weights at the requested size.  `A_log` is initialised
/// log-spaced exactly as the JAX reference (`log(1..=N)` broadcast over D) so `exp` stays
/// finite; all other parameters are small random normals.  Compute cost is value-independent.
fn build<B: Backend>(cfg: &SelectiveSsmConfig, d: usize, dev: &B::Device) -> SelectiveSsm<B> {
    let n = cfg.hidden_features;
    let a_cols = if cfg.use_complex_ssm { n / 2 } else { n };
    let a_vals: Vec<f32> = (0..d)
        .flat_map(|_| (1..=a_cols).map(|j| (j as f32).ln()))
        .collect();
    let weights = SsmWeights {
        conv: Tensor::random([cfg.shift_conv_size, 1, d], Distribution::Normal(0.0, 0.1), dev)
            .require_grad(),
        bc: lin(d, 2 * n, dev),
        a_log: Tensor::from_data(TensorData::new(a_vals, [d, a_cols]), dev).require_grad(),
        d: Tensor::ones([d], dev).require_grad(),
        dt: lin(d, cfg.dt_rank, dev),
        dt_proj: if cfg.dt_proj { Some(lin(cfg.dt_rank, d, dev)) } else { None },
        theta: if cfg.use_complex_ssm { Some(lin(d, n / 2, dev)) } else { None },
    };
    SelectiveSsm { cfg: cfg.clone(), weights }
}

/// Force the lazy backend to finish pending work by reading a scalar reduction of `t`.
fn sync<B: Backend, const R: usize>(t: Tensor<B, R>) -> f32 {
    t.sum().into_data().to_vec::<f32>().unwrap()[0]
}

#[allow(clippy::too_many_arguments)]
fn cfg_for(
    mode: &str,
    d: usize,
    n: usize,
    chunk: usize,
    use_remat: bool,
    algo: ScanAlgo,
) -> SelectiveSsmConfig {
    SelectiveSsmConfig {
        hidden_features: n,
        reverse: false,
        complement: false,
        use_complex_ssm: mode == "complex",
        chunk_size: chunk,
        n_channel_groups: 1,
        use_remat,
        scan_algo: algo,
        dt_rank: (d + 15) / 16, // ceil(D/16), matching JAX 'auto'
        dt_proj: true,
        shift_conv_size: 3,
        activation: "silu".to_string(),
    }
}

/// Forward-only timing on a plain (non-autodiff) backend: matches inference / a jitted
/// forward with no tape.  Returns average milliseconds per forward.
#[allow(clippy::too_many_arguments)]
fn time_forward<B: ScanBackend>(
    mode: &str,
    b: usize,
    l: usize,
    d: usize,
    n: usize,
    chunk: usize,
    use_remat: bool,
    algo: ScanAlgo,
    iters: usize,
    dev: &B::Device,
) -> f64 {
    let cfg = cfg_for(mode, d, n, chunk, use_remat, algo);
    let model = build::<B>(&cfg, d, dev);
    let x = Tensor::<B, 3>::random([b, l, d], Distribution::Normal(0.0, 1.0), dev);
    // Warm up (shader compilation / allocation).
    for _ in 0..2 {
        let _ = sync(model.forward(x.clone()));
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = sync(model.forward(x.clone()));
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

/// Forward+backward timing on an Autodiff backend (loss = sum(y^2), grads to params+input).
#[allow(clippy::too_many_arguments)]
fn time_fwd_bwd<B: AutodiffBackend + ScanBackend>(
    mode: &str,
    b: usize,
    l: usize,
    d: usize,
    n: usize,
    chunk: usize,
    use_remat: bool,
    algo: ScanAlgo,
    iters: usize,
    dev: &B::Device,
) -> f64 {
    let cfg = cfg_for(mode, d, n, chunk, use_remat, algo);
    let model = build::<B>(&cfg, d, dev);
    let step = |x: Tensor<B, 3>| {
        let out = model.forward(x.clone());
        let loss = out.powf_scalar(2.0).sum();
        let grads = loss.backward();
        // Read a grad scalar to force the backward to actually run.
        x.grad(&grads).map(sync).unwrap_or(0.0)
    };
    let x = Tensor::<B, 3>::random([b, l, d], Distribution::Normal(0.0, 1.0), dev).require_grad();
    for _ in 0..2 {
        let _ = step(x.clone());
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = step(x.clone());
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 9 {
        eprintln!(
            "usage: scaling <cpu|gpu> <real|complex> <B> <L> <D> <N> <chunk> <iters> [remat|ref] [matrix|hillis]"
        );
        std::process::exit(2);
    }
    let device = a[1].as_str();
    let mode = a[2].clone();
    let b: usize = a[3].parse().unwrap();
    let l: usize = a[4].parse().unwrap();
    let d: usize = a[5].parse().unwrap();
    let n: usize = a[6].parse().unwrap();
    let chunk: usize = a[7].parse().unwrap();
    let iters: usize = a[8].parse().unwrap();
    // Scan variant: "remat" (default) recomputes chunk intermediates in backward; "ref" is
    // the reference scan that retains them.
    let scan = a.get(9).map(|s| s.as_str()).unwrap_or("remat");
    let use_remat = scan != "ref";
    // Within-chunk algorithm: "matrix" (default), "hillis", or "cubecl" (needs --features cubecl).
    let algo_str = a.get(10).map(|s| s.as_str()).unwrap_or("matrix");
    let algo = match algo_str {
        "hillis" => ScanAlgo::Hillis,
        "cubecl" => ScanAlgo::Cubecl,
        _ => ScanAlgo::Matrix,
    };

    let (fwd_ms, fwdbwd_ms) = match device {
        "cpu" => {
            use burn::backend::{Autodiff, NdArray};
            let dev = Default::default();
            let f = time_forward::<NdArray>(&mode, b, l, d, n, chunk, use_remat, algo, iters, &dev);
            let fb =
                time_fwd_bwd::<Autodiff<NdArray>>(&mode, b, l, d, n, chunk, use_remat, algo, iters, &dev);
            (f, fb)
        }
        "gpu" => {
            #[cfg(feature = "wgpu")]
            {
                use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
                let dev = WgpuDevice::default();
                let f = time_forward::<Wgpu>(&mode, b, l, d, n, chunk, use_remat, algo, iters, &dev);
                let fb =
                    time_fwd_bwd::<Autodiff<Wgpu>>(&mode, b, l, d, n, chunk, use_remat, algo, iters, &dev);
                (f, fb)
            }
            #[cfg(not(feature = "wgpu"))]
            {
                eprintln!("gpu requested but built without --features wgpu");
                std::process::exit(3);
            }
        }
        other => {
            eprintln!("unknown device {other:?}");
            std::process::exit(2);
        }
    };

    println!(
        "RESULT impl=rust,device={device},mode={mode},scan={scan},algo={algo_str},B={b},L={l},D={d},N={n},chunk={chunk},fwd_ms={fwd_ms:.4},fwdbwd_ms={fwdbwd_ms:.4},host_peak_mib={:.1}",
        peak_rss_mib()
    );
}
