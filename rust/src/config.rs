//! Configuration structs.  Field names and semantics match the JAX implementation
//! (`jax/src/selectssm/selectssm.py`) and the fixture manifest exactly, so the two
//! ports read in parallel.  These are deserialized straight from `fixtures/manifest.json`.

use serde::Deserialize;

fn one() -> usize { 1 }
fn three() -> usize { 3 }
fn yes() -> bool { true }
fn silu() -> String { "silu".to_string() }
// Hillis–Steele is the default within-chunk scan: ~3–4× faster than the matrix form at equal
// memory/parity, pure burn ops (all backends), and free of the matrix form's `cs²` GPU-buffer
// blow-up.  `ScanAlgo::Cubecl` is faster still where the `cubecl` feature + wgpu are available.
fn hillis() -> ScanAlgo { ScanAlgo::Hillis }

/// Within-chunk scan algorithm.  All variants compute the same linear SSM (parity within the
/// usual floating-point tolerance), trading work, memory, and parallelism:
///
/// * `Matrix` — materialise the `cs×cs` decay matrix and contract it; `O(cs²)` work/transient
///   memory per chunk, GPU-friendly (one big matmul-like op) but the most memory-hungry.
/// * `Hillis` — a Hillis–Steele parallel prefix scan; `O(cs·log cs)` work, no `cs×cs` tensor.
///   The portable (all-backends) analogue of `jax.lax.associative_scan`.
/// * `Cubecl` — a custom GPU scan kernel (one thread per state channel, sequential over the
///   chunk).  Requires the `cubecl` feature and the wgpu backend inside the rematerializing
///   path; everywhere else it transparently falls back to `Hillis`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScanAlgo {
    Matrix,
    Hillis,
    Cubecl,
}

/// Configuration for [`crate::SelectiveSsm`] (mirrors the JAX `SelectiveSSM` module).
#[derive(Debug, Clone, Deserialize)]
pub struct SelectiveSsmConfig {
    /// State width N (must be even when `use_complex_ssm`).
    pub hidden_features: usize,
    #[serde(default)]
    pub reverse: bool,
    #[serde(default)]
    pub complement: bool,
    /// Enable the complex-SSM "RoPE trick".
    #[serde(default)]
    pub use_complex_ssm: bool,
    pub chunk_size: usize,
    #[serde(default = "one")]
    pub n_channel_groups: usize,
    /// Use the rematerializing chunked scan (recompute chunk intermediates in the backward
    /// pass) instead of the reference scan that retains them.  Default on; bitwise-identical
    /// forward output, far lower training memory.  See [`crate::remat`].
    #[serde(default = "yes")]
    pub use_remat: bool,
    /// Within-chunk scan algorithm (see [`ScanAlgo`]).  Default `Hillis`.
    #[serde(default = "hillis")]
    pub scan_algo: ScanAlgo,
    /// Resolved (integer) low-rank dt projection width R.
    pub dt_rank: usize,
    #[serde(default = "yes")]
    pub dt_proj: bool,
    #[serde(default = "three")]
    pub shift_conv_size: usize,
    #[serde(default = "silu")]
    pub activation: String,
}

/// Configuration for [`crate::BidirectionalMamba`] (mirrors the JAX `BidirectionalMamba`).
#[derive(Debug, Clone, Deserialize)]
pub struct BidirectionalMambaConfig {
    pub hidden_features: usize,
    pub expansion_factor: f64,
    pub dt_rank: usize,
    #[serde(default)]
    pub complement: bool,
    #[serde(default)]
    pub tie_in_proj: bool,
    #[serde(default)]
    pub tie_gate: bool,
    #[serde(default = "yes")]
    pub concatenate_fwd_rev: bool,
    #[serde(default = "silu")]
    pub activation: String,
    #[serde(default)]
    pub norm_type: String,
    #[serde(default)]
    pub mlp_layer: bool,
    #[serde(default)]
    pub use_complex_ssm: bool,
    pub chunk_size: usize,
    #[serde(default = "one")]
    pub n_channel_groups: usize,
    #[serde(default = "yes")]
    pub use_remat: bool,
    #[serde(default = "hillis")]
    pub scan_algo: ScanAlgo,
    #[serde(default = "yes")]
    pub dt_proj: bool,
    #[serde(default = "three")]
    pub shift_conv_size: usize,
}

impl BidirectionalMambaConfig {
    /// Build the inner `SelectiveSsmConfig` for the forward (`reverse=false`) or reverse half.
    pub fn inner_ssm(&self, reverse: bool) -> SelectiveSsmConfig {
        SelectiveSsmConfig {
            hidden_features: self.hidden_features,
            reverse,
            complement: reverse && self.complement,
            use_complex_ssm: self.use_complex_ssm,
            chunk_size: self.chunk_size,
            n_channel_groups: self.n_channel_groups,
            use_remat: self.use_remat,
            scan_algo: self.scan_algo,
            dt_rank: self.dt_rank,
            dt_proj: self.dt_proj,
            shift_conv_size: self.shift_conv_size,
            activation: self.activation.clone(),
        }
    }
}
