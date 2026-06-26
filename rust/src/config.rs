//! Configuration structs.  Field names and semantics match the JAX implementation
//! (`jax/src/selectssm/selectssm.py`) and the fixture manifest exactly, so the two
//! ports read in parallel.  These are deserialized straight from `fixtures/manifest.json`.

use serde::Deserialize;

fn one() -> usize { 1 }
fn three() -> usize { 3 }
fn yes() -> bool { true }
fn silu() -> String { "silu".to_string() }

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
            dt_rank: self.dt_rank,
            dt_proj: self.dt_proj,
            shift_conv_size: self.shift_conv_size,
            activation: self.activation.clone(),
        }
    }
}
