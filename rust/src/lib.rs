//! Selective SSM (Mamba) ported to Rust/burn, with the Mamba-3 complex-SSM "RoPE trick".
//!
//! This is a port of the JAX/Flax `selectssm` package (see `../jax`).  It implements the
//! chunked-scan strategy only: matmul-form computation within each chunk plus a plain
//! sequential loop across chunks (no associative-scan / prefix-sum primitive is assumed).
//! The complex-SSM RoPE trick carries the cumulative rotation angle across chunks explicitly.
//!
//! Backends: `ndarray` (CPU, deterministic, default) and `wgpu` (WebGPU, feature `wgpu`),
//! both wrapped in `burn::backend::Autodiff` so gradients are available and testable.

// The wgpu backend nests backend types deeply; bump the recursion limit for it.
#![recursion_limit = "256"]

pub mod config;
pub mod loader;
pub mod chunked_scan;
pub mod selective_ssm;
pub mod bidirectional;

#[cfg(feature = "rcps")]
pub mod rcps;

pub use config::{BidirectionalMambaConfig, SelectiveSsmConfig};
pub use selective_ssm::SelectiveSsm;
pub use bidirectional::BidirectionalMamba;
