//! `RcpsWrapper` --- reverse-complement parameter sharing, ported from the JAX `RCPSWrapper`.
//!
//! Severable: this module is only compiled with the `rcps` cargo feature.  It wraps a
//! `(B, L, D) -> (B, L, D)` inner module (here `BidirectionalMamba`) for exact RC
//! equivariance by running it on the sense strand and on the channel-flipped antisense
//! strand with shared weights, then concatenating.  There is no scan content of its own;
//! it is reshape / flip / weight-tying only.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::bidirectional::BidirectionalMamba;
use crate::config::BidirectionalMambaConfig;
use crate::loader::{Store, P};

/// Flip along both the sequence (dim 1) and channel (dim 2) axes.
fn rc_conjugate<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    x.flip([1, 2])
}

pub struct RcpsWrapper<B: Backend> {
    pub inner: BidirectionalMamba<B>,
}

impl<B: Backend> RcpsWrapper<B> {
    /// Load with the inner module under the `inner/` parameter prefix (matching Flax).
    pub fn load(store: &Store<B>, cfg: BidirectionalMambaConfig) -> Self {
        RcpsWrapper {
            inner: BidirectionalMamba::load(store, "inner/", cfg),
        }
    }

    pub fn named(&self, out: &mut Vec<(String, P<B>)>) {
        self.inner.named("inner/", out);
    }

    /// Forward pass, `(B, L, 2D) -> (B, L, 2D)`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [bb, l, dd] = x.dims();
        let d = dd / 2;
        let sense = x.clone().slice([0..bb, 0..l, 0..d]);
        let antisense = x.slice([0..bb, 0..l, d..dd]);

        let sense_out = self.inner.forward(sense);
        let antisense_out = rc_conjugate(self.inner.forward(rc_conjugate(antisense)));

        Tensor::cat(vec![sense_out, antisense_out], 2)
    }
}
