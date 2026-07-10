//! A tiny, fully seeded splitmix64 RNG for deterministic from-scratch weight initialization.
//!
//! This is deliberately NOT a cryptographic or high-quality statistical generator — it is a
//! reproducible source of `f32` draws so that `SelectiveSsm::init` / `BidirectionalMamba::init`
//! produce the same weights given the same seed.  Bit-matching JAX's PRNG is explicitly a
//! non-goal (JAX uses threefry); what matters is that the init *distributions* are correct
//! (see [`crate::loader`] init methods).  The generator is splitmix64 (Steele et al. 2014),
//! the same one the parity-learning integration test used before it was promoted here.

/// A seeded splitmix64 pseudo-random generator.
pub struct Rng(pub u64);

impl Rng {
    /// Construct from a seed.
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    /// Next raw 64-bit value (splitmix64 step).
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform draw in `[0, 1)` (24 bits of mantissa).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A uniform draw in `[lo, hi)`.
    pub fn uniform_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// A standard normal draw (Box–Muller, one of the pair).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// A single random bit as `0.0` / `1.0` (used by the parity test's data generator).
    pub fn bit(&mut self) -> f32 {
        (self.next_u64() & 1) as f32
    }
}
