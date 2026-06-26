# selectssm (JAX/Flax)

Selective SSM (Mamba) implementation in JAX/Flax, with multiple scanning strategies
for memory/compute tradeoffs and an optional complex-SSM "RoPE trick" for state tracking.

Based on [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
and [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569).

## Modules

- **`SelectiveSSM`** — Unidirectional selective state space model with causal convolution
  and configurable scanning strategy. With `reverse=True` it runs anti-causally (flips the
  sequence, scans, flips back).
- **`BidirectionalMamba`** — Non-causal wrapper combining a forward and a reverse
  `SelectiveSSM` with gated output projection. Every output position depends on the full
  input, so do not plug this into an autoregressive head.
- **`RCPSWrapper`, `RCPSNorm`, `RCPSEmbedding`, `RCPSLMHead`** — Reverse-complement
  parameter-sharing layers (after Schiff et al., Caduceus, ICML 2024). `RCPSWrapper` makes
  any `(B,L,D) → (B,L,D)` module exactly RC-equivariant.

## Complex SSM ("RoPE trick") — `use_complex_ssm`

`SelectiveSSM(use_complex_ssm=True)` enables a complex-valued diagonal SSM, implemented as a
real recurrence via the Mamba-3 data-dependent RoPE trick (arXiv:2603.15569, Prop. 2/3). This
is **not** standard fixed-frequency positional RoPE — the rotation angle is produced
data-dependently from the input and is what enables state-tracking tasks (e.g. parity) that a
purely real SSM cannot represent.

How it works:

1. A signed rotation angle `theta` (one per `(real, imag)` state pair) is projected from the
   input, analogous to how `dt` is produced.
2. The cumulative angle `Phi_t = sum_{s<=t} theta_s` is formed as a prefix sum, **carried
   across chunks** inside `ssm_chunked_scan`.
3. Both the `B` and `C` projections are rotated by `R(-Phi)` before the (unchanged) real
   recurrence. The `y = Cᵀh` contraction then realises the relative rotation `R(Phi_s - Phi_t)`
   between input and output positions — exactly as RoPE does for query/key, but with
   data-dependent angles.
4. The decay magnitude is tied across each state pair so that each 2×2 block is a scaled
   rotation, making the factorization exact.

Properties:

- **Off by default**, and feature-off is bitwise-identical to the plain real SSM.
- Requires `hidden_features` (N) to be even.
- Supported with the chunked scan only; combining it with `recursive_scan` or
  `custom_vjp_scan` raises an error.
- Composes with the reverse / bidirectional paths and **exactly** with `RCPSWrapper`.

## Scanning strategies

| Strategy | Flag | Description |
|---|---|---|
| Chunked associative scan | default | `jax.lax.associative_scan` within chunks, `jax.lax.scan` across chunks, with `@jax.remat` |
| Recursive scan | `recursive_scan=True` | Recursively splits sequence for lower peak memory |
| Custom VJP scan | `custom_vjp_scan=True` | Recursive scan with hand-written backward pass for minimal memory |

## Installation

This package lives in the `jax/` subdirectory of the repository:

```bash
pip install "git+ssh://git@github.com/ihh/selectssm.git#subdirectory=jax"
```

or, from a checkout:

```bash
pip install ./jax              # or: cd jax && pip install -e .
```

## Usage

```python
from selectssm import SelectiveSSM, BidirectionalMamba, RCPSWrapper

# Unidirectional (causal)
ssm = SelectiveSSM(hidden_features=16)

# Complex SSM via the RoPE trick (enables state tracking)
ssm_cplx = SelectiveSSM(hidden_features=16, use_complex_ssm=True)

# Bidirectional (non-causal) — no equivariance guarantee
bimamba = BidirectionalMamba(hidden_features=16, expansion_factor=2.0)

# Exactly RC-equivariant: wrap any (B,L,D)->(B,L,D) module in RCPSWrapper.
rc_bimamba = RCPSWrapper(
    module_cls=BidirectionalMamba,
    module_kwargs={'hidden_features': 16, 'expansion_factor': 2.0},
)
```

## Testing

```bash
pip install -e './jax[test]'
cd jax && pytest                      # all tests
pytest -m "not slow"                  # skip the parity training test
```

The parity test (`tests/test_parity.py`) trains a tiny one-layer model on short sequences and
evaluates on longer, held-out ones: `use_complex_ssm=True` learns parity and length-generalizes,
while `use_complex_ssm=False` stays near chance.

## Cross-language fixtures

`scripts/generate_fixtures.py` writes golden vectors (inputs, parameters, forward outputs, and
gradients) to the repo-level `fixtures/` directory. These are the shared source of truth that
the Rust port asserts parity against. Regenerate with:

```bash
pip install -e './jax[fixtures]'
python jax/scripts/generate_fixtures.py
```
