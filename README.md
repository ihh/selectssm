# selectssm

Selective SSM (Mamba) implementation in JAX/Flax, with multiple scanning strategies for memory/compute tradeoffs.

Based on [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).

## Modules

- **`SelectiveSSM`** — Unidirectional selective state space model with causal convolution and configurable scanning strategy. With `reverse=True` it runs anti-causally (flips the sequence, scans, flips back).
- **`BidirectionalMamba`** — Non-causal wrapper combining a forward and a reverse `SelectiveSSM` with gated output projection. Every output position depends on the full input, so do not plug this into an autoregressive head.
- **`RCPSWrapper`, `RCPSNorm`, `RCPSEmbedding`, `RCPSLMHead`** — Reverse-complement parameter-sharing layers (after Schiff et al., Caduceus, ICML 2024). `RCPSWrapper` makes any `(B,L,D) → (B,L,D)` module exactly RC-equivariant by running it on the sense strand and on the channel-flipped antisense strand with shared weights, then concatenating.

### A note on RC equivariance

Two RC-handling paths are available, and they are not equivalent:

- `BidirectionalMamba(complement=True, tie_in_proj=True, tie_gate=True, concatenate_fwd_rev=True)` — an *approximate*, learned RC-handling scheme. The inner SSM's `A`, `BC`, `dt`, and depthwise conv parameters are not constrained to be channel-flip-symmetric, so the operator is not exactly RC-equivariant; the symmetry has to be learned.
- `RCPSWrapper(module_cls=BidirectionalMamba, module_kwargs={…})` — *exact* RC equivariance, by construction, for any inner module. This is the recommended path when equivariance is required.

## Scanning strategies

| Strategy | Flag | Description |
|---|---|---|
| Chunked associative scan | default | `jax.lax.associative_scan` within chunks, `jax.lax.scan` across chunks, with `@jax.remat` |
| Recursive scan | `recursive_scan=True` | Recursively splits sequence for lower peak memory |
| Custom VJP scan | `custom_vjp_scan=True` | Recursive scan with hand-written backward pass for minimal memory |

## Installation

```bash
pip install git+ssh://git@github.com/ihh/selectssm.git
```

## Usage

```python
from selectssm import SelectiveSSM, BidirectionalMamba, RCPSWrapper

# Unidirectional (causal)
ssm = SelectiveSSM(hidden_features=16)

# Bidirectional (non-causal) — no equivariance guarantee
bimamba = BidirectionalMamba(hidden_features=16, expansion_factor=2.0)

# Exactly RC-equivariant: wrap any (B,L,D)->(B,L,D) module in RCPSWrapper.
# Input/output have 2*D channels (sense || antisense).
rc_bimamba = RCPSWrapper(
    module_cls=BidirectionalMamba,
    module_kwargs={'hidden_features': 16, 'expansion_factor': 2.0},
)
```
