# selectssm

Selective SSM (Mamba) implementation in JAX/Flax, with multiple scanning strategies for memory/compute tradeoffs.

Based on [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).

## Modules

- **`SelectiveSSM`** — Unidirectional selective state space model with causal convolution, configurable scanning strategy, and optional RC-complement reversal.
- **`BidirectionalMamba`** — Bidirectional wrapper combining forward and reverse `SelectiveSSM` instances with gated output projection. Supports RC-equivariance for RNA via tied projections and complement reversal.

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
from selectssm import SelectiveSSM, BidirectionalMamba

# Unidirectional
ssm = SelectiveSSM(hidden_features=16)

# Bidirectional with RC-equivariance
bimamba = BidirectionalMamba(
    hidden_features=16,
    expansion_factor=2.0,
    complement=True,
    tie_in_proj=True,
    tie_gate=True,
    concatenate_fwd_rev=True,
)
```
