"""Reverse Complement Parameter Sharing (RCPS) wrappers for RC-equivariant sequence models.

These modules enforce exact RC equivariance (a Z/2 symmetry) by doubling the channel
dimension to simultaneously carry sense and antisense representations, processed by
shared weights.  The antisense half is conjugated --- flipped along both the sequence
and channel axes --- before and after each layer, so the overall model commutes with
the reverse-complement operation on one-hot-encoded (or more generally, ordered-alphabet-
embedded) DNA sequences.

The group action is:

    RC: (B, L, D) -> (B, L, D)
    RC(x)[b, l, d] = x[b, L-1-l, D-1-d]

i.e. simultaneous reversal of the sequence and channel axes.  When D equals the
alphabet size and x is one-hot, this is equivalent to reversing the sequence and
replacing each base with its complement (given that the alphabet is ordered so that
complement pairs are symmetric about the midpoint, e.g. [A, C, G, T] or equivalently
any ordering where complement(i) = D-1-i).

For an arbitrary learned embedding of dimension D the channel-flip still defines
a valid Z/2 action; equivariance with respect to it guarantees that the model treats
a sequence and its reverse-complement identically (up to the same conjugation at
the output).

Reference: Schiff et al., "Caduceus: Bi-Directional Equivariant Long-Range DNA
Sequence Modeling", ICML 2024.
"""

from typing import Sequence, Callable
from dataclasses import field

import flax.linen as nn
import jax.numpy as jnp


def _rc_conjugate(x):
    """Flip along both sequence (axis -2) and channel (axis -1) axes."""
    return jnp.flip(x, axis=(-2, -1))


class RCPSWrapper(nn.Module):
    """Wraps any (B, L, D) -> (B, L, D) module for RC equivariance via channel doubling.

    Input/output shape: (B, L, 2*D).
    First D channels = sense strand, last D = antisense.
    Both halves are processed by the same inner module (shared weights).
    The antisense half is RC-conjugated (flipped along sequence and channel axes)
    before and after processing.

    The inner module receives tensors of shape (B, L, D) --- half the input width.
    It need not be RC-aware itself; the wrapper handles the symmetry.

    Args:
        module_cls: The nn.Module class to wrap (e.g. BidirectionalMamba).
        module_kwargs: Keyword arguments forwarded to module_cls().
    """
    module_cls: type
    module_kwargs: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, x, **kwargs):
        # x: (B, L, 2D)
        D = x.shape[-1] // 2
        sense = x[..., :D]       # (B, L, D)
        antisense = x[..., D:]   # (B, L, D)

        inner = self.module_cls(**self.module_kwargs, name='inner')
        sense_out = inner(sense, **kwargs)                                # (B, L, D)
        antisense_out = _rc_conjugate(inner(_rc_conjugate(antisense), **kwargs))  # (B, L, D)

        return jnp.concatenate([sense_out, antisense_out], axis=-1)  # (B, L, 2D)


class RCPSNorm(nn.Module):
    """RC-equivariant normalization.

    Applies a single shared normalization module to both halves, conjugating the
    antisense half so that learnable per-channel parameters (scale/bias) are applied
    in the corresponding order.

    Args:
        norm_cls: Normalization class (e.g. nn.RMSNorm, nn.LayerNorm).
        norm_kwargs: Extra kwargs for norm_cls.
    """
    norm_cls: type = None  # default: nn.RMSNorm
    norm_kwargs: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, x, **kwargs):
        D = x.shape[-1] // 2
        sense = x[..., :D]
        antisense = x[..., D:]

        norm_cls = self.norm_cls or nn.RMSNorm
        norm = norm_cls(**self.norm_kwargs, name='norm')
        sense = norm(sense, **kwargs)
        antisense = _rc_conjugate(norm(_rc_conjugate(antisense), **kwargs))

        return jnp.concatenate([sense, antisense], axis=-1)


class RCPSEmbedding(nn.Module):
    """RC-equivariant embedding layer.

    Maps token IDs (B, L) -> (B, L, 2*d_model) by concatenating:
      - sense:     embed(input_ids)
      - antisense: RC_conjugate(embed(RC(input_ids)))

    where RC(input_ids) reverses the sequence and maps each token through
    complement_map.

    Args:
        vocab_size: Number of tokens.
        d_model: Embedding dimension (output has 2*d_model channels).
        complement_map: Array of length vocab_size where complement_map[i] is the
            complement token ID for token i.  Should be an involution (applying it
            twice gives the identity) and a fixed point for non-base tokens
            (pad, mask, N, etc.).
    """
    vocab_size: int
    d_model: int
    complement_map: Sequence[int]

    @nn.compact
    def __call__(self, input_ids):
        # input_ids: (B, L) integer token IDs
        embed = nn.Embed(self.vocab_size, self.d_model, name='embed')

        sense = embed(input_ids)  # (B, L, D)

        cmap = jnp.array(self.complement_map, dtype=input_ids.dtype)
        rc_ids = jnp.flip(cmap[input_ids], axis=-1)   # reverse sequence + complement tokens
        antisense = _rc_conjugate(embed(rc_ids))       # conjugate the embedding

        return jnp.concatenate([sense, antisense], axis=-1)  # (B, L, 2D)


class RCPSLMHead(nn.Module):
    """RC-equivariant language model head.

    Input: (B, L, 2*D).  Output: (B, L, vocab_size).

    Produces logits by summing:
      - sense-strand logits (standard linear projection)
      - antisense-strand logits (channel-flipped, with complement-reindexed weights)

    This closes the equivariance loop: the model's prediction for base b at
    position i on the sense strand is reinforced by its prediction for
    complement(b) at the corresponding antisense position.

    Args:
        vocab_size: Number of output tokens.
        complement_map: Same as in RCPSEmbedding.
    """
    vocab_size: int
    complement_map: Sequence[int]

    @nn.compact
    def __call__(self, x):
        # x: (B, L, 2D)
        D = x.shape[-1] // 2
        sense = x[..., :D]        # (B, L, D)
        antisense = x[..., D:]    # (B, L, D)

        kernel = self.param('kernel', nn.initializers.lecun_normal(), (D, self.vocab_size))
        bias = self.param('bias', nn.initializers.zeros, (self.vocab_size,))

        cmap = jnp.array(self.complement_map)

        fwd_logits = sense @ kernel + bias                              # (B, L, V)
        rc_logits = jnp.flip(antisense, axis=-1) @ kernel[:, cmap] + bias  # (B, L, V)

        return fwd_logits + rc_logits
