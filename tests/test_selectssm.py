"""Basic smoke tests for selectssm package."""

import jax
import jax.numpy as jnp
import flax.linen as nn
import pytest

from selectssm import SelectiveSSM, BidirectionalMamba, ssm_chunked_scan
from selectssm import ssm_recursive_scan, ssm_scan
from selectssm import RCPSWrapper, RCPSNorm, RCPSEmbedding, RCPSLMHead


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


class TestSelectiveSSM:
    def test_forward_shape(self, rng):
        model = SelectiveSSM(hidden_features=16)
        x = jax.random.normal(rng, (2, 32, 64))
        params = model.init(rng, x)
        y = model.apply(params, x)
        assert y.shape == x.shape

    def test_recursive_scan(self, rng):
        model = SelectiveSSM(hidden_features=16, recursive_scan=True)
        x = jax.random.normal(rng, (2, 32, 64))
        params = model.init(rng, x)
        y = model.apply(params, x)
        assert y.shape == x.shape

    def test_custom_vjp_scan(self, rng):
        model = SelectiveSSM(hidden_features=16, custom_vjp_scan=True)
        x = jax.random.normal(rng, (2, 32, 64))
        params = model.init(rng, x)
        y = model.apply(params, x)
        assert y.shape == x.shape


class TestBidirectionalMamba:
    def test_forward_shape(self, rng):
        model = BidirectionalMamba(hidden_features=16, expansion_factor=2.0)
        x = jax.random.normal(rng, (2, 32, 64))
        params = model.init(rng, x)
        y = model.apply(params, x)
        assert y.shape == x.shape

    def test_complement_mode(self, rng):
        model = BidirectionalMamba(
            hidden_features=16,
            expansion_factor=2.0,
            complement=True,
            tie_in_proj=True,
            tie_gate=True,
        )
        x = jax.random.normal(rng, (2, 32, 64))
        params = model.init(rng, x)
        y = model.apply(params, x)
        assert y.shape == x.shape

    def test_concatenate_fwd_rev(self, rng):
        model = BidirectionalMamba(
            hidden_features=16,
            expansion_factor=2.0,
            concatenate_fwd_rev=True,
        )
        x = jax.random.normal(rng, (2, 32, 64))
        params = model.init(rng, x)
        y = model.apply(params, x)
        assert y.shape == x.shape

    def test_gradient_flow(self, rng):
        model = BidirectionalMamba(hidden_features=16, expansion_factor=2.0)
        x = jax.random.normal(rng, (2, 16, 32))
        params = model.init(rng, x)

        def loss_fn(params):
            y = model.apply(params, x)
            return jnp.sum(y ** 2)

        grads = jax.grad(loss_fn)(params)
        leaves = jax.tree_util.tree_leaves(grads)
        assert all(jnp.isfinite(g).all() for g in leaves)
        assert any(jnp.abs(g).max() > 0 for g in leaves)


class TestRCPS:
    def test_wrapper_shape(self, rng):
        inner = nn.Dense(features=32)
        model = RCPSWrapper(inner_module=inner)
        x = jax.random.normal(rng, (2, 16, 64))  # 2D=64 -> D=32
        params = model.init(rng, x)
        y = model.apply(params, x)
        assert y.shape == x.shape

    def test_equivariance(self, rng):
        """RC(f(x)) == f(RC(x)) for RCPSWrapper."""
        inner = nn.Dense(features=32)
        model = RCPSWrapper(inner_module=inner)
        x = jax.random.normal(rng, (1, 16, 64))
        params = model.init(rng, x)

        def rc(z):
            return jnp.flip(z, axis=(-2, -1))

        y1 = rc(model.apply(params, x))
        y2 = model.apply(params, rc(x))
        assert jnp.allclose(y1, y2, atol=1e-5)

    def test_norm_shape(self, rng):
        model = RCPSNorm()
        x = jax.random.normal(rng, (2, 16, 64))
        params = model.init(rng, x)
        y = model.apply(params, x)
        assert y.shape == x.shape

    def test_embedding(self, rng):
        cmap = [3, 2, 1, 0]  # ACGT complement
        model = RCPSEmbedding(vocab_size=4, features=16, complement_map=cmap)
        tokens = jnp.array([[0, 1, 2, 3]])
        params = model.init(rng, tokens)
        y = model.apply(params, tokens)
        assert y.shape == (1, 4, 32)  # 2*D = 2*16 = 32

    def test_lm_head(self, rng):
        cmap = [3, 2, 1, 0]
        model = RCPSLMHead(vocab_size=4, complement_map=cmap)
        x = jax.random.normal(rng, (1, 8, 32))  # 2D=32
        params = model.init(rng, x)
        logits = model.apply(params, x)
        assert logits.shape == (1, 8, 4)
