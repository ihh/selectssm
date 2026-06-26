"""Tests for the complex-SSM "RoPE trick" (Mamba-3, arXiv:2603.15569, Prop. 2/3).

These cover, for ``use_complex_ssm``:

* feature-off structural identity (no new parameters; ``A_log`` keeps its full
  width), so the real SSM is unchanged.  Bitwise identity of outputs/gradients
  versus the pre-change implementation was additionally verified out-of-band by
  capturing a baseline before the change and asserting ``np.array_equal``.
* feature-on shape-correctness and differentiability (including w.r.t. ``theta``);
* the padding path for sequence lengths that are not a multiple of the chunk size;
* composition with the reverse / bidirectional paths and the RCPS wrapper;
* a direct check that the RoPE-trick chunked scan equals an explicit naive complex
  (block-rotation) recurrence --- i.e. that the algebra and sign conventions are
  correct, not merely shape-consistent.
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import pytest

from selectssm import SelectiveSSM, BidirectionalMamba, RCPSWrapper
from selectssm.selectssm import ssm_chunked_scan, _apply_rotary


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


def _grads(model, x, seed=0):
    params = model.init(jax.random.PRNGKey(seed), x)
    y = model.apply(params, x)

    def loss(p):
        return jnp.sum(model.apply(p, x) ** 2)

    g = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(g)
    return params, y, leaves


def _param_keys(params):
    return {"/".join(str(k.key) for k in path)
            for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]}


# ----------------------------------------------------------------------------
# Naive reference: Proposition 2, eq. (9) --- rotate the STATE each step by the
# block-diagonal R(+theta), with a per-pair-shared decay magnitude and *unrotated*
# B, C.  This is the complex SSM written as a real recurrence WITHOUT the RoPE
# trick.  ssm_chunked_scan (which uses the trick) must reproduce it exactly.
# ----------------------------------------------------------------------------
def naive_block_rotation(x, A, B, C, dt, theta):
    x = np.asarray(x, np.float64); A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64); C = np.asarray(C, np.float64)
    dt = np.asarray(dt, np.float64); theta = np.asarray(theta, np.float64)
    Bb, L, D = x.shape
    N = A.shape[-1]; H = N // 2
    y = np.zeros((Bb, L, D))
    for b in range(Bb):
        h = np.zeros((D, N))
        for t in range(L):
            cos = np.cos(theta[b, t]); sin = np.sin(theta[b, t])
            re, im = h[:, :H].copy(), h[:, H:].copy()
            h_rot = np.concatenate([re * cos - im * sin, re * sin + im * cos], axis=-1)
            dA = np.exp(dt[b, t][:, None] * A)
            dB = dt[b, t][:, None] * B[b, t][None, :] * x[b, t][:, None]
            h = dA * h_rot + dB
            y[b, t] = h @ C[b, t]
    return y


class TestComplexOff:
    def test_no_theta_param_and_full_width_A(self, rng):
        x = jax.random.normal(rng, (2, 24, 32))
        off = SelectiveSSM(hidden_features=16, use_complex_ssm=False).init(rng, x)
        on = SelectiveSSM(hidden_features=16, use_complex_ssm=True).init(rng, x)
        koff, kon = _param_keys(off), _param_keys(on)
        # feature-off introduces no parameters that feature-on lacks the other way:
        assert not any("theta" in k for k in koff)
        assert any("theta" in k for k in kon)
        # A_log is full width (D, N) when off, half width (D, N/2) when on
        assert off["params"]["A_log"].shape == (32, 16)
        assert on["params"]["A_log"].shape == (32, 8)

    def test_theta_zero_is_bitwise_real(self, rng):
        # Rotation by a zero angle is the exact identity, so theta=0 must reproduce
        # the plain real scan (theta=None) bit-for-bit.
        ks = jax.random.split(rng, 5)
        x = jax.random.normal(ks[0], (2, 12, 8))
        A = -jnp.exp(jax.random.normal(ks[1], (8, 16)))
        B = jax.random.normal(ks[2], (2, 12, 16))
        C = jax.random.normal(ks[3], (2, 12, 16))
        dt = jax.nn.softplus(jax.random.normal(ks[4], (2, 12, 8)))
        y_real = ssm_chunked_scan(x, A, B, C, dt, theta=None, chunk_size=4)
        y_zero = ssm_chunked_scan(x, A, B, C, dt, theta=jnp.zeros((2, 12, 8)), chunk_size=4)
        assert np.array_equal(np.asarray(y_real), np.asarray(y_zero))

    def test_odd_hidden_features_raises(self, rng):
        x = jax.random.normal(rng, (2, 8, 32))
        with pytest.raises(ValueError, match="even"):
            SelectiveSSM(hidden_features=15, use_complex_ssm=True).init(rng, x)

    @pytest.mark.parametrize("flag", ["recursive_scan", "custom_vjp_scan"])
    def test_mutually_exclusive_with_other_scans(self, rng, flag):
        x = jax.random.normal(rng, (2, 8, 32))
        with pytest.raises(ValueError, match="chunked scan"):
            SelectiveSSM(hidden_features=16, use_complex_ssm=True, **{flag: True}).init(rng, x)


class TestComplexOn:
    def test_forward_shape_and_diff(self, rng):
        x = jax.random.normal(rng, (2, 24, 32))
        params, y, leaves = _grads(SelectiveSSM(hidden_features=16, use_complex_ssm=True), x)
        assert y.shape == x.shape
        assert all(jnp.isfinite(g).all() for g in leaves)
        assert any(jnp.abs(g).max() > 0 for g in leaves)
        # the theta projection actually receives gradient
        theta_grad = jax.grad(lambda p: jnp.sum(
            SelectiveSSM(hidden_features=16, use_complex_ssm=True).apply(p, x) ** 2))(params)
        assert jnp.abs(theta_grad["params"]["theta"]["kernel"]).max() > 0

    @pytest.mark.parametrize("chunk_size", [5, 7])
    def test_padding_path_nonmultiple_chunk(self, rng, chunk_size):
        # L=24 is not a multiple of 5 or 7 -> internal padding path.
        x = jax.random.normal(rng, (2, 24, 32))
        _, y, leaves = _grads(
            SelectiveSSM(hidden_features=16, use_complex_ssm=True, chunk_size=chunk_size), x)
        assert y.shape == x.shape
        assert all(jnp.isfinite(g).all() for g in leaves)

    def test_reverse(self, rng):
        x = jax.random.normal(rng, (2, 24, 32))
        _, y, _ = _grads(SelectiveSSM(hidden_features=16, use_complex_ssm=True, reverse=True), x)
        assert y.shape == x.shape

    def test_bidirectional(self, rng):
        x = jax.random.normal(rng, (2, 16, 32))
        _, y, _ = _grads(
            BidirectionalMamba(hidden_features=8, expansion_factor=2.0,
                               ssm_args={"use_complex_ssm": True}), x)
        assert y.shape == x.shape

    def test_rcps_equivariance_exact(self, rng):
        model = RCPSWrapper(module_cls=BidirectionalMamba,
                            module_kwargs={"hidden_features": 8, "expansion_factor": 2.0,
                                           "ssm_args": {"use_complex_ssm": True}})
        x = jax.random.normal(rng, (1, 16, 32))
        params = model.init(rng, x)

        def rc(z):
            return jnp.flip(z, axis=(-2, -1))

        y1 = rc(model.apply(params, x))
        y2 = model.apply(params, rc(x))
        assert jnp.allclose(y1, y2, atol=1e-4)


class TestRopeMath:
    @pytest.mark.parametrize("chunk_size", [12, 6, 4, 3, 5])  # 5 exercises padding
    def test_matches_naive_complex_recurrence(self, chunk_size):
        ks = jax.random.split(jax.random.PRNGKey(0), 6)
        Bb, L, D, N = 2, 12, 8, 8
        H = N // 2
        x = jax.random.normal(ks[0], (Bb, L, D))
        A_half = -jnp.exp(jax.random.normal(ks[1], (D, H)) * 0.5)
        A = jnp.concatenate([A_half, A_half], axis=-1)  # block-shared decay
        B = jax.random.normal(ks[2], (Bb, L, N))
        C = jax.random.normal(ks[3], (Bb, L, N))
        dt = jax.nn.softplus(jax.random.normal(ks[4], (Bb, L, D))) * 0.1
        theta = jax.random.normal(ks[5], (Bb, L, H)) * 1.3
        y_rope = np.asarray(ssm_chunked_scan(x, A, B, C, dt, theta=theta, chunk_size=chunk_size))
        y_naive = naive_block_rotation(x, A, B, C, dt, theta)
        rel = np.abs(y_rope - y_naive).max() / (np.abs(y_naive).max() + 1e-9)
        assert rel < 1e-5, f"chunk_size={chunk_size} rel={rel}"

    def test_apply_rotary_is_rotation(self):
        # _apply_rotary applies R(-Phi); applying R(+Phi) afterwards must recover v.
        v = jax.random.normal(jax.random.PRNGKey(1), (3, 8))
        Phi = jax.random.normal(jax.random.PRNGKey(2), (3, 4))
        rot = _apply_rotary(v, Phi)
        back = _apply_rotary(rot, -Phi)
        assert jnp.allclose(v, back, atol=1e-5)
        # norm preserved per pair
        assert jnp.allclose(jnp.sum(v ** 2, -1), jnp.sum(rot ** 2, -1), atol=1e-5)
