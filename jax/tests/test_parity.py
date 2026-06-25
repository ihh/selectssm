"""Functional state-tracking test: PARITY with length generalization.

This is the qualitative correctness signal for the complex-SSM "RoPE trick"
(Mamba-3, arXiv:2603.15569, Table 5b), following the Chomsky-hierarchy protocol of
Grazzi et al. (2025): a tiny one-layer model is trained on short binary sequences
and evaluated on *longer*, held-out ones.

* ``use_complex_ssm=True`` (rotational dynamics) learns parity and generalizes to
  sequences longer than any seen in training.
* ``use_complex_ssm=False`` (purely real SSM) fails --- it stays near chance.

The asymmetry is the gate.  A real SSM that also solved parity would mean the
rotation is leaking expressivity it should not; a complex SSM that failed would
mean the rotation is broken.  Step counts are kept modest for CI; the
solves-vs-chance gap, not a precise number, is what is asserted.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import pytest

from selectssm import SelectiveSSM


class ParityModel(nn.Module):
    """Embed bits -> one causal SelectiveSSM layer -> per-position parity logits."""
    d_model: int = 32
    hidden_features: int = 16
    use_complex_ssm: bool = False

    @nn.compact
    def __call__(self, ids):
        x = nn.Embed(2, self.d_model)(ids)
        x = SelectiveSSM(hidden_features=self.hidden_features,
                         use_complex_ssm=self.use_complex_ssm, name="ssm")(x)
        x = nn.RMSNorm()(x)
        return nn.Dense(2)(x)  # running-parity logits at every position


def _make_batch(key, n, L):
    bits = jax.random.bernoulli(key, 0.5, (n, L)).astype(jnp.int32)
    parity = jnp.cumsum(bits, axis=1) % 2
    return bits, parity


def _train_and_eval(use_complex_ssm, *, steps=500, bs=64, lmin=8, ltr=20, lr=3e-3,
                    eval_len=32, seed=0):
    key = jax.random.PRNGKey(seed)
    model = ParityModel(use_complex_ssm=use_complex_ssm)
    k0, key = jax.random.split(key)
    params = model.init(k0, jnp.zeros((1, ltr), jnp.int32))
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def loss_fn(p, ids, tgt):
        logits = model.apply(p, ids)
        return optax.softmax_cross_entropy_with_integer_labels(logits, tgt).mean()

    @jax.jit
    def step(p, opt_state, ids, tgt):
        loss, grads = jax.value_and_grad(loss_fn)(p, ids, tgt)
        updates, opt_state = opt.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss

    for _ in range(steps):
        key, kb, kl = jax.random.split(key, 3)
        # even lengths in [lmin, ltr] keep the number of jit recompiles small
        L = int(jax.random.randint(kl, (), lmin // 2, ltr // 2 + 1)) * 2
        ids, tgt = _make_batch(kb, bs, L)
        params, opt_state, _ = step(params, opt_state, ids, tgt)

    # evaluate on sequences LONGER than anything seen in training
    ke = jax.random.PRNGKey(999 + eval_len)
    ids, tgt = _make_batch(ke, 512, eval_len)
    pred = jnp.argmax(model.apply(params, ids), -1)
    return float((pred == tgt).mean())


@pytest.mark.slow
def test_parity_complex_solves_real_fails():
    # trained on lengths <= 20, evaluated on length 32 (longer than training)
    acc_on = _train_and_eval(use_complex_ssm=True, eval_len=32)
    acc_off = _train_and_eval(use_complex_ssm=False, eval_len=32)
    assert acc_on > 0.9, f"complex SSM failed to length-generalize on parity: {acc_on:.3f}"
    assert acc_off < 0.7, f"real SSM unexpectedly solved parity: {acc_off:.3f}"
    assert acc_on - acc_off > 0.25
