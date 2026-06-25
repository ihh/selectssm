"""Generate cross-language golden vectors (the parity oracle).

JAX is treated as the reference implementation.  From seeded inputs and seeded
parameters this script serializes, for a set of model configurations:

  * the input ``x``,
  * every parameter (flattened Flax path -> tensor),
  * the forward output ``y``,
  * the gradient of a fixed scalar loss ``L = sum(y**2)`` w.r.t. the input and
    w.r.t. every parameter (a VJP).

Each case becomes one ``fixtures/<name>.safetensors`` file plus an entry in
``fixtures/manifest.json`` recording the fully-resolved config.  The Rust test
suite loads these and asserts forward (~1e-4 rel) and gradient (~1e-3 rel) parity.

The cases cover ``{causal, reverse, bidirectional, rcps} x {complex off, on}`` and
include one sequence length that is NOT a multiple of the chunk size (the internal
padding path).  Everything is seeded; this fixture set is the shared source of truth.

Run from the repo root:  ``python jax/scripts/generate_fixtures.py``
"""

import json
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from selectssm import SelectiveSSM, BidirectionalMamba, RCPSWrapper

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"


def flat_params(params):
    """Flatten a Flax param pytree to {"/".join(path-without-leading-'params'): array}."""
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        keys = [str(k.key) for k in path]
        if keys and keys[0] == "params":
            keys = keys[1:]
        out["/".join(keys)] = np.asarray(leaf, dtype=np.float32)
    return out


def loss_fn_factory(model):
    def loss(params, x):
        return jnp.sum(model.apply(params, x) ** 2)
    return loss


def make_case(name, kind, model, x, seed, config):
    """Init, run forward, take grads w.r.t. input and params; return a flat tensor dict + meta."""
    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, x)
    y = model.apply(params, x)

    loss = loss_fn_factory(model)
    grad_params = jax.grad(loss, argnums=0)(params, x)
    grad_x = jax.grad(loss, argnums=1)(params, x)

    tensors = {"input": np.asarray(x, np.float32),
               "output": np.asarray(y, np.float32),
               "grad_input": np.asarray(grad_x, np.float32)}
    for k, v in flat_params(params).items():
        tensors["param/" + k] = v
    for k, v in flat_params(grad_params).items():
        tensors["grad_param/" + k] = v

    meta = {
        "name": name,
        "kind": kind,
        "seed": seed,
        "input_shape": list(x.shape),
        "loss": "sum(y**2)",
        "config": config,
        "param_shapes": {k: list(v.shape) for k, v in flat_params(params).items()},
        "file": name + ".safetensors",
    }
    return tensors, meta


def selective_config(**over):
    cfg = dict(hidden_features=8, reverse=False, complement=False, use_complex_ssm=False,
               chunk_size=4, n_channel_groups=1, dt_rank=2, dt_proj=True,
               dt_min=0.001, dt_max=0.1, shift_conv_size=3, activation="silu")
    cfg.update(over)
    return cfg


def bidir_config(**over):
    cfg = dict(hidden_features=8, expansion_factor=2.0, dt_rank=2, complement=False,
               tie_in_proj=False, tie_gate=False, concatenate_fwd_rev=True,
               activation="silu", norm_type="rms", mlp_layer=False,
               ssm=selective_config(hidden_features=8, dt_rank=2))
    # the inner SSM inherits expansion-resolved width; record the salient flags only
    cfg.update(over)
    return cfg


def build_cases():
    cases = []
    B = 2

    # ---- SelectiveSSM: causal / reverse x complex off / on ----
    for reverse in (False, True):
        for complex_on in (False, True):
            name = f"selective_{'rev' if reverse else 'fwd'}_{'cplx' if complex_on else 'real'}"
            cfg = selective_config(reverse=reverse, use_complex_ssm=complex_on)
            model = SelectiveSSM(hidden_features=cfg["hidden_features"], reverse=reverse,
                                 use_complex_ssm=complex_on, chunk_size=cfg["chunk_size"],
                                 dt_rank=cfg["dt_rank"])
            x = jax.random.normal(jax.random.PRNGKey(100 + len(cases)), (B, 16, 8))
            cases.append((name, "selective_ssm", model, x, 100 + len(cases), cfg))

    # ---- SelectiveSSM: non-multiple-of-chunk length (padding path), complex on ----
    cfg = selective_config(use_complex_ssm=True, chunk_size=6)  # L=20 not a multiple of 6
    model = SelectiveSSM(hidden_features=cfg["hidden_features"], use_complex_ssm=True,
                         chunk_size=6, dt_rank=cfg["dt_rank"])
    x = jax.random.normal(jax.random.PRNGKey(200), (B, 20, 8))
    cases.append(("selective_fwd_cplx_pad", "selective_ssm", model, x, 200, cfg))

    # ---- BidirectionalMamba: complex off / on ----
    for complex_on in (False, True):
        name = f"bidirectional_{'cplx' if complex_on else 'real'}"
        cfg = bidir_config()
        cfg["use_complex_ssm"] = complex_on
        model = BidirectionalMamba(hidden_features=8, expansion_factor=2.0, dt_rank=2,
                                   ssm_args={"use_complex_ssm": complex_on, "chunk_size": 4})
        x = jax.random.normal(jax.random.PRNGKey(300 + len(cases)), (B, 16, 8))
        cases.append((name, "bidirectional", model, x, 300 + len(cases), cfg))

    # ---- RCPSWrapper(BidirectionalMamba): complex off / on ----
    for complex_on in (False, True):
        name = f"rcps_{'cplx' if complex_on else 'real'}"
        cfg = bidir_config()
        cfg["use_complex_ssm"] = complex_on
        cfg["rcps"] = True
        model = RCPSWrapper(module_cls=BidirectionalMamba,
                            module_kwargs={"hidden_features": 8, "expansion_factor": 2.0,
                                           "dt_rank": 2,
                                           "ssm_args": {"use_complex_ssm": complex_on,
                                                        "chunk_size": 4}})
        x = jax.random.normal(jax.random.PRNGKey(400 + len(cases)), (B, 16, 16))  # 2D channels
        cases.append((name, "rcps", model, x, 400 + len(cases), cfg))

    return cases


def main():
    try:
        from safetensors.numpy import save_file
    except ImportError as e:
        raise SystemExit("safetensors is required: pip install -e 'jax[fixtures]'") from e

    FIXTURES_DIR.mkdir(exist_ok=True)
    manifest = {"description": "Golden vectors generated by jax/scripts/generate_fixtures.py "
                               "(JAX reference). Forward and gradient parity oracle for the Rust port.",
                "loss": "sum(y**2)", "cases": []}

    for name, kind, model, x, seed, config in build_cases():
        tensors, meta = make_case(name, kind, model, x, seed, config)
        save_file(tensors, str(FIXTURES_DIR / meta["file"]))
        manifest["cases"].append(meta)
        n_params = sum(1 for k in tensors if k.startswith("param/"))
        print(f"  {name:28s} {kind:14s} x{list(x.shape)}  params={n_params}  "
              f"|y|={float(jnp.sum(model.apply(model.init(jax.random.PRNGKey(seed), x), x)**2)):.4f}")

    (FIXTURES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest['cases'])} cases + manifest.json to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
