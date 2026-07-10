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
    """Init, run forward, take grads w.r.t. input and params; return a flat tensor dict + meta.

    Supports modules with auxiliary collections (e.g. BatchNorm's ``batch_stats``): the loss
    and gradients are taken only w.r.t. the ``params`` collection (at inference, ``train=False``,
    the running stats are constants), and the auxiliary collections are exported alongside the
    params under the same ``param/`` namespace so the Rust loader can read them.
    """
    rng = jax.random.PRNGKey(seed)
    variables = model.init(rng, x)
    # Split trainable params from any auxiliary collections (batch_stats, ...).
    if isinstance(variables, dict) and "params" in variables and len(variables) > 1:
        params = {"params": variables["params"]}
        aux = {k: v for k, v in variables.items() if k != "params"}
    else:
        params = variables
        aux = {}

    def apply_all(p):
        return model.apply({**p, **aux}, x, train=False) if aux else model.apply(p, x)

    y = apply_all(params)

    def loss(p, xx):
        return jnp.sum((model.apply({**p, **aux}, xx, train=False) if aux else model.apply(p, xx)) ** 2)

    grad_params = jax.grad(loss, argnums=0)(params, x)
    grad_x = jax.grad(loss, argnums=1)(params, x)

    tensors = {"input": np.asarray(x, np.float32),
               "output": np.asarray(y, np.float32),
               "grad_input": np.asarray(grad_x, np.float32)}
    for k, v in flat_params(params).items():
        tensors["param/" + k] = v
    for k, v in flat_params(grad_params).items():
        tensors["grad_param/" + k] = v
    # Auxiliary collections (e.g. batch_stats mean/var) share the param/ namespace; the Flax
    # path (minus the leading collection key) is already stripped by flat_params.
    for coll_name, coll in aux.items():
        for k, v in flat_params({"params": coll}).items():
            tensors["param/" + k] = v

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
               shift_conv_size=3, activation="silu")
    cfg.update(over)
    return cfg


def bidir_config(**over):
    # Flat config carrying everything the Rust forward needs (inner-SSM fields included).
    cfg = dict(hidden_features=8, expansion_factor=2.0, dt_rank=2, complement=False,
               tie_in_proj=False, tie_gate=False, concatenate_fwd_rev=True,
               activation="silu", norm_type="rms", bn_momentum=0.9, mlp_layer=False,
               dense_expansion=2, mlp_dropout_rate=0.1, use_complex_ssm=False,
               chunk_size=4, n_channel_groups=1, dt_proj=True, shift_conv_size=3)
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

    # ---- SelectiveSSM: activation coverage (relu / gelu) ----
    for act, seed in (("relu", 210), ("gelu", 211)):
        cfg = selective_config(activation=act)
        model = SelectiveSSM(hidden_features=cfg["hidden_features"], chunk_size=cfg["chunk_size"],
                             dt_rank=cfg["dt_rank"], activation=act)
        x = jax.random.normal(jax.random.PRNGKey(seed), (B, 16, 8))
        cases.append((f"selective_{act}", "selective_ssm", model, x, seed, cfg))

    # ---- SelectiveSSM: dt_proj=False, dt_rank=1 (broadcast) and dt_rank=2 (D-divisible repeat) ----
    for dtr, tag, seed in ((1, "r1", 220), (2, "rk", 221)):
        cfg = selective_config(dt_proj=False, dt_rank=dtr)
        model = SelectiveSSM(hidden_features=cfg["hidden_features"], chunk_size=cfg["chunk_size"],
                             dt_rank=dtr, dt_proj=False)
        x = jax.random.normal(jax.random.PRNGKey(seed), (B, 16, 8))  # D=8, divisible by 1 and 2
        cases.append((f"selective_dtproj_off_{tag}", "selective_ssm", model, x, seed, cfg))

    # ---- SelectiveSSM: n_channel_groups=2, D=8 (grouped-scan validation path) ----
    cfg = selective_config(n_channel_groups=2)
    model = SelectiveSSM(hidden_features=cfg["hidden_features"], chunk_size=cfg["chunk_size"],
                         dt_rank=cfg["dt_rank"], n_channel_groups=2)
    x = jax.random.normal(jax.random.PRNGKey(230), (B, 16, 8))
    cases.append(("selective_groups", "selective_ssm", model, x, 230, cfg))

    # ---- BidirectionalMamba: complex off / on ----
    # NB: seeds are pinned explicitly (305/306) rather than `300 + len(cases)` so that inserting
    # new cases above does NOT shift these seeds and silently drift the committed goldens.
    for complex_on, seed in ((False, 305), (True, 306)):
        name = f"bidirectional_{'cplx' if complex_on else 'real'}"
        cfg = bidir_config()
        cfg["use_complex_ssm"] = complex_on
        model = BidirectionalMamba(hidden_features=8, expansion_factor=2.0, dt_rank=2,
                                   ssm_args={"use_complex_ssm": complex_on, "chunk_size": 4})
        x = jax.random.normal(jax.random.PRNGKey(seed), (B, 16, 8))
        cases.append((name, "bidirectional", model, x, seed, cfg))

    # ---- BidirectionalMamba: normalization-type coverage ----
    # layernorm, no-norm ("none"), empty-norm ("") at D=8.
    for norm, seed in (("layer", 500), ("none", 501), ("", 502)):
        tag = {"layer": "layernorm", "none": "none", "": "emptynorm"}[norm]
        cfg = bidir_config(norm_type=norm)
        model = BidirectionalMamba(hidden_features=8, expansion_factor=2.0, dt_rank=2,
                                   norm_type=norm, ssm_args={"chunk_size": 4})
        x = jax.random.normal(jax.random.PRNGKey(seed), (B, 16, 8))
        cases.append((f"bidirectional_{tag}", "bidirectional", model, x, seed, cfg))

    # groupnorm: JAX `nn.GroupNorm()` uses num_groups=32, so the channel dim must be a multiple
    # of 32 for the layer to build.  Use D=32 (still small; B=2, L=16).
    cfg = bidir_config(norm_type="group")
    model = BidirectionalMamba(hidden_features=8, expansion_factor=2.0, dt_rank=2,
                               norm_type="group", ssm_args={"chunk_size": 4})
    x = jax.random.normal(jax.random.PRNGKey(520), (B, 16, 32))
    cases.append(("bidirectional_groupnorm", "bidirectional", model, x, 520, cfg))

    # batchnorm at inference (train=False): running mean/var come from init (0/1); exported as
    # extra tensors so the Rust loader normalizes by them.
    cfg = bidir_config(norm_type="batch")
    model = BidirectionalMamba(hidden_features=8, expansion_factor=2.0, dt_rank=2,
                               norm_type="batch", ssm_args={"chunk_size": 4})
    x = jax.random.normal(jax.random.PRNGKey(530), (B, 16, 8))
    cases.append(("bidirectional_batchnorm", "bidirectional", model, x, 530, cfg))

    # ---- BidirectionalMamba: MLP sub-layer (train=False -> dropout is identity) ----
    cfg = bidir_config(mlp_layer=True)
    model = BidirectionalMamba(hidden_features=8, expansion_factor=2.0, dt_rank=2,
                               mlp_layer=True, ssm_args={"chunk_size": 4})
    x = jax.random.normal(jax.random.PRNGKey(540), (B, 16, 8))
    cases.append(("bidirectional_mlp", "bidirectional", model, x, 540, cfg))

    # ---- BidirectionalMamba: tied in_proj + gate, add (not concat), complement ----
    cfg = bidir_config(tie_in_proj=True, tie_gate=True, concatenate_fwd_rev=False, complement=True)
    model = BidirectionalMamba(hidden_features=8, expansion_factor=2.0, dt_rank=2,
                               tie_in_proj=True, tie_gate=True, concatenate_fwd_rev=False,
                               complement=True, ssm_args={"chunk_size": 4})
    x = jax.random.normal(jax.random.PRNGKey(550), (B, 16, 8))
    cases.append(("bidirectional_tied", "bidirectional", model, x, 550, cfg))

    # ---- RCPSWrapper(BidirectionalMamba): complex off / on ----
    # Seeds pinned (407/408, the original `400 + len(cases)` values) so the committed goldens
    # do not drift when cases are inserted above.
    for complex_on, seed in ((False, 407), (True, 408)):
        name = f"rcps_{'cplx' if complex_on else 'real'}"
        cfg = bidir_config()
        cfg["use_complex_ssm"] = complex_on
        cfg["rcps"] = True
        model = RCPSWrapper(module_cls=BidirectionalMamba,
                            module_kwargs={"hidden_features": 8, "expansion_factor": 2.0,
                                           "dt_rank": 2,
                                           "ssm_args": {"use_complex_ssm": complex_on,
                                                        "chunk_size": 4}})
        x = jax.random.normal(jax.random.PRNGKey(seed), (B, 16, 16))  # 2D channels
        cases.append((name, "rcps", model, x, seed, cfg))

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
