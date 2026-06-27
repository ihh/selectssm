#!/usr/bin/env python3
"""Performance-scaling micro-benchmark worker for the JAX/Flax ``SelectiveSSM``.

Runs ONE configuration (from argv) and prints a single ``RESULT ...`` line of ``key=value``
pairs: average forward latency, average forward+backward latency, host peak RSS, and JAX's
own reported peak device bytes.  One config per process so peak-memory numbers are clean.

The ``scan`` argument selects which of the three scanning strategies the layer uses:
``chunked`` (the default in ``SelectiveSSM``), ``recursive``, or ``custom_vjp``.  This lets
the orchestrator test the claim that the chunked scan is the most efficient.  Complex mode
(the RoPE trick) is only implemented for the chunked scan.

argv: <gpu|cpu> <real|complex> <chunked|recursive|custom_vjp> <B> <L> <D> <N> <chunk> <iters>
Driven by ../scripts/benchmark_scaling.py (also runnable standalone).
"""
import os
import sys
import time
import resource

# Make device memory reflect real usage (so external nvidia-smi sampling is meaningful)
# rather than XLA's default 75% pre-grab.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

device = sys.argv[1]
if device == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
from selectssm.selectssm import SelectiveSSM

mode = sys.argv[2]
scan = sys.argv[3]
B, L, D, N, chunk, iters = (int(x) for x in sys.argv[4:10])

ssm_kwargs = dict(hidden_features=N, use_complex_ssm=(mode == "complex"))
if scan == "chunked":
    ssm_kwargs.update(chunk_size=chunk)
elif scan == "recursive":
    ssm_kwargs.update(recursive_scan=True)
elif scan == "custom_vjp":
    ssm_kwargs.update(custom_vjp_scan=True)
else:
    raise SystemExit(f"unknown scan {scan!r}")

model = SelectiveSSM(**ssm_kwargs)

key = jax.random.PRNGKey(0)
kx, kp = jax.random.split(key)
x = jax.random.normal(kx, (B, L, D), dtype=jnp.float32)
params = model.init(kp, x)

fwd = jax.jit(lambda p, x: model.apply(p, x))


def loss_fn(p, x):
    return jnp.sum(fwd_raw(p, x) ** 2)


fwd_raw = lambda p, x: model.apply(p, x)
grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))


def timeit(fn, n):
    # Warm up (compilation + allocation), then time n synced iterations.
    jax.block_until_ready(fn())
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) * 1000.0 / n


fwd_ms = timeit(lambda: fwd(params, x), iters)
fwdbwd_ms = timeit(lambda: grad_fn(params, x), iters)

host_peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
jax_peak_mib = 0.0
try:
    stats = jax.devices()[0].memory_stats()
    jax_peak_mib = stats.get("peak_bytes_in_use", 0) / (1024.0 * 1024.0)
except Exception:
    pass

print(
    f"RESULT impl=jax,device={device},mode={mode},scan={scan},"
    f"B={B},L={L},D={D},N={N},chunk={chunk},"
    f"fwd_ms={fwd_ms:.4f},fwdbwd_ms={fwdbwd_ms:.4f},"
    f"host_peak_mib={host_peak_mib:.1f},jax_peak_mib={jax_peak_mib:.1f}"
)
