#!/usr/bin/env python3
"""Orchestrate the selectssm performance-scaling sweep across both implementations.

For each configuration it launches a single-config worker as a subprocess (the Rust
`scaling` example or the JAX `bench_worker.py`), samples that PID's GPU memory from
`nvidia-smi` while it runs to capture peak VRAM, parses the worker's `RESULT` line for
forward / forward+backward latency, and collects everything into a CSV + markdown tables.

One process per config keeps peak-memory numbers clean and isolates OOMs (a crashed config
is recorded as a failure and the sweep continues).

Usage:
  python3 scripts/benchmark_scaling.py                 # full sweep on GPU 0
  python3 scripts/benchmark_scaling.py --gpu 1         # use GPU 1
  python3 scripts/benchmark_scaling.py --lengths 256 512 1024
  python3 scripts/benchmark_scaling.py --out results.csv
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUST_BIN = os.path.join(ROOT, "rust", "target", "release", "examples", "scaling")
JAX_WORKER = os.path.join(ROOT, "jax", "scripts", "bench_worker.py")

RESULT_RE = re.compile(r"RESULT\s+(.*)")


def parse_result(text):
    m = RESULT_RE.search(text)
    if not m:
        return None
    out = {}
    for kv in m.group(1).split(","):
        k, _, v = kv.partition("=")
        out[k.strip()] = v.strip()
    return out


def sample_vram(pid, gpu, stop, peak):
    """Poll nvidia-smi for `pid`'s GPU memory (MiB) until `stop` is set; record max in peak[0]."""
    q = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    while not stop.is_set():
        try:
            out = subprocess.run(q, capture_output=True, text=True, timeout=5).stdout
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                    peak[0] = max(peak[0], float(parts[1]))
        except Exception:
            pass
        time.sleep(0.05)


def run_one(cmd, env, gpu, timeout):
    """Run a worker subprocess, sampling its peak VRAM. Returns (result_dict_or_None, vram_mib)."""
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stop = threading.Event()
    peak = [0.0]
    sampler = threading.Thread(target=sample_vram, args=(proc.pid, gpu, stop, peak), daemon=True)
    sampler.start()
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        out = (out or "") + "\n[TIMEOUT]"
    finally:
        stop.set()
        sampler.join(timeout=1)
    res = parse_result(out or "")
    if res is None:
        tail = "\n".join((out or "").strip().splitlines()[-3:])
        print(f"    FAILED ({tail[:200]})", file=sys.stderr)
    return res, peak[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--state", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--rust-iters", type=int, default=15)
    ap.add_argument("--jax-iters", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(ROOT, "fixtures", "benchmark_scaling.csv"))
    args = ap.parse_args()

    B, D, N, C = args.batch, args.dim, args.state, args.chunk

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # (impl, scan, mode) suites.  Rust's scan column is the rematerializing custom op
    # ("remat") or the reference scan that retains the tape ("ref"); JAX's is one of its
    # three scanning strategies.
    suites = [
        ("rust", "remat", "real"),
        ("rust", "ref", "real"),
        ("jax", "chunked", "real"),
        ("jax", "recursive", "real"),
        ("jax", "custom_vjp", "real"),
        ("rust", "remat", "complex"),
        ("rust", "ref", "complex"),
        ("jax", "chunked", "complex"),
    ]

    rows = []
    for impl, scan, mode in suites:
        print(f"== {impl} / {scan} / {mode} ==")
        for L in args.lengths:
            if impl == "rust":
                cmd = [RUST_BIN, "gpu", mode, str(B), str(L), str(D), str(N), str(C),
                       str(args.rust_iters), scan]
            else:
                cmd = [sys.executable, JAX_WORKER, "gpu", mode, scan,
                       str(B), str(L), str(D), str(N), str(C), str(args.jax_iters)]
            res, vram = run_one(cmd, env, args.gpu, args.timeout)
            if res is None:
                rows.append(dict(impl=impl, scan=scan, mode=mode, B=B, L=L, D=D, N=N, chunk=C,
                                 fwd_ms="", fwdbwd_ms="", vram_mib="", status="FAIL"))
                print(f"  L={L:<6} FAIL")
                continue
            row = dict(impl=impl, scan=scan, mode=mode, B=B, L=L, D=D, N=N, chunk=C,
                       fwd_ms=float(res["fwd_ms"]), fwdbwd_ms=float(res["fwdbwd_ms"]),
                       vram_mib=round(vram, 1), status="ok")
            rows.append(row)
            print(f"  L={L:<6} fwd={row['fwd_ms']:8.3f}ms  fwd+bwd={row['fwdbwd_ms']:9.3f}ms  vram={row['vram_mib']:8.1f}MiB")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["impl", "scan", "mode", "B", "L", "D", "N", "chunk", "fwd_ms", "fwdbwd_ms", "vram_mib", "status"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
