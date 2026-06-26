import logging
from typing import Any, Callable, Sequence, Union, Tuple
from dataclasses import field

import math
from functools import reduce

import einops
import flax.linen as nn

import jax
import jax.numpy as jnp

from .ssmrecscan import ssm_recursive_scan, ssm_scan

def inverse_softplus(x):
    return x + jnp.log(1 - jnp.exp(-x))

def debug_log(fmt: str, *args, **kwargs):
  jax.debug.callback(
      lambda *args, **kwargs: logging.warning(fmt.format(*args, **kwargs)),
      *args, **kwargs)

def largest_factor_up_to(b,n):
    if n < 2:
        return n
    k = b
    while n % k != 0:
        k -= 1
    return k

# Apply a data-dependent rotary embedding to a state-space projection.
#
# This is the "RoPE trick" of Mamba-3 (arXiv:2603.15569, Prop. 3): a complex
# diagonal SSM is equivalent to a real SSM whose B and C projections are rotated
# by the cumulative transition phase.  Each projection v (of state width N) is
# split in half --- the first N/2 entries are the real parts, the last N/2 the
# imaginary parts --- and every (real, imag) pair is rotated by R(-Phi):
#
#     R(-Phi) = [[ cos Phi,  sin Phi],
#                [-sin Phi,  cos Phi]]
#
# Both B and C are rotated by the same cumulative angle Phi_t = sum_{s<=t} theta_s;
# the y = C^T h contraction then realises the relative rotation R(Phi_s - Phi_t)
# between input position s and output position t, exactly as RoPE does for the
# query/key dot product (Su et al. 2023), but with data-dependent angles.
#
# v: (..., N);  Phi: (..., N/2).  Returns the rotated projection, same shape as v.
def _apply_rotary (v, Phi):
    h = v.shape[-1] // 2
    v_re = v[..., :h]
    v_im = v[..., h:]
    cos = jnp.cos (Phi)
    sin = jnp.sin (Phi)
    return jnp.concatenate ([v_re * cos + v_im * sin,
                             -v_re * sin + v_im * cos], axis=-1)

# x: (B, L, D)
# Acoeff: (D, N)
# Bcoeff: (B, L, N)
# Ccoeff: (B, L, N)
# dt: (B, L, D) or (B, L, 1);  can assume (B, L, D) and rely on broadcasting
# theta: (B, L, N/2) or None.  Data-dependent per-step rotation angle for the
#   complex-SSM "RoPE trick".  When None, the scan is the plain real SSM and the
#   result is bitwise-identical to the original implementation.  When provided,
#   the cumulative angle Phi_t = sum_{s<=t} theta_s is formed as a prefix sum that
#   is carried across chunks (so it composes with the chunked-scan machinery), and
#   B and C are rotated by R(-Phi) before the real recurrence.
def ssm_chunked_scan (x, Acoeff, Bcoeff, Ccoeff, dt, theta=None, chunk_size: int = None, n_channel_groups: int = 1):
    B = x.shape[-3]
    L = x.shape[-2]
    D = x.shape[-1]
    N = Acoeff.shape[-1]

    if n_channel_groups is not None:
        K = n_channel_groups
    else:
        K = 1
    if D % K != 0:
        raise ValueError(f"n_channel_groups={n_channel_groups} must divide D={D}")

    if chunk_size is None:
        chunk_size = largest_factor_up_to(int(math.sqrt(K*L)),L)

    # If the chunk size does not divide L, pad the sequence up to the next multiple
    # with zeros and discard the padded outputs.  Padding at the end is safe because
    # the scan is causal: padded steps have dt=0 (hence dA=exp(0)=1, dB=0, theta=0),
    # so they neither decay the state nor inject input nor advance the rotation angle,
    # and they come strictly after every real position.  When chunk_size divides L
    # (the default, and every previously-supported case) no padding is applied and
    # the computation is bitwise-identical to before.
    pad = (-L) % chunk_size
    if pad:
        x = jnp.pad (x, ((0, 0), (0, pad), (0, 0)))
        Bcoeff = jnp.pad (Bcoeff, ((0, 0), (0, pad), (0, 0)))
        Ccoeff = jnp.pad (Ccoeff, ((0, 0), (0, pad), (0, 0)))
        dt = jnp.pad (dt, ((0, 0), (0, pad), (0, 0)))
        if theta is not None:
            theta = jnp.pad (theta, ((0, 0), (0, pad), (0, 0)))
    Lp = L + pad
    n_chunks = Lp // chunk_size

    # Transpose length & batch dimensions to make the scan over length, and split into chunks
    # This is a bit inefficient, but taking dynamic slices appears to be worse
    x_chunks = einops.rearrange (x, 'b (c l) (k d) -> c k l b d', c=n_chunks, k=K)
    A_blocks = einops.rearrange (Acoeff, '(k d) n -> k d n', k=K)
    B_chunks = einops.rearrange (Bcoeff, 'b (c l) n -> c l b n', c=n_chunks)
    C_chunks = einops.rearrange (Ccoeff, 'b (c l) n -> c l b n', c=n_chunks)
    dt_chunks = einops.rearrange (dt, 'b (c l) (k d) -> c k l b d', c=n_chunks, k=K)

    # Function to do an associative scan for a single chunk
    # We decorate this with @jax.remat to flag that we are OK with re-performing this scan whenever needed
    @jax.remat
    def scan_chunk (carry, chunk):
        # For the purposes of shape annotation within this code we write D instead of D/K
        g_init, h_init = carry  # (1, B, D, N)  (1, B, D, N)

        x_chunk, A_block, B_chunk, C_chunk, dt_chunk = chunk
        # dA = exp(A*dt) [zero-order hold]
        A_dt = jnp.einsum('dn,lbd->lbdn', A_block, dt_chunk)  # negative
        dA = jnp.exp(A_dt)
        # dB = B*dt*x [Euler step] ... this is more efficient than, but an approximation to, the zero-order hold
        # The purported long-range benefits of the zero-order hold are outweighed by their extra compute & memory costs, especially when striping Mamba with attention, since attention can handle long-range dependencies on its own.
        dB = jnp.einsum ('lbn,lbd,lbd->lbdn', B_chunk, x_chunk, dt_chunk)  # (chunk_size, B, D, N)
        # The associative scan is a product of matrices of the form ((g,h),(0,1)) where g_i=exp(A*dt)x_i and h_i=B*dt*x_i
        # Since matrices of this form are are closed under multiplication, we can represent all intermediate products in the same way
        @jax.remat
        def associative_scan_fn (l, r):  # l, r, and return value are tuples of the form ((B,D,N), (B,D,N))
            g_l, h_l = l
            g_r, h_r = r
            return tuple((g_l*g_r, g_r*h_l + h_r))
        gs, hs = jax.lax.associative_scan (associative_scan_fn, (dA, dB))  # (chunk_size, B, D, N)  (chunk_size, B, D, N)
        hs = gs * h_init + hs  # Incorporate h_init here so that it is reflected in y_chunk
        # We only need to keep the last state of gs, so we can discard the rest. Otherwise we would incorporate g_init here, like so:
        # gs = g_init * gs
        y_chunk = jnp.einsum ('lbn,lbdn->lbd', C_chunk, hs)  # (chunk_size, B, D)
        return (gs[-1:,...] * g_init, hs[-1:,...]), y_chunk  # note g_init incorporated here

    # A wrapper that splits the dimensions into K blocks and does the inner associative scan for each block, re-using B and C (which don't change across dimensions)
    @jax.remat
    def scan_chunk_mapped (carry, chunk):
        g_init, h_init, angle_acc = carry  # (K,1,B,D/K,N) (K,1,B,D/K,N) (B,N/2)

        x_chunk, B_chunk, C_chunk, dt_chunk, theta_chunk = chunk   # (K,L,B,D/K), (L,B,N), (L,B,N), (K,L,B,D/K), (L,B,N/2)|None
        # Complex-SSM "RoPE trick": form the cumulative rotation angle as a prefix
        # sum carried across chunks, and rotate B and C (shared across the K channel
        # groups) by R(-Phi) before the real recurrence.
        if theta_chunk is not None:
            Phi = angle_acc[None, ...] + jnp.cumsum (theta_chunk, axis=0)  # (L,B,N/2)  inclusive prefix sum
            new_angle_acc = angle_acc + jnp.sum (theta_chunk, axis=0)      # (B,N/2)    angle carried to next chunk
            B_chunk = _apply_rotary (B_chunk, Phi)
            C_chunk = _apply_rotary (C_chunk, Phi)
        else:
            new_angle_acc = angle_acc

        @jax.remat
        def scan_chunk_wrapper (block):
            dA_init_block, dB_init_block, x_chunk_block, A_block, dt_chunk_block = block
            return scan_chunk ((dA_init_block, dB_init_block), (x_chunk_block, A_block, B_chunk, C_chunk, dt_chunk_block))
        (g_final, h_final), y_chunk = jax.lax.map (scan_chunk_wrapper, (g_init, h_init, x_chunk, A_blocks, dt_chunk))
        return (g_final, h_final, new_angle_acc), y_chunk


    # Perform the scan over chunks recurrently (with rematerialization as noted above), with each chunk being an associative scan
    theta_chunks = einops.rearrange (theta, 'b (c l) h -> c l b h', c=n_chunks) if theta is not None else None
    angle_init = jnp.zeros ((B, N // 2)) if theta is not None else jnp.zeros ((B, 0))
    (_A_final, _h_final, _angle_final), y_chunks = jax.lax.scan (
        scan_chunk_mapped,
        (jnp.ones((K,1,B,D//K,N)), jnp.zeros((K,1,B,D//K,N)), angle_init),
        (x_chunks, B_chunks, C_chunks, dt_chunks, theta_chunks))  # (K, n_chunks, B, D//K)

    y = einops.rearrange (y_chunks, 'c k l b d -> b (c l) (k d)')  # (B, Lp, D)
    if pad:
        y = y[:, :L, :]
    return y  # (B, L, D)


class SelectiveSSM(nn.Module):
    """ A variation on MAMBA: https://arxiv.org/pdf/2312.00752.pdf """

    reverse: bool = False
    complement: bool = False  # only checked if reverse is true

    hidden_features: int = 16  # N
    chunk_size: int = None
    n_channel_groups: int = None

    # Complex-SSM "RoPE trick" (Mamba-3, arXiv:2603.15569, Prop. 3).  When True,
    # a data-dependent rotation angle theta is projected from the input and the
    # complex diagonal transition is realised by rotating the B and C projections
    # by the cumulative angle (see ssm_chunked_scan / _apply_rotary).  This adds
    # the rotational dynamics needed for state-tracking tasks (e.g. parity) that a
    # purely real SSM cannot represent.  Off by default; when False the model is
    # bitwise-identical to the plain real SSM.  Supported only with the chunked
    # scan (the default); combining it with recursive_scan or custom_vjp_scan
    # raises an error.  Requires hidden_features (N) to be even.
    use_complex_ssm: bool = False

    dt_rank: Union[int, str] = 'auto'  # R
    dt_proj: bool = True   # whether to use a linear projection (vs broadcast) to map dt_rank to D

    dt_min: float = 0.001  # 1/(long-range context length)
    dt_max: float = 0.1    # 1/(short-range context length)

    shift_conv_size: int = 3

    activation: str = "silu"

    diagnostics: dict = field(default_factory=dict)

    # If GPU memory is a bottleneck, enable recursive_scan.
    # custom_vjp_scan
    recursive_scan: bool = False
    min_recursion_length: int = 2
    recursive_split: int = 2

    custom_vjp_scan: bool = False

    @nn.compact
    def __call__(
        self,
        x,  # (B, L, D)
        train: bool = False,
    ):
        B = x.shape[-3]
        L = x.shape[-2]
        D = x.shape[-1]  # if called by BidirectionalMamba, this is actually E*D

        N = self.hidden_features

        if self.dt_rank == 'auto':
            dt_rank = math.ceil(D / 16)
        else:
            dt_rank = self.dt_rank

        if self.diagnostics and 'ssm_input_norm' in self.diagnostics:
            self.sow("diagnostics", "ssm_input_mean", jnp.mean(x))
            self.sow("diagnostics", "ssm_input_sd", jnp.std(x))

        if self.reverse:
            x = jnp.flip (x, axis=(-2,-1) if self.complement else -2)

        u = nn.Conv(features=D, feature_group_count=D,
            kernel_size=(self.shift_conv_size,), strides=(1,),
            padding=[(self.shift_conv_size - 1, 0)],  # causal: left-pad only
            use_bias=False, name="shift_conv",
            kernel_init=nn.initializers.lecun_normal())(x)  # (B, L, D)

        if self.diagnostics and 'ssm_coeffs' in self.diagnostics:
            self.sow("diagnostics", "conv_mean", jnp.mean(u))
            self.sow("diagnostics", "conv_sd", jnp.std(u))

        if self.activation == "gelu":
            u = nn.gelu(u)
        elif self.activation == "relu":
            u = nn.relu(u)
        elif self.activation == "silu":
            u = nn.silu(u)
        elif self.activation is not None:
            raise Exception(f"Unknown activation: {self.activation}")

        # Initialize A nonrandomly with evenly spaced eigenvalues; keep parameterization in log space to guarantee A<0
        if self.use_complex_ssm:
            # Complex SSM: each state pair (i, i+N/2) shares its decay magnitude --- the
            # real part of a single complex eigenvalue --- so that exp(A*dt) restricted to
            # each 2x2 rotation block is a scaled identity that commutes with the rotation.
            # This is what makes the RoPE-trick factorization (Prop. 3) exact.  A_log is
            # therefore parameterized at half width (D, N/2) and tiled to (D, N).
            A_log = self.param ('A_log', lambda rng: jnp.log (jnp.repeat (jnp.arange(start=1,stop=N//2+1,dtype=jnp.float32)[None,:], D, axis=0)))  # (D, N/2)
            Acoeff = -jnp.exp (jnp.concatenate ([A_log, A_log], axis=-1))  # (D, N)
        else:
            Acoeff = -jnp.exp (self.param ('A_log', lambda rng: jnp.log (jnp.repeat (jnp.arange(start=1,stop=N+1,dtype=jnp.float32)[None,:], D, axis=0))))  # (D, N)
        Bcoeff, Ccoeff = jnp.split (nn.Dense (features=2*N, name='BC', use_bias=True, kernel_init=nn.initializers.lecun_normal()) (u), 2, axis=-1)  # (B, L, N) *2
        Dcoeff = self.param ('D', lambda rng: jnp.ones((D,)))  # (D,)

        dt_bias_init = lambda rng, shape, dtype: inverse_softplus (jax.random.uniform (rng, shape=shape, dtype=dtype, minval=self.dt_min, maxval=self.dt_max))
        dt = nn.Dense (features=dt_rank, use_bias=True, name='dt',
                       kernel_init=nn.initializers.lecun_normal(),
                       bias_init=nn.initializers.zeros if self.dt_proj else dt_bias_init) (u)  # (B, L, dt_rank)

        if self.diagnostics and 'ssm_coeffs' in self.diagnostics:
            self.sow("diagnostics", "dt_lowrank_mean", jnp.mean(dt))
            self.sow("diagnostics", "dt_lowrank_sd", jnp.std(dt))

        if self.dt_proj:
            dt = nn.Dense (features=D, use_bias=True, kernel_init=nn.initializers.lecun_normal(), bias_init=dt_bias_init, name='dt_proj') (dt)  # (B, L, D)
        else:
            if dt_rank > 1:  # if dt_rank is 1, we can just rely on broadcasting, and save memory
                if D % dt_rank != 0:
                    raise ValueError(f"dt_rank={dt_rank} must divide D={D}")
                dt = jnp.repeat (dt, D // dt_rank, axis=-1)  # (B, L, D)
        dt = nn.activation.softplus (dt)  # (B, L, D) or (B, L, 1)

        if self.diagnostics and 'ssm_coeffs' in self.diagnostics:
            self.sow("diagnostics", "activated_conv_mean", jnp.mean(u))
            self.sow("diagnostics", "activated_conv_sd", jnp.std(u))
            self.sow("diagnostics", "dt_mean", jnp.mean(dt))
            self.sow("diagnostics", "dt_sd", jnp.std(dt))
            self.sow("diagnostics", "A_mean", jnp.mean(Acoeff))
            self.sow("diagnostics", "A_sd", jnp.std(Acoeff))
            self.sow("diagnostics", "B_sd", jnp.std(Bcoeff))
            self.sow("diagnostics", "C_sd", jnp.std(Ccoeff))

        # Complex-SSM "RoPE trick": project a data-dependent rotation angle theta,
        # one per (real, imag) state pair, analogous to how dt is produced.  No
        # activation is applied --- the angle is signed, since negative effective
        # eigenvalues (rotations through pi) are exactly what unlocks state tracking.
        theta = None
        if self.use_complex_ssm:
            if N % 2 != 0:
                raise ValueError(f"use_complex_ssm requires an even hidden_features (N); got N={N}")
            theta = nn.Dense (features=N // 2, use_bias=True, name='theta',
                              kernel_init=nn.initializers.lecun_normal(),
                              bias_init=nn.initializers.zeros) (u)  # (B, L, N/2)
            if self.diagnostics and 'ssm_coeffs' in self.diagnostics:
                self.sow("diagnostics", "theta_mean", jnp.mean(theta))
                self.sow("diagnostics", "theta_sd", jnp.std(theta))

        # Perform SSM scan
        if self.custom_vjp_scan:
            if self.use_complex_ssm:
                raise ValueError("use_complex_ssm is only supported with the chunked scan; set custom_vjp_scan=False")
            y = ssm_scan (u, Acoeff, Bcoeff, Ccoeff, dt, min_recursion_length=self.min_recursion_length, recursive_split=self.recursive_split)  # (B, L, D)
        elif self.recursive_scan:
            if self.use_complex_ssm:
                raise ValueError("use_complex_ssm is only supported with the chunked scan; set recursive_scan=False")
            y = ssm_recursive_scan (u, Acoeff, Bcoeff, Ccoeff, dt, min_recursion_length=self.min_recursion_length, recursive_split=self.recursive_split)  # (B, L, D)
        else:
            y = ssm_chunked_scan (u, Acoeff, Bcoeff, Ccoeff, dt, theta=theta, chunk_size=self.chunk_size, n_channel_groups=self.n_channel_groups)  # (B, L, D)

        if self.reverse:
            y = jnp.flip (y, axis=(-2,-1) if self.complement else -2)

        if self.diagnostics and 'ssm_residual' in self.diagnostics:
            self.sow("diagnostics", "ssm_residual_mean", jnp.mean(y))
            self.sow("diagnostics", "ssm_residual_sd", jnp.std(y))

        # Add in the skip connection term
        y = y + jnp.einsum ('bld,d->bld', u, Dcoeff)

        if self.diagnostics and 'ssm_output_norm' in self.diagnostics:
            self.sow("diagnostics", "ssm_output_mean", jnp.mean(y))
            self.sow("diagnostics", "ssm_output_sd", jnp.std(y))

        return y

class BidirectionalMamba(nn.Module):
    """Non-causal bidirectional Mamba block.

    Combines a forward and a reverse `SelectiveSSM` with a gated output projection.
    Every output position depends on the full input sequence, so this module is
    unsuitable as a drop-in for an autoregressive language-model head.

    The ``complement=True, tie_in_proj=True, tie_gate=True, concatenate_fwd_rev=True``
    configuration does NOT yield exact reverse-complement equivariance — the inner
    SSM's parameters (``A``, ``BC``, ``dt``, depthwise conv) are not constrained
    to be channel-flip-symmetric, so any RC symmetry is only approximate/learned.
    For exact RC equivariance, wrap this module (or any other) in `RCPSWrapper`
    from :mod:`selectssm.rcps`.
    """

    hidden_features: int   # N
    expansion_factor: float  # E

    dt_rank: Union[int, str] = 'auto'

    # For an RC-equivariant model, set all of {complement,tie_in_proj,tie_gate,concatenate_fwd_rev} to True
    complement: bool = False
    tie_in_proj: bool = False
    tie_gate: bool = False
    concatenate_fwd_rev: bool = True

    activation: str = "silu"
    norm_type: str = "rms"

    bn_momentum: float = 0.9

    mlp_layer: bool = False
    dense_expansion: int = 2
    mlp_dropout_rate: float = 0.1

    ssm_args: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, x, train: bool = False):

        input_features = x.shape[-1]  # D

        if self.dt_rank == 'auto':
            dt_rank = math.ceil(input_features / 16)
        else:
            dt_rank = self.dt_rank

        if self.activation == "gelu":
            activate = nn.gelu
        elif self.activation == "silu":
            activate = nn.silu
        elif self.activation == "relu":
            activate = nn.relu
        else:
            raise Exception(f"Unknown activation: {self.activation}")

        skip = x
        if 'skip' in self.diagnostics and train:
            self.sow ("diagnostics", "skip_mean", jnp.mean(skip))
            self.sow ("diagnostics", "skip_sd", jnp.std(skip))

        # normalize
        if self.norm_type == "batch":
            x = nn.BatchNorm(momentum=self.bn_momentum, use_running_average=not train)(x)
        elif self.norm_type == "layer":
            x = nn.LayerNorm()(x)
        elif self.norm_type == "group":
            x = nn.GroupNorm()(x)
        elif self.norm_type == "rms":
            x = nn.RMSNorm()(x)

        ED = math.ceil (self.expansion_factor * input_features)
        # project to expanded dimension
        n_in_proj = 1 if self.tie_in_proj else 2
        n_gate = 1 if self.tie_gate else 2
        [xf, _xr, zf, _zr] = jnp.split (nn.Dense (features=((n_in_proj+n_gate)*ED), name='in_proj', kernel_init=nn.initializers.lecun_normal()) (x), [k*ED for k in [1,n_in_proj,n_in_proj+1]], axis=-1)
        xr = xf if self.tie_in_proj else _xr
        zr = zf if self.tie_gate else _zr

        # forward and backward SSM
        ssm = SelectiveSSM
        xf = ssm(hidden_features=self.hidden_features, reverse=False, dt_rank=dt_rank, diagnostics=self.diagnostics, **self.ssm_args, name='ssm_fwd') (xf, train)
        xr = ssm(hidden_features=self.hidden_features, reverse=True, complement=self.complement, dt_rank=dt_rank, diagnostics=self.diagnostics, **self.ssm_args, name='ssm_rev') (xr, train)

        if 'gate' in self.diagnostics and train:
            self.sow ("diagnostics", "gate_fwd_mean", jnp.mean(zf))
            self.sow ("diagnostics", "gate_fwd_sd", jnp.std(zf))
            self.sow ("diagnostics", "gate_rev_mean", jnp.mean(zr))
            self.sow ("diagnostics", "gate_rev_sd", jnp.std(zr))

        # concatenate (or add) forward and backward channels, multiplied by respective activated gates
        if self.concatenate_fwd_rev:
            x = jnp.concatenate ([xf * activate(zf), xr * activate(zr)], axis=-1)
        else:
            x = xf * activate(zf) + xr * activate(zr)

        if 'gated' in self.diagnostics and train:
            self.sow ("diagnostics", "gated_mean", jnp.mean(x))
            self.sow ("diagnostics", "gated_sd", jnp.std(x))

        # project back down
        x = nn.Dense (features=input_features, name='out_proj', kernel_init=nn.initializers.lecun_normal()) (x)

        # residual add
        if 'residual' in self.diagnostics and train:
            self.sow ("diagnostics", "residual_mean", jnp.mean(x))
            self.sow ("diagnostics", "residual_sd", jnp.std(x))

        x = skip + x

        # MLP layer (optional)
        if self.mlp_layer:
            skip = x
            x = nn.Dense(self.dense_expansion*input_features, name="mlp", kernel_init=nn.initializers.lecun_normal())(x)
            x = nn.Dropout(rate=self.mlp_dropout_rate, deterministic=not train)(x)
            x = activate(x)
            x = nn.Dense(input_features, name="mlp_proj", kernel_init=nn.initializers.lecun_normal())(x)
            x = nn.Dropout(rate=self.mlp_dropout_rate, deterministic=not train)(x)
            x = skip + x

        return x
