"""Torch-free, deterministic weight-norm fold for the decoder conversion.

The decoder checkpoint stores weight-normalized convolutions as ``v`` and ``g``; the engine needs
``w = g * v / ||v||``, with the norm taken over every dimension except dim 0.

The default method, ``cuda-order``, evaluates exactly the binary32 arithmetic of PyTorch's CUDA kernel
(``weight_norm_fwd_first_dim_kernel`` in aten/src/ATen/native/cuda/WeightNorm.cu, torch 2.7.1), which folded
the tested bundle:

- one CUDA block of 256 threads per row; thread ``t`` accumulates ``s = fma(v[i], v[i], s)`` over
  ``i = t, t + 256, ...`` (nvcc contracts ``s += v * v`` into a fused multiply-add);
- the 256 partial sums are added pairwise, ``x[t] += x[t + k]`` for k = 128, 64, ..., 1;
- ``norm = sqrtf(sum)``, ``rnorm = 1.f / norm`` and ``w = (g * v) * rnorm``.

Every binary32 operation is computed as the binary64 operation followed by one rounding to binary32. For
+, -, *, / and sqrt of binary32 operands this equals the correctly rounded binary32 result, because binary64
carries more than 2 * 24 + 2 significand bits (double rounding is then innocuous; S. A. Figueroa, "When is
double rounding innocuous?", 1995). The fused multiply-add is computed exactly with an error-free
transformation (below). Only IEEE-754 binary64 add, multiply, divide, square root and the binary64 to binary32
conversion are used, element by element, never a library reduction such as ``np.sum``, whose summation order
depends on the numpy version and the SIMD width. The result is therefore the same on every CPU, operating
system and numpy version.

``exact`` is the alternative measured against it: an exactly rounded sum of squares (``math.fsum``), then
binary64 sqrt, divide and multiply, and one rounding to binary32.
"""
from __future__ import annotations

import math

import numpy as np

F32 = np.dtype("<f4")
F64 = np.dtype("<f8")
THREADS = 256  # BLOCK in WeightNorm.cu
METHODS = ("cuda-order", "exact")
DEFAULT_METHOD = "cuda-order"


def _r32(x: np.ndarray) -> np.ndarray:
    """binary64 -> binary32, round to nearest even."""
    return x.astype(F32)


def add32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _r32(a.astype(F64) + b.astype(F64))


def mul32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _r32(a.astype(F64) * b.astype(F64))


def div32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _r32(a.astype(F64) / b.astype(F64))


def sqrt32(a: np.ndarray) -> np.ndarray:
    return _r32(np.sqrt(a.astype(F64)))


def fma32(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Correctly rounded binary32 ``a * b + c``.

    ``a * b`` is exact in binary64 (24 + 24 significand bits). ``s = a * b + c`` in binary64 may be inexact; TwoSum
    gives its exact error ``e`` (``s + e == a * b + c`` exactly). Rounding ``s`` to binary32 is then correct unless
    ``s`` lies exactly halfway between two binary32 values while ``e != 0``: the exact value is then on the side of
    ``e``, and the rounding goes to that neighbour instead of to the even one.
    """
    p = a.astype(F64) * b.astype(F64)
    c64 = c.astype(F64)
    s = p + c64
    bv = s - c64
    e = (c64 - (s - bv)) + (p - bv)
    r = _r32(s)
    r64 = r.astype(F64)
    toward = np.where(s > r64, np.float32(np.inf), np.float32(-np.inf)).astype(F32)
    other = np.nextafter(r, toward)
    halfway = (s != r64) & (s == (r64 + other.astype(F64)) * 0.5) & (e != 0)
    if halfway.any():
        upper = np.maximum(r, other)
        lower = np.minimum(r, other)
        r = np.where(halfway, np.where(e > 0, upper, lower), r)
    return r


def _rows(v: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.ascontiguousarray(v, dtype=F32)
    g = np.ascontiguousarray(g, dtype=F32)
    rows = v.shape[0]
    if rows == 0 or g.size != rows:
        raise ValueError(f"weight norm: g has {g.size} values for {rows} rows")
    if not (np.isfinite(v).all() and np.isfinite(g).all()):
        raise ValueError("weight norm: non-finite input")
    return v.reshape(rows, -1), g.reshape(rows, 1)


def fold_cuda_order(v: np.ndarray, g: np.ndarray, threads: int = THREADS, fused: bool = True) -> np.ndarray:
    """The CUDA kernel's arithmetic (see the module docstring). ``threads``/``fused`` exist for the measurements."""
    flat, g2 = _rows(v, g)
    rows, n = flat.shape
    steps = -(-n // threads)
    lanes = np.zeros((rows, steps * threads), F32)
    lanes[:, :n] = flat  # threads past the row end add nothing: fma(0, 0, s) == s exactly
    lanes = lanes.reshape(rows, steps, threads)
    acc = np.zeros((rows, threads), F32)
    for j in range(steps):
        x = lanes[:, j, :]
        acc = fma32(x, x, acc) if fused else add32(acc, mul32(x, x))
    width = threads
    while width > 1:
        half = width // 2
        acc = add32(acc[:, :half], acc[:, half:width])
        width = half
    norm = sqrt32(acc[:, 0])
    rnorm = div32(np.ones_like(norm), norm)
    w = mul32(mul32(g2, flat), rnorm[:, None])
    return w.reshape(v.shape)


def fold_exact(v: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Correctly rounded sum of squares (math.fsum; every square is exact in binary64), binary64 sqrt, divide and
    multiply, then one rounding to binary32."""
    flat, g2 = _rows(v, g)
    sq = flat.astype(F64) ** 2
    sums = np.array([math.fsum(row) for row in sq.tolist()], dtype=F64)
    scale = g2[:, 0].astype(F64) / np.sqrt(sums)
    return _r32(flat.astype(F64) * scale[:, None]).reshape(v.shape)


def fold(v: np.ndarray, g: np.ndarray, method: str = DEFAULT_METHOD) -> np.ndarray:
    """``w = g * v / ||v||`` over every dimension but 0, as little-endian float32."""
    if method == "cuda-order":
        w = fold_cuda_order(v, g)
    elif method == "exact":
        w = fold_exact(v, g)
    else:
        raise ValueError(f"unknown weight-norm method {method!r} (known: {', '.join(METHODS)})")
    return np.ascontiguousarray(w, dtype=F32)
