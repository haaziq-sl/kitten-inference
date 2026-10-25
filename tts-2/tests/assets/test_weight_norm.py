"""Torch-free weight-norm fold: exact binary32 arithmetic, the CUDA kernel's order, and a pinned digest."""
from __future__ import annotations

import hashlib
import math
from fractions import Fraction

import numpy as np
import pytest

from kt2n_assets import convert_s3gen, weight_norm


def round32(x: Fraction) -> float:
    """Correctly rounded binary32 (round half to even) of an exact rational (normal range)."""
    if x == 0:
        return 0.0
    sign = -1 if x < 0 else 1
    x = abs(x)
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** e > x:
        e -= 1
    ulp = Fraction(2) ** (max(e, -126) - 23)
    q, rem = divmod(x, ulp)
    if rem * 2 > ulp or (rem * 2 == ulp and q % 2 == 1):
        q += 1
    return sign * float(q * ulp)


def sqrt_round32(x: float) -> float:
    """Correctly rounded binary32 sqrt of a positive binary32 value: the y whose two half-way points bracket sqrt(x),
    decided with exact squares (sqrt(x) is never exactly half-way between two binary32 values)."""
    y, target = np.float32(math.sqrt(x)), Fraction(x)
    while True:
        below = np.nextafter(y, np.float32(-np.inf))
        above = np.nextafter(y, np.float32(np.inf))
        if ((Fraction(float(below)) + Fraction(float(y))) / 2) ** 2 > target:
            y = below
        elif ((Fraction(float(y)) + Fraction(float(above))) / 2) ** 2 < target:
            y = above
        else:
            return float(y)


def f32(x):
    return np.asarray(x, dtype=np.float32)


def test_binary32_primitives_are_ieee():
    rng = np.random.default_rng(1)
    a = (rng.standard_normal(4096) * 10.0 ** rng.integers(-6, 6, 4096)).astype(np.float32)
    b = (rng.standard_normal(4096) * 10.0 ** rng.integers(-6, 6, 4096)).astype(np.float32)
    assert np.array_equal(weight_norm.add32(a, b), a + b)
    assert np.array_equal(weight_norm.mul32(a, b), a * b)
    assert np.array_equal(weight_norm.div32(a, b), a / b)
    assert np.array_equal(weight_norm.sqrt32(np.abs(a)), np.sqrt(np.abs(a)))
    for x, y in zip(a[:300], b[:300]):
        assert float(weight_norm.add32(f32([x]), f32([y]))[0]) == round32(Fraction(float(x)) + Fraction(float(y)))
        assert float(weight_norm.div32(f32([x]), f32([y]))[0]) == round32(Fraction(float(x)) / Fraction(float(y)))
    for x in np.abs(a[:300]) + np.float32(1e-30):
        assert float(weight_norm.sqrt32(f32([x]))[0]) == sqrt_round32(float(x))


def test_fma32_is_correctly_rounded():
    rng = np.random.default_rng(2)
    a = (rng.standard_normal(3000) * 0.05).astype(np.float32)
    b = (rng.standard_normal(3000) * 3).astype(np.float32)
    c = rng.standard_normal(3000).astype(np.float32)
    got = weight_norm.fma32(a, b, c)
    for x, y, z, r in zip(a, b, c, got):
        assert float(r) == round32(Fraction(float(x)) * Fraction(float(y)) + Fraction(float(z)))


def test_fma32_double_rounding_case():
    # (1 - 2^-23) * (1 + 2^-23) + (2^24 + 2) = 2^24 + 3 - 2^-46 exactly. binary64 rounds that to 2^24 + 3, halfway
    # between the binary32 values 2^24 + 2 and 2^24 + 4; rounding again (to even) would give 2^24 + 4. The exact
    # value is below the halfway point, so the correctly rounded result is 2^24 + 2.
    a, b, c = f32([1 - 2.0 ** -23]), f32([1 + 2.0 ** -23]), f32([2.0 ** 24 + 2])
    assert float(np.float32(np.float64(a[0]) * np.float64(b[0]) + np.float64(c[0]))) == 2.0 ** 24 + 4
    assert float(weight_norm.fma32(a, b, c)[0]) == 2.0 ** 24 + 2
    assert float(weight_norm.fma32(-a, b, -c)[0]) == -(2.0 ** 24 + 2)


def cuda_kernel_reference(v: np.ndarray, g: np.ndarray, threads: int = 256) -> np.ndarray:
    """weight_norm_fwd_first_dim_kernel, one scalar operation at a time, every rounding done exactly."""
    rows = v.shape[0]
    flat = v.reshape(rows, -1)
    out = np.empty_like(flat)
    for row in range(rows):
        n = flat.shape[1]
        partial = []
        for tid in range(threads):
            s = 0.0
            for i in range(tid, n, threads):
                x = Fraction(float(flat[row, i]))
                s = round32(x * x + Fraction(s))
            partial.append(s)
        width = threads
        while width > 1:
            width //= 2
            partial = [round32(Fraction(partial[t]) + Fraction(partial[t + width])) for t in range(width)]
        norm = sqrt_round32(partial[0])
        rnorm = round32(Fraction(1) / Fraction(norm))
        gr = Fraction(float(g.reshape(-1)[row]))
        for i in range(n):
            out[row, i] = round32(Fraction(round32(gr * Fraction(float(flat[row, i])))) * Fraction(rnorm))
    return out.reshape(v.shape)


@pytest.mark.parametrize("shape", [(3, 5), (2, 256), (2, 300), (2, 4, 133), (1, 1)])
def test_cuda_order_fold_matches_the_kernel_step_by_step(shape):
    rng = np.random.default_rng(sum(shape))
    v = (rng.standard_normal(shape) * 0.03).astype(np.float32)
    g = (rng.random((shape[0], 1, 1)[:len(shape)]) + 0.5).astype(np.float32)
    got = weight_norm.fold(v, g)
    assert got.dtype == np.dtype("<f4") and got.shape == v.shape
    assert np.array_equal(got.view("<u4"), cuda_kernel_reference(v, g).view("<u4"))


def test_exact_fold_is_within_one_ulp_of_the_true_value():
    rng = np.random.default_rng(5)
    v = (rng.standard_normal((4, 37)) * 0.1).astype(np.float32)
    g = rng.random(4).astype(np.float32) + np.float32(0.5)
    got = weight_norm.fold(v, g, "exact")
    for row in range(4):
        norm = math.sqrt(math.fsum(float(x) ** 2 for x in v[row]))
        for x, w in zip(v[row], got[row]):
            true = float(g[row]) * float(x) / norm
            assert abs(float(w) - true) <= float(np.spacing(np.float32(abs(true))))


def test_fold_ignores_memory_layout_and_rejects_bad_input():
    rng = np.random.default_rng(6)
    v = (rng.standard_normal((6, 9, 4)) * 0.02).astype(np.float32)
    g = rng.random((6, 1, 1)).astype(np.float32)
    strided = np.asfortranarray(v)
    assert np.array_equal(weight_norm.fold(strided, g), weight_norm.fold(v, g))
    with pytest.raises(ValueError, match="non-finite"):
        weight_norm.fold(np.full((2, 3), np.nan, np.float32), np.ones(2, np.float32))
    with pytest.raises(ValueError, match="rows"):
        weight_norm.fold(v, np.ones(5, np.float32))
    with pytest.raises(ValueError, match="unknown weight-norm method"):
        weight_norm.fold(v, g, "approximate")


def _synthetic(rows: int, cols: int, salt: int) -> np.ndarray:
    """Platform-independent values: integer hashing, exact binary64 scaling, one rounding to binary32."""
    i = np.arange(rows * cols, dtype=np.uint64)
    h = (i * np.uint64(2654435761) + np.uint64(salt)) % np.uint64(1 << 32)
    return ((h.astype(np.float64) / 2.0 ** 32 - 0.5) * 0.125).astype(np.float32).reshape(rows, cols)


def test_pinned_fold_digest():
    # The same bytes on every OS, CPU and numpy version (checked in CI on Linux x86-64/arm64, Windows and macOS).
    v = _synthetic(96, 1543, 7)
    g = (np.abs(_synthetic(96, 1, 11)) + np.float32(0.25)).reshape(96, 1, 1)
    digest = hashlib.sha256(weight_norm.fold(v, g).tobytes()).hexdigest()
    assert digest == PINNED_FOLD_SHA256


# torch 2.7.1+cu128 torch._weight_norm(v, g, 0) on an RTX 4090 gives the same bytes.
PINNED_FOLD_SHA256 = "cdcea6f42d6ea4e56781fe61c4cdd1aeb2b04f9d343479cac7e3575997884122"


def test_converter_records_the_numpy_method_and_a_location_independent_source_name():
    meta = convert_s3gen._converter("numpy")
    assert meta["weight_norm_device"] == "numpy" and meta["weight_norm_method"] == weight_norm.DEFAULT_METHOD
    assert "weight_norm_method" not in convert_s3gen._converter("cpu")


def test_independent_bf16_rounding_matches_the_converter():
    rng = np.random.default_rng(8)
    x = np.concatenate([
        (rng.standard_normal(200000) * 10.0 ** rng.integers(-30, 30, 200000)).astype(np.float32),
        # exact ties between two bf16 values, both parities, both signs
        ((np.arange(1, 5000, dtype=np.uint32) << np.uint32(16)) | np.uint32(0x8000)).view(np.float32),
        ((np.arange(1, 5000, dtype=np.uint32) << np.uint32(16)) | np.uint32(0x80008000)).view(np.float32),
        np.array([0.0, -0.0, 1.0, np.finfo(np.float32).tiny, 3.4e38], np.float32)])
    assert np.array_equal(convert_s3gen.bf16_rne_reference(x), convert_s3gen.bf16_rne_bits(x).reshape(-1))


def test_manifest_digest_ignores_windows_line_ends(tmp_path):
    lf = convert_s3gen.MANIFEST.read_bytes().replace(b"\r\n", b"\n")
    crlf = tmp_path / "s3gen_tensors.tsv"
    crlf.write_bytes(lf.replace(b"\n", b"\r\n"))
    assert convert_s3gen.manifest_sha256(crlf) == convert_s3gen.manifest_sha256(convert_s3gen.MANIFEST)
    assert convert_s3gen.manifest_sha256(crlf) == hashlib.sha256(lf).hexdigest()
    assert convert_s3gen.read_manifest(crlf) == convert_s3gen.read_manifest(convert_s3gen.MANIFEST)


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cpu"])
def test_torch_folds_record_the_manifest_bytes_as_earlier_versions(tmp_path, device):
    # Earlier versions hashed the checkout's bytes; a torch fold keeps that, so it rebuilds their file on a CRLF checkout.
    lf = convert_s3gen.MANIFEST.read_bytes().replace(b"\r\n", b"\n")
    crlf = tmp_path / "s3gen_tensors.tsv"
    crlf.write_bytes(lf.replace(b"\n", b"\r\n"))
    assert convert_s3gen.manifest_sha256(crlf, device) == hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()
    lf_file = tmp_path / "lf.tsv"
    lf_file.write_bytes(lf)
    assert convert_s3gen.manifest_sha256(lf_file, device) == convert_s3gen.manifest_sha256(lf_file, "numpy")
