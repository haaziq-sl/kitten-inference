"""Shared fixtures for the converter tests (CI-safe: synthetic weights only)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO / "tools"
DATA_DIR = REPO / "tests" / "data"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class PackGolden:
    """Recorded outputs of the reference exact ternary packer (``tests/data/pack_golden.json``).

    The reference ``pack_exact`` and ``unpack`` were run once on every seeded
    input of ``test_pack.py`` and on the synthetic checkpoint's projections. Each
    record maps the sha256 of an input to the sha256 of the reference output (or
    to the error the reference raised), so the tests need neither the reference
    code nor torch.
    """

    def __init__(self, path: Path):
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.pack_entries = doc["pack_exact"]
        self.unpack_entries = doc["unpack"]
        self._pack = {(e["input_sha256"], tuple(e["input_shape"])): e for e in self.pack_entries}
        self._unpack = {(e["packed_sha256"], tuple(e["packed_shape"]), e["scales_sha256"],
                         tuple(e["scales_shape"]), e["k"]): e for e in self.unpack_entries}

    def pack_exact(self, bits: np.ndarray) -> dict:
        """The reference result for BF16 ``bits`` (uint16): ``packed``/``scales`` digests or ``error``."""
        bits = np.ascontiguousarray(bits, dtype="<u2")
        key = (sha256_array(bits), tuple(bits.shape))
        if key not in self._pack:
            pytest.fail(f"no golden pack_exact vector for this {list(bits.shape)} input (sha256 {key[0]})")
        return self._pack[key]

    def unpack(self, packed: np.ndarray, scales: np.ndarray, k: int) -> dict:
        """The reference ``unpack`` output digest for int32 ``packed`` codes and BF16 ``scales`` bits."""
        packed = np.ascontiguousarray(packed, dtype="<i4")
        scales = np.ascontiguousarray(scales, dtype="<u2")
        key = (sha256_array(packed), tuple(packed.shape), sha256_array(scales), tuple(scales.shape), int(k))
        if key not in self._unpack:
            pytest.fail(f"no golden unpack vector for codes {list(packed.shape)} / scales {list(scales.shape)}")
        return self._unpack[key]["output"]


@pytest.fixture(scope="session")
def golden():
    return PackGolden(DATA_DIR / "pack_golden.json")


@pytest.fixture(scope="session")
def synth_src(tmp_path_factory):
    """A synthetic Qwen3-shaped checkpoint (2 layers, hidden 256, inter 768, vocab 1024)."""
    from synth import make_checkpoint

    return make_checkpoint(tmp_path_factory.mktemp("synth"))


@pytest.fixture(scope="session")
def synth_kt2n(synth_src, tmp_path_factory):
    from kt2n_assets.convert import convert

    out = tmp_path_factory.mktemp("kt2n") / "model.kt2n"
    summary = convert(synth_src, out, log=lambda _m: None)
    return out, summary


@pytest.fixture
def cli_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TOOLS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    return env
