"""Reproduce the original native cloning exports from upstream checkpoints.

Preparation runs on the CPU. All exported tensor bytes must match the pinned
native contracts; inference does not import this module or its dependencies.

This is the torch exporter (tools/requirements-export-clone.txt). Ordinary preparation
does not need it: clone_assets.public_clone_exports() writes the same files from
the model repository's speaker/model.safetensors without torch. This module
recomputes the derived tensors and checks them against the pins, from either the
gated upstream checkpoint (--pyannote-model, --download) or the public weights
(--speaker-safetensors), and writes the reference files that ship beside it.
"""
from __future__ import annotations

import argparse
import collections
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import typing

from .clone_assets import (COMPUTED_CONSTS, CONSTS_METADATA, CONTRACTS, DERIVED_PYANNOTE,  # noqa: F401
                           PYANNOTE_CHECKPOINT_SHA256, PYANNOTE_REPO, PYANNOTE_REVISION, REFERENCE_CONSTS,
                           REFERENCE_SINCNET, checked_tensors, public_speaker_state, pyannote_metadata,
                           reference_constants, write_checked, write_reference)
from .convert_s3gen import PINNED_REVISION, PINNED_SHA256
from .safetensors_raw import SafetensorsFile, sha256_file

ACOUSTIC_REPO = "ResembleAI/chatterbox-turbo"
ACOUSTIC_FILE = "s3gen_meanflow.safetensors"


class _TrainingMetadata:
    """Discarded checkpoint metadata, never used to construct a network.

The pinned Lightning checkpoint carries callback/configuration classes. Mapping
those specific names to an inert record lets PyTorch's weights-only loader read
its tensor state without importing an obsolete training or service stack.
"""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass


_METADATA_GLOBALS = frozenset({
    "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
    "pytorch_lightning.callbacks.early_stopping.EarlyStopping",
    "pyannote.audio.core.task.Resolution", "pyannote.audio.core.task.Problem",
    "pyannote.audio.core.task.Specifications", "pyannote.audio.core.model.Introspection",
    "omegaconf.nodes.AnyNode", "omegaconf.listconfig.ListConfig",
    "omegaconf.dictconfig.DictConfig", "omegaconf.base.ContainerMetadata", "omegaconf.base.Metadata",
})


def checkpoint_state(path: Path):
    """Read only the exact approved checkpoint, with a fixed metadata allowlist."""
    if sha256_file(path) != PYANNOTE_CHECKPOINT_SHA256:
        raise ValueError("pyannote checkpoint checksum does not match the pinned upstream revision")
    import torch

    allowed = {name: _TrainingMetadata for name in _METADATA_GLOBALS}
    allowed.update({"collections.defaultdict": collections.defaultdict, "typing.Any": typing.Any,
                    "builtins.dict": dict, "builtins.list": list, "builtins.int": int})
    names = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    unknown = set(names) - allowed.keys()
    if unknown:
        raise ValueError("unsupported checkpoint metadata: " + ", ".join(sorted(unknown)))
    with torch.serialization.safe_globals([(allowed[name], name) for name in names]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["state_dict"]


def public_state(path: Path):
    """The FP32 state of speaker/model.safetensors (each tensor checked against its pin) as torch tensors."""
    import torch

    return {name: torch.from_numpy(value) for name, value in public_speaker_state(path).items()}


def speaker_tensors(path: Path):
    """Use the checkpoint's FP32 state (or the public speaker/model.safetensors) and the original SincNet filterbank."""
    state = public_state(path) if Path(path).suffix == ".safetensors" else checkpoint_state(path)
    import torch
    from asteroid_filterbanks import ParamSincFB

    prefix = "sincnet.conv1d.0.filterbank."
    filters = ParamSincFB(80, 251, stride=10, sample_rate=16000)
    loaded = filters.load_state_dict({name[len(prefix):]: value for name, value in state.items()
                                      if name.startswith(prefix)}, strict=False)
    # The upstream checkpoint carries all four filterbank tensors. The public speaker/model.safetensors leaves out
    # the two constructor buffers, n_ and window_; ParamSincFB's own values stand in for them (both are checked
    # against the pins when the export is written).
    if loaded.unexpected_keys or set(loaded.missing_keys) - {"n_", "window_"}:
        raise ValueError(f"unexpected SincNet filterbank state: missing {loaded.missing_keys}, "
                         f"unexpected {loaded.unexpected_keys}")
    for key in loaded.missing_keys:
        state[prefix + key] = getattr(filters, key)
    with torch.no_grad():
        state["sincnet.conv1d.0.filters"] = filters.filters()
    # The original Model.state_dict export excludes the training loss and the
    # integer batch counters. Selecting the complete pinned set does the same.
    return {name: state[name].detach().cpu().numpy().astype("<f4")
            for name in CONTRACTS["pyannote"]}


# The front-end constants (COMPUTED_CONSTS) are computed as before, checked against the reference values shipped
# beside clone_assets.py (the pinned bytes, computed once on an Intel CPU), and the reference values are written,
# so every CPU writes the same, pinned constants (see clone_assets.py).
CONST_TOLERANCE = 1e-4  # largest accepted absolute difference from the reference values (a few float32 steps;
                        # a wrong torch/torchaudio/librosa/scipy version differs by far more)


def computed_constants() -> dict:
    """Run the original front-end constant computations (no checkpoint needed)."""
    import librosa
    import numpy as np
    import torch
    import torchaudio
    from scipy.signal import firwin
    from torchaudio.compliance import kaldi
    from torchaudio.functional.functional import _get_sinc_resample_kernel

    poly = firwin(61, 1.0 / 3, window=("kaiser", 5.0)).astype(np.float32)
    poly *= 2
    kernel, width = _get_sinc_resample_kernel(24000, 16000, 8000, dtype=torch.float32)
    if width != 10:
        raise ValueError("unexpected torchaudio resampler width")
    bins, _ = kaldi.get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
    return {
        "mel24_basis": librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80,
                                           fmin=0, fmax=8000).astype(np.float32),
        "mel24_window": torch.hann_window(1920).numpy().astype(np.float32),
        "poly_2_3_taps": poly,
        "ta_24_16_kernel_cpu": kernel.numpy().astype(np.float32),
        # Resample constructs its kernel in float64 and stores it as float32.
        # The original GPU-side exporter only moved this buffer to CUDA;
        # copying that buffer back preserves these exact CPU-created bytes.
        "ta_24_16_kernel_dev": torchaudio.transforms.Resample(24000, 16000).kernel.numpy(),
        "kaldi_mel_banks": torch.nn.functional.pad(bins, (0, 1), value=0).numpy().astype(np.float32),
        "kaldi_povey_400": torch.hann_window(400, periodic=False,
                                           dtype=torch.float64).pow(0.85).float().numpy(),
    }


def canonical_constants(computed: dict, reference: dict, tolerance: float = CONST_TOLERANCE) -> dict:
    """Return the reference values after checking that this CPU computed the same constants within tolerance."""
    import numpy as np

    out = {}
    for name in COMPUTED_CONSTS:
        got, ref = np.asarray(computed[name]), reference[name]
        if got.shape != ref.shape:
            raise ValueError(f"clone consts: {name} has shape {list(got.shape)}, the reference has "
                             f"{list(ref.shape)}; check the torch/torchaudio/librosa/scipy versions "
                             "in requirements-cloning.txt")
        diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        worst = float(diff.max()) if diff.size else 0.0
        if not np.all(np.isfinite(got)) or worst > tolerance:
            raise ValueError(f"clone consts: {name} computed on this machine differs from the reference values "
                             f"by up to {worst:.3g} (limit {tolerance:g}); check the torch/torchaudio/librosa/scipy "
                             "versions in requirements-cloning.txt")
        if worst > 0:
            print(f"clone consts: {name}: {int(np.count_nonzero(diff))} of {diff.size} values differ from the "
                  f"reference by up to {worst:.3g} on this CPU; writing the reference values", file=sys.stderr)
        out[name] = ref.copy()
    return out


def write_reference_constants(arrays: dict, path: Path = REFERENCE_CONSTS) -> None:
    """Write the reference file (done once, from a machine that reproduces every pin)."""
    write_reference(arrays, path)


def frontend_tensors(acoustic: Path):
    """Run the original frontend export operations, with no approximation."""
    if sha256_file(acoustic) != PINNED_SHA256:
        raise ValueError("acoustic checkpoint checksum does not match the pinned upstream revision")
    tensors = canonical_constants(computed_constants(), reference_constants())
    # load_state_dict in the original S3Tokenizer exporter replaces both
    # constructor buffers with these checkpoint tensors. In particular, a
    # freshly generated Hann window is NOT bit-identical to tokenizer.window.
    with SafetensorsFile(acoustic) as source:
        for output, name in (("s3_mel_filters", "tokenizer._mel_filters"),
                             ("s3_window", "tokenizer.window")):
            if source.by_name[name].dtype != "F32":
                raise ValueError(f"acoustic {name} must be FP32")
            tensors[output] = source.array(name).copy()
    return tensors


_write_checked = write_checked


def export(checkpoint: Path, acoustic: Path, destination: Path, sincnet_reference: Path | None = None) -> dict:
    """Publish a local directory only after all 58 tensor checks pass.

    ``checkpoint`` is the gated upstream pytorch_model.bin or the public speaker/model.safetensors. With
    ``sincnet_reference`` the derived SincNet tensors are also written there (after the checks passed)."""
    if destination.exists():
        raise ValueError("export destination already exists; choose a new directory")
    public = Path(checkpoint).suffix == ".safetensors"
    speaker = speaker_tensors(checkpoint)
    consts = frontend_tensors(acoustic)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clone-export-", dir=destination.parent) as scratch:
        stage = Path(scratch) / "exports"
        stage.mkdir()
        files = [
            write_checked(stage / "voicebuild_consts.safetensors", consts, "consts", CONSTS_METADATA),
            write_checked(stage / "voicebuild_pyannote.safetensors", speaker, "pyannote", pyannote_metadata()),
        ]
        report = {"version": 1, "pyannote_repo": PYANNOTE_REPO, "pyannote_revision": PYANNOTE_REVISION,
                  "pyannote_checkpoint_sha256": PYANNOTE_CHECKPOINT_SHA256,
                  "pyannote_input": {"file": Path(checkpoint).name, "sha256": sha256_file(checkpoint),
                                     "kind": "public speaker/model.safetensors" if public else "upstream checkpoint"},
                  "acoustic_repo": ACOUSTIC_REPO, "acoustic_revision": PINNED_REVISION,
                  "acoustic_sha256": PINNED_SHA256, "files": files,
                  "tensor_validation": "native-cloning-fp32-v1", "legacy_speaker_bank": "absent",
                  "preparation_device": "cpu",
                  "dependencies": {name: importlib.metadata.version(name) for name in (
                      "torch", "torchaudio", "numpy", "scipy", "librosa", "asteroid-filterbanks")}}
        (stage / "export.json").write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
        stage.rename(destination)
    if sincnet_reference is not None:
        write_reference({name: speaker[name] for name in DERIVED_PYANNOTE}, sincnet_reference)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pyannote-model", type=Path,
                        help="Authorized pinned pytorch_model.bin, or its directory")
    source.add_argument("--download", action="store_true",
                        help="Download the gated pinned model using your existing Hugging Face login/HF_TOKEN")
    source.add_argument("--speaker-safetensors", type=Path,
                        help="The public speaker/model.safetensors of KittenML/kitten-tts-2 (no login)")
    parser.add_argument("--write-sincnet-reference", type=Path, metavar="JSON",
                        help=f"Also write the derived SincNet tensors as a reference file (like {REFERENCE_SINCNET.name})")
    parser.add_argument("--acoustic-source", type=Path, help="Pinned public s3gen_meanflow.safetensors")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face cache directory")
    parser.add_argument("--offline", action="store_true", help="Use local files or cached downloads only")
    parser.add_argument("--out", type=Path, required=True, help="New export directory")
    args = parser.parse_args(argv)
    checkpoint = args.pyannote_model or args.speaker_safetensors
    acoustic = args.acoustic_source
    if args.out.exists():
        parser.error("export destination already exists; choose a new directory")
    if checkpoint is not None and checkpoint.is_dir():
        checkpoint /= "pytorch_model.bin"
    if checkpoint is None or acoustic is None:
        from huggingface_hub import hf_hub_download
        common = {"cache_dir": args.cache_dir, "local_files_only": args.offline}
        if checkpoint is None:
            checkpoint = Path(hf_hub_download(PYANNOTE_REPO, "pytorch_model.bin",
                                              revision=PYANNOTE_REVISION, token=True, **common))
        if acoustic is None:
            acoustic = Path(hf_hub_download(ACOUSTIC_REPO, ACOUSTIC_FILE,
                                            revision=PINNED_REVISION, **common))
    extra = {"sincnet_reference": args.write_sincnet_reference} if args.write_sincnet_reference else {}
    print(json.dumps(export(checkpoint, acoustic, args.out, **extra), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
