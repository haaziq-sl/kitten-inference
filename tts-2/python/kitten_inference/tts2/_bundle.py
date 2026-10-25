"""Reads a prepared model directory (``config.json`` with its ``native`` manifest), as kitten-tts-native does."""
from __future__ import annotations

import hashlib
import json
import re
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SAFE = re.compile(r"[A-Za-z0-9_.\-/]+")


class BundleError(RuntimeError):
    """The prepared model directory is missing a file, or a file has the wrong size or SHA-256."""


@dataclass
class Bundle:
    root: Path
    model_sha: str                       # the KittenML/kitten-tts-2 revision it was prepared from ("" = unknown)
    default_voice: str
    lm: Path
    decoder: Path
    voices: Path
    tokenizer: Path
    text_data: Path
    cloning_dir: Optional[Path]          # None = prepared without the cloning files
    files: List[Tuple[Path, int, str]] = field(default_factory=list)  # (path, size, sha256) of every engine file

    @property
    def has_cloning(self) -> bool:
        return self.cloning_dir is not None


def _relative(root: Path, value, what: str) -> Path:
    if not isinstance(value, str) or not value or value.startswith("/") or value.endswith("/") or "//" in value \
            or not _SAFE.fullmatch(value) or any(p in (".", "..") for p in value.split("/")):
        raise BundleError(f"invalid path for {what} in {root / 'config.json'}: {value!r}")
    return root / value


def _entry(root: Path, block, what: str) -> Tuple[Path, int, str]:
    if not isinstance(block, dict):
        raise BundleError(f"{root / 'config.json'} has no {what} entry; prepare the directory again")
    path = _relative(root, block.get("file"), what)
    size, digest = block.get("size"), block.get("sha256")
    if not isinstance(size, int) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise BundleError(f"bad {what} entry in {root / 'config.json'}")
    return path, size, digest


def load(root: Path, *, need_cloning: bool = False) -> Bundle:
    """Reads ``root/config.json`` and checks that every listed file exists with its size (SHA-256: verify())."""
    root = Path(root)
    config_path = root / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BundleError(f"missing {config_path}: the model directory is not prepared") from None
    except (OSError, ValueError) as exc:
        raise BundleError(f"cannot read {config_path}: {exc}") from None
    if not isinstance(config, dict) or config.get("type") != "KITTEN2":
        raise BundleError(f"{config_path} is not a Kitten TTS 2 model directory")
    native = config.get("native")
    if not isinstance(native, dict) or native.get("version") != 1 or native.get("decoder", "default") != "default":
        raise BundleError(f"{config_path} has no prepared engine files (version 1, default decoder)")
    files = [_entry(root, native.get(key), key) for key in ("lm", "acoustic", "voices", "tokenizer")]
    lm, decoder, voices, tokenizer = (f[0] for f in files)
    text = native.get("text")
    if not isinstance(text, dict) or not isinstance(text.get("files"), list):
        raise BundleError(f"{config_path} has no text data entry")
    text_dir = _relative(root, text.get("directory"), "text")
    files += [_entry(root, f, "text") for f in text["files"]]
    cloning_dir = None
    cloning = native.get("cloning")
    if isinstance(cloning, dict):
        cloning_files = [_entry(root, f, "cloning") for f in cloning.get("files", [])]
        names = {p.name for p, _, _ in cloning_files}
        required = {"s3gen.safetensors", "voicebuild_consts.safetensors", "voicebuild_pyannote.safetensors"}
        if required <= names:
            cloning_dir = _relative(root, cloning.get("directory"), "cloning")
            if need_cloning:
                files += cloning_files
    if need_cloning and cloning_dir is None:
        raise BundleError(f"{root} has no voice cloning files")
    for path, size, _ in files:
        try:
            actual = path.stat().st_size
        except OSError:
            raise BundleError(f"missing model file: {path}") from None
        if actual != size:
            raise BundleError(f"model file has the wrong size: {path} ({actual} bytes, expected {size})")
    source = native.get("source") if isinstance(native.get("source"), dict) else {}
    revision = source.get("revision", "")
    default_voice = config.get("default_voice") if isinstance(config.get("default_voice"), str) else ""
    return Bundle(root=root, model_sha=revision if re.fullmatch(r"[0-9a-f]{40}", revision or "") else "",
                  default_voice=default_voice or "Bruno", lm=lm, decoder=decoder, voices=voices,
                  tokenizer=tokenizer, text_data=text_dir, cloning_dir=cloning_dir, files=files)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class Verifier:
    """Checks every file's SHA-256 in background threads (largest files first), like kitten-tts-native does
    while its engine starts. ``result()`` waits and raises BundleError for the first mismatch."""

    def __init__(self, bundle: Bundle, threads: int = 2, files: Optional[List[Tuple[Path, int, str]]] = None):
        self._files = sorted(bundle.files if files is None else files, key=lambda f: -f[1])
        self._errors: Dict[int, str] = {}
        self._next = 0
        self._lock = threading.Lock()
        self._threads = [threading.Thread(target=self._work, daemon=True, name="kitten-tts2-verify")
                         for _ in range(max(1, min(threads, len(self._files))))]
        for t in self._threads:
            t.start()

    def _work(self) -> None:
        while True:
            with self._lock:
                if self._next >= len(self._files):
                    return
                k = self._next
                self._next += 1
            path, _, expected = self._files[k]
            try:
                if _sha256(path) != expected:
                    self._errors[k] = f"model file is damaged (SHA-256 mismatch): {path}"
            except OSError as exc:
                self._errors[k] = f"cannot read model file {path}: {exc}"

    def result(self) -> None:
        for t in self._threads:
            t.join()
        if self._errors:
            raise BundleError(self._errors[min(self._errors)])


def voice_catalogue(voices_file: Path) -> List[dict]:
    """The voice records of a voices.kt2n file (its JSON metadata only; no tensor data is read)."""
    with open(voices_file, "rb") as f:
        header = f.read(64)
        if len(header) < 64 or header[:8] not in (b"KT2NVOX1", b"KT2NENG1"):
            raise BundleError(f"not a voices file: {voices_file}")
        _magic, _version, _flags, meta_off, meta_len = struct.unpack("<8sIIQQ", header[:32])
        f.seek(meta_off)
        meta = json.loads(f.read(meta_len).decode("utf-8"))
    voices = meta.get("voices")
    if not isinstance(voices, list):
        raise BundleError(f"no voices in {voices_file}")
    return voices


def public_voices(voices_file: Path) -> List[Tuple[str, List[str], str]]:
    """(id, other accepted names, language) of every public voice in catalogue order, as --list-voices prints."""
    out = []
    for rec in voice_catalogue(voices_file):
        if rec.get("hidden") is True:
            continue
        vid = rec.get("id")
        if not isinstance(vid, str):
            continue
        aliases = []
        for name in rec.get("public_names") or []:
            if isinstance(name, str) and name != vid and name not in aliases:
                aliases.append(name)
        language = rec.get("language") if isinstance(rec.get("language"), str) else ""
        out.append((vid, aliases, language))
    return out
