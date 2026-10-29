"""Prepare a self-contained native bundle from the public Kitten TTS 2 model.

Only preparation needs Python model tooling (numpy, tokenizers, huggingface_hub;
no PyTorch). Runtime consumes the checked native manifest, model files and
prepared voices without network access.

The model comes from the latest Hugging Face revision unless --revision pins one,
and the prepared manifest records the exact commit. Every other download is
pinned: the decoder and the NeMo data wheel by sha256, and the English grammar
files from KittenML/kitten-text-processing at a fixed commit, each checked
against a recorded sha256.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

from . import convert, convert_s3gen, voices
from .safetensors_raw import sha256_file

# The source tree (engines/tts2/) when this runs from tools/kt2n_assets; None when the package is installed elsewhere.
ROOT = Path(__file__).resolve().parents[2]
ROOT = ROOT if (ROOT / "CMakeLists.txt").is_file() and (ROOT / "tools").is_dir() else None
MODEL_REPO = "KittenML/kitten-tts-2"
ACOUSTIC_REPO = "ResembleAI/chatterbox-turbo"
ACOUSTIC_FILE = "s3gen_meanflow.safetensors"
SPEAKER_FILE = "speaker/model.safetensors"  # pyannote/embedding weights (MIT) inside the model repository
# What preparation reads from the model repository (for callers that download the snapshot themselves).
MODEL_ALLOW_PATTERNS = ["config.json", "lm/config.json", "lm/model.safetensors", "lm/tokenizer.json",
                        "cpp/default/voices.json", "voices/voices.json", "voices/*.npz"]
CLONING_ALLOW_PATTERNS = [SPEAKER_FILE]
LM_SIDE_FILES = ("config.json", "tokenizer.json")  # the lm/ files besides the weights that model.kt2n records
NEMO_WHEEL_URL = "https://files.pythonhosted.org/packages/b6/08/ca954a6b77d259561f56580adfd52eb69f54dc2469e1184157708480e7ed/nemo_text_processing-1.2.0-py3-none-any.whl"
NEMO_WHEEL_SHA256 = "dd78657a0f031023de8b2c8fee6b55d71f444485826f7b507881e407afa5d804"
# English compact grammars from the public kitten-text-processing repository.
GRAMMAR_REPO = "KittenML/kitten-text-processing"
GRAMMAR_COMMIT = "1988d61ca82d51ad11cb4a541ac61e1f46677eb4"
GRAMMAR_PATH = "kitten_text_processing/data/en"
GRAMMAR_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}/{name}"
GRAMMAR_FILES = {
    "metadata.json": "71eeab7698c6caa6bee301a1b95d95c9bedcac1835072947a85a5f9cb68b8d10",
    "tagger.fst.gz": "8bd9b7ccca17c15c68d6efce6ebd080ee676ebacf25087e60be73b94a23636f3",
    "verbalizer.fst.gz": "85e4dc57484408fe0675e59cc050d92d7e51bbb38b07164387ca7d1e40336070",
    "post.fst.gz": "32f2438d980349dfe82f2deaa03675376322d0daf67929df865d9903f66447c7",
}
TEXT_CONVERTER = "kt2n_prepare_text_data" + (".exe" if os.name == "nt" else "")
WHITELIST_TABLES = ("whitelist/tts.tsv", "whitelist/alternatives.tsv",
                    "whitelist/alternatives_all_format.tsv", "whitelist/lj_speech.tsv",
                    "whitelist/symbol.tsv", "suppletive.tsv")
TOKEN_MAP = {"audio_id_base": 151675, "num_audio_tokens": 6561, "speech_start_id": 151669,
             "speech_end_id": 151670, "text_start_id": 151671, "text_end_id": 151672,
             "start_id": 151673, "stop_id": 151674, "final_seg_id": 158236,
             "reference_speech_start_id": 158237, "reference_speech_end_id": 158238,
             "reference_text_start_id": 158239, "reference_text_end_id": 158240}


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)


def record(path: Path, root: Path) -> dict:
    return {"file": path.relative_to(root).as_posix(), "size": path.stat().st_size,
            "sha256": sha256_file(path)}


def check_config(config: dict) -> None:
    if config.get("type") != "KITTEN2" or config.get("sample_rate") != 24000:
        raise ValueError("native inference requires the KITTEN2 24 kHz model")
    if config.get("token_map") != TOKEN_MAP:
        raise ValueError("model token map is incompatible with this native engine")


def local_file(root: Path, name: str) -> Path:
    root = root.resolve()
    lexical = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"missing or unsafe model file: {name}")
    path = lexical.resolve()
    # Hugging Face snapshots use symlinks to content-addressed sibling blobs.
    # Permit that specific cache layout, while still rejecting arbitrary escapes.
    cached = (root.parent.name == "snapshots" and re.fullmatch(r"[a-f0-9]{40}", root.name)
              and any(path.is_relative_to(p) for p in
                      (root.parent.parent / "blobs", root.parent.parent.parent / "blobs")))
    if not (path.is_relative_to(root) or cached) or not path.is_file():
        raise ValueError(f"missing or unsafe model file: {name}")
    # Keep the snapshot path: callers also resolve sibling model/voice assets.
    path = lexical
    return path


def word_tables(wheel: bytes) -> tuple[bytes, bytes]:
    if hashlib.sha256(wheel).hexdigest() != NEMO_WHEEL_SHA256:
        raise ValueError("NeMo data wheel checksum mismatch")
    base = "nemo_text_processing/text_normalization/en/data/"
    whitelist, words = set(), set()
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for name in WHITELIST_TABLES:
            for line in archive.read(base + name).decode("utf-8").splitlines():
                left = line.split("\t", 1)[0].strip()
                if left:
                    whitelist.add(left.casefold())
        for name in sorted(archive.namelist()):
            if not name.startswith(base) or not name.endswith(".tsv"):
                continue
            for line in archive.read(name).decode("utf-8").splitlines():
                for word in re.split(r"[\s\-./]+", line.split("\t", 1)[0]):
                    word = word.strip(",;:'\"()").casefold()
                    if word:
                        words.add(word)
    if not whitelist or not words:
        raise ValueError("empty normalization word tables")
    encode = lambda values: ("\n".join(sorted(values)) + "\n").encode("utf-8")
    return encode(whitelist), encode(words)


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


NEMO_WHEEL_NAME = "nemo_text_processing-1.2.0.whl"


def nemo_wheel_path(cache: Path) -> Path:
    """Where pinned_wheel keeps the cached NeMo data wheel (so a caller can pass it back as --nemo-wheel)."""
    return cache / NEMO_WHEEL_NAME


def pinned_wheel(cache: Path, offline: bool, fetch=download) -> bytes:
    """The NeMo data wheel, downloaded once into ``cache`` and sha256-checked; a bad cached copy is downloaded again."""
    path = nemo_wheel_path(cache)
    if path.is_file():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() == NEMO_WHEEL_SHA256:
            return data
    if offline:
        raise ValueError(f"offline mode needs --nemo-wheel or the cached NeMo data wheel {path} "
                         "(missing or checksum mismatch)")
    data = fetch(NEMO_WHEEL_URL)
    if hashlib.sha256(data).hexdigest() != NEMO_WHEEL_SHA256:
        raise ValueError(f"downloaded NeMo data wheel checksum mismatch ({NEMO_WHEEL_URL})")
    atomic_bytes(path, data)
    return data


def pinned_grammar(cache: Path, offline: bool, fetch=download) -> Path:
    """The English grammar files at GRAMMAR_COMMIT, downloaded once into ``cache`` and sha256-checked."""
    destination = cache / f"kitten-text-processing-{GRAMMAR_COMMIT}" / "en"
    for name, digest in GRAMMAR_FILES.items():
        path = destination / name
        if path.is_file() and sha256_file(path) == digest:
            continue
        if offline:
            raise ValueError(f"offline mode needs --grammar-dir or the cached grammar file {path}")
        url = GRAMMAR_URL.format(repo=GRAMMAR_REPO, commit=GRAMMAR_COMMIT, path=GRAMMAR_PATH, name=name)
        data = fetch(url)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"downloaded grammar file checksum mismatch: {name} ({url})")
        atomic_bytes(path, data)
    return destination


def converter_candidates(base: Path) -> list[Path]:
    """build/bin, then the preset build directories: build/default/bin, then other build/*/bin."""
    presets = sorted(p for p in base.glob("build/*/bin") if p.parent.name != "default")
    return [base / "build/bin", base / "build/default/bin", *presets]


def find_text_converter(explicit: Path | None) -> Path | None:
    """--text-converter, else a built kt2n_prepare_text_data (source tree, then cwd), else PATH."""
    if explicit is not None:
        return explicit if explicit.is_file() else None
    roots = ([ROOT] if ROOT is not None else []) + [Path.cwd()]
    for build in (candidate for root in roots for candidate in converter_candidates(root)):
        candidate = build / TEXT_CONVERTER
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which(TEXT_CONVERTER)
    return Path(found) if found else None


def check_weight_norm_device(device: str) -> None:
    """Fail before any download when the requested weight-norm fold device is unusable.

    The default numpy fold needs no torch; only --weight-norm-device cuda|cpu imports it."""
    if device == "numpy":
        return
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"error: --weight-norm-device {device} folds the decoder weight norms with torch, which is "
                         "not installed. Install torch 2.7.1 (for cuda: python -m pip install torch==2.7.1 "
                         "--index-url https://download.pytorch.org/whl/cu128), or leave out --weight-norm-device: "
                         f"the default numpy fold needs no torch ({exc})") from None
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"error: --weight-norm-device cuda folds the decoder weight norms on CUDA, but this torch {torch.__version__} "
            f"cannot use CUDA (built for CUDA: {torch.version.cuda or 'no'}). Install a CUDA build of torch 2.7.1 "
            "on a machine with an NVIDIA GPU and driver, or leave out --weight-norm-device (the default numpy fold "
            "needs no torch and no GPU).")
    if device.startswith("cuda") and getattr(torch.version, "hip", None):
        # ROCm torch answers torch.cuda.is_available() too, but folds on an AMD GPU with other arithmetic.
        raise SystemExit(
            f"error: this torch {torch.__version__} is a ROCm (AMD) build; folding the decoder weight norms on it gives "
            "weights that differ from the tested bundle. Leave out --weight-norm-device (the default numpy fold needs "
            "no torch and no GPU).")


def bundle_size(root: Path) -> tuple[int, int]:
    files = [p for p in root.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def human_size(n: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n} bytes"


def shell_quote(text: str, windows: bool | None = None) -> str:
    """Quote one argument for the user's shell: POSIX quoting, or the Windows command-line rules on Windows."""
    if os.name == "nt" if windows is None else windows:
        return subprocess.list2cmdline([text])
    return shlex.quote(text)


def print_summary(out: Path, config_path: Path, *, cloning: bool) -> None:
    count, total = bundle_size(out)
    print(config_path)
    print(f"Bundle: {out} ({count} files, {human_size(total)})")
    assets = shell_quote(str(out))
    print(f"Next: kitten-tts-native --assets {assets} --text 'Hello from Kitten TTS.' --voice Bruno "
          "--output hello.wav")
    if cloning:
        print(f"Clone a voice: kitten-tts-native --assets {assets} --text 'Hello from my own voice.' "
              "--reference-audio reference.wav --reference-transcript 'The words in the recording.' "
              "--output clone.wav")


def prepare_text(grammar_dir: Path, converter: Path, wheel: bytes, destination: Path) -> dict:
    """Map public English compact graphs to native OpenFst and exact word tables."""
    source = grammar_dir / "en" if (grammar_dir / "en").is_dir() else grammar_dir
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("language") != "en" or metadata.get("nemo_version") != "1.2.0":
        raise ValueError("native grammar conversion requires English NeMo 1.2.0 data")
    whitelist, words = word_tables(wheel)
    destination.mkdir(parents=True, exist_ok=True)
    sources = {}
    with tempfile.TemporaryDirectory(prefix="kt2n-grammar-") as scratch:
        for name, target in (("tagger", "tagger"), ("verbalizer", "verbalizer"), ("post", "post_processor")):
            compressed = (source / (name + ".fst.gz")).read_bytes()
            digest = hashlib.sha256(compressed).hexdigest()
            if digest != metadata["graphs"][name]["sha256"]:
                raise ValueError(f"public grammar checksum mismatch: {name}")
            inp = Path(scratch) / (name + ".compact")
            inp.write_bytes(gzip.decompress(compressed))
            out = destination / (target + ".fst.partial")
            subprocess.run([str(converter.resolve()), str(inp), str(out)], check=True)
            out.replace(destination / (target + ".fst"))
            sources[name] = digest
    atomic_bytes(destination / "nemo_whitelist.txt", whitelist)
    atomic_bytes(destination / "whitelist_words.txt", whitelist)
    atomic_bytes(destination / "data_words.txt", words)
    return {"grammar_sha256": sources, "nemo_data_sha256": NEMO_WHEEL_SHA256}


def prepare_voices(voice_file: Path, tokenizer_file: Path, config: dict, destination: Path,
                   checkpoint_sha: str, acoustic_sha: str, *, model_dir: Path | None = None) -> None:
    from tokenizers import Tokenizer
    import numpy as np
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    rows = json.loads(voice_file.read_text(encoding="utf-8"))
    raw = {}
    if model_dir is not None:
        public_file = local_file(model_dir, config["voices"])
        public = json.loads(public_file.read_text(encoding="utf-8"))
        export_file = local_file(model_dir, config["cpp"]["decoders"]["default"]["voices"]["file"])
        original = json.loads(export_file.read_text(encoding="utf-8"))
        for name, row in rows.items():
            # Only the unchanged public builtin carries the catalogue embedding;
            # a prepared/custom override must not inherit somebody else's raw vector.
            if name in public and row == original.get(name):
                artifact = local_file(model_dir, str(Path(config["voices"]).parent / public[name]["artifacts"]))
                with np.load(artifact, allow_pickle=False) as data:
                    raw[name] = np.asarray(data["embedding"], dtype="<f4").reshape(-1).copy()
    catalogue = voices.from_cpp_voices(rows, tokenizer, config["token_map"], raw_embeddings=raw)
    blob = voices.write(catalogue, checkpoint=checkpoint_sha, s3gen_sha256=acoustic_sha,
                        source={"voices_sha256": sha256_file(voice_file)},
                        peak_norm={"schema": "kitten-public-voices.v1", "default_gain_db": 0.0})
    voices.read(blob)
    atomic_bytes(destination, blob)


def acoustic_source(args) -> Path:
    acoustic = args.acoustic_source
    if acoustic is None:
        from huggingface_hub import hf_hub_download
        acoustic = Path(hf_hub_download(ACOUSTIC_REPO, ACOUSTIC_FILE, revision=convert_s3gen.PINNED_REVISION,
                        cache_dir=args.cache_dir, local_files_only=args.offline, token=False))
    if sha256_file(acoustic) != convert_s3gen.PINNED_SHA256:
        raise ValueError("full decoder source checksum mismatch")
    return acoustic


def speaker_source(args, repo: str, revision: str, model: Path | None) -> Path:
    """speaker/model.safetensors from the same model snapshot: --model-dir, else the Hugging Face cache/download
    (public; no login)."""
    if model is not None:
        return local_file(model, SPEAKER_FILE)
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(repo, revision=revision, cache_dir=args.cache_dir,
                                      local_files_only=args.offline, token=False, allow_patterns=CLONING_ALLOW_PATTERNS))
    return local_file(snapshot, SPEAKER_FILE)


def prepare_cloning(args, acoustic: Path, manifest: dict, model: Path | None = None) -> dict:
    from .clone_assets import prepare_clone_files
    catalogue = voices.read(local_file(args.out, manifest["voices"]["file"]).read_bytes())
    bank = [v for v in catalogue.voices if v["kind"] == "builtin"]
    if not bank:
        raise ValueError("cloning requires builtin voices with their original raw speaker embeddings")
    for voice in bank:
        raw = catalogue.arrays[voice["id"]]["raw_speaker"]
        if raw.shape != (512,) or not (raw != 0).any():
            raise ValueError("cloning speaker bank contains an empty or zero vector")
    destination = args.out / "native/cloning"
    if args.clone_consts and args.clone_pyannote:
        source = prepare_clone_files(acoustic, args.clone_consts, args.clone_pyannote, destination)
    elif not args.pyannote_model:
        # Default: the pyannote/embedding weights in the model repository plus pinned derived constants; no torch,
        # no login. The files are byte-identical to the torch exporter's (export_clone.py).
        from .clone_assets import public_clone_exports
        repo, revision = manifest["source"]["model_repo"], manifest["source"]["revision"]
        speaker = speaker_source(args, repo, revision, model)
        with tempfile.TemporaryDirectory(prefix="kt2n-clone-export-") as scratch:
            exports = Path(scratch) / "exports"
            public = public_clone_exports(speaker, acoustic, exports)
            source = prepare_clone_files(acoustic, exports / "voicebuild_consts.safetensors",
                                         exports / "voicebuild_pyannote.safetensors", destination)
        source.update(public, pyannote_source=f"{repo}@{revision}/{SPEAKER_FILE}")
    else:
        # --pyannote-model: the original exporter on the gated upstream checkpoint (needs torch, torchaudio,
        # librosa, scipy and asteroid-filterbanks; see export_clone.py).
        from .export_clone import export
        checkpoint = args.pyannote_model
        if checkpoint.is_dir():
            checkpoint = checkpoint / "pytorch_model.bin"
        with tempfile.TemporaryDirectory(prefix="kt2n-clone-export-") as scratch:
            exports = Path(scratch) / "exports"
            export(checkpoint, acoustic, exports)
            source = prepare_clone_files(acoustic, exports / "voicebuild_consts.safetensors",
                                         exports / "voicebuild_pyannote.safetensors", destination)
    return {"version": 1, "directory": "native/cloning",
            "files": [record(path, args.out) for path in sorted(destination.iterdir())],
            "source": source, "speaker_bank_voices": len(bank)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, help="Existing public model checkout/cache (no model download)")
    p.add_argument("--repo", default=MODEL_REPO, help=f"Hugging Face model repository (default: {MODEL_REPO})")
    p.add_argument("--revision", default="main",
                   help="Model revision: a branch, tag or commit (default: main, the latest)")
    p.add_argument("--cache-dir", type=Path,
                   help="Download cache (default: the Hugging Face cache, and ~/.cache/kitten-tts for the "
                        "grammar files and the NeMo data wheel)")
    p.add_argument("--offline", action="store_true", help="Use local files/cache only")
    p.add_argument("--out", type=Path, required=True, help="Bundle directory to create (or to update with "
                   "--voices-only / --cloning-only)")
    p.add_argument("--voices", type=Path,
                   help="voices.json in the model's cpp/default layout, including prepared references")
    p.add_argument("--voices-only", action="store_true", help="Update voice catalogue in an existing native bundle")
    p.add_argument("--with-cloning", "--cloning", action="store_true",
                   help="Include the optional voice-cloning networks (their weights come from the same model "
                        f"revision, {SPEAKER_FILE}; no Hugging Face login and no torch)")
    p.add_argument("--cloning-only", action="store_true", help="Add cloning to an existing native bundle")
    p.add_argument("--clone-consts", type=Path, help="Original voicebuild_consts.safetensors export")
    p.add_argument("--clone-pyannote", type=Path, help="Original authorized voicebuild_pyannote.safetensors export")
    p.add_argument("--pyannote-model", type=Path,
                   help="Optional: the pinned upstream pyannote/embedding checkpoint (pytorch_model.bin or its "
                        "directory) instead of the model repository's copy; needs the export_clone.py dependencies "
                        "(torch). The result is the same")
    p.add_argument("--acoustic-source", type=Path, help="Pinned full decoder safetensors; otherwise use HF cache/download")
    p.add_argument("--grammar-dir", type=Path,
                   help="Local kitten-text-processing data directory (or its en/ subdirectory) for offline use; "
                        f"otherwise the pinned English grammar files of {GRAMMAR_REPO} at {GRAMMAR_COMMIT[:7]} "
                        "are downloaded and checked")
    p.add_argument("--text-converter", "--text-data-tool", dest="text_converter", type=Path,
                   help=f"Built {TEXT_CONVERTER} executable (default: build/bin/, build/default/bin/ or "
                        "another build/*/bin/ of this source tree or the current directory, then PATH)")
    p.add_argument("--nemo-wheel", type=Path, help="Pinned NeMo 1.2.0 wheel (read data only; never installed)")
    p.add_argument("--weight-norm-device", choices=("numpy", "cuda", "cpu"), default="numpy",
                   help="How to fold the decoder weight norms. numpy (default): no torch or GPU, the same files on "
                        "every machine, and the same folded weights (so the same audio) as cuda. Its files differ "
                        "from earlier versions' in metadata only, but a voice saved with a model prepared by an "
                        "earlier version does not load with them. cuda (torch 2.7.1+cu128 and an NVIDIA GPU): the "
                        "files earlier versions wrote; they record the GPU model and the torch version, so they come "
                        "out the same again only on the same GPU model with the same torch. cpu (torch): folds "
                        "slightly differently")
    a = p.parse_args(argv)
    a.out = a.out.resolve()
    config_path = a.out / "config.json"
    if a.cloning_only:
        a.with_cloning = True
    if bool(a.clone_consts) != bool(a.clone_pyannote):
        p.error("--clone-consts and --clone-pyannote must be supplied together")
    if a.pyannote_model and a.clone_pyannote:
        p.error("choose --pyannote-model or the two prepared clone exports")
    if (a.pyannote_model or a.clone_consts) and not a.with_cloning:
        p.error("clone sources require --with-cloning or --cloning-only")
    if a.voices_only and a.with_cloning:
        p.error("update voices and cloning in separate commands")
    if a.cloning_only:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        check_config(cfg)
        manifest = cfg["native"]
        if manifest.get("version") != 1 or manifest.get("decoder") != "default" or "cloning" in manifest:
            raise ValueError("incompatible bundle or cloning already prepared")
        for key in ("lm", "acoustic", "voices", "tokenizer"):
            expected = manifest[key]
            if record(local_file(a.out, expected["file"]), a.out) != expected:
                raise ValueError(f"bundle checksum mismatch: {key}")
        acoustic = acoustic_source(a)
        manifest["cloning"] = prepare_cloning(a, acoustic, manifest,
                                              a.model_dir.resolve() if a.model_dir else None)
        atomic_bytes(config_path, (json.dumps(cfg, indent=2) + "\n").encode())
        print_summary(a.out, config_path, cloning=True)
        return 0
    if a.voices_only:
        if not a.voices:
            p.error("--voices-only requires --voices")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        check_config(cfg)
        manifest = cfg["native"]
        if manifest.get("version") != 1 or manifest.get("decoder") != "default":
            raise ValueError("incompatible native bundle")
        for key in ("lm", "acoustic", "tokenizer"):
            expected = manifest[key]
            got = record(local_file(a.out, expected["file"]), a.out)
            if got != expected:
                raise ValueError(f"bundle checksum mismatch: {key}")
        target = local_file(a.out, manifest["voices"]["file"])
        prepare_voices(a.voices, local_file(a.out, manifest["tokenizer"]["file"]), cfg, target,
                       manifest["lm"]["sha256"], manifest["acoustic"]["sha256"])
        manifest["voices"] = record(target, a.out)
        atomic_bytes(config_path, (json.dumps(cfg, indent=2) + "\n").encode())
        print_summary(a.out, config_path, cloning=False)
        return 0
    if config_path.exists():
        raise ValueError("output bundle already exists; choose a new --out or use --voices-only")
    converter = find_text_converter(a.text_converter)
    if converter is None:
        p.error(f"--text-converter {a.text_converter} is not a file" if a.text_converter else
                f"cannot find {TEXT_CONVERTER}: build this project first (cmake --preset default writes it to "
                "build/default/bin/), put it on PATH, or pass --text-converter")
    check_weight_norm_device(a.weight_norm_device)
    # Small pinned inputs first, so a bad cache or network fails before the large downloads.
    conversion_cache = (a.cache_dir or Path.home() / ".cache/kitten-tts") / "native-conversion"
    if a.nemo_wheel:
        wheel = a.nemo_wheel.read_bytes()
        if hashlib.sha256(wheel).hexdigest() != NEMO_WHEEL_SHA256:
            raise ValueError(f"NeMo data wheel checksum mismatch: {a.nemo_wheel}")
    else:
        wheel = pinned_wheel(conversion_cache, a.offline)
    word_tables(wheel)
    grammar_dir = a.grammar_dir if a.grammar_dir else pinned_grammar(conversion_cache, a.offline)
    if a.model_dir:
        model = a.model_dir.resolve()
    else:
        from huggingface_hub import snapshot_download
        model = Path(snapshot_download(a.repo, revision=a.revision, cache_dir=a.cache_dir,
                      local_files_only=a.offline, token=False,
                      allow_patterns=MODEL_ALLOW_PATTERNS + (CLONING_ALLOW_PATTERNS if a.with_cloning else [])))
    config = json.loads(local_file(model, "config.json").read_text(encoding="utf-8"))
    check_config(config)
    lm_dir = local_file(model, config["lm_dir"] + "/model.safetensors").parent
    voice_file = a.voices or local_file(model, config["cpp"]["decoders"]["default"]["voices"]["file"])
    acoustic = acoustic_source(a)
    dest = a.out / "native"
    dest.mkdir(parents=True, exist_ok=True)
    source_text = prepare_text(grammar_dir, converter, wheel, dest / "text")
    with tempfile.TemporaryDirectory(prefix="kt2n-config-") as scratch:
        runtime = Path(scratch) / "runtime_config.json"
        runtime.write_bytes((json.dumps({"type": config["type"], "token_map": config["token_map"],
                                      "generation": config.get("generation", {})})).encode("utf-8"))
        # Only the files preparation downloads: a shared Hugging Face cache may hold more of the same revision.
        convert.convert(lm_dir, dest / "model.kt2n", runtime_config=str(runtime), side_files=LM_SIDE_FILES)
    convert_s3gen.convert(str(acoustic), str(dest / "s3gen.kt2n"),
                         manifest_path=Path(__file__).with_name("s3gen_tensors.tsv"),
                         expect_sha256=convert_s3gen.PINNED_SHA256, revision=convert_s3gen.PINNED_REVISION,
                         device=a.weight_norm_device, extra_groups=[])
    tokenizer = local_file(model, config["lm_dir"] + "/tokenizer.json")
    shutil.copyfile(tokenizer, dest / "tokenizer.json")
    prepare_voices(voice_file, tokenizer, config, dest / "voices.kt2n",
                   sha256_file(dest / "model.kt2n"), sha256_file(dest / "s3gen.kt2n"), model_dir=model)
    manifest = {"version": 1, "decoder": "default",
                "lm": record(dest / "model.kt2n", a.out), "acoustic": record(dest / "s3gen.kt2n", a.out),
                "voices": record(dest / "voices.kt2n", a.out), "tokenizer": record(dest / "tokenizer.json", a.out),
                "text": {"directory": "native/text", "files": [record(f, a.out) for f in sorted((dest / "text").iterdir())]},
                "source": {"model_repo": a.repo,
                           "revision": model.name if re.fullmatch(r"[a-f0-9]{40}", model.name) else a.revision,
                           "lm_sha256": sha256_file(lm_dir / "model.safetensors"),
                           "acoustic_sha256": convert_s3gen.PINNED_SHA256,
                           "weight_norm_device": a.weight_norm_device, **source_text}}
    # A standalone bundle contains native files only. When merging into an HF
    # model repository, publishers copy native/ and only the native config key.
    config.pop("cpp", None)
    if a.with_cloning:
        manifest["cloning"] = prepare_cloning(a, acoustic, manifest, model)
    config["native"] = manifest
    atomic_bytes(config_path, (json.dumps(config, indent=2) + "\n").encode())
    print_summary(a.out, config_path, cloning=a.with_cloning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
