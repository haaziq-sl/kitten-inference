"""``kitten-tts2``: runs the bundled kitten-tts-native, preparing the model first and adding ``--assets``.

    kitten-tts --model KittenML/kitten-tts-2 --text "Hello" --voice Bruno --output hello.wav    # any kitten-tts-native options
    kitten-tts --model KittenML/kitten-tts-2 --prepare-only [--revision R] [--with-cloning]      # prepare, print the model directory

Options of this wrapper (removed before the native tool runs): --revision R (model revision; default the latest),
--cache-dir DIR, --backend cuda|rocm, --prepare-only, --with-cloning.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# kitten-tts-native's options that take a value (cli/cli_core.cc); every other --option is a flag.
NATIVE_VALUE_OPTIONS = {
    "--assets", "--text", "--text-file", "--voice", "--output", "--format", "--stream", "--device", "--max-streams",
    "--repeat", "--threads", "--data", "--report", "--seed", "--preset", "--temperature", "--top-p", "--min-p",
    "--top-k", "--max-tokens", "--reference-audio", "--reference-transcript", "--save-voice", "--load-voice",
    "--voice-cache", "--voice-cache-voices", "--voice-cache-mib", "--memory-profile",
}
OWN_VALUE_OPTIONS = {"--revision", "--cache-dir", "--backend"}
OWN_FLAGS = {"--prepare-only", "--with-cloning"}
USAGE = """kitten-tts: Kitten TTS 2 command line (runs the bundled kitten-tts-native).

  kitten-tts --model KittenML/kitten-tts-2 --text "Hello from Kitten TTS 2." --voice Bruno --output hello.wav
  kitten-tts --model KittenML/kitten-tts-2 --list-voices
  kitten-tts --model KittenML/kitten-tts-2 --prepare-only [--revision R] [--with-cloning]

The model is downloaded (about 4.6 GB, kept in the Hugging Face cache) and prepared (2 GB) on first use, once per
model revision, and --assets is added for you. Options of this command, besides the ones below:
  --revision R          model revision on Hugging Face (default: the latest)
  --cache-dir DIR       where prepared models (and then also the downloads) are kept
                        (default: ~/.cache/kitten-inference/tts2)
  --backend cuda|rocm   NVIDIA or AMD engine (default: the one whose GPU driver is found; NVIDIA first)
  --prepare-only        prepare the model, print its directory and exit
  --with-cloning        with --prepare-only: also prepare the voice cloning files

"""


class UsageError(Exception):
    pass


def split_args(argv: Sequence[str]) -> Tuple[dict, List[str]]:
    """Separates this wrapper's options from kitten-tts-native's (whose values are never taken for options)."""
    own: dict = {}
    native: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        key, eq, inline = arg.partition("=") if arg.startswith("--") else (arg, "", "")
        if key in OWN_FLAGS:
            if eq:
                raise UsageError(f"{key} does not take a value")
            own[key] = True
        elif key in OWN_VALUE_OPTIONS:
            if eq:
                value = inline
            elif i + 1 < len(argv):
                i += 1
                value = argv[i]
            else:
                raise UsageError(f"{key} needs a value")
            if not value:
                raise UsageError(f"{key} needs a value")
            own[key] = value
        else:
            native.append(arg)
            if key in NATIVE_VALUE_OPTIONS and not eq and i + 1 < len(argv):
                i += 1
                native.append(argv[i])
        i += 1
    return own, native


def native_options(native: Sequence[str]) -> set:
    """The option names in a native argument list (values skipped)."""
    names = set()
    i = 0
    while i < len(native):
        arg = native[i]
        if arg.startswith("--") or arg == "-h":
            key, eq, _ = arg.partition("=")
            names.add("--help" if key == "-h" else key)
            if key in NATIVE_VALUE_OPTIONS and not eq:
                i += 1
        i += 1
    return names


def _device(native: Sequence[str]) -> int:
    """The --device value of a native argument list (0 when absent or not a number: the tool reports that)."""
    for i, arg in enumerate(native):
        value = arg.partition("=")[2] if arg.startswith("--device=") else (
            native[i + 1] if arg == "--device" and i + 1 < len(native) else None)
        if value is not None:
            try:
                return max(0, int(value))
            except ValueError:
                return 0
    return 0


def _value(native: Sequence[str], option: str) -> Optional[str]:
    """The value of ``option`` in a native argument list (the last one wins, as in kitten-tts-native)."""
    found = None
    for i, arg in enumerate(native):
        if arg.startswith(option + "="):
            found = arg.partition("=")[2]
        elif arg == option and i + 1 < len(native):
            found = native[i + 1]
    return found


def _voice_identity(path: Optional[str]) -> Optional[Tuple[str, str]]:
    """The prepared model a saved voice file belongs to, as kitten-tts-native compares it; None when the file cannot
    be read as a voice (the native tool reports that)."""
    from . import _bundle
    if not path:
        return None
    try:
        p = Path(path)
        if p.stat().st_size > 64 << 20:
            return None
        identity = _bundle.identity(_bundle.blob_meta(p.read_bytes()))
    except (OSError, ValueError, _bundle.BundleError):
        return None
    return identity if all(identity) else None


def _model_identity(directory: Path) -> Optional[Tuple[str, str]]:
    from . import _bundle
    try:
        return _bundle.identity(_bundle.file_meta(_bundle.load(directory).voices))
    except (OSError, ValueError, _bundle.BundleError):
        return None


def _revisions_for(root: Path, identity: Tuple[str, str]) -> List[str]:
    """The prepared revisions in ``root`` that a voice with ``identity`` belongs to."""
    found = []
    try:
        configs = sorted(root.glob("*/config.json"))
    except OSError:
        return found
    for config in configs:
        if _model_identity(config.parent) == identity:
            found.append(config.parent.name)
    return found


class _VoiceNeedsRevision(Exception):
    pass


def _voice_message(path: str, revision: str, other: Optional[str], root: Path) -> str:
    if other:
        return (f"the voice {path} was made with the model revision {other}, which is still prepared in {root}; this "
                f"run uses {revision}. A saved voice works only with the model it was made with: add --revision "
                f"{other}, or clone the voice again with this model")
    return (f"the voice {path} was made with another model than revision {revision}, and that model is not prepared "
            f"in {root}. A saved voice works only with the model it was made with: pass its revision with "
            "--revision (kitten-tts --model KittenML/kitten-tts-2 --save-voice prints it), or clone the voice again with this model")


def _runtime_missing(backend, names: set) -> Optional[str]:
    """One line when the GPU runtime the native tool needs is not on this machine, instead of its raw error. The ROCm
    tool is linked against libamdhip64.so.7, so without ROCm it cannot start at all (the loader's error, exit 127);
    the CUDA tool loads the NVIDIA driver at run time, which --help and --version do not need."""
    from . import _gpu
    if backend.name == "rocm":
        found = _gpu.runtime("rocm")
        if found is None:
            return (f"the AMD ROCm 7.x runtime ({_gpu.HIP_RUNTIME}) was not found, and the AMD engine needs it: "
                    f"install ROCm 7.x as in step 1 of {_gpu.ROCM_DOCS}")
        if os.path.isabs(found):
            _library_folder(os.path.dirname(found))
        return None
    if names & {"--help", "--version"}:
        return None
    if _gpu.runtime("cuda") is None:
        return _gpu.no_driver_message()
    return None


def _library_folder(folder: str) -> None:
    """Puts a ROCm folder first in LD_LIBRARY_PATH: the ROCm runtime was found only there (no ld.so.conf entry), and
    the native tool looks there only that way (besides /opt/rocm-7.1.0/lib, its library's RUNPATH)."""
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if folder not in current.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = folder + (os.pathsep + current if current else "")


def _package_version() -> str:
    """The installed kitten-inference's version (a post release changes only that one)."""
    import kitten_inference
    from . import __version__
    return str(getattr(kitten_inference, "__version__", __version__))


def _run(cli: str, args: List[str]) -> int:
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name != "nt":
        os.execv(cli, [cli, *args])  # replaces this process: signals and the exit status are the tool's own
    # Windows: run it as a child in this console; Ctrl-C reaches it directly, so this process ignores it meanwhile.
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return subprocess.call([cli, *args])
    finally:
        signal.signal(signal.SIGINT, previous)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from . import _backend
    too_old = _backend.python_message()  # before anything else: the other modules need Python 3.9
    if too_old is not None:
        print(f"kitten-tts: {too_old}", file=sys.stderr)
        return 1
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        own, native = split_args(args)
    except UsageError as exc:
        print(f"kitten-tts: {exc}\nRun 'kitten-tts --model KittenML/kitten-tts-2 --help' for usage.", file=sys.stderr)
        return 2
    names = native_options(native)
    from . import _gpu, _prepare
    if not args or "--help" in names:
        print(USAGE, end="")
        try:
            backend = _backend.find(own.get("--backend"))
        except (ImportError, ValueError) as exc:  # the usage above is still the answer to --help
            print(f"kitten-tts: {exc}", file=sys.stderr)
            return 0
        missing = _runtime_missing(backend, names | {"--help"})
        if missing is not None:  # the usage above is still the answer
            print(f"kitten-tts: {missing}", file=sys.stderr)
            return 0
        return _run(str(backend.cli), ["--help"])
    if own.get("--with-cloning") and not own.get("--prepare-only"):
        print("kitten-tts: --with-cloning goes with --prepare-only (cloning files are added on their own when "
              "--reference-audio is used)", file=sys.stderr)
        return 2
    if own.get("--prepare-only") and native:
        print(f"kitten-tts: --prepare-only takes only --revision, --cache-dir, --backend and --with-cloning "
              f"(got {' '.join(native)})", file=sys.stderr)
        return 2
    try:
        backend = _backend.find(own.get("--backend"))
    except _backend.EngineNotInstalled as exc:
        if "--version" in names and not _backend._engine_platform():
            # A platform without a Kitten TTS 2 engine (Kitten TTS 0.8 runs there): the version is still the answer.
            print(f"kitten-tts {_package_version()} (no Kitten TTS 2 engine on this platform)")
            print(f"kitten-tts: {exc}", file=sys.stderr)
            return 0
        print(f"kitten-tts: {exc}", file=sys.stderr)
        return 1
    except (ImportError, ValueError) as exc:
        print(f"kitten-tts: {exc}", file=sys.stderr)
        return 1
    preparing = bool(own.get("--prepare-only")) or "--assets" not in names
    if preparing and "--version" not in names:
        libraries = _backend.modules_message()  # preparing a model needs them; --assets runs only the native tool
        if libraries is not None:
            print(f"kitten-tts: {libraries}", file=sys.stderr)
            return 1
    gpu_checked = False
    if not own.get("--prepare-only") and _backend.fallback(backend) is not None:
        # Both GPU runtimes are here and the NVIDIA engine was picked by itself: see now whether its driver starts,
        # since the native tool cannot switch engines (the AMD engine is used when it does not).
        try:
            backend = _backend.check_gpu(backend, _device(native), log=_prepare._log)
        except _gpu.GPUError as exc:
            print(f"kitten-tts: {exc}", file=sys.stderr)
            return 1
        gpu_checked = True
    # Said before anything is downloaded: the native tool could not start without its GPU runtime. Preparing alone
    # needs none (the text-data helper runs without it).
    missing = None if own.get("--prepare-only") else _runtime_missing(backend, names)
    if "--version" in names:
        print(f"kitten-tts {_package_version()} ({backend.label} {backend.version})")
        if missing is not None:
            print(f"kitten-tts: {missing}", file=sys.stderr)
            return 1
        return _run(str(backend.cli), native)
    if (own.get("--with-cloning") or "--reference-audio" in names) and not backend.can_clone:
        # Before the cloning files are downloaded and prepared (1.1 GB): this engine could not use them.
        print(f"kitten-tts: voice cloning needs the NVIDIA (CUDA) engine; the {backend.label} has no voice cloning. "
              "Clone and save the voice on an NVIDIA GPU (kitten-tts --model KittenML/kitten-tts-2 --reference-audio ... --save-voice FILE), then "
              "use it here with --load-voice FILE", file=sys.stderr)
        return 2
    if missing is not None:
        print(f"kitten-tts: {missing}", file=sys.stderr)
        return 1
    if preparing:
        # A saved voice works only with the model it was made with (the native tool would only print its status code).
        voice_path = _value(native, "--load-voice")
        voice = _voice_identity(voice_path)

        def before_download():
            # ensure() calls this only when the revision to run is not prepared yet. When the voice belongs to a
            # revision that is, that one is what the user needs: say so now, instead of downloading and preparing a
            # model the voice cannot use.
            if voice is not None:
                other = _revisions_for(_prepare.cache_root(own.get("--cache-dir")), voice)
                if other:
                    raise _VoiceNeedsRevision(other[0])
            # A missing driver or GPU is reported now, not after minutes of work. The native tool checks again when
            # it starts.
            if not gpu_checked:
                _gpu.check(backend.name, _device(native), log=_prepare._log)

        try:
            directory = _prepare.ensure(text_data_tool=backend.text_data_tool, revision=own.get("--revision"),
                                        cache_dir=own.get("--cache-dir"),
                                        with_cloning=bool(own.get("--with-cloning")) or "--reference-audio" in names,
                                        before_download=None if own.get("--prepare-only") else before_download)
        except KeyboardInterrupt:
            print("kitten-tts: cancelled", file=sys.stderr)
            return 130
        except _VoiceNeedsRevision as needed:
            asked = f"revision {own['--revision']}" if own.get("--revision") else "the latest revision"
            print(f"kitten-tts: the voice {voice_path} was made with the model revision {needed}, which is prepared "
                  f"in {_prepare.cache_root(own.get('--cache-dir'))}; {asked} is not prepared yet, and the voice would "
                  f"not work with it. Add --revision {needed} to use the voice (kitten-tts --model KittenML/kitten-tts-2 --prepare-only prepares "
                  "the latest revision)", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"kitten-tts: {exc}", file=sys.stderr)
            return 1
        if own.get("--prepare-only"):
            print(directory)
            return 0
        if voice is not None and _model_identity(Path(directory)) not in (None, voice):
            model = Path(directory)
            others = [r for r in _revisions_for(model.parent, voice) if r != model.name]
            message = _voice_message(voice_path, model.name, others[0] if others else None, model.parent)
            print(f"kitten-tts: {message}", file=sys.stderr)
            return 1
        if "--save-voice" in names:
            revision = Path(directory).name
            print(f"kitten-tts: model revision {revision}; a voice saved now loads only with this revision "
                  f"(kitten-tts --model KittenML/kitten-tts-2 --revision {revision} --load-voice ... once Hugging Face has a newer one)",
                  file=sys.stderr)
        native = ["--assets", str(directory), *native]
    elif own.get("--revision") or own.get("--cache-dir"):
        print("kitten-tts: --revision and --cache-dir have no effect with --assets", file=sys.stderr)
        return 2
    return _run(str(backend.cli), native)


if __name__ == "__main__":
    raise SystemExit(main())
