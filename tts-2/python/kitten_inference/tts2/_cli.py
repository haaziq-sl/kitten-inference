"""``kitten-tts2``: runs the bundled kitten-tts-native, preparing the model first and adding ``--assets``.

    kitten-tts2 --text "Hello" --voice Bruno --output hello.wav    # any kitten-tts-native options
    kitten-tts2 --prepare-only [--revision R] [--with-cloning]      # prepare, print the model directory

Options of this wrapper (removed before the native tool runs): --revision R (model revision; default the latest),
--cache-dir DIR, --backend cuda|rocm, --prepare-only, --with-cloning.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
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
USAGE = """kitten-tts2: Kitten TTS 2 command line (runs the bundled kitten-tts-native).

  kitten-tts2 --text "Hello from Kitten TTS 2." --voice Bruno --output hello.wav
  kitten-tts2 --list-voices
  kitten-tts2 --prepare-only [--revision R] [--with-cloning]

The model is downloaded (about 4.6 GB, kept in the Hugging Face cache) and prepared (2 GB) on first use, once per
model revision, and --assets is added for you. Options of this command, besides the ones below:
  --revision R          model revision on Hugging Face (default: the latest)
  --cache-dir DIR       where prepared models (and then also the downloads) are kept
                        (default: ~/.cache/kitten-inference/tts2)
  --backend cuda|rocm   which installed engine to use (default: the installed one)
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
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        own, native = split_args(args)
    except UsageError as exc:
        print(f"kitten-tts2: {exc}\nRun 'kitten-tts2 --help' for usage.", file=sys.stderr)
        return 2
    names = native_options(native)
    from . import __version__, _backend, _prepare
    if not args or "--help" in names:
        print(USAGE, end="")
        try:
            backend = _backend.find(own.get("--backend"))
        except (ImportError, ValueError) as exc:  # the usage above is still the answer to --help
            print(f"kitten-tts2: {exc}", file=sys.stderr)
            return 0
        return _run(str(backend.cli), ["--help"])
    if own.get("--with-cloning") and not own.get("--prepare-only"):
        print("kitten-tts2: --with-cloning goes with --prepare-only (cloning files are added on their own when "
              "--reference-audio is used)", file=sys.stderr)
        return 2
    if own.get("--prepare-only") and native:
        print(f"kitten-tts2: --prepare-only takes only --revision, --cache-dir, --backend and --with-cloning "
              f"(got {' '.join(native)})", file=sys.stderr)
        return 2
    try:
        backend = _backend.find(own.get("--backend"))
    except (ImportError, ValueError) as exc:
        print(f"kitten-tts2: {exc}", file=sys.stderr)
        return 1
    if "--version" in names:
        print(f"kitten-tts2 {__version__} ({backend.distribution} {backend.version})")
        return _run(str(backend.cli), native)
    if own.get("--prepare-only") or "--assets" not in names:
        from . import _gpu
        # Before the first download, unless only preparing (that needs no GPU): a missing driver or GPU is reported
        # now, not after minutes of work. The native tool checks again when it starts.
        gpu_check = None if own.get("--prepare-only") else (
            lambda: _gpu.check(backend.name, _device(native), log=_prepare._log))
        try:
            directory = _prepare.ensure(text_data_tool=backend.text_data_tool, revision=own.get("--revision"),
                                        cache_dir=own.get("--cache-dir"),
                                        with_cloning=bool(own.get("--with-cloning")) or "--reference-audio" in names,
                                        before_download=gpu_check)
        except KeyboardInterrupt:
            print("kitten-tts2: cancelled", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"kitten-tts2: {exc}", file=sys.stderr)
            return 1
        if own.get("--prepare-only"):
            print(directory)
            return 0
        native = ["--assets", str(directory), *native]
    elif own.get("--revision") or own.get("--cache-dir"):
        print("kitten-tts2: --revision and --cache-dir have no effect with --assets", file=sys.stderr)
        return 2
    return _run(str(backend.cli), native)


if __name__ == "__main__":
    raise SystemExit(main())
