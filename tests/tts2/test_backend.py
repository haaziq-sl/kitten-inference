"""Engine discovery: the installed engines, the backend= argument, $KITTEN_TTS2_BACKEND, the GPU runtime probe, the
retry with the AMD engine, and what each platform and Python is told."""
import os
import platform
import sys

import pytest

from kitten_inference.tts2 import _backend, _gpu

PLATFORMS = "Kitten TTS 2 runs on Linux x86-64 (NVIDIA or AMD Radeon GPU) and Windows x64 (NVIDIA GPU); this is "
REINSTALL = 'pip install --force-reinstall "kitten-inference"'
ENGINE_FILES = f"This install is missing its Kitten TTS 2 engine files. Reinstall it: {REINSTALL}"


def _on(monkeypatch, system, machine):
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(platform, "machine", lambda: machine)


def _runtimes(monkeypatch, *present):
    """Only the GPU runtimes of the engines named in ``present`` load (the HIP runtime also by a ROCm folder path)."""
    def load(name):
        kind = "rocm" if os.path.basename(name) == _gpu.HIP_RUNTIME else "cuda"
        if kind not in present:
            raise OSError(f"{name}: cannot open shared object file: No such file or directory")
        return object()
    monkeypatch.setattr(_gpu, "_cdll", load)


@pytest.mark.parametrize("system, machine, shown", [
    ("darwin", "arm64", "macOS arm64"), ("darwin", "x86_64", "macOS x86_64"), ("linux", "aarch64", "Linux aarch64"),
    ("win32", "ARM64", "Windows ARM64"), ("android", "aarch64", "Android aarch64")])
def test_platform_without_an_engine(no_backends, monkeypatch, system, machine, shown):
    _on(monkeypatch, system, machine)
    expected = PLATFORMS + shown + ". Kitten TTS 0.8 (kitten_inference.tts08) runs here."
    assert _backend.no_engine_message() == expected
    with pytest.raises(_backend.EngineNotInstalled) as e:
        _backend.find()
    assert str(e.value) == expected and isinstance(e.value, ImportError)
    with pytest.raises(_backend.EngineNotInstalled, match="this is " + shown):
        _backend.find("cuda")


@pytest.mark.parametrize("system, machine", [("linux", "x86_64"), ("win32", "AMD64")])
def test_engine_files_missing(no_backends, monkeypatch, system, machine):
    _on(monkeypatch, system, machine)
    assert _backend.no_engine_message() == ENGINE_FILES
    with pytest.raises(_backend.EngineNotInstalled) as e:
        _backend.find()
    assert str(e.value) == ENGINE_FILES


def test_requested_engine_missing(no_backends, fake_backend, monkeypatch):
    _on(monkeypatch, "linux", "x86_64")
    fake_backend("cuda")
    assert _backend.no_engine_message() is None
    with pytest.raises(_backend.EngineNotInstalled) as e:
        _backend.find("rocm")
    assert str(e.value) == ENGINE_FILES


def test_rocm_on_windows(no_backends, fake_backend, monkeypatch):
    _on(monkeypatch, "win32", "AMD64")
    fake_backend("cuda")
    message = ("the AMD (ROCm) engine runs on Linux only; on Windows, run Kitten TTS 2 inside WSL2: "
               "https://github.com/KittenML/kitten-inference/blob/main/engines/tts2/docs/amd-radeon.md")
    with pytest.raises(_backend.EngineNotInstalled) as e:
        _backend.find("rocm")
    assert str(e.value) == message
    monkeypatch.setenv("KITTEN_TTS2_BACKEND", "rocm")
    with pytest.raises(_backend.EngineNotInstalled) as e:
        _backend.find()
    assert str(e.value) == message


def test_python_38(no_backends, fake_backend, request):
    fake_backend("cuda")
    assert _backend.python_message() is None
    request.getfixturevalue("python38")
    message = ("Kitten TTS 2 needs Python 3.9 or newer (this is Python 3.8). Kitten TTS 0.8 (kitten_inference.tts08) "
               "works on Python 3.8.")
    assert _backend.python_message() == message
    with pytest.raises(ImportError) as e:
        _backend.require_python_api()
    assert str(e.value) == message


def test_missing_libraries(no_backends, fake_backend, hide_modules):
    fake_backend("cuda")
    _backend.require_python_api()
    hide_modules("numpy", "tokenizers")
    assert _backend.missing_modules() == ["numpy", "tokenizers"]
    message = ('Kitten TTS 2 needs more Python packages (not installed: numpy, tokenizers). Install them: '
               'pip install "kitten-inference"')
    with pytest.raises(ImportError) as e:
        _backend.require_python_api()
    assert str(e.value) == message


def test_checks_run_in_order(no_backends, monkeypatch, hide_modules, request):
    # Python version, then an engine for this platform, then the libraries: each message is the one that helps.
    _on(monkeypatch, "darwin", "arm64")
    hide_modules("numpy")
    with pytest.raises(_backend.EngineNotInstalled, match="this is macOS arm64"):
        _backend.require_python_api()
    request.getfixturevalue("python38")
    with pytest.raises(ImportError, match="needs Python 3.9"):
        _backend.require_python_api()


def test_kittentts2_reports_missing_backend(no_backends):
    with pytest.raises(ImportError) as e:
        from kitten_inference.tts2 import KittenTTS2
        KittenTTS2()
    assert str(e.value) == _backend.no_engine_message()


def test_found_backend(no_backends, fake_backend):
    module = fake_backend("cuda")
    b = _backend.find()
    assert (b.name, b.label, b.abi, b.version) == ("cuda", "NVIDIA (CUDA) engine", 2, "0.2.0")
    assert b.cli == module.CLI and b.text_data_tool == module.TEXT_DATA_TOOL and b.lib == module.LIB
    assert b.can_clone and b.runtime_found is True


def test_rocm_backend_cannot_clone(no_backends, fake_backend):
    fake_backend("rocm", caps=())
    b = _backend.find()
    assert b.name == "rocm" and b.label == "AMD (ROCm) engine" and not b.can_clone


@pytest.mark.skipif(sys.platform == "win32", reason="platform: the AMD engine is Linux only")
def test_env_selects_backend(no_backends, fake_backend, monkeypatch):
    fake_backend("cuda")
    fake_backend("rocm", caps=())
    monkeypatch.setenv("KITTEN_TTS2_BACKEND", "rocm")
    assert _backend.find().name == "rocm"
    monkeypatch.setenv("KITTEN_TTS2_BACKEND", "CUDA")
    assert _backend.find().name == "cuda"
    assert _backend.find("rocm").name == "rocm"  # the argument wins over the environment
    assert _backend.find("rocm").runtime_found is None  # asked for: nothing probed


def test_both_installed_prefers_present_runtime(no_backends, fake_backend, monkeypatch):
    fake_backend("cuda")
    fake_backend("rocm", caps=())
    _runtimes(monkeypatch, "cuda", "rocm")
    assert (_backend.find().name, _backend.find().runtime_found) == ("cuda", True)
    _runtimes(monkeypatch, "rocm")
    assert (_backend.find().name, _backend.find().runtime_found) == ("rocm", True)
    # Neither: the NVIDIA engine, marked, so that the messages name both vendors.
    _runtimes(monkeypatch)
    assert (_backend.find().name, _backend.find().runtime_found) == ("cuda", False)


@pytest.mark.skipif(sys.platform == "win32", reason="platform: the AMD engine is Linux only")
def test_rocm_found_in_its_own_folder(no_backends, fake_backend, monkeypatch, tmp_path):
    # ROCm installed without AMD's ld.so.conf step: libamdhip64.so.7 loads only by its path in the ROCm folder.
    fake_backend("cuda")
    fake_backend("rocm", caps=())
    root = tmp_path / "rocm-7.2.0"
    monkeypatch.setenv("ROCM_PATH", str(root))
    wanted = os.path.join(str(root), "lib", _gpu.HIP_RUNTIME)

    def load(name):
        if name != wanted:
            raise OSError(f"{name}: cannot open shared object file: No such file or directory")
        return object()
    monkeypatch.setattr(_gpu, "_cdll", load)
    assert _gpu.runtime("rocm") == wanted
    assert (_backend.find().name, _backend.find().runtime_found) == ("rocm", True)


@pytest.mark.skipif(sys.platform == "win32", reason="platform: the AMD engine is Linux only")
def test_rocm_folders(monkeypatch):
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm")
    monkeypatch.setattr(_gpu.glob, "glob", lambda pattern: ["/opt/rocm-7.1.0", "/opt/rocm-7.10.1", "/opt/rocm-7.2.0"])
    assert _gpu.hip_candidates() == ["libamdhip64.so.7", "/opt/rocm/lib/libamdhip64.so.7",
                                     "/opt/rocm-7.10.1/lib/libamdhip64.so.7", "/opt/rocm-7.2.0/lib/libamdhip64.so.7",
                                     "/opt/rocm-7.1.0/lib/libamdhip64.so.7"]


def test_eager_loading_is_asked_for_before_the_probe(no_backends, fake_backend, monkeypatch):
    fake_backend("cuda")
    fake_backend("rocm", caps=())
    environ = {}
    monkeypatch.setattr(os, "environ", environ)
    seen = []
    monkeypatch.setattr(_backend, "_driver_present", lambda name: seen.append(dict(environ)) or False)
    assert _backend.find().name == "cuda"
    assert seen[0] == {"CUDA_MODULE_LOADING": "EAGER"}


class _NoDevice:
    """An NVIDIA driver library that sees no GPU."""

    def cuInit(self, flags):
        return _gpu.CUDA_ERROR_NO_DEVICE

    def cuDriverGetVersion(self, out):
        return 0


def _start_failure(monkeypatch, rocm=lambda device: 1):
    # The real check of the NVIDIA driver: its error also gives advice, the AMD route included (the HIP runtime loads).
    real = _gpu.check_cuda

    def load(name):
        return object() if os.path.basename(name) == _gpu.HIP_RUNTIME else _NoDevice()
    monkeypatch.setattr(_gpu, "check_cuda", lambda device, log=None: real(device, log, load=load))
    monkeypatch.setattr(_gpu, "check_rocm", rocm)


def test_nvidia_picked_by_itself_falls_back_to_amd(no_backends, fake_backend, monkeypatch):
    # An AMD machine with a leftover NVIDIA driver library: the NVIDIA engine is picked first and cannot start.
    fake_backend("cuda")
    fake_backend("rocm", caps=())
    _runtimes(monkeypatch, "cuda", "rocm")
    _start_failure(monkeypatch)
    notes = []
    picked = _backend.find()
    assert picked.name == "cuda"
    used = _backend.check_gpu(picked, 0, log=notes.append)
    assert (used.name, used.label, used.runtime_found) == ("rocm", "AMD (ROCm) engine", True)
    # What went wrong, without the NVIDIA error's advice (which already names the AMD route).
    assert notes == ["the NVIDIA driver sees no GPU (CUDA_ERROR_NO_DEVICE); using the AMD (ROCm) engine instead"]


def test_no_fallback_when_asked_for_or_without_rocm(no_backends, fake_backend, monkeypatch):
    fake_backend("cuda")
    fake_backend("rocm", caps=())
    _runtimes(monkeypatch, "cuda", "rocm")
    _start_failure(monkeypatch)
    with pytest.raises(_gpu.DriverStartError, match="check that the GPU is attached"):  # what the user asked for
        _backend.check_gpu(_backend.find("cuda"), 0)
    _runtimes(monkeypatch, "cuda")  # no ROCm runtime
    with pytest.raises(_gpu.DriverStartError):
        _backend.check_gpu(_backend.find(), 0)


def test_other_gpu_errors_are_not_retried(no_backends, fake_backend, monkeypatch):
    fake_backend("cuda")
    fake_backend("rocm", caps=())
    _runtimes(monkeypatch, "cuda", "rocm")

    def device_out_of_range(device, log=None):
        raise _gpu.GPUError("there is no NVIDIA GPU 3: the driver sees 1 (device= counts from 0)")
    monkeypatch.setattr(_gpu, "check_cuda", device_out_of_range)
    with pytest.raises(_gpu.GPUError, match="no NVIDIA GPU 3"):
        _backend.check_gpu(_backend.find(), 3)


def test_fallback_that_fails_too_names_both(no_backends, fake_backend, monkeypatch):
    fake_backend("cuda")
    fake_backend("rocm", caps=())
    _runtimes(monkeypatch, "cuda", "rocm")

    def no_amd_gpu(device):
        raise _gpu.GPUError("ROCm sees no AMD GPU")
    _start_failure(monkeypatch, rocm=no_amd_gpu)
    with pytest.raises(_gpu.GPUError) as e:
        _backend.check_gpu(_backend.find(), 0)
    assert str(e.value) == ("the NVIDIA driver sees no GPU (CUDA_ERROR_NO_DEVICE); the AMD (ROCm) engine cannot start "
                            "either: ROCm sees no AMD GPU")


def test_unknown_backend_name(no_backends):
    with pytest.raises(ValueError, match="'cuda'.*'rocm'"):
        _backend.find("metal")


def test_abi_mismatch(no_backends, fake_backend):
    fake_backend("cuda", abi=3)
    with pytest.raises(_backend.EngineMismatch) as e:
        _backend.find()
    assert str(e.value) == ("the NVIDIA (CUDA) engine in this install has engine ABI 3, but this kitten-inference "
                            f"needs ABI 2: reinstall it in a fresh environment: {REINSTALL}")


def test_library_abi_check(tmp_path):
    from kitten_inference.tts2 import _abi
    with pytest.raises(_abi.LibraryError, match="not found"):
        _abi.load(tmp_path / "libkt2n_engine.so")
    bogus = tmp_path / "bogus.so"
    bogus.write_bytes(b"not a library")
    with pytest.raises(_abi.LibraryError, match="cannot load"):
        _abi.load(bogus)
