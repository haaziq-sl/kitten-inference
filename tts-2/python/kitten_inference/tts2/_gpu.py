"""A quick look at the GPU driver before the first download: no driver, no visible GPU or a CUDA 11 driver is
reported at once, instead of after minutes of downloading and preparing the model.

It asks the driver directly (libcuda / nvcuda.dll for NVIDIA, the HIP runtime for AMD), as the engine does when it
starts. It never decides for the engine in unclear cases: an answer it does not understand lets the engine start and
report for itself. The Linux x86-64 install has both engines, so its messages also name the AMD route.
"""
from __future__ import annotations

import ctypes
import glob
import os
import re
import sys
from typing import Callable, List, Optional

CUDA_ERROR_NO_DEVICE = 100
HIP_ERROR_NO_DEVICE = 100
CUDA_12 = 12000      # cuDriverGetVersion: a CUDA 12.8 program needs a CUDA 12 driver at least
CUDA_12_8 = 12080    # R570: the driver the engine is built and documented for
CUDA_DRIVER = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
HIP_RUNTIME = "libamdhip64.so.7"
NEED_DRIVER = "Kitten TTS 2 needs an NVIDIA GPU with the NVIDIA driver R570 or newer"
ROCM_DOCS = "https://github.com/KittenML/kitten-inference/blob/main/tts-2/docs/amd-radeon.md"
USE_ROCM = "--backend rocm or KITTEN_TTS2_BACKEND=rocm"


class GPUError(RuntimeError):
    """This machine has no usable GPU for the installed engine (no driver, no GPU visible, or a too old driver)."""


class DriverStartError(GPUError):
    """The NVIDIA driver is missing or cannot start (no driver, no GPU, a cuInit error): the cases in which an
    automatic choice of the NVIDIA engine is retried with the AMD one (_backend.check_gpu). ``reason`` is what went
    wrong without the advice, for the messages of that retry."""

    def __init__(self, message: str, reason: Optional[str] = None):
        super().__init__(message)
        self.reason = message if reason is None else reason


def _cdll(name: str):
    return ctypes.CDLL(name)


def _version_key(path: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", os.path.basename(path)))


def hip_candidates() -> List[str]:
    """Where the HIP runtime is looked for: the loader's own search, then (Linux) $ROCM_PATH/lib, /opt/rocm/lib and
    /opt/rocm-7*/lib (newest first), where ROCm 7 installs it. Without AMD's ld.so.conf step it is only there; the
    engine library itself also looks in /opt/rocm-7.1.0/lib (its RUNPATH)."""
    found = [HIP_RUNTIME]
    if sys.platform == "win32":
        return found
    roots = [os.environ.get("ROCM_PATH", ""), "/opt/rocm",
             *sorted(glob.glob("/opt/rocm-7*"), key=_version_key, reverse=True)]
    for root in roots:
        path = os.path.join(root, "lib", HIP_RUNTIME) if root else ""
        if path and path not in found:
            found.append(path)
    return found


def _open_runtime(name: str, load=None):
    """(what loaded, the library) for the GPU runtime of engine ``name``, or None when this machine does not have it.
    A HIP runtime loaded from a ROCm folder stays loaded, so the engine library loaded later in this process finds it
    by name."""
    load = load or _cdll
    for candidate in [CUDA_DRIVER] if name == "cuda" else hip_candidates():
        try:
            return candidate, load(candidate)
        except OSError:
            continue
    return None


def runtime(name: str, load=None) -> Optional[str]:
    """The GPU runtime library the engine ``name`` ("cuda" or "rocm") needs, as found on this machine (its name, or
    its path in a ROCm folder), or None."""
    found = _open_runtime(name, load)
    return found[0] if found else None


def _rocm_engine() -> bool:
    """Whether this install has the AMD engine (the Linux x86-64 wheels)."""
    from . import _backend
    try:
        return _backend._import("rocm") is not None
    except Exception:  # a broken engine package is reported when it is used
        return False


def _amd_hint(load=None) -> str:
    """The AMD route, added to the messages of an NVIDIA driver that is missing or cannot start."""
    if sys.platform == "win32":
        return f" With an AMD Radeon GPU, run Kitten TTS 2 inside WSL2: {ROCM_DOCS}"
    if not _rocm_engine():
        return ""
    if runtime("rocm", load) is not None:
        return f" This machine has the AMD ROCm 7 runtime: {USE_ROCM} uses its AMD GPU."
    return f" For an AMD Radeon RX 7000 or RX 9000, install ROCm 7 ({ROCM_DOCS}) and use {USE_ROCM}."


def _no_driver() -> str:
    return f"no NVIDIA driver found on this machine ({CUDA_DRIVER} is missing)"


def no_driver_message(load=None) -> str:
    """What to do on a machine without the NVIDIA driver."""
    head = f"{_no_driver()}. {NEED_DRIVER}"
    if sys.platform != "win32" and _rocm_engine() and runtime("rocm", load) is None:
        return (f"{head}, or an AMD Radeon RX 7000 or RX 9000 with ROCm 7 ({HIP_RUNTIME} is missing too; setup: "
                f"{ROCM_DOCS}). In a container, attach the GPU (docker run --gpus all, or --device /dev/kfd --device "
                f"/dev/dri for AMD). With both GPUs, {USE_ROCM} picks the AMD one.")
    hint = _amd_hint(load)
    if not hint and sys.platform != "win32":
        hint = " In a container, start it with the GPU attached (docker run --gpus all ...)."
    return f"{head}.{hint}"


def _version_text(v: int) -> str:
    return f"{v // 1000}.{v % 1000 // 10}"


def check_cuda(device: int = 0, log: Optional[Callable[[str], None]] = None, load=None) -> int:
    """Raises GPUError when no CUDA engine can run here; returns the number of visible NVIDIA GPUs."""
    load = load or _cdll
    try:
        cuda = load(CUDA_DRIVER)
    except OSError:
        raise DriverStartError(no_driver_message(load), _no_driver()) from None
    try:
        status = int(cuda.cuInit(0))
        version = ctypes.c_int(0)
        version_status = int(cuda.cuDriverGetVersion(ctypes.byref(version)))
    except AttributeError:  # not the driver library we know: let the engine report
        return -1
    if status == CUDA_ERROR_NO_DEVICE:
        reason = "the NVIDIA driver sees no GPU (CUDA_ERROR_NO_DEVICE)"
        raise DriverStartError(f"{reason}: check that the GPU is attached and visible (CUDA_VISIBLE_DEVICES, docker "
                               f"run --gpus all).{_amd_hint(load)}", reason)
    if status != 0:
        reason = f"the NVIDIA driver could not start (cuInit error {status}{_cuda_name(cuda, status)})"
        raise DriverStartError(f"{reason}. {NEED_DRIVER}.{_amd_hint(load)}", reason)
    if version_status == 0 and 0 < version.value < CUDA_12:
        raise GPUError(f"the NVIDIA driver supports CUDA {_version_text(version.value)} only. {NEED_DRIVER} "
                       "(CUDA 12.8): update the driver")
    if version_status == 0 and version.value < CUDA_12_8 and log is not None:
        log(f"the NVIDIA driver supports CUDA {_version_text(version.value)}; Kitten TTS 2 is built for R570 (CUDA "
            "12.8) or newer and may not start with it: if it fails to start, update the driver")
    count = ctypes.c_int(0)
    if int(cuda.cuDeviceGetCount(ctypes.byref(count))) == 0:
        if count.value <= device:
            raise GPUError(f"there is no NVIDIA GPU {device}: the driver sees {count.value} "
                           f"(device= counts from 0)")
        return count.value
    return -1


def _cuda_name(cuda, status: int) -> str:
    try:
        name = ctypes.c_char_p()
        if int(cuda.cuGetErrorName(status, ctypes.byref(name))) == 0 and name.value:
            return ", " + name.value.decode("ascii", "replace")
    except (AttributeError, OSError):
        pass
    return ""


def check_rocm(device: int = 0, load=None) -> int:
    """Raises GPUError when ROCm sees no AMD GPU; returns the number of visible GPUs (-1 when it cannot tell)."""
    found = _open_runtime("rocm", load)
    if found is None:  # the engine library reports a missing ROCm runtime itself
        return -1
    hip = found[1]
    try:
        count = ctypes.c_int(0)
        status = int(hip.hipGetDeviceCount(ctypes.byref(count)))
    except (OSError, AttributeError):  # not the runtime library we know: let the engine report
        return -1
    if status == HIP_ERROR_NO_DEVICE or (status == 0 and count.value == 0):
        raise GPUError("ROCm sees no AMD GPU: check that this user can open /dev/kfd and /dev/dri (the render and "
                       "video groups; /dev/dxg under WSL2), that HIP_VISIBLE_DEVICES does not hide it, or in a "
                       "container pass --device /dev/kfd --device /dev/dri")
    if status == 0 and count.value <= device:
        raise GPUError(f"there is no AMD GPU {device}: ROCm sees {count.value} (device= counts from 0)")
    return count.value if status == 0 else -1


def _atoi(text: str) -> int:
    """C's atoi(): the leading integer of ``text``, 0 when there is none."""
    m = re.match(r"\s*([+-]?\d+)", text)
    return int(m.group(1)) if m else 0


def request_eager_loading(backend: str) -> None:
    """Asks the driver to load every GPU module when the context is made, as the engine does at create
    (request_eager_module_loading in src/engine/engine.cc): CUDA_MODULE_LOADING=EAGER, and on ROCm also
    HIP_ENABLE_DEFERRED_LOADING=0. The driver reads these once, when this process first initialises it, and the check
    below does that before the engine starts; without them first requests of the process pay lazy module loading
    inside their kernel launches (the engine measured first audio 35 -> 44 ms on an RTX 4090). As in the engine, a
    value already set wins and KT2N_CUDA_EAGER=0 keeps the driver's lazy default."""
    if _atoi(os.environ.get("KT2N_CUDA_EAGER", "1")) == 0:
        return
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    if backend == "rocm":
        os.environ.setdefault("HIP_ENABLE_DEFERRED_LOADING", "0")


def check(backend: str, device: int = 0, log: Optional[Callable[[str], None]] = None) -> int:
    """The check for ``backend`` ("cuda" or "rocm"). It is the first GPU call of the process, so it first asks for
    the engine's eager module loading (request_eager_loading)."""
    request_eager_loading(backend)
    if backend == "rocm":
        return check_rocm(device)
    return check_cuda(device, log)
