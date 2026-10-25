"""The GPU check that runs before the first download: no driver, no visible GPU, a CUDA 11 driver, a bad device
number (fake driver libraries), and the real answer on a machine without an NVIDIA driver (CI runners)."""
import ctypes
import os
import types

import pytest

from kitten_inference.tts2 import _gpu


class FakeCuda:
    def __init__(self, init=0, version=12080, count=1):
        self.init, self.version, self.count = init, version, count

    def cuInit(self, flags):
        return self.init

    def cuDriverGetVersion(self, out):
        out._obj.value = self.version
        return 0

    def cuDeviceGetCount(self, out):
        out._obj.value = self.count
        return 0

    def cuGetErrorName(self, status, out):
        out._obj.value = b"CUDA_ERROR_UNKNOWN"
        return 0


def _loader(lib):
    def load(name):
        if lib is None:
            raise OSError(f"{name}: cannot open shared object file")
        return lib
    return load


def test_no_driver():
    with pytest.raises(_gpu.GPUError, match="no NVIDIA driver found.*R570"):
        _gpu.check_cuda(0, load=_loader(None))


def test_no_visible_gpu():
    with pytest.raises(_gpu.GPUError, match="sees no GPU"):
        _gpu.check_cuda(0, load=_loader(FakeCuda(init=_gpu.CUDA_ERROR_NO_DEVICE)))


def test_driver_fails_to_start():
    with pytest.raises(_gpu.GPUError, match="cuInit error 999, CUDA_ERROR_UNKNOWN"):
        _gpu.check_cuda(0, load=_loader(FakeCuda(init=999)))


def test_cuda_11_driver_is_refused():
    with pytest.raises(_gpu.GPUError, match="supports CUDA 11.8 only"):
        _gpu.check_cuda(0, load=_loader(FakeCuda(version=11080)))


def test_older_cuda_12_driver_gets_a_note_only():
    notes = []
    assert _gpu.check_cuda(0, notes.append, load=_loader(FakeCuda(version=12040))) == 1
    assert len(notes) == 1 and "CUDA 12.4" in notes[0] and "R570" in notes[0]
    notes.clear()
    assert _gpu.check_cuda(0, notes.append, load=_loader(FakeCuda(version=13000, count=2))) == 2
    assert notes == []


def test_device_number_out_of_range():
    with pytest.raises(_gpu.GPUError, match="no NVIDIA GPU 1: the driver sees 1"):
        _gpu.check_cuda(1, load=_loader(FakeCuda()))


def test_unknown_library_lets_the_engine_decide():
    assert _gpu.check_cuda(0, load=_loader(types.SimpleNamespace())) == -1


class FakeHip:
    def __init__(self, status=0, count=1):
        self.status, self.count = status, count

    def hipGetDeviceCount(self, out):
        out._obj.value = self.count
        return self.status


def test_rocm():
    assert _gpu.check_rocm(0, load=_loader(FakeHip(count=2))) == 2
    with pytest.raises(_gpu.GPUError, match="ROCm sees no AMD GPU.*/dev/kfd"):
        _gpu.check_rocm(0, load=_loader(FakeHip(status=_gpu.HIP_ERROR_NO_DEVICE, count=0)))
    with pytest.raises(_gpu.GPUError, match="ROCm sees no AMD GPU"):
        _gpu.check_rocm(0, load=_loader(FakeHip(count=0)))
    with pytest.raises(_gpu.GPUError, match="no AMD GPU 3"):
        _gpu.check_rocm(3, load=_loader(FakeHip(count=1)))
    assert _gpu.check_rocm(0, load=_loader(None)) == -1  # the engine library reports a missing ROCm itself
    assert _gpu.check_rocm(0, load=_loader(FakeHip(status=1))) == -1


@pytest.mark.skipif(os.environ.get("KT2N_EXPECT_NO_NVIDIA_DRIVER") != "1",
                    reason="set KT2N_EXPECT_NO_NVIDIA_DRIVER=1 on a machine without the NVIDIA driver")
def test_this_machine_has_no_nvidia_driver():
    # CI runners: the real lookup of libcuda.so.1 / nvcuda.dll fails, before any download.
    with pytest.raises(_gpu.GPUError, match="no NVIDIA driver found"):
        _gpu.check("cuda", 0)


def test_kittentts2_checks_the_gpu_before_downloading(no_backends, fake_backend, monkeypatch):
    from kitten_inference.tts2 import _abi, _api, _prepare
    fake_backend("cuda")
    monkeypatch.setattr(_abi, "load", lambda path: types.SimpleNamespace())
    prepared = []
    monkeypatch.setattr(_prepare, "ensure", lambda **kw: prepared.append(kw))
    monkeypatch.setattr(_gpu, "check_cuda", lambda device, log=None: (_ for _ in ()).throw(
        _gpu.GPUError(f"no NVIDIA driver found (device {device})")))
    with pytest.raises(_gpu.GPUError, match="device 2"):
        _api.KittenTTS2(device=2)
    assert prepared == []  # nothing was downloaded
