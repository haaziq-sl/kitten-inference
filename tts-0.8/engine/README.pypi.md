# Kitten Inference

`kitten-inference` is the native runtime used by KittenTTS. It provides the
compiled `model_inference` Python extension that runs KittenTTS neural-network
graphs with C++ kernels.

Most users should install KittenTTS instead of installing this package directly:

```bash
pip install kittentts
```

KittenTTS depends on `kitten-inference`, so pip selects the native wheel that
matches your Python version, operating system, and CPU architecture.

## Direct Install

Direct installation is useful for runtime smoke tests and debugging:

```bash
pip install kitten-inference
```

```python
import model_inference as mi

assert mi.set_backend("cpu")
print(mi.active_backend())
```

## Wheel Coverage

The release publishes platform-specific wheels for the KittenTTS native runtime.

| Platform | Python |
|---|---|
| Linux x86_64, glibc 2.17+ and 2.28+ | CPython 3.8-3.14 |
| Linux ARM64 / aarch64, glibc 2.17+ and 2.28+ | CPython 3.8-3.14 |
| Windows x86_64 | CPython 3.8-3.14 |
| Windows ARM64 | CPython 3.9-3.14 |
| macOS Apple Silicon / ARM64 | CPython 3.8-3.14 |
| macOS Intel / x86_64 | CPython 3.8-3.14 |
| Android ARM64 / Termux | CPython 3.13 experimental |

Android is currently experimental and non-blocking for releases. The desktop
and Linux wheels are the primary supported install targets for KittenTTS users.

## Notes

- This package is not a standalone text-to-speech API. Use `kittentts` for the
  high-level `KittenTTS(...).generate(...)` interface.
- KittenTTS speed control is backed by native duration scaling exposed through
  `InferenceModel.set_speed(...)`.
- The import name is `model_inference`.
- ARM64 wheels are built for modern ARMv8.2 dot-product-capable cores.
- CUDA wheels are not part of the public wheel matrix yet.
