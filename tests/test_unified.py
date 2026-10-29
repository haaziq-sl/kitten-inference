"""Contract tests for the public interface and model dispatch."""
from pathlib import Path
from types import SimpleNamespace
import importlib
import numpy as np
import pytest
from kitten_inference import KittenTTS
from kitten_inference import _cli


@pytest.fixture(params=['tts08', 'tts2'])
def model(request, monkeypatch):
    calls = []
    class Fake:
        sample_rate = 22050 if request.param == 'tts08' else 24000
        model = SimpleNamespace(sample_rate=sample_rate)
        available_voices = ['Bella']
        def __init__(self, *args, **kwargs):
            calls.append(('init', args, kwargs))
        def generate(self, text, **kwargs):
            calls.append(('generate', text, kwargs))
            return np.zeros(12, np.float32)
        def stream(self, text, **kwargs):
            calls.append(('stream', text, kwargs))
            try:
                yield np.zeros(3, np.float32)
                yield np.ones(2, np.float32)
            finally:
                calls.append(('cancel',))
        generate_stream = stream
        def list_voices(self): return ['Bruno']
        def can_clone(self): return True
        def close(self): calls.append(('close',))
        def clone_voice(self, *args, **kwargs):
            calls.append(('clone', args, kwargs))
            return 'voice'
    if request.param == 'tts08':
        module = importlib.import_module('kitten_inference.tts08')
        monkeypatch.setattr(module, 'KittenTTS', Fake)
        name = 'KittenML/kitten-tts-nano-0.8'
    else:
        module = importlib.import_module('kitten_inference.tts2')
        # Avoid invoking the module's dependency/platform checks in a contract test.
        monkeypatch.setitem(vars(module), 'KittenTTS2', Fake)
        name = 'KittenML/kitten-tts-2'
    return KittenTTS(name), calls


def test_explicit_model():
    with pytest.raises(TypeError): KittenTTS()
    with pytest.raises(ValueError): KittenTTS('')


def test_audio_options_and_default_voice(model):
    tts, calls = model
    audio = tts.generate('hello', speed=1.2, normalize=False)
    assert audio.dtype == np.float32 and audio.ndim == 1
    options = calls[-1][2]
    assert options['speed'] == 1.2 and 'voice' not in options
    assert options['clean_text' if tts.family == 'tts08' else 'normalize'] is False
    assert tts.list_voices() == (['Bella'] if tts.family == 'tts08' else ['Bruno'])


def test_file_uses_model_sample_rate(model, tmp_path):
    import soundfile as sf
    tts, calls = model
    path = tts.generate_to_file('hello', tmp_path / 'speech.wav')
    assert sf.info(path).samplerate == tts.sample_rate
    if tts.family == 'tts08': assert calls[-1][2]['clean_text'] is True
    tts.save_wav(tmp_path / 'chunks.wav', [np.zeros(3), np.zeros(4)])
    assert sf.info(tmp_path / 'chunks.wav').frames == 7


def test_stream_cancellation_and_close(model):
    tts, calls = model
    stream = tts.stream('hello')
    assert next(stream).shape == (3,)
    stream.close()
    assert calls[-1] == ('cancel',)
    tts.close(); tts.close()
    with pytest.raises(RuntimeError, match='closed'): tts.generate('hello')
    assert calls.count(('close',)) == (1 if tts.family == 'tts2' else 0)


def test_capabilities_and_model_specific_operations(model):
    tts, calls = model
    assert tts.capabilities['streaming']
    if tts.family == 'tts08':
        assert not tts.can_clone()
        with pytest.raises(NotImplementedError): tts.clone_voice('ref.wav', 'text')
        with pytest.raises(TypeError, match='seed'): tts.generate('hello', seed=3)
    else:
        assert tts.can_clone()
        assert tts.clone_voice('ref.wav', 'text', language='en') == 'voice'
        assert calls[-1] == ('clone', ('ref.wav', 'text'), {'language': 'en'})


def test_cli_preserves_advanced_gpu_arguments(monkeypatch):
    from kitten_inference.tts2 import _cli as gpu
    seen = []
    monkeypatch.setattr(gpu, 'main', lambda args: seen.extend(args) or 7)
    args = ['--reference-audio', 'a file.wav', '--voice-cache-mib', '300', '--prepare-only', '--with-cloning']
    assert _cli.main(['--model', 'kitten-tts-2'] + args) == 7
    assert seen == args


def test_cli_help_needs_no_model_or_engine(capsys):
    assert _cli.main(['--help']) == 0
    assert '--model' in capsys.readouterr().out


def test_dependency_markers():
    from importlib.metadata import requires
    from packaging.requirements import Requirement
    reqs = list(map(Requirement, requires('kitten-inference')))
    for version, platform, machine, gpu in [('3.8','linux','x86_64',False), ('3.12','linux','x86_64',True),
        ('3.12','win32','AMD64',True), ('3.12','darwin','arm64',False), ('3.12','linux','aarch64',False)]:
        env = dict(python_version=version, sys_platform=platform, platform_machine=machine, extra='')
        selected = {r.name.replace('_','-') for r in reqs if r.marker is None or r.marker.evaluate(env)}
        assert {'numpy','huggingface-hub','soundfile','phonemizer','espeakng-loader'} <= selected
        assert ('tokenizers' in selected) == gpu
        assert ('filelock' in selected) == gpu


def test_selected_model_version_reaches_native_command(monkeypatch):
    from kitten_inference.tts2 import _cli as gpu
    seen = []
    monkeypatch.setattr(gpu, 'main', lambda args: seen.extend(args) or 0)
    assert _cli.main(['--model', 'kitten-tts-2', '--version']) == 0
    assert seen == ['--version']
