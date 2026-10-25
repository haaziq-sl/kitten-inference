"""GPU integration tests of kitten_inference.tts2 (run: python -m pytest tests/gpu).

They need an installed engine package (kitten-inference-tts2-cu12 / -rocm, or a development wheel from
make_dev_backend.py) and a GPU; otherwise they are skipped. Environment:
  KT2N_TEST_REVISION   model revision to prepare (default: the program's pinned revision below)
  KT2N_TEST_CACHE      cache directory for prepared models (default: the package default)
  KT2N_TEST_CLONING=1  also run voice cloning (CUDA engine and about 11 GB of free GPU memory)
  KT2N_TEST_MEMORY_FREE=1  this process is alone on the GPU (the Ctrl-C test then also checks the memory)
  KT2N_REQUIRE_GPU=1   fail instead of skipping when no engine or GPU is available
"""
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

REVISION = os.environ.get("KT2N_TEST_REVISION", "c93cb7c271baa51d9ae8204d5277988906f235a2")
CACHE = os.environ.get("KT2N_TEST_CACHE") or None
TEXT = "The ferry left the harbour a little after dawn. It costs 25 dollars and 50 cents."
SEED = 7


def _skip(reason):
    if os.environ.get("KT2N_REQUIRE_GPU") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def backend():
    from kitten_inference.tts2 import _backend
    try:
        return _backend.find()
    except ImportError as exc:
        _skip(f"no engine installed: {exc}")


@pytest.fixture(scope="session")
def tts(backend):
    from kitten_inference.tts2 import KittenTTS2
    t0 = time.monotonic()
    t = KittenTTS2(revision=REVISION, cache_dir=CACHE)
    print(f"\nKittenTTS2() ready in {time.monotonic() - t0:.1f} s: {t.model_dir}", file=sys.stderr)
    yield t
    t.close()


def _wav_samples(path):
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
        return np.frombuffer(w.readframes(w.getnframes()), "<i2")


def _sane(audio, text=TEXT):
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert np.all(np.isfinite(audio))
    seconds = audio.size / 24000
    assert 0.25 * len(text.split()) / 3 < seconds < 3 * len(text.split()), seconds  # ~3 words/s, wide margins
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert 0.01 < rms < 0.5, rms
    return seconds


def test_list_voices(tts):
    voices = tts.list_voices()
    assert "Bruno" in voices and len(voices) >= 40 and len(set(voices)) == len(voices)


def test_generate_complete_is_seeded_and_matches_the_cli(tts, tmp_path):
    a = tts.generate(TEXT, voice="Bruno", seed=SEED)
    _sane(a)
    b = tts.generate(TEXT, voice="Bruno", seed=SEED)
    assert np.array_equal(a, b), "the same seed gave different audio with stream=False"
    path = tts.save_wav(tmp_path / "py.wav", a)
    cli = shutil.which("kitten-tts2")
    assert cli, "kitten-tts2 is not on PATH"
    out = tmp_path / "cli.wav"
    run = subprocess.run([cli, "--revision", REVISION, *(["--cache-dir", CACHE] if CACHE else []), "--text", TEXT,
                          "--voice", "Bruno", "--seed", str(SEED), "--stream", "false", "--output", str(out)],
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    assert np.array_equal(_wav_samples(path), _wav_samples(out)), "Python and kitten-tts2 audio differ"
    report = json.loads([l for l in run.stderr.splitlines() if l.startswith("{")][-1])
    assert report["engine"]["output"] == "complete"


def test_unseeded_requests_differ(tts):
    """Without a seed each request samples on its own (a "try again" gives another take); with one it repeats."""
    a = tts.generate("Try this sentence again.", voice="Bruno")
    b = tts.generate("Try this sentence again.", voice="Bruno")
    assert not (a.size == b.size and np.array_equal(a, b)), "two unseeded requests gave the same audio"
    c = tts.generate("Try this sentence again.", voice="Bruno", seed=21)
    d = tts.generate("Try this sentence again.", voice="Bruno", seed=21)
    assert np.array_equal(c, d)


def test_revision_is_the_commit(tts):
    assert tts.revision == REVISION and tts.model_dir.name == REVISION


def test_generate_other_voice_and_options(tts):
    voices = tts.list_voices()
    other = next(v for v in voices if v != "Bruno")
    a = tts.generate("Hello there.", voice=other, seed=1, preset="stable", temperature=0.7, top_k=40)
    _sane(a, "Hello there.")
    b = tts.generate("Hello there.", voice="Bruno", seed=1, preset="stable", temperature=0.7, top_k=40)
    assert not np.array_equal(a[: min(a.size, b.size)], b[: min(a.size, b.size)])


def test_stream(tts):
    t0 = time.monotonic()
    first = None
    chunks = []
    for chunk in tts.stream(TEXT, voice="Bruno", seed=SEED):
        if first is None:
            first = time.monotonic() - t0
        assert chunk.dtype == np.float32 and chunk.ndim == 1 and chunk.size > 0
        chunks.append(chunk)
    total = time.monotonic() - t0
    audio = np.concatenate(chunks)
    seconds = _sane(audio)
    print(f"\nstream: {len(chunks)} chunks, first audio {first * 1000:.0f} ms, {seconds:.2f} s of audio in "
          f"{total:.2f} s", file=sys.stderr)
    assert len(chunks) >= 2 and first < total
    whole = tts.generate(TEXT, voice="Bruno", seed=SEED, stream=True)
    _sane(whole)


def test_stream_closed_early_cancels(tts):
    it = tts.stream(TEXT * 4, voice="Bruno")
    next(it)
    it.close()
    a = tts.generate("Still working.", voice="Bruno", seed=3)  # the engine is free again
    _sane(a, "Still working.")


def test_errors(tts):
    with pytest.raises(ValueError, match="unknown voice"):
        tts.generate("x", voice="NoSuchVoice")
    with pytest.raises(ValueError):
        tts.generate("   ", voice="Bruno")
    with pytest.raises(ValueError):
        tts.generate("x", voice="Bruno", temperature=9.0)
    with pytest.raises(TypeError):
        tts.generate("x", voice="Bruno", temprature=0.5)


def test_chunks_reach_a_stream_while_another_thread_waits(tts, monkeypatch):
    """Thread B's generate() waits behind stream A (one stream at a time) and polls the engine meanwhile. Each of
    A's chunks must reach A as soon as it is polled, whichever thread polled it (engine event time -> delivery)."""
    import threading
    from kitten_inference.tts2 import _engine
    lags = []
    real_events = _engine.Engine.events

    def events(self, ident):
        for ev in real_events(self, ident):
            if ev.kind == 1 and ev.at_ns:  # audio
                lags.append((ident, (time.monotonic_ns() - ev.at_ns) / 1e6))
            yield ev

    monkeypatch.setattr(_engine.Engine, "events", events)
    for _ in range(3):
        it = tts.stream(TEXT * 2, voice="Bruno", seed=SEED)
        next(it)
        a = lags[-1][0]
        waiter = threading.Thread(target=lambda: tts.generate("A second request.", voice="Bruno", seed=3))
        waiter.start()
        for _ in it:
            pass
        waiter.join(60)
    mine = sorted(lag for ident, lag in lags if ident == a)
    print(f"\nchunk hand-off with a waiting thread: median {mine[len(mine) // 2]:.2f} ms, max {mine[-1]:.2f} ms "
          f"({len(mine)} chunks)", file=sys.stderr)
    assert mine[-1] < 25, mine  # 0.1 ms in practice; the earlier pump held chunks for 160 ms (median)


def test_gpu_check_sees_this_gpu():
    from kitten_inference.tts2 import _gpu
    assert _gpu.check("cuda", 0) >= 1
    with pytest.raises(_gpu.GPUError, match="no NVIDIA GPU 7"):
        _gpu.check("cuda", 7)
    code = "from kitten_inference.tts2 import _gpu; _gpu.check('cuda', 0)"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert run.returncode == 1 and "sees no GPU" in run.stderr, run.stderr[-500:]


def test_ctrl_c_during_start_frees_the_engine(tmp_path):
    """A real interrupt (as Ctrl-C) while kt2n_engine_create runs: no engine may stay behind, and the next
    KittenTTS2() in the same process starts normally."""
    script = Path(__file__).with_name("ctrl_c_start.py")
    out = tmp_path / "r.json"
    run = subprocess.run([sys.executable, str(script), REVISION, str(CACHE or ""), str(out)], capture_output=True,
                         text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    r = json.loads(out.read_text())
    print(f"\nCtrl-C during start: {r}", file=sys.stderr)
    assert r["interrupted"] and r["interrupt_during_create"], r
    assert r["next_started"], r
    # The engine create made was destroyed: the memory is back where close() left it (other processes aside).
    assert r["after_interrupted_start_mib"] - r["after_close_mib"] < 512, r


def test_save_wav_list_of_chunks(tts, tmp_path):
    chunks = list(tts.stream("Two short chunks.", voice="Bruno"))
    path = tts.save_wav(tmp_path / "s.wav", chunks)
    assert _wav_samples(path).size == sum(c.size for c in chunks)


def test_cli_list_voices_and_prepare_only(tts):
    cli = shutil.which("kitten-tts2")
    cache = ["--cache-dir", CACHE] if CACHE else []
    run = subprocess.run([cli, "--prepare-only", "--revision", REVISION, *cache], capture_output=True, text=True)
    assert run.returncode == 0 and Path(run.stdout.strip()) == tts.model_dir, run.stderr
    run = subprocess.run([cli, "--revision", REVISION, *cache, "--list-voices"], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    names = [line.split("\t")[0] for line in run.stdout.splitlines()]
    assert names == tts.list_voices()


@pytest.fixture(scope="session")
def reference(tmp_path_factory, tts):
    """A 6-8 s reference recording (a built-in voice, saved as MP3 through ffmpeg) and its transcript."""
    if shutil.which("ffmpeg") is None:
        _skip("ffmpeg is not installed")
    text = "My grandmother kept a small garden behind the house, full of tomatoes, beans and sunflowers."
    audio = tts.generate(text, voice="Bruno", seed=11)
    work = tmp_path_factory.mktemp("clone")
    wav = tts.save_wav(work / "reference.wav", audio)
    mp3 = work / "reference.mp3"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav), "-ac", "2", "-ar", "44100",
                    str(mp3)], check=True)
    return mp3, text


@pytest.mark.skipif(os.environ.get("KT2N_TEST_CLONING") != "1", reason="set KT2N_TEST_CLONING=1 (about 11 GB of free GPU memory)")
def test_clone_save_load(tts, reference, tmp_path):
    if not tts.can_clone:
        _skip("this engine has no voice cloning")
    mp3, transcript = reference
    t0 = time.monotonic()
    voice = tts.clone_voice(mp3, transcript)
    print(f"\nclone_voice: {time.monotonic() - t0:.1f} s (includes the engine restart), {len(voice.artifact)} bytes",
          file=sys.stderr)
    a = tts.generate("This is my cloned voice speaking.", voice=voice, seed=5)
    _sane(a, "This is my cloned voice speaking.")
    saved = tts.save_voice(voice, tmp_path / "me.voice")
    loaded = tts.load_voice(saved)
    assert loaded == voice
    b = tts.generate("This is my cloned voice speaking.", voice=loaded, seed=5)
    assert np.array_equal(a, b), "a saved and loaded voice gave other audio than the cloned one"
    # The saved voice names the prepared model it belongs to, as the engine checks it.
    from kitten_inference.tts2 import _bundle
    assert _bundle.identity(_bundle.blob_meta(voice.artifact)) == tts._model_identity(tts.model_dir)
    # A voice of another prepared model is refused before the engine is touched, with the fix in the message.
    meta = _bundle.blob_meta(voice.artifact)
    other = voice.artifact.replace(meta["checkpoint"].encode(), ("0" * len(meta["checkpoint"])).encode(), 1)
    (tmp_path / "other.voice").write_bytes(other)
    with pytest.raises(ValueError, match=r"works only with the model revision it was made with"):
        tts.load_voice(tmp_path / "other.voice")
    from kitten_inference.tts2 import Voice, EngineError
    try:  # what the engine itself says (bypassing the check above)
        tts._engine.voice_load(Voice(other).artifact)
        engine_answer = "loaded"
    except EngineError as exc:
        engine_answer = f"status {exc.status}: {exc.message}"
    print(f"\nengine on a voice of another model: {engine_answer}", file=sys.stderr)
    assert engine_answer.startswith(("status 210", "status 213")), engine_answer
    c = tts.generate("Back to a built-in voice.", voice="Bruno", seed=5)  # built-in voices still work
    _sane(c, "Back to a built-in voice.")


@pytest.mark.skipif(os.environ.get("KT2N_TEST_CLONING") != "1", reason="set KT2N_TEST_CLONING=1 (about 11 GB of free GPU memory)")
def test_loaded_voice_in_a_new_process(tts, reference, tmp_path):
    """A voice saved here loads and speaks in a fresh KittenTTS2 that never cloned (saved-voice engine)."""
    if not tts.can_clone:
        _skip("this engine has no voice cloning")
    mp3, transcript = reference
    voice = tts.clone_voice(mp3, transcript)
    path = tts.save_voice(voice, tmp_path / "v.voice")
    expected = tts.generate("A voice from a file.", voice=voice, seed=9)
    code = (f"import numpy as np; from kitten_inference.tts2 import KittenTTS2; "
            f"t = KittenTTS2(revision={REVISION!r}, cache_dir={CACHE!r}); v = t.load_voice({str(path)!r}); "
            f"a = t.generate('A voice from a file.', voice=v, seed=9); np.save({str(tmp_path / 'a.npy')!r}, a)")
    tts.close()  # the new process has the GPU to itself (on a 12 GB card two engines do not fit)
    try:
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    finally:
        tts.__init__(revision=REVISION, cache_dir=CACHE)  # the shared object, started again for the next tests
    assert run.returncode == 0, run.stderr[-2000:]
    assert np.array_equal(np.load(tmp_path / "a.npy"), expected)


def test_ffmpeg_missing_message(tts, reference, monkeypatch):
    if not tts.can_clone:
        _skip("this engine has no voice cloning")
    from kitten_inference.tts2 import FFmpegNotFound
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(FFmpegNotFound, match="ffmpeg was not found"):
        tts.clone_voice(reference[0], reference[1])


@pytest.mark.skipif(os.environ.get("KT2N_TEST_CLONING") != "1", reason="set KT2N_TEST_CLONING=1 (about 11 GB of free GPU memory)")
def test_two_custom_voices(tts, reference, tmp_path):
    """The engine keeps one custom voice: switching evicts the previous one, except while a stream still uses it."""
    if not tts.can_clone:
        _skip("this engine has no voice cloning")
    mp3, transcript = reference
    v1 = tts.clone_voice(mp3, transcript)
    other = next(v for v in tts.list_voices() if v != "Bruno")
    wav = tts.save_wav(tmp_path / "ref2.wav", tts.generate(transcript, voice=other, seed=12))
    v2 = tts.clone_voice(wav, transcript)
    assert v1 != v2
    a1 = tts.generate("First voice.", voice=v1, seed=4)
    s = tts.stream(TEXT * 3, voice=v2)
    next(s)  # v2 is in use by a suspended stream
    with pytest.raises(RuntimeError, match="still in use"):
        tts.generate("First voice.", voice=v1, seed=4)
    s.close()
    assert np.array_equal(tts.generate("First voice.", voice=v1, seed=4), a1)  # reloaded: same audio
    _sane(tts.generate("Second voice.", voice=v2, seed=4), "Second voice.")


def test_zz_close_while_streaming(tts):
    """close() from another thread while a stream waits for audio: the stream ends with an error, nothing crashes.
    Runs last: it closes the shared engine."""
    import threading
    it = tts.stream(TEXT * 6, voice="Bruno")
    next(it)
    closer = threading.Thread(target=tts.close)
    closer.start()
    with pytest.raises(RuntimeError, match="closed"):
        for _ in it:
            pass
    closer.join(30)
    assert not closer.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        tts.generate("x", voice="Bruno")
