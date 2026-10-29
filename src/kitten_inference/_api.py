"""One interface over the native Kitten model families."""
from pathlib import Path


def model_family(model):
    if not isinstance(model, (str, Path)) or not str(model).strip():
        raise ValueError('model must be a model name or local model directory')
    name = str(model).rstrip('/').lower()
    if name in ('kittenml/kitten-tts-2', 'kitten-tts-2'):
        return 'tts2'
    # Existing 0.8 loaders validate native local and remote configurations.
    return 'tts08'


class KittenTTS:
    """Load an explicitly selected model. See docs/api.md for model-specific options."""

    def __init__(self, model, *, backend=None, cache_dir=None, **model_options):
        self.family = model_family(model)
        self._closed = False
        if self.family == 'tts2':
            from .tts2 import KittenTTS2
            self._model = KittenTTS2(backend=backend, cache_dir=cache_dir, **model_options)
        else:
            if model_options:
                raise TypeError('TTS 0.8 does not support: ' + ', '.join(sorted(model_options)))
            from .tts08 import KittenTTS as TTS08
            self._model = TTS08(str(model), backend=backend, cache_dir=cache_dir)

    def _open(self):
        if self._closed:
            raise RuntimeError('this KittenTTS is closed')
        return self._model

    @property
    def sample_rate(self):
        m = self._open()
        return m.sample_rate if self.family == 'tts2' else m.model.sample_rate

    @property
    def capabilities(self):
        return {'streaming': True,
                'streaming_mode': 'incremental' if self.family == 'tts2' else 'text_chunks',
                'voice_cloning': self.can_clone(),
                'normalization_spans': self.family == 'tts08'}

    def can_clone(self):
        m = self._open()
        return m.can_clone() if self.family == 'tts2' else False

    def default_voice(self):
        m = self._open()
        return m.default_voice() if self.family == 'tts2' else 'expr-voice-5-m'

    def list_voices(self):
        m = self._open()
        return m.list_voices() if self.family == 'tts2' else list(m.available_voices)

    def _options(self, options, *, file=False):
        opts = dict(options)
        if self.family == 'tts08':
            unknown = set(opts) - {'speed', 'normalize'}
            if unknown:
                raise TypeError('TTS 0.8 does not support: ' + ', '.join(sorted(unknown)))
            opts['clean_text'] = opts.pop('normalize', file)
        return opts

    def generate(self, text, voice=None, **options):
        m = self._open()
        kwargs = self._options(options)
        if voice is not None:
            kwargs['voice'] = voice
        return m.generate(text, **kwargs)

    def stream(self, text, voice=None, **options):
        m = self._open()
        kwargs = self._options(options)
        if voice is not None:
            kwargs['voice'] = voice
        iterator = (m.stream if self.family == 'tts2' else m.generate_stream)(text, **kwargs)
        try:
            for chunk in iterator:
                self._open()
                yield chunk
        finally:
            iterator.close()

    def save_wav(self, path, audio):
        from ._audio import save_wav
        return save_wav(path, audio, self.sample_rate)

    def generate_to_file(self, text, output_path, voice=None, **options):
        # Preserve 0.8's file-generation normalization default.
        if self.family == 'tts08':
            options.setdefault('normalize', True)
        audio = self.generate(text, voice=voice, **options)
        import soundfile as sf
        sf.write(output_path, audio, self.sample_rate)
        return Path(output_path)

    def _tts2(self):
        m = self._open()
        if self.family != 'tts2':
            raise NotImplementedError('this operation requires Kitten TTS 2')
        return m

    def clone_voice(self, reference_audio_path, transcript, **options):
        return self._tts2().clone_voice(reference_audio_path, transcript, **options)

    def save_voice(self, voice, path):
        return self._tts2().save_voice(voice, path)

    def load_voice(self, path):
        return self._tts2().load_voice(path)

    @property
    def revision(self):
        return self._tts2().revision

    @property
    def model_dir(self):
        return self._tts2().model_dir

    def build_info(self):
        return self._tts2().build_info()

    def stats(self):
        return self._tts2().stats()

    def normalize_text(self, text, locale='en-US', return_spans=False):
        self._open()
        if self.family != 'tts08':
            raise NotImplementedError('standalone normalization is available for TTS 0.8')
        return self._model.normalize_text(text, locale=locale, return_spans=return_spans)

    def close(self):
        if not self._closed:
            self._closed = True
            model = self._model
            self._model = None
            if self.family == 'tts2':
                model.close()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()
