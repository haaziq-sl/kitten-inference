"""Run after the documented installation, without adding test dependencies.

Writes audio and a JSON verdict using only the installed package and its dependencies.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--project', default='kitten-inference')
    parser.add_argument('--version', required=True)
    parser.add_argument('--backend')
    parser.add_argument('--clone', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = {'platform': platform.platform(), 'machine': platform.machine(), 'python': sys.version,
              'project': args.project, 'version': metadata.version(args.project), 'model': args.model,
              'checks': {}, 'status': 'running'}
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        result['checks'][name] = 'pass'
    try:
        import numpy as np
        import soundfile as sf
        from kitten_inference import KittenTTS
        check('installed_version', result['version'] == args.version)
        packages = {d.metadata['Name'].lower().replace('_', '-') for d in metadata.distributions() if d.metadata['Name']}
        check('no_pytorch', 'torch' not in packages)
        text = 'The ferry left the harbour a little after dawn. It costs 25 dollars and 50 cents.'
        with KittenTTS(args.model, backend=args.backend) as tts:
            voices = tts.list_voices()
            check('voices', bool(voices))
            voice = 'Bruno' if tts.family == 'tts2' else 'Jasper'
            check('documented_voice', voice in voices)
            options = {'seed': 7} if tts.family == 'tts2' else {}
            audio = tts.generate(text, voice=voice, **options)
            check('audio', audio.ndim == 1 and audio.dtype == np.float32 and len(audio) > 2400
                  and bool(np.isfinite(audio).all()) and float(np.max(np.abs(audio))) > 0.001)
            tts.save_wav(args.out / 'speech.wav', audio)
            check('wav', sf.info(args.out / 'speech.wav').samplerate == tts.sample_rate)
            chunks = list(tts.stream(text, voice=voice, **options))
            check('stream', bool(chunks) and all(c.dtype == np.float32 and c.ndim == 1 for c in chunks))
            tts.save_wav(args.out / 'stream.wav', chunks)
            stream = tts.stream(text, voice=voice, **options)
            next(stream)
            stream.close()
            check('after_cancel', len(tts.generate('Speech after cancellation.', voice=voice, **options)) > 0)
            tts.generate_to_file('Hello from Kitten TTS.', args.out / 'file.wav', voice=voice)
            check('file_output', sf.info(args.out / 'file.wav').frames > 0)
            result['capabilities'] = tts.capabilities
            if tts.family == 'tts08':
                check('normalization', 'dollars' in tts.normalize_text('It costs $25.'))
            else:
                result['revision'] = tts.revision
                result['build_info'] = tts.build_info()
                check('seed_repeat', bool(np.array_equal(audio, tts.generate(text, voice=voice, seed=7))))
            if args.clone:
                check('cloning_supported', tts.can_clone())
                custom = tts.clone_voice(args.out / 'speech.wav', text)
                saved = tts.save_voice(custom, args.out / 'voice.kt2n')
                restored = tts.load_voice(saved)
                custom_audio = tts.generate('This is the saved voice speaking.', voice=restored, seed=7)
                check('saved_voice_generation', len(custom_audio) > 0 and bool(np.isfinite(custom_audio).all()))
                tts.save_wav(args.out / 'cloned.wav', custom_audio)
        check('context_cleanup', True)
        result['status'] = 'pass'
    except Exception as exc:
        result['status'] = 'fail'
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        result['files'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir()
                           if p.is_file() and p.name != 'verdict.json'}
        (args.out / 'verdict.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
