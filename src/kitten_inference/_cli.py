"""Unified command line, with native TTS 2 advanced options preserved."""
import argparse
import sys
from pathlib import Path
from ._api import KittenTTS, model_family


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='kitten-tts', add_help=False)
    parser.add_argument('--model', required=True, help='Hugging Face model name or local TTS 0.8 directory')
    if argv == ['--version']:
        print(__import__('kitten_inference').__version__)
        return 0
    if argv in (['--help'], ['-h'], []):
        parser.print_help()
        print('\nSelect --model MODEL --help for synthesis and advanced options.')
        return 0 if argv else 2
    args, rest = parser.parse_known_args(argv)
    if model_family(args.model) == 'tts2':
        from .tts2._cli import main as native_main
        return native_main(rest)
    cpu = argparse.ArgumentParser(prog='kitten-tts --model ' + args.model)
    texts = cpu.add_mutually_exclusive_group()
    texts.add_argument('--text')
    texts.add_argument('--text-file', type=Path)
    cpu.add_argument('--voice')
    cpu.add_argument('--output', type=Path)
    cpu.add_argument('--backend', choices=['cpu', 'metal'])
    cpu.add_argument('--cache-dir')
    cpu.add_argument('--speed', type=float, default=1.0)
    cpu.add_argument('--normalize', choices=['true', 'false'], default='true')
    cpu.add_argument('--list-voices', action='store_true')
    opts = cpu.parse_args(rest)
    if not opts.list_voices and (opts.text is None and opts.text_file is None or opts.output is None):
        cpu.error('synthesis requires --text or --text-file, and --output')
    try:
        with KittenTTS(args.model, backend=opts.backend, cache_dir=opts.cache_dir) as tts:
            if opts.list_voices:
                print('\n'.join(tts.list_voices()))
            else:
                text = opts.text if opts.text is not None else opts.text_file.read_text(encoding='utf-8')
                tts.generate_to_file(text, opts.output, voice=opts.voice, speed=opts.speed,
                                     normalize=opts.normalize == 'true')
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f'kitten-tts: {exc}', file=sys.stderr)
        return 1
    return 0
