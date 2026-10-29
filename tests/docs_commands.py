"""Replay a README install block in the platform's documented shell.

Only the candidate distribution name/version and virtual-environment activation differ
from the published README. No test libraries are installed before the walkthrough.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=['tts08', 'tts2'], required=True)
    p.add_argument('--project', default='kitten-inference')
    p.add_argument('--version', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.project) or not re.fullmatch(r'\d+(?:\.\d+)+(?:\.post\d+)?', args.version):
        p.error('invalid candidate package or version')
    source = Path(__file__).resolve().parents[1] / 'README.md'
    heading = '### Kitten TTS 0.8: the CPU' if args.family == 'tts08' else '### Kitten TTS 2: NVIDIA and AMD GPUs'
    section = source.read_text(encoding='utf-8').split(heading, 1)[1]
    block = re.search(r'```sh\n(.*?)\n```', section, re.S).group(1)
    block = block.replace('pip install "kitten-inference"', 'pip install "' + args.project + '==' + args.version + '"')
    args.out.mkdir(parents=True, exist_ok=True)
    folder = args.out.resolve()
    windows = sys.platform == 'win32'
    if windows:
        block = block.replace('python3 -m venv .venv && . .venv/bin/activate', 'py -m venv .venv\ncall .venv\\Scripts\\activate.bat')
        text = '@echo on\n' + '\n'.join(line + '\nif errorlevel 1 exit /b %errorlevel%' for line in block.splitlines()) + '\n'
        path = folder / 'documented-install.cmd'
        path.write_text(text, encoding='utf-8')
        command = ['cmd.exe', '/d', '/c', str(path)]
    else:
        path = folder / 'documented-install.sh'
        path.write_text('set -e\nset -x\n' + block + '\n', encoding='utf-8')
        command = ['/bin/zsh' if sys.platform == 'darwin' else '/bin/bash', str(path)]
    with (folder / 'transcript.log').open('w', encoding='utf-8') as log:
        run = subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT)
    (folder / 'docs-verdict.json').write_text(json.dumps({
        'status': 'pass' if run.returncode == 0 else 'fail', 'exit_code': run.returncode,
        'platform': platform.platform(), 'shell': command[0], 'source': 'README.md',
        'family': args.family, 'project': args.project, 'version': args.version,
        'substitutions': ['candidate distribution name and exact version', 'documented Windows activation' if windows else 'none'],
    }, indent=2) + '\n')
    print((folder / 'transcript.log').read_text(encoding='utf-8', errors='replace'))
    return run.returncode


if __name__ == '__main__':
    raise SystemExit(main())
