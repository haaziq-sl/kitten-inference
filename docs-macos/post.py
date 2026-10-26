#!/usr/bin/env python3
"""After a session: check the WAV files, the venvs and the caches it left. Not a doc command.

  python3 -I post.py OUTDIR      (OUTDIR holds the driver's steps.json)
"""
import hashlib, json, os, re, subprocess, sys, wave
from array import array
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
out = Path(sys.argv[1])
S = json.loads((out / 'steps.json').read_text(encoding='utf-8'))
root, home = Path(S['root']), Path(S['home'])
dist = os.environ['DIST']
norm = lambda n: re.sub(r'[-_.]+', '-', n).lower()
res = {'session': S['session'], 'units': {}}


def wav(p):
    if not p.exists():
        return {'exists': False}
    with wave.open(str(p)) as w:
        n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    r = {'exists': True, 'bytes': p.stat().st_size, 'rate': sr, 'channels': ch, 'sample_bytes': sw, 'frames': n,
         'seconds': round(n / sr, 3) if sr else None}
    if sw == 2:
        a = array('h', raw)
        r['peak'] = max((abs(x) for x in a), default=0) / 32768
        r['rms'] = round((sum(x * x for x in a) / max(len(a), 1)) ** 0.5 / 32768, 5)
    elif sw == 4:
        a = array('i', raw)
        r['peak'] = max((abs(x) for x in a), default=0) / 2 ** 31
    return r


def sh(*a, **kw):
    p = subprocess.run(a, capture_output=True, text=True, **kw)
    return p.returncode, (p.stdout + p.stderr).strip()


for d in sorted({Path(x) for x in [str(p.parent) for p in root.rglob('hello.wav')] + [str(p.parent.parent) for p in root.rglob('.venv/bin')]}):
    u = {'dir': str(d.relative_to(root)), 'hello.wav': wav(d / 'hello.wav')}
    py = d / '.venv' / 'bin' / 'python'
    if py.exists():
        u['venv_python'] = sh(str(py), '--version')[1]
        u['pip_version'] = sh(str(py), '-m', 'pip', '--version')[1].split(' from ')[0]
        rc, txt = sh(str(py), '-m', 'pip', 'list', '--format=json', '--disable-pip-version-check')
        try:
            pkgs = {p['name']: p['version'] for p in json.loads(txt)}
        except Exception:
            pkgs = {'_raw': txt[-500:]}
        u['packages'] = pkgs
        ours = [f'{n}=={v}' for n, v in pkgs.items() if norm(n).startswith(norm(dist))]
        u['test_release_installed'] = ours
        if ours:
            wd = out / ('wheels-' + d.name)
            rc, txt = sh(str(py), '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:', '--disable-pip-version-check',
                         '-d', str(wd), *ours)
            u['download_rc'] = rc
            u['wheels'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(wd.glob('*.whl'))}
            for f in wd.glob('*.whl'):
                f.unlink()
        rc, txt = sh(str(py), '-c', 'import kitten_inference, kitten_inference.tts08 as t; print(kitten_inference.__file__); print(t.__file__)')
        u['import'] = {'rc': rc, 'out': txt[-400:]}
    res['units'][u['dir']] = u

log = (out / 'session.log').read_bytes().decode('utf-8', 'replace') if (out / 'session.log').exists() else ''
res['wheel_files_in_log'] = sorted(set(re.findall(re.escape(dist.replace('-', '_')) + r'[A-Za-z0-9_]*-[0-9][^\s/\'"]*\.whl', log)))
hub = home / '.cache' / 'huggingface' / 'hub'
res['hf_cache'] = {}
if hub.exists():
    for repo in sorted(hub.glob('models--*')):
        snaps = sorted(p.name for p in (repo / 'snapshots').glob('*'))
        blobs = sum(f.stat().st_size for f in (repo / 'blobs').glob('*') if f.is_file())
        files = sorted(str(f.relative_to(repo / 'snapshots')) for f in (repo / 'snapshots').rglob('*') if f.is_file() or f.is_symlink())
        res['hf_cache'][repo.name] = {'snapshots': snaps, 'blob_bytes': blobs, 'files': files[:60]}
ki = home / '.cache' / 'kitten-inference'
res['kitten_inference_cache'] = {str(p.relative_to(ki)): sum(1 for _ in p.rglob('*')) for p in ki.glob('*')} if ki.exists() else None
res['home_top'] = sorted(str(p.relative_to(home)) for p in home.glob('*')) + sorted(str(p.relative_to(home)) for p in home.glob('.*'))
for c in root.rglob('claims.json'):
    (out / 'claims.json').write_bytes(c.read_bytes())
(out / 'post.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
print(json.dumps(res, indent=1, ensure_ascii=False)[:6000])
