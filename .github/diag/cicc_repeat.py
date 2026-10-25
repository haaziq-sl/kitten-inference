#!/usr/bin/env python3
"""cicc_repeat.py BUILD_DIR SOURCE ARCH N OUT_DIR [--wrapper "setarch x86_64 -R"] [--jobs J]

Compiles SOURCE (a .cu file of the build, e.g. src/render/hift/hift.cu) N times for one architecture (ARCH, e.g.
sm_90) with the build's own nvcc command from BUILD_DIR/compile_commands.json, keeping nvcc's intermediate files
(--keep), and records for each compile: the module id that names the anonymous namespace, the sha256 of the PTX that
cicc wrote, and the SASS hash of every function in the cubin that ptxas wrote (instruction lines, anonymous-namespace
names normalised, as tts-2/packaging/sass_manifest.py hashes them). Linux and Windows (the command line is edited as a
string and run through the shell, so the build's quoting stays as it is). --wrapper runs each compile under a command,
e.g. with address-space layout randomisation off. Writes OUT_DIR/summary.json and one gzipped PTX per distinct PTX."""
import argparse, collections, gzip, hashlib, json, os, re, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('build'); p.add_argument('source'); p.add_argument('arch'); p.add_argument('n', type=int); p.add_argument('out')
p.add_argument('--wrapper', default=''); p.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
a = p.parse_args()
out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
cc = json.load(open(Path(a.build) / 'compile_commands.json'))
src_tail = a.source.replace('\\', '/')
e = next(x for x in cc if x['file'].replace('\\', '/').endswith(src_tail))
cmd = e['command'] if 'command' in e else subprocess.list2cmdline(e['arguments'])
num = a.arch.split('_')[1]
# drop the build's architectures, output and dependency-file options; keep everything else as written
cmd = re.sub(r'"?(?:--generate-code|-gencode)[= ]arch=compute_\d+,code=\[?[^\s"\]]*\]?"?', ' ', cmd)
cmd = re.sub(r'\s-o\s+("[^"]*"|\S+)', ' ', cmd)
cmd = re.sub(r'\s--dependency-output\s+("[^"]*"|\S+)|\s--generate-dependencies-with-compile|\s-MD\b|\s-MT\s+\S+|\s-MF\s+\S+', ' ', cmd)
windows = os.name == 'nt'
ANON = re.compile(r'_GLOBAL__N__[0-9a-f]+_\d+_[A-Za-z0-9_]+?_[0-9a-f]{8}')
cuobjdump = shutil.which('cuobjdump') or 'cuobjdump'


def sass(cubin):
    text = subprocess.run([cuobjdump, '-sass', str(cubin)], capture_output=True, text=True).stdout
    res, h = {}, None
    for line in text.splitlines():
        m = re.match(r'\s*Function : (\S+)', line)
        if m:
            h = res.setdefault(ANON.sub('_GLOBAL__N_', m.group(1)), hashlib.sha256()); continue
        s = line.strip()
        if h is not None and s.startswith('/*'):
            h.update(ANON.sub('_GLOBAL__N_', s).encode() + b'\n')
    return {k: v.hexdigest()[:16] for k, v in res.items()}


def one(i):
    kd = out / f'keep-{i}'
    if kd.exists(): shutil.rmtree(kd)
    kd.mkdir()
    obj = kd / ('out.obj' if windows else 'out.o')
    full = f'{a.wrapper} {cmd} --generate-code=arch=compute_{num},code=[sm_{num}] --keep --keep-dir "{kd}" -o "{obj}"'
    r = subprocess.run(full, shell=True, cwd=e['directory'], capture_output=True, text=True)
    if r.returncode:
        return i, {'error': (r.stdout + r.stderr)[-1500:]}
    ptx = next(kd.glob('*.ptx')); cub = next(kd.glob('*.cubin')); mid = next(kd.glob('*.module_id'), None)
    data = ptx.read_bytes()
    res = {'module_id': mid.read_text().strip() if mid else '', 'ptx': hashlib.sha256(data).hexdigest()[:16],
           'ptx_bytes': len(data), 'sass': sass(cub)}
    keep = out / f'ptx-{res["ptx"]}.ptx.gz'
    if not keep.exists():
        keep.write_bytes(gzip.compress(data, 9))
    shutil.rmtree(kd, ignore_errors=True)
    return i, res


with ThreadPoolExecutor(min(a.jobs, a.n)) as ex:
    runs = dict(ex.map(one, range(a.n)))
ok = {i: r for i, r in runs.items() if 'error' not in r}
per = collections.defaultdict(collections.Counter)
for r in ok.values():
    for k, v in r['sass'].items():
        per[k][v] += 1
summary = {
    'source': a.source, 'arch': a.arch, 'compiles': a.n, 'ok': len(ok), 'wrapper': a.wrapper, 'command': cmd,
    'module_ids': dict(collections.Counter(r['module_id'] for r in ok.values())),
    'ptx_variants': dict(collections.Counter(r['ptx'] for r in ok.values())),
    'functions': len(per), 'varying_functions': {k: dict(c) for k, c in per.items() if len(c) > 1},
    'errors': {i: r['error'] for i, r in runs.items() if 'error' in r},
    'runs': {i: {'ptx': r['ptx'], 'sass': r['sass']} for i, r in sorted(ok.items())},
}
(out / 'summary.json').write_text(json.dumps(summary, indent=1))
print(f'{a.source} {a.arch} x{a.n} (wrapper: {a.wrapper or "none"}): {len(ok)} ok; module ids {summary["module_ids"]}; '
      f'PTX variants {summary["ptx_variants"]}; functions {len(per)}, varying {len(summary["varying_functions"])}')
for k, c in summary['varying_functions'].items():
    print('  varies:', k, c)
for i, err in summary['errors'].items():
    print('  error', i, err[-600:])
sys.exit(0 if ok else 1)
