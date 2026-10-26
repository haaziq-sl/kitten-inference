#!/usr/bin/env python3
"""Test-PyPI release tooling for the five kitten-inference distributions (test names: $TEST_PREFIX + real name).
(Copy used by the macOS docs test to rebuild the test-release wheels from CI wheels; the job checks every sha256.)
Stdlib only (Python >= 3.11). Never uploads (infra/upload_wheel.py does, from the laptop, run by the coordinator).

The five distributions and their test names (import names never change: everything stays `kitten_inference`):
  kitten-inference            -> $TEST_PREFIX + kitten-inference
  kitten-inference-tts08      -> $TEST_PREFIX + kitten-inference-tts08
  kitten-inference-tts2       -> $TEST_PREFIX + kitten-inference-tts2
  kitten-inference-tts2-cu12  -> $TEST_PREFIX + kitten-inference-tts2-cu12
  kitten-inference-tts2-rocm  -> $TEST_PREFIX + kitten-inference-tts2-rocm

  test_release.py retag WHEEL... --out DIR [--version DIST=VER ...]
      Already-built wheels -> test names WITHOUT rebuilding: renames the .dist-info (and .data) directory, rewrites
      METADATA `Name:` (+ `Version:` only if --version is given) and every `Requires-Dist:` on one of the five names, and
      rewrites RECORD. Every other zip member is copied as its raw compressed stream, so the payload (engine library, CLI,
      Python files) is byte-identical to the reviewed wheel; the tool proves it (RECORD hashes before == after) and writes
      DIR/retag-report.json.
  test_release.py prepare --repo REPO --ref REF --out DIR [--version DIST=VER ...] [--all-versions VER] [--exclude PATH]
      Source route: `git archive REF` into DIR/src (a build copy; the repo is never touched), rename each of the five
      distributions found in any pyproject.toml, rewrite their dependency/extra requirements on the five names, set versions
      given as arguments. .py literals equal to a distribution name are reported (not changed; --rewrite-metadata-calls
      rewrites the ones that are importlib.metadata call arguments). Writes DIR/rename-report.json + DIR/rename.diff and re-parses every changed pyproject to prove only those
      fields changed.
  test_release.py build --src DIR/src --out DIR/dist [--dist NAME ...] [--sdist]
      `python -m build --wheel` (+ --sdist) for the renamed pure-Python projects (default: the umbrella and tts2).
      Rented boxes only: refuses on the laptop.
  test_release.py check FILE... [--expect DIST=VER ...] [--no-external]
      Independent check of wheels/sdists: zip CRCs, file name vs METADATA, dist-info name, RECORD covers every member
      with valid sha256/size, Name is a test name, no Requires-Dist on a real (non-test) name; then `twine check`
      and `wheel unpack` (when installed; --no-external skips them). Writes nothing; exit 1 on any failure.

Typical flow (coordinator): CI/build boxes produce the reviewed wheels -> `retag` them on a box -> `check` -> copy the
retagged files + retag-report.json to the laptop -> `infra/upload_wheel.py --dry-run FILES` -> `infra/upload_wheel.py FILES`
-> `infra/verify_published.py FILES`.
"""
import argparse, base64, csv, difflib, hashlib, io, json, os, re, shutil, socket, struct, subprocess, sys, tarfile, \
    tempfile, zipfile, zlib
try:
    import tomllib
except ImportError:  # only the source route needs it
    tomllib = None
from pathlib import Path

REAL = ['kitten-inference', 'kitten-inference-tts08', 'kitten-inference-tts2', 'kitten-inference-tts2-cu12',
        'kitten-inference-tts2-rocm']
TEST_PREFIX = os.environ['TEST_PREFIX']
TEST = {n: TEST_PREFIX + n for n in REAL}
LAPTOP_HOSTS = set()
CHUNK = 1 << 20


def norm(name):
    """PEP 503 normalisation."""
    return re.sub(r'[-_.]+', '-', name).lower()


def wheel_norm(name):
    """Distribution part of a wheel / dist-info name (PEP 427 + binary distribution format)."""
    return re.sub(r'[-_.]+', '_', name).lower()


REAL_BY_NORM = {norm(n): n for n in REAL}
TEST_NORMS = {norm(v) for v in TEST.values()}


def req_name(req):
    m = re.match(r'\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)', req)
    return m.group(1) if m else None


def rename_req(req):
    """'kitten-inference-tts2>=0.2.0,<0.3; extra == "x"' -> '<test prefix>kitten-inference-tts2>=0.2.0,<0.3; ...' (or None)."""
    n = req_name(req)
    if n and norm(n) in REAL_BY_NORM:
        i = req.index(n)
        return req[:i] + TEST[REAL_BY_NORM[norm(n)]] + req[i + len(n):]
    return None


def parse_versions(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition('=')
        if not v or norm(k) not in REAL_BY_NORM:
            raise SystemExit(f'--version wants DIST=VER with DIST one of {REAL}; got {it!r}')
        if not re.fullmatch(r'\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?', v):
            raise SystemExit(f'not a plain PEP 440 version: {v!r}')
        out[REAL_BY_NORM[norm(k)]] = v
    return out


def sha256_b64(data):
    return 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- wheels: read helpers

WHEEL_RE = re.compile(r'^(?P<dist>[^-]+)-(?P<ver>[^-]+)(-(?P<build>\d[^-]*))?-(?P<tags>[^-]+-[^-]+-[^-]+)\.whl$')


def dist_info_dir(names):
    d = {n.split('/', 1)[0] for n in names if n.split('/', 1)[0].endswith('.dist-info') and '/' in n}
    if len(d) != 1:
        raise ValueError(f'expected exactly one .dist-info directory, found {sorted(d)}')
    return d.pop()


def member_digest(zf, zi):
    """(sha256 digest, size) of a zip member, streamed (members can be hundreds of MB)."""
    h, n = hashlib.sha256(), 0
    with zf.open(zi) as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
            n += len(b)
    return h.digest(), n


def record_hash(digest):
    return 'sha256=' + base64.urlsafe_b64encode(digest).rstrip(b'=').decode()


def header_value(meta: bytes, key: str):
    head = meta.replace(b'\r\n', b'\n').split(b'\n\n', 1)[0].decode('utf-8')
    vals = [l.split(':', 1)[1].strip() for l in head.split('\n') if l.lower().startswith(key.lower() + ':')]
    return vals


def read_record(data: bytes):
    rows = list(csv.reader(io.StringIO(data.decode('utf-8'))))
    return [r for r in rows if r]


# ---------------------------------------------------------------- raw zip copy (no recompression)

LFH = struct.Struct('<4sHHHHHLLLHH')            # local file header
CDH = struct.Struct('<4sBBHHHHHLLLHHHHHLL')     # central directory header
EOCD = struct.Struct('<4sHHHHLLH')


def dos_time(dt6):
    y, mo, d, h, mi, s = dt6
    return (h << 11) | (mi << 5) | (s // 2), ((max(y, 1980) - 1980) << 9) | (mo << 5) | d


def raw_member(src_f, zi):
    """(local header length, offset of the compressed data) of a member, from its local header."""
    src_f.seek(zi.header_offset)
    h = LFH.unpack(src_f.read(LFH.size))
    if h[0] != b'PK\x03\x04':
        raise ValueError(f'bad local header for {zi.filename}')
    return zi.header_offset + LFH.size + h[9] + h[10]


def write_zip(src_path, out_path, plan):
    """plan: list of (ZipInfo, new_name, new_bytes_or_None). None = copy the raw compressed stream unchanged."""
    cds = []
    with open(src_path, 'rb') as src, open(out_path, 'wb') as out:
        for zi, name, new in plan:
            if max(zi.file_size, zi.compress_size, zi.header_offset) >= 0xFFFFFFFF:
                raise ValueError('zip64 member: not supported')
            fname = name.encode('utf-8')
            flags = (zi.flag_bits & ~0x08) | (0x800 if not name.isascii() else 0)
            t, d = dos_time(zi.date_time)
            if new is None:
                method, crc, csize, usize = zi.compress_type, zi.CRC, zi.compress_size, zi.file_size
            else:
                method = zi.compress_type if zi.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) else zipfile.ZIP_DEFLATED
                crc, usize = zlib.crc32(new) & 0xFFFFFFFF, len(new)
                if method == zipfile.ZIP_DEFLATED:
                    c = zlib.compressobj(9, zlib.DEFLATED, -15)
                    payload = c.compress(new) + c.flush()
                else:
                    payload = new
                csize = len(payload)
                flags &= ~0x06  # compression-option bits refer to the old stream
            offset = out.tell()
            out.write(LFH.pack(b'PK\x03\x04', 20, flags, method, t, d, crc, csize, usize, len(fname), 0) + fname)
            if new is None:
                src.seek(raw_member(src, zi))
                left = csize
                while left:
                    b = src.read(min(CHUNK, left))
                    if not b:
                        raise ValueError(f'truncated member {zi.filename}')
                    out.write(b)
                    left -= len(b)
            else:
                out.write(payload)
            cds.append(CDH.pack(b'PK\x01\x02', zi.create_version, zi.create_system, 20, flags, method, t, d, crc, csize,
                                usize, len(fname), 0, 0, 0, zi.internal_attr, zi.external_attr, offset) + fname)
            if out.tell() >= 0xFFFFFFFF:
                raise ValueError('archive larger than 4 GiB: zip64 not supported')
        cd_start = out.tell()
        for c in cds:
            out.write(c)
        out.write(EOCD.pack(b'PK\x05\x06', 0, 0, len(cds), len(cds), out.tell() - cd_start, cd_start, 0))


# ---------------------------------------------------------------- retag

def rewrite_metadata(meta: bytes, new_name, new_version, new_body=None):
    """Line-level edit of the header block; the description body is left byte-identical unless new_body is given
    (then it replaces the body and the Description-Content-Type is set to text/markdown)."""
    nl = b'\r\n' if b'\r\n' in meta else b'\n'
    head, sep, body = meta.partition(nl + nl)
    lines, changes, seen_name = head.split(nl), [], False
    if new_body is not None:
        lines = [l for l in lines if not l.lower().startswith(b'description-content-type:')]
        lines.append(b'Description-Content-Type: text/markdown')
        changes.append({'before': f'<description body {len(body)} bytes>', 'after': f'<description body {len(new_body)} bytes>'})
        body, sep = new_body.replace(b'\r\n', b'\n').replace(b'\n', nl), nl + nl
    for i, line in enumerate(lines):
        s = line.decode('utf-8')
        key = s.split(':', 1)[0].lower() if ':' in s else ''
        if key == 'name' and not seen_name:
            seen_name = True
            new = f'Name: {new_name}'
        elif key == 'version' and new_version:
            new = f'Version: {new_version}'
        elif key == 'requires-dist':
            r = rename_req(s.split(':', 1)[1].strip())
            new = f'Requires-Dist: {r}' if r else s
        else:
            continue
        if new != s:
            changes.append({'before': s, 'after': new})
            lines[i] = new.encode('utf-8')
    if not seen_name:
        raise ValueError('METADATA has no Name header')
    return nl.join(lines) + sep + body, changes


def retag_one(wheel: Path, out_dir: Path, versions, descriptions=None):
    m = WHEEL_RE.match(wheel.name)
    if not m:
        raise SystemExit(f'not a wheel file name: {wheel.name}')
    with zipfile.ZipFile(wheel) as zf:
        bad = zf.testzip()
        if bad:
            raise SystemExit(f'{wheel.name}: CRC error in {bad}')
        infos = zf.infolist()
        names = [i.filename for i in infos]
        di = dist_info_dir(names)
        meta = zf.read(f'{di}/METADATA')
        old_name = header_value(meta, 'Name')[0]
        old_ver = header_value(meta, 'Version')[0]
        if norm(old_name) not in REAL_BY_NORM:
            raise SystemExit(f'{wheel.name}: Name {old_name!r} is not one of the five distributions')
        real = REAL_BY_NORM[norm(old_name)]
        new_name, new_ver = TEST[real], versions.get(real)
        if norm(m['dist']) != norm(old_name) or m['ver'] != old_ver:
            raise SystemExit(f'{wheel.name}: file name does not match METADATA ({old_name} {old_ver})')
        old_prefix = f'{wheel_norm(old_name)}-{old_ver}'
        if di != old_prefix + '.dist-info':
            # tolerate a dist-info dir spelled with the unnormalised name (older tools)
            old_prefix = di[:-len('.dist-info')]
        new_prefix = f'{wheel_norm(new_name)}-{new_ver or old_ver}'
        record_name = f'{di}/RECORD'
        record = read_record(zf.read(record_name))
        rec = {r[0]: r for r in record}
        # 1. the input's RECORD must already describe its payload exactly
        payload_ok = 0
        for zi in infos:
            if zi.is_dir() or zi.filename == record_name:
                continue
            r = rec.get(zi.filename)
            if not r or not r[1]:
                if zi.filename.startswith(di + '/RECORD.'):
                    continue  # signatures: dropped below
                raise SystemExit(f'{wheel.name}: {zi.filename} missing from RECORD')
            dg, size = member_digest(zf, zi)
            if record_hash(dg) != r[1] or (r[2] and int(r[2]) != size):
                raise SystemExit(f'{wheel.name}: RECORD hash/size mismatch for {zi.filename}')
            payload_ok += 1
        new_meta, meta_changes = rewrite_metadata(meta, new_name, new_ver, (descriptions or {}).get(real))

    def mapped(n):
        for old, new in ((old_prefix + '.dist-info', new_prefix + '.dist-info'), (old_prefix + '.data', new_prefix + '.data')):
            if n == old or n.startswith(old + '/'):
                return new + n[len(old):]
        return n

    dropped = [n for n in names if n.startswith(di + '/RECORD.')]
    new_record_rows = []
    for r in record:
        if r[0] in dropped:
            continue
        if r[0] == f'{di}/METADATA':
            new_record_rows.append([mapped(r[0]), sha256_b64(new_meta), str(len(new_meta))])
        elif r[0] == record_name:
            new_record_rows.append([mapped(r[0]), '', ''])
        else:
            new_record_rows.append([mapped(r[0])] + r[1:])
    sio = io.StringIO()
    csv.writer(sio, lineterminator='\n').writerows(new_record_rows)
    new_record = sio.getvalue().encode('utf-8')
    plan = []
    for zi in infos:
        if zi.filename in dropped:
            continue
        new = new_meta if zi.filename == f'{di}/METADATA' else new_record if zi.filename == record_name else None
        plan.append((zi, mapped(zi.filename), new))
    out_name = f"{wheel_norm(new_name)}-{new_ver or old_ver}{'-' + m['build'] if m['build'] else ''}-{m['tags']}.whl"
    out = out_dir / out_name
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    tmp = out.with_suffix('.whl.part')
    write_zip(wheel, tmp, plan)
    # 2. prove: same member list (modulo the dist-info rename), and every payload member's bytes unchanged
    with zipfile.ZipFile(wheel) as a, zipfile.ZipFile(tmp) as b:
        if b.testzip():
            raise SystemExit(f'{out_name}: CRC error after retag')
        an = [mapped(n) for n in a.namelist() if n not in dropped]
        if an != b.namelist():
            raise SystemExit(f'{out_name}: member list changed')
        same_raw = 0
        for zi in a.infolist():
            if zi.filename in dropped or zi.filename in (f'{di}/METADATA', record_name):
                continue
            zb = b.getinfo(mapped(zi.filename))
            if (zi.CRC, zi.file_size, zi.compress_size, zi.compress_type, zi.external_attr) != \
                    (zb.CRC, zb.file_size, zb.compress_size, zb.compress_type, zb.external_attr):
                raise SystemExit(f'{out_name}: member {zi.filename} changed')
            if not zi.is_dir() and member_digest(a, zi) != member_digest(b, zb):
                raise SystemExit(f'{out_name}: content of {zi.filename} changed')
            same_raw += 1
    tmp.replace(out)
    return {'input': wheel.name, 'input_sha256': file_sha256(wheel), 'input_bytes': wheel.stat().st_size,
            'output': out.name, 'output_sha256': file_sha256(out), 'output_bytes': out.stat().st_size,
            'name': [old_name, new_name], 'version': [old_ver, new_ver or old_ver],
            'dist_info': [di, new_prefix + '.dist-info'], 'metadata_changes': meta_changes,
            'members_byte_identical': same_raw, 'payload_files_hash_checked': payload_ok,
            'changed_members': [f'{new_prefix}.dist-info/METADATA', f'{new_prefix}.dist-info/RECORD'],
            'dropped_signatures': dropped}


def cmd_retag(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    versions = parse_versions(a.version)
    descriptions = {}
    for item in a.description or []:
        dist, _, path = item.partition('=')
        real = REAL_BY_NORM.get(norm(dist))
        if not real or not path:
            raise SystemExit(f'--description wants DIST=README.md with one of the five real names: {item!r}')
        descriptions[real] = Path(path).read_bytes()
    rows = [retag_one(Path(w), out, versions, descriptions) for w in a.wheels]
    rep = out / 'retag-report.json'
    old = json.loads(rep.read_text()) if rep.exists() else []
    rep.write_text(json.dumps(old + rows, indent=1) + '\n')
    for r in rows:
        print(f"{r['input']} -> {r['output']}  ({r['members_byte_identical']} members byte-identical; "
              f"{len(r['metadata_changes'])} METADATA lines changed; {r['output_bytes'] - r['input_bytes']:+d} bytes)")
    print('report', rep)


# ---------------------------------------------------------------- prepare (source route)

def toml_string_spans(text, value):
    """Every quoted occurrence of `value` as a TOML basic or literal string."""
    pats = [json.dumps(value), "'" + value + "'"]
    return [(m.start(), m.end(), m.group(0)) for p in pats for m in re.finditer(re.escape(p), text)]


def edit_pyproject(path: Path, versions, all_version):
    text = path.read_text(encoding='utf-8')
    data = tomllib.loads(text)
    proj = data.get('project') or {}
    name = proj.get('name')
    if not name or norm(name) not in REAL_BY_NORM:
        reqs = [r for r in proj.get('dependencies', []) + [x for v in (proj.get('optional-dependencies') or {}).values() for x in v]
                if rename_req(r)]
        return None if not reqs else {'path': str(path), 'warning': f'non-target project {name!r} requires {reqs} (left unchanged)'}
    real = REAL_BY_NORM[norm(name)]
    changes, new_text = [], text
    # [project] table boundaries: only edit inside it (and its optional-dependencies sub-table)
    def table_span(hdr):
        m = re.search(r'(?m)^\s*\[' + re.escape(hdr) + r'\]\s*(#.*)?$', new_text)
        if not m:
            return None
        n = re.search(r'(?m)^\s*\[(?!' + re.escape(hdr) + r'\])', new_text[m.end():])
        return m.end(), m.end() + n.start() if n else len(new_text)

    def sub_in(span, pattern, repl, what):
        nonlocal new_text
        s, e = span
        seg, n = re.subn(pattern, repl, new_text[s:e], count=1)
        if n != 1:
            raise SystemExit(f'{path}: could not rewrite {what}')
        new_text = new_text[:s] + seg + new_text[e:]

    span = table_span('project')
    if span is None:
        raise SystemExit(f'{path}: [project] table not found as a header (inline tables not supported)')
    sub_in(span, r'(?m)^(\s*name\s*=\s*)(["\'])' + re.escape(name) + r'\2', lambda m: m.group(1) + m.group(2) + TEST[real] + m.group(2), 'name')
    changes.append({'field': 'project.name', 'before': name, 'after': TEST[real]})
    want = versions.get(real) or all_version
    if want and want == proj.get('version'):
        want = None  # already that version: nothing to change
    if want:
        if 'version' not in proj:
            raise SystemExit(f'{path}: version is dynamic ({proj.get("dynamic")}); make it static or omit --version for {real}')
        span = table_span('project')
        sub_in(span, r'(?m)^(\s*version\s*=\s*)(["\'])' + re.escape(proj['version']) + r'\2',
               lambda m: m.group(1) + m.group(2) + want + m.group(2), 'version')
        changes.append({'field': 'project.version', 'before': proj['version'], 'after': want})
    reqs = [('project.dependencies', r) for r in proj.get('dependencies', [])]
    reqs += [(f'project.optional-dependencies.{k}', r) for k, v in (proj.get('optional-dependencies') or {}).items() for r in v]
    done = set()
    for field, r in reqs:
        new = rename_req(r)
        if not new:
            continue
        changes.append({'field': field, 'before': r, 'after': new})
        if r in done:   # the same string in several extras: all occurrences were rewritten the first time
            continue
        done.add(r)
        spans = []
        for hdr in ('project', 'project.optional-dependencies'):
            sp = table_span(hdr)
            if sp:
                spans += [(sp[0] + s, sp[0] + e, q) for s, e, q in toml_string_spans(new_text[sp[0]:sp[1]], r)]
        if not spans:
            raise SystemExit(f'{path}: requirement {r!r} not found as a plain quoted string')
        for s, e, q in sorted(set(spans), reverse=True):
            new_text = new_text[:s] + q[0] + new + q[-1] + new_text[e:]
    # prove: re-parse; the only differences are the ones listed
    after = tomllib.loads(new_text)
    expect = json.loads(json.dumps(data))
    expect['project']['name'] = TEST[real]
    if want:
        expect['project']['version'] = want
    if 'dependencies' in proj:
        expect['project']['dependencies'] = [rename_req(r) or r for r in proj['dependencies']]
    if 'optional-dependencies' in proj:
        expect['project']['optional-dependencies'] = {k: [rename_req(r) or r for r in v] for k, v in proj['optional-dependencies'].items()}
    if after != expect:
        raise SystemExit(f'{path}: re-parse shows unexpected differences; refusing')
    path.write_text(new_text, encoding='utf-8')
    return {'path': str(path), 'distribution': real, 'test_name': TEST[real], 'changes': changes,
            'version': want or proj.get('version')}


def cmd_prepare(a):
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists and is not empty')
    src = out / 'src'
    src.mkdir(parents=True)
    commit = subprocess.check_output(['git', '-C', a.repo, 'rev-parse', a.ref + '^{commit}'], text=True).strip()
    with tempfile.TemporaryFile() as t:
        subprocess.run(['git', '-C', a.repo, 'archive', '--format=tar', commit], stdout=t, check=True)
        t.seek(0)
        with tarfile.open(fileobj=t) as tf:
            tf.extractall(src, filter='data')
    tracked = lambda p: p.is_file() and (p.name == 'pyproject.toml' or p.suffix == '.py')
    pristine = {p.relative_to(src): p.read_bytes() for p in src.rglob('*') if tracked(p)}
    versions = parse_versions(a.version)
    report = {'repo': os.path.abspath(a.repo), 'ref': a.ref, 'commit': commit, 'pyprojects': [], 'py_literals': [], 'warnings': []}
    excluded = []
    for p in sorted(src.rglob('pyproject.toml')):
        rel = p.relative_to(src)
        if any(rel.match(g) or str(rel) == g for g in a.exclude or []):
            excluded.append(str(rel))
            continue
        r = edit_pyproject(p, versions, a.all_versions)
        if r and 'warning' in r:
            report['warnings'].append(r['warning'])
        elif r:
            r['path'] = str(p.relative_to(src))
            report['pyprojects'].append(r)
    found = {r['distribution'] for r in report['pyprojects']}
    missing = [n for n in versions if n not in found]
    if missing:
        raise SystemExit(f'--version given for distributions with no pyproject at {a.ref}: {missing}')
    report['excluded'] = excluded
    if len(found) != len(report['pyprojects']):
        dup = {}
        for r in report['pyprojects']:
            dup.setdefault(r['distribution'], []).append(r['path'])
        raise SystemExit('several pyproject.toml files declare the same distribution: '
                         + '; '.join(f'{k}: {v}' for k, v in dup.items() if len(v) > 1) + ' (leave stale ones out with --exclude PATH)')
    # Python sources: exact string literals equal to a distribution name. They are only REPORTED by default: most are not
    # metadata lookups (a cache directory "kitten-inference", a descriptive field, test expectations) and the retag route
    # never changes code either. --rewrite-metadata-calls rewrites only a literal that is the first argument of an
    # importlib.metadata call (version/distribution/metadata/files/requires), e.g. metadata.version("kitten-inference-tts2").
    names = '|'.join(re.escape(n) for n in sorted(REAL, key=len, reverse=True))
    lit = re.compile(r'''(["'])(''' + names + r''')\1''')
    call = re.compile(r'''(\b(?:version|distribution|metadata|files|requires)\(\s*)(["'])(''' + names + r''')\2''')
    for p in sorted(src.rglob('*.py')):
        s = p.read_text(encoding='utf-8', errors='surrogateescape')
        if not lit.search(s):
            continue
        new_lines = []
        for i, line in enumerate(s.splitlines(keepends=True), 1):
            nl = call.sub(lambda m: m.group(1) + m.group(2) + TEST[m.group(3)] + m.group(2), line) if a.rewrite_metadata_calls else line
            if lit.search(line):
                report['py_literals'].append({'file': str(p.relative_to(src)), 'line': i, 'text': line.strip(),
                                              'metadata_call': bool(call.search(line)), 'rewritten': nl != line,
                                              'after': nl.strip() if nl != line else None})
            new_lines.append(nl)
        if new_lines and ''.join(new_lines) != s:
            p.write_text(''.join(new_lines), encoding='utf-8', errors='surrogateescape')
    # unified diff of everything that changed in the build copy
    diff = []
    for rel, before in sorted(pristine.items()):
        now_b = (src / rel).read_bytes()
        if now_b != before:
            diff += difflib.unified_diff(before.decode('utf-8', 'replace').splitlines(True),
                                         now_b.decode('utf-8', 'replace').splitlines(True), f'a/{rel}', f'b/{rel}')
    (out / 'rename.diff').write_text(''.join(diff))
    (out / 'rename-report.json').write_text(json.dumps(report, indent=1) + '\n')
    for r in report['pyprojects']:
        print(f"{r['path']}: {r['distribution']} -> {r['test_name']} {r['version']}; " +
              '; '.join(f"{c['field']}: {c['before']} -> {c['after']}" for c in r['changes'][1:]))
    for r in report['py_literals']:
        tag = 'REWRITTEN' if r['rewritten'] else ('metadata call, NOT rewritten' if r['metadata_call'] else 'literal, left as is')
        print(f"  {tag}: {r['file']}:{r['line']}: {r['text'][:140]}" + (f"  ->  {r['after'][:140]}" if r['rewritten'] else ''))
    calls = [r for r in report['py_literals'] if r['metadata_call'] and not r['rewritten'] and '/tests/' not in '/' + r['file']]
    if calls:
        print(f'WARNING {len(calls)} importlib.metadata lookup(s) by real distribution name outside tests/ (they find nothing '
              'under the test names; --rewrite-metadata-calls rewrites them in this build copy)')
    for w in report['warnings']:
        print('WARNING', w)
    print(f"prepared {src} from {a.ref} ({commit[:12]}); {len(report['pyprojects'])} distributions renamed; "
          f"report {out / 'rename-report.json'}, diff {out / 'rename.diff'} ({sum(1 for l in diff if l[:1] in '+-' and l[:3] not in ('+++', '---'))} lines)")


def cmd_build(a):
    if socket.gethostname() in LAPTOP_HOSTS:
        raise SystemExit('Refusing: builds run on rented boxes only (laptop rule)')
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = [REAL_BY_NORM[norm(d)] for d in (a.dist or ['kitten-inference', 'kitten-inference-tts2'])]
    projects = {}
    for p in src.rglob('pyproject.toml'):
        n = (tomllib.loads(p.read_text()).get('project') or {}).get('name', '')
        if norm(n) in TEST_NORMS:
            projects[n] = p.parent
    built = []
    for real in wanted:
        d = projects.get(TEST[real])
        if d is None:
            raise SystemExit(f'no renamed project for {real} under {src} (run prepare first)')
        before = set(out.iterdir())
        cmd = [sys.executable, '-m', 'build', '--wheel'] + (['--sdist'] if a.sdist else []) + ['--outdir', str(out), str(d)]
        print('+', ' '.join(cmd))
        subprocess.run(cmd, check=True)
        built += sorted(set(out.iterdir()) - before)
    for f in built:
        print(f'{f.name} sha256 {file_sha256(f)}')
    return cmd_check(argparse.Namespace(files=[str(f) for f in built], expect=None, no_external=False))


# ---------------------------------------------------------------- check

def check_wheel(p: Path, expect):
    errs = []
    m = WHEEL_RE.match(p.name)
    if not m:
        return [f'bad wheel file name'], {}
    with zipfile.ZipFile(p) as zf:
        bad = zf.testzip()
        if bad:
            errs.append(f'CRC error in {bad}')
        names = zf.namelist()
        try:
            di = dist_info_dir(names)
        except ValueError as e:
            return [str(e)], {}
        meta = zf.read(f'{di}/METADATA')
        name, ver = header_value(meta, 'Name')[0], header_value(meta, 'Version')[0]
        if wheel_norm(name) != m['dist'] or ver != m['ver']:
            errs.append(f'file name {m["dist"]} {m["ver"]} != METADATA {name} {ver}')
        if di != f'{wheel_norm(name)}-{ver}.dist-info':
            errs.append(f'dist-info dir {di} != {wheel_norm(name)}-{ver}.dist-info')
        for d in {n.split('/', 1)[0] for n in names if n.split('/', 1)[0].endswith('.data')}:
            if d != f'{wheel_norm(name)}-{ver}.data':
                errs.append(f'.data dir {d} does not match the distribution')
        rec = read_record(zf.read(f'{di}/RECORD'))
        listed = {r[0] for r in rec}
        for n in names:
            if n.endswith('/'):
                continue
            if n not in listed and not n.startswith(di + '/RECORD.'):
                errs.append(f'{n} not in RECORD')
        for r in rec:
            if r[0] == f'{di}/RECORD':
                continue
            if r[0] not in names:
                errs.append(f'RECORD lists missing {r[0]}')
                continue
            dg, size = member_digest(zf, zf.getinfo(r[0]))
            if not r[1] or record_hash(dg) != r[1] or int(r[2] or -1) != size:
                errs.append(f'RECORD hash/size mismatch for {r[0]}')
        wheelfile = zf.read(f'{di}/WHEEL').decode()
        tags = header_value(wheelfile.encode(), 'Tag')
        py, abi, plats = m['tags'].split('-')
        want_tags = {f'{x}-{y}-{z}' for x in py.split('.') for y in abi.split('.') for z in plats.split('.')}
        if set(tags) != want_tags:
            errs.append(f'WHEEL Tag {tags} != file name tags {sorted(want_tags)}')
        reqs = header_value(meta, 'Requires-Dist')
        top = sorted({n.split('/', 1)[0] for n in names if '/' in n and not n.split('/', 1)[0].endswith(('.dist-info', '.data'))})
        execs = sorted(n for n in names if (zf.getinfo(n).external_attr >> 16) & 0o111 and not n.endswith('/'))
    errs += common_meta_checks(name, ver, reqs, expect)
    return errs, {'name': name, 'version': ver, 'tags': m['tags'], 'top_level': top, 'requires_dist': reqs,
                  'extras': header_value(meta, 'Provides-Extra'), 'executables': execs, 'members': len(names)}


def check_sdist(p: Path, expect):
    errs = []
    with tarfile.open(p) as tf:
        pk = [m for m in tf.getmembers() if m.name.count('/') == 1 and m.name.endswith('/PKG-INFO')]
        if len(pk) != 1:
            return ['expected one top-level PKG-INFO'], {}
        meta = tf.extractfile(pk[0]).read()
    name, ver = header_value(meta, 'Name')[0], header_value(meta, 'Version')[0]
    if not p.name.startswith(f'{wheel_norm(name)}-{ver}.') and not p.name.startswith(f'{name}-{ver}.'):
        errs.append(f'file name does not match {name} {ver}')
    reqs = header_value(meta, 'Requires-Dist')
    errs += common_meta_checks(name, ver, reqs, expect)
    return errs, {'name': name, 'version': ver, 'requires_dist': reqs, 'extras': header_value(meta, 'Provides-Extra')}


def common_meta_checks(name, ver, reqs, expect):
    errs = []
    if norm(name) not in TEST_NORMS:
        errs.append(f'Name {name!r} is not a test name')
    for r in reqs:
        n = req_name(r) or ''
        if norm(n) in REAL_BY_NORM:
            errs.append(f'Requires-Dist on the real name: {r}')
    if expect:
        real = name[len(TEST_PREFIX):] if name.startswith(TEST_PREFIX) else name
        want = expect.get(REAL_BY_NORM.get(norm(real), ''))
        if want and want != ver:
            errs.append(f'version {ver} != expected {want}')
    return errs


def external_checks(files):
    out = {}
    try:
        r = subprocess.run([sys.executable, '-m', 'twine', 'check', '--strict', *files], capture_output=True, text=True)
        out['twine_check'] = 'PASSED' if r.returncode == 0 else 'FAILED: ' + (r.stdout + r.stderr)[-1500:]
    except FileNotFoundError:
        out['twine_check'] = 'not run (no twine)'
    if 'No module named twine' in out.get('twine_check', ''):
        out['twine_check'] = 'not run (no twine)'
    wheels = [f for f in files if f.endswith('.whl')]
    if wheels:
        with tempfile.TemporaryDirectory() as td:
            res = []
            for w in wheels:
                r = subprocess.run([sys.executable, '-m', 'wheel', 'unpack', '-d', td, w], capture_output=True, text=True)
                if 'No module named wheel' in r.stderr:
                    res = ['not run (no wheel)']
                    break
                res.append(f"{Path(w).name}: {'OK' if r.returncode == 0 else 'FAILED ' + (r.stdout + r.stderr)[-500:]}")
            out['wheel_unpack'] = res
    return out


def cmd_check(a):
    expect = parse_versions(a.expect) if a.expect else {}
    failed, rows = False, []
    for f in a.files:
        p = Path(f)
        errs, info = check_wheel(p, expect) if p.suffix == '.whl' else check_sdist(p, expect)
        rows.append({'file': p.name, 'sha256': file_sha256(p), 'bytes': p.stat().st_size, **info, 'errors': errs})
        failed |= bool(errs)
        print(f"{'OK  ' if not errs else 'FAIL'} {p.name} ({p.stat().st_size / 1e6:.1f} MB) {info.get('name')} {info.get('version')}"
              + (f" requires {info['requires_dist']}" if info.get('requires_dist') else '') + ''.join(f'\n     - {e}' for e in errs))
    if not a.no_external:
        ext = external_checks(a.files)
        for k, v in ext.items():
            print(k, v if isinstance(v, str) else '; '.join(v))
            vals = [v] if isinstance(v, str) else v
            failed |= any('FAILED' in x for x in vals)
    print('CHECK', 'FAILED' if failed else 'PASSED')
    return 1 if failed else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='op', required=True)
    p = sp.add_parser('retag'); p.add_argument('wheels', nargs='+'); p.add_argument('--out', required=True)
    p.add_argument('--version', action='append')
    p.add_argument('--description', action='append', help='DIST=README.md: replace that distribution\'s long description (dist-info only)')
    p = sp.add_parser('prepare'); p.add_argument('--repo', required=True); p.add_argument('--ref', required=True)
    p.add_argument('--out', required=True); p.add_argument('--version', action='append'); p.add_argument('--all-versions')
    p.add_argument('--exclude', action='append', help='pyproject.toml path (relative to the repo) to leave untouched')
    p.add_argument('--rewrite-metadata-calls', action='store_true')
    p = sp.add_parser('build'); p.add_argument('--src', required=True); p.add_argument('--out', required=True)
    p.add_argument('--dist', action='append'); p.add_argument('--sdist', action='store_true')
    p = sp.add_parser('check'); p.add_argument('files', nargs='+'); p.add_argument('--expect', action='append')
    p.add_argument('--no-external', action='store_true')
    a = ap.parse_args()
    raise SystemExit({'retag': cmd_retag, 'prepare': cmd_prepare, 'build': cmd_build, 'check': cmd_check}[a.op](a) or 0)
