#!/usr/bin/env python3
"""Run the doc lines of one session on a macOS runner, like a new user would.

  python3 -I driver.py SESSION OUTDIR

Every doc line comes byte for byte from the files in doc/ (copied from the README code blocks). The only change is
the one allowed for test releases: in `pip install "kitten-inference[...]"` the distribution name becomes $DIST.
Lines marked "setup" are not from the docs (opening a new folder or a new terminal); they are printed as such.

Shell kinds:
  bash-script  /bin/bash runs a generated script, stdin /dev/null, stops at the first failing line
  bash-pty     interactive /bin/bash in a real pseudo-terminal, one line at a time
  zsh-pty      interactive /bin/zsh (the macOS default shell) in a real pseudo-terminal, one line at a time
  zsh-login-pty  the same as a login shell (what Terminal starts), with the stock macOS PATH
Each session gets an empty HOME, so pip, Hugging Face and package caches start empty.
"""
import json, os, pty, re, select, shlex, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE / 'doc'
DIST = os.environ['DIST']
SYSTEM_PATH = '/usr/bin:/bin:/usr/sbin:/sbin'
STEP_TIMEOUT = 1200

SESSIONS = {
    # name: (shell, path, units, stop_on_fail)
    'S1-bash':             ('bash-script',   'runner', ['top'], True),
    'S2-bash-pkg':         ('bash-script',   'runner', ['pkg', 'claims'], True),
    'S3-bash-deep':        ('bash-script',   'runner', ['deep'], True),
    'S4-bash-pty-replay':  ('bash-pty',      'runner', ['top', 'pkg'], True),
    'S5-zsh-pty':          ('zsh-pty',       'runner', ['top', 'pkg'], False),
    'S6-zsh-login-system': ('zsh-login-pty', 'system', ['top', 'pkg'], False),
    'S7-bash-system':      ('bash-script',   'system', ['top'], True),
    'S8-bash-system-oldpip': ('bash-script', 'system', ['oldpip'], True),
    # machine check before the release exists: no doc lines
    'P0-smoke-zsh':        ('zsh-pty',       'runner', ['smoke'], False),
    'P0-smoke-zsh-login':  ('zsh-login-pty', 'system', ['smoke'], False),
    'P0-smoke-bash':       ('bash-pty',      'runner', ['smoke'], False),
}


def doc_lines(name):
    out = []
    for line in (DOC / name).read_text(encoding='utf-8').splitlines():
        if line.startswith('pip install "kitten-inference['):
            line = line.replace('pip install "kitten-inference[', f'pip install "{DIST}[', 1)
        out.append(line)
    return out


def units_to_steps(units, root):
    """List of (kind, line, unit_dir). kind is 'doc' or 'setup'."""
    steps, active = [], False
    top = doc_lines('top_tts08_block.sh')
    steps.append(('info', 'command -v python3; python3 --version; echo "shell: $0 $ZSH_VERSION$BASH_VERSION"', root))
    for u in units:
        d = root / u
        if u == 'top':
            d.mkdir(parents=True)
            (d / 'use_from_python.py').write_text((DOC / 'top_use_from_python_tts08.py').read_text(encoding='utf-8'), encoding='utf-8')
            if active:
                steps.append(('setup', 'deactivate', d))
            steps.append(('setup', f'cd {shlex.quote(str(d))}', d))
            steps += [('doc', l, d) for l in top]
            steps.append(('doc', 'python use_from_python.py', d))
            active = True
        elif u == 'pkg':
            d.mkdir(parents=True)
            (d / 'use.py').write_text((DOC / 'pkg_use.py').read_text(encoding='utf-8'), encoding='utf-8')
            if active:
                steps.append(('setup', 'deactivate', d))
            steps.append(('setup', f'cd {shlex.quote(str(d))}', d))
            steps += [('doc', l, d) for l in doc_lines('pkg_install_block.sh')]
            steps.append(('doc', 'python use.py', d))
            active = True
        elif u == 'claims':
            d = root / 'pkg'
            (d / 'claims.py').write_text((HERE / 'claims.py').read_text(encoding='utf-8'), encoding='utf-8')
            steps.append(('check', 'python claims.py', d))
        elif u == 'deep':
            # tts-0.8/README.md "Platforms": deep folders work. A user folder with a long path and a non-ASCII name.
            d = root / 'deep' / 'José' / ('x' * 60) / ('y' * 60) / ('z' * 60) / ('w' * 60)
            d.mkdir(parents=True)
            steps.append(('setup', f'cd {shlex.quote(str(d))}', d))
            steps += [('doc', l, d) for l in top]
        elif u == 'smoke':
            for l in ['echo one # two', 'setopt 2>/dev/null | tr "\\n" " "; echo', 'echo "$PATH"', 'tty']:
                steps.append(('info', l, root))
        elif u == 'oldpip':
            # README.md: "With a pip you cannot upgrade, write "kitten-inference[tts-0-8]"" (no pip upgrade line).
            d.mkdir(parents=True)
            steps.append(('setup', f'cd {shlex.quote(str(d))}', d))
            steps.append(('doc', top[0], d))
            steps.append(('doc', f'pip install "{DIST}[tts-0-8]"', d))
            steps.append(('doc', top[3], d))
            active = True
    return steps


def session_env(home, path_kind, shell):
    env = dict(os.environ)
    dropped = []
    for k in list(env):
        if k.startswith(('PYTHON', 'PIP_', 'VIRTUAL_ENV', 'HF_', 'HUGGINGFACE', 'XDG_', 'ZDOTDIR', 'CONDA', 'OMP_')):
            dropped.append(k)
            del env[k]
    print('@@INFO runner variables not passed to the session:', ' '.join(sorted(dropped)) or '(none)', flush=True)
    env.update(HOME=str(home), LANG='en_US.UTF-8', LC_ALL='en_US.UTF-8', TERM='xterm-256color')
    env['SHELL'] = '/bin/zsh' if shell.startswith('zsh') else '/bin/bash'
    if path_kind == 'system':
        env['PATH'] = SYSTEM_PATH
    if os.environ.get('SESSION_PIP_FIND_LINKS'):
        # Allowed deviation for this test release: the TTS 0.8 test wheels come from a folder, everything else from pypi.org.
        env['PIP_FIND_LINKS'] = os.environ['SESSION_PIP_FIND_LINKS']
        print('@@INFO deviation: PIP_FIND_LINKS=' + env['PIP_FIND_LINKS'], flush=True)
    env.pop('SESSION_PIP_FIND_LINKS', None)
    for k in ('DIST', 'GITHUB_TOKEN', 'ACTIONS_RUNTIME_TOKEN', 'ACTIONS_ID_TOKEN_REQUEST_TOKEN'):
        env.pop(k, None)
    return env


def log(fh, s):
    sys.stdout.write(s)
    sys.stdout.flush()
    fh.write(s.encode())
    fh.flush()


def run_script(steps, env, out, stop):
    lines = ['set +e']
    for i, (kind, line, d) in enumerate(steps, 1):
        lines.append(f'printf "%s\\n" {shlex.quote(f"@@STEP {i} {kind}: {line}")}')
        lines.append(f'__t=$(date +%s)')
        lines.append(line)
        lines.append(f'__rc=$?; echo "@@EXIT {i} $__rc $(( $(date +%s) - __t ))s"')
        if (stop and kind in ('doc', 'check')) or kind == 'setup':
            lines.append(f'[ $__rc -eq 0 ] || exit 100')
    script = out / 'session.sh'
    script.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with open(out / 'session.log', 'wb') as fh:
        p = subprocess.Popen(['/bin/bash', '--noprofile', '--norc', str(script)], env=env, cwd=env['HOME'],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for chunk in iter(lambda: p.stdout.read1(65536), b''):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            fh.write(chunk)
        rc = p.wait()
    text = (out / 'session.log').read_bytes().decode('utf-8', 'replace')
    return parse(text, steps), rc


def parse(text, steps):
    res = []
    for i, (kind, line, d) in enumerate(steps, 1):
        m = re.search(r'@@EXIT %d (\d+)(?: (\d+)s)?' % i, text)
        res.append({'i': i, 'kind': kind, 'line': line, 'rc': int(m.group(1)) if m else None,
                    'seconds': int(m.group(2)) if m and m.group(2) else None})
    return res


def run_pty(argv, steps, env, out, stop):
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(env['HOME'])
        os.execve(argv[0], argv, env)
    try:
        import fcntl, struct, termios
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack('HHHH', 50, 160, 0, 0))
    except Exception:
        pass
    fh = open(out / 'session.log', 'wb')
    buf = bytearray()

    def pump(pattern, timeout):
        rx = re.compile(pattern)
        end = time.time() + timeout
        while time.time() < end:
            m = rx.search(buf.decode('utf-8', 'replace'))
            if m:
                return m
            r, _, _ = select.select([fd], [], [], 1.0)
            if r:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    return None
                if not data:
                    return None
                buf.extend(data)
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
                fh.write(data)
                fh.flush()
        return None

    def send(s):
        os.write(fd, s.encode() + b'\n')

    send('echo "@@READY $((40+2))"')
    if not pump(r'@@READY 42', 60):
        log(fh, '\n@@HARNESS shell did not start\n')
    res = []
    for i, (kind, line, d) in enumerate(steps, 1):
        log(fh, f'\n@@STEP {i} {kind}: {line}\n')
        t = time.time()
        send(line)
        send(f'echo "@@EXIT {i} $?"')
        m = pump(r'@@EXIT %d (\d+)' % i, STEP_TIMEOUT)
        rc = int(m.group(1)) if m else None
        res.append({'i': i, 'kind': kind, 'line': line, 'rc': rc, 'seconds': round(time.time() - t)})
        if rc is None:
            log(fh, f'\n@@HARNESS step {i} timed out\n')
            break
        if rc != 0 and ((stop and kind in ('doc', 'check')) or kind == 'setup'):
            break
    send('exit')
    time.sleep(2)
    try:
        os.kill(pid, signal.SIGHUP)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    fh.close()
    return res, 0


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    name, out = sys.argv[1], Path(sys.argv[2]).resolve()
    shell, path_kind, units, stop = SESSIONS[name]
    root = Path(os.environ['RUNNER_TEMP']) / 'docs' / name
    home = root / 'home'
    home.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)
    env = session_env(home, path_kind, shell)
    steps = units_to_steps(units, root)
    print(f'@@SESSION {name} shell={shell} path={path_kind} home={home}', flush=True)
    who = subprocess.run(['/bin/bash', '--noprofile', '--norc', '-c',
                          'command -v python3; python3 --version; command -v pip pip3 || true'],
                         env=env, capture_output=True, text=True)
    print('@@INFO python3 for this shell before the doc lines (bash lookup):', who.stdout.replace('\n', ' | '), flush=True)
    if shell == 'bash-script':
        res, rc = run_script(steps, env, out, stop)
    else:
        argv = {'bash-pty': ['/bin/bash', '-i'], 'zsh-pty': ['/bin/zsh', '-i'],
                'zsh-login-pty': ['/bin/zsh', '-il']}[shell]
        res, rc = run_pty(argv, steps, env, out, stop)
    first_fail = next((r for r in res if r['rc'] != 0 and r['kind'] != 'info'), None)
    summary = {'session': name, 'shell': shell, 'path': path_kind, 'units': units, 'stop_on_fail': stop,
               'home': str(home), 'root': str(root), 'steps': res,
               'all_ran': len(res) == len(steps), 'first_failure': first_fail}
    (out / 'steps.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    print('\n@@RESULT ' + json.dumps({'session': name, 'first_failure': first_fail,
                                      'rcs': [r['rc'] for r in res], 'n_steps': len(steps)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
