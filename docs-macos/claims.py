# Tester checks of what tts-0.8/README.md says, using only the API it documents. Not a doc command.
# Run inside the venv made by the README's Install block, after its Use block.
import json, platform, statistics, sys, time, traceback

import numpy as np
from kitten_inference.tts08 import KittenTTS

SR = 24000
out = {'python': sys.version.split()[0], 'machine': platform.machine(), 'mac_ver': platform.mac_ver()[0]}


def info(a):
    a = np.asarray(a)
    return {'dtype': str(a.dtype), 'ndim': a.ndim, 'samples': int(a.shape[-1]), 'seconds': round(a.shape[-1] / SR, 3),
            'rms': round(float(np.sqrt(np.mean(a.astype(np.float64) ** 2))), 5), 'finite': bool(np.isfinite(a).all())}


def check(name, fn):
    t = time.time()
    try:
        out[name] = fn()
    except Exception as e:
        out[name] = {'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-1500:]}
    print(f'{name} ({time.time() - t:.1f} s):', json.dumps(out[name])[:600], flush=True)


def timed(tts, text, voice, n=5):
    tts.generate(text, voice=voice)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        a = tts.generate(text, voice=voice)
        ts.append(time.perf_counter() - t)
    return {'median_s': round(statistics.median(ts), 4), 'audio_s': round(a.shape[-1] / SR, 3),
            'rtf': round(statistics.median(ts) / (a.shape[-1] / SR), 4)}


TEXT = 'This high-quality voice runs without a GPU.'
VOICES = ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']
models = {}


def load(name, **kw):
    t = time.time()
    m = KittenTTS(name, **kw)
    return m, round(time.time() - t, 2)


def c_voices():
    m, s = load('KittenML/kitten-tts-mini-0.8')
    models['mini'] = m
    v = list(m.available_voices)
    return {'available_voices': v, 'same_as_readme': sorted(v) == sorted(VOICES), 'load_s': s}


def c_generate():
    a = models['mini'].generate(TEXT, voice='Jasper')
    return dict(info(a), float32=str(np.asarray(a).dtype) == 'float32')


def c_every_voice():
    return {v: info(models['mini'].generate('Hello there.', voice=v))['seconds'] for v in VOICES}


def c_speed():
    m = models['mini']
    a = m.generate('Hello, world.', voice='Luna', speed=1.0)
    b = m.generate('Hello, world.', voice='Luna', speed=1.2)
    return {'speed_1.0_s': info(a)['seconds'], 'speed_1.2_s': info(b)['seconds'], 'faster': info(b)['samples'] < info(a)['samples']}


def c_clean_text():
    m = models['mini']
    t = 'The ticket costs $25 and the trip is 12 km.'
    a = m.generate(t, voice='Bella', clean_text=True)
    b = m.generate(t, voice='Bella')
    return {'clean_text_s': info(a)['seconds'], 'plain_s': info(b)['seconds'], 'rms': info(a)['rms']}


def c_stream():
    chunks = list(models['mini'].generate_stream('This is the first sentence. And here is the second one.', voice='Hugo'))
    return {'chunks': len(chunks), 'seconds': [info(c)['seconds'] for c in chunks], 'dtypes': sorted({str(np.asarray(c).dtype) for c in chunks})}


def c_all_models():
    r = {}
    for n in ['KittenML/kitten-tts-micro-0.8', 'KittenML/kitten-tts-nano-0.8', 'KittenML/kitten-tts-mini-0.8-int8',
              'KittenML/kitten-tts-micro-0.8-int8', 'KittenML/kitten-tts-nano-0.8-int8']:
        m, s = load(n)
        r[n] = dict(info(m.generate(TEXT, voice='Kiki')), load_s=s)
        del m
    return r


def c_backends():
    # README: Apple Silicon uses Metal for the full-size models and the CPU for int8 by default; backend= is respected;
    # Metal and CPU audio can differ slightly (clip lengths within about 3.5%).
    r = {}
    for base in ['KittenML/kitten-tts-nano-0.8', 'KittenML/kitten-tts-nano-0.8-int8']:
        d, _ = load(base)
        me, _ = load(base, backend='metal')
        cp, _ = load(base, backend='cpu')
        a, b, c = (np.asarray(x.generate(TEXT, voice='Jasper')) for x in (d, me, cp))
        r[base] = {'default': info(a), 'metal': info(b), 'cpu': info(c),
                   'default_equals_metal': a.shape == b.shape and bool(np.array_equal(a, b)),
                   'default_equals_cpu': a.shape == c.shape and bool(np.array_equal(a, c)),
                   'metal_vs_cpu_length_pct': round(100.0 * (b.shape[-1] - c.shape[-1]) / c.shape[-1], 2)}
        r[base]['timing_default'] = timed(d, TEXT, 'Jasper')
        r[base]['timing_metal'] = timed(me, TEXT, 'Jasper')
        r[base]['timing_cpu'] = timed(cp, TEXT, 'Jasper')
        del d, me, cp
    return r


for name, fn in [('voices', c_voices), ('generate', c_generate), ('every_voice', c_every_voice), ('speed', c_speed),
                 ('clean_text', c_clean_text), ('generate_stream', c_stream), ('all_models', c_all_models),
                 ('backends', c_backends)]:
    check(name, fn)

with open('claims.json', 'w') as f:
    json.dump(out, f, indent=1)
errors = [k for k, v in out.items() if isinstance(v, dict) and 'error' in v]
print('CLAIMS errors:', errors or 'none')
sys.exit(1 if errors else 0)
