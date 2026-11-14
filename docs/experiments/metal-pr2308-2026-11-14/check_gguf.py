"""Bounded real-GGUF functional qualification; never a throughput measurement.

Protocol: same 16-token prefix and four forced steps as checkpoint, exact A/B
bit equality, CPU exact with f32 KV against unchanged 2e-4 screening tolerance.
Independent lock-byte prediction uses the real tensor table and runtime mapping.
"""
import array
import importlib.util
import json
import math
import mmap
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models/gguf'
OUT = ROOT / 'real-gguf'
assert (ROOT / 'gguf-preparation/import-verified.json').exists()
OUT.mkdir()
BASE_ENV = {'PATH': os.environ['PATH'], 'HOME': os.environ['HOME'], 'QWEN_EXPERT_CACHE_BYTES': '536870912'}
ENGINE = ROOT / 'worktrees/candidate/c/qwen36'

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')

def run(name, command, extra=None, timeout=1800):
    dest = OUT / name
    dest.mkdir()
    env = dict(BASE_ENV, **(extra or {}))
    record = {'command': [str(x) for x in command], 'environment': env,
              'start': time.time(), 'kind': 'functional_check_not_performance'}
    with (dest / 'stdout.txt').open('x') as stdout, (dest / 'stderr.txt').open('x') as stderr:
        process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        record['pid'] = process.pid
        save(dest / 'running.json', record)
        print('START', name, process.pid, flush=True)
        try:
            record['returncode'] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            record.update(returncode=process.returncode, timeout=True)
        record['end'] = time.time()
        save(dest / 'result.json', record)
    print('END', name, record['returncode'], flush=True)
    assert record['returncode'] == 0 and not record.get('timeout'), name
    return dest

def read_logits(path):
    with path.open('rb') as source:
        header = source.readline().decode().strip()
        values = array.array('f'); values.frombytes(source.read())
    assert values and all(math.isfinite(x) or x == -math.inf for x in values)
    return header, values

def compare(left, right):
    ah, a = read_logits(left); bh, b = read_logits(right)
    assert ah == bh and len(a) == len(b)
    assert all(math.isfinite(x) == math.isfinite(y) for x, y in zip(a, b))
    finite = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    gap = max(abs(x-y) for x, y in finite)
    scale = max(abs(x) for x, y in finite)
    return {'header': ah, 'values': len(a), 'bitwise_equal': left.read_bytes() == right.read_bytes(),
            'max_abs': gap, 'relative_max': gap / max(scale, 1e-30),
            'different_values': sum(x != y for x, y in zip(a, b))}

save(OUT / 'protocol.json', {'scope': '16 prompt tokens + 4 forced steps; small screening, not whole-model quality certification',
     'exact_ab': 'bitwise equality', 'cpu_reference': 'COLI_METAL=0 GQ_EXACT=1 Q36_KV=f32',
     'cpu_relative_max_threshold': 2e-4, 'fast_ab': 'Record and diagnose any new difference',
     'locks': 'Exact sum of page-rounded mapped dense tensor ranges, excluding F32 copies; balanced unlocks and GPU product counts >0',
     'cli': [8, 1], 'stateful': 'Same 10 comparisons as checkpoint, pin0/1, three turns and reset'})

# Parse the imported real model, rather than using fixture dimensions/names.
spec = importlib.util.spec_from_file_location('campaign_gguf_import', ROOT / 'worktrees/candidate/c/tools/gguf_import.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
mapping = json.loads((MODEL / 'gguf.json').read_text())
path = MODEL / mapping['gguf']
parts, readers, meta, tensors = module.open_model(path)
for reader in readers:
    reader.close()
assert len(parts) == 1 and meta['general.architecture'] == 'qwen35moe'
cfg = json.loads((MODEL / 'config.json').read_text())
cfg = cfg.get('text_config', cfg)
types = cfg['layer_types']
assert len(types) == cfg['num_hidden_layers']
names = ['output.weight' if 'output.weight' in tensors else 'token_embd.weight']
common = ['ffn_gate_inp', 'ffn_gate_shexp', 'ffn_up_shexp', 'ffn_down_shexp']
for index, kind in enumerate(types):
    assert kind in ['full_attention', 'linear_attention']
    own = ['attn_q', 'attn_k', 'attn_v', 'attn_output'] if kind == 'full_attention' else ['attn_qkv', 'attn_gate', 'ssm_out']
    names.extend(f'blk.{index}.{name}.weight' for name in own + common)
page = mmap.PAGESIZE
ranges = []
for name in names:
    t = tensors[name]
    start, end = t['abs_offset'], t['abs_offset'] + t['nbytes']
    ranges.append({'name': name, 'ggml_type': t['ggml_type'], 'offset': start, 'bytes': t['nbytes'],
                   'locked_bytes': 0 if t['ggml_type'] == 0 else ((end + page-1) // page - start // page) * page})
expected = sum(t['locked_bytes'] for t in ranges)
mapped = sum(t['bytes'] for t in ranges if t['ggml_type'] != 0)
save(OUT / 'expected-lock-ranges.json', {'page_bytes': page, 'locked_bytes_sum_not_unique_pages': expected,
     'mapped_dense_bytes': mapped, 'ranges': ranges,
     'source': 'qwen36.c gg_name mapping; qwen36_graph.h qg_gguf_pages and qg_dense; real GGUF tensor table'})

products = re.compile(r'GGUF: (\d+) products with the file\'s dense matrices and (\d+) with its routed experts ran on the GPU')
locked = re.compile(r'GGUF: the dense matrices locked in memory \((\d+) bytes\)')
counts = re.compile(r'GGUF: the dense matrices locked for (\d+) prompts, unlocked after (\d+) first decode steps and at the end of (\d+) requests; (\d+) still locked at exit')
def check_counters(dest, n_new):
    err = (dest / 'stderr.txt').read_text()
    assert 'graph ready' in err and 'dense=mapped' in err
    assert 'could not be locked' not in err
    p, l, c = products.search(err), locked.search(err), counts.search(err)
    assert p and l and c, err[-4000:]
    assert all(int(x) > 0 for x in p.groups())
    assert int(l.group(1)) == expected
    assert f'({mapped} bytes mapped)' in err
    got = tuple(map(int, c.groups()))
    assert got == ((1, 1, 0, 0) if n_new >= 2 else (1, 0, 1, 0)), got
    return {'gpu_products_dense_expert': list(map(int, p.groups())), 'locked_bytes': int(l.group(1)),
            'locks_decode_unlock_end_unlock_remaining': got}

prompt = OUT / 'smoke-prompt.txt'; prompt.write_text('The capital of Italy is')
diagnostics = {}
for n_new in [8, 1]:
    name = 'cli-B-' + str(n_new)
    dest = run(name, [ENGINE, '8', '4', prompt], {'SNAP': str(MODEL), 'N_NEW': str(n_new),
               'ENC_DEBUG': '1', 'DUMP': str(OUT / ('smoke-dump-' + str(n_new))), 'COLI_TIMERS': '1'})
    diagnostics[name] = check_counters(dest, n_new)
save(OUT / 'cli-lock-results.json', diagnostics)

cases = [('A-exact', 'A', {'QWEN36_METAL_EXACT': '1'}), ('B-exact', 'B', {'QWEN36_METAL_EXACT': '1'}),
         ('B-cpu', 'B', {'COLI_METAL': '0', 'GQ_EXACT': '1', 'Q36_KV': 'f32'}),
         ('A-fast', 'A', {}), ('B-fast', 'B', {})]
for name, variant, env in cases:
    dest = OUT / name
    run(name, [ROOT / ('binaries/' + variant + '/qwen36-bench'), '--model', MODEL,
              '--text', ROOT / 'harness/promessi_sposi.txt', '--lengths', '16', '--gen', '4',
              '--teacher', '--nll', dest / 'nll.txt', '--logits', dest / 'logits.bin', '--steps', dest / 'steps.txt'], env)
    err = (dest / 'stderr.txt').read_text()
    assert ('[bench] CPU ready' if name == 'B-cpu' else '[bench] Metal graph ready') in err
results = {}
for left, right in [('A-exact', 'B-exact'), ('B-exact', 'B-cpu'), ('A-fast', 'B-fast'), ('B-exact', 'B-fast')]:
    results[left + '__' + right] = compare(OUT / left / 'logits.bin', OUT / right / 'logits.bin')
save(OUT / 'comparisons.json', results)
assert results['A-exact__B-exact']['bitwise_equal'], 'Exact A/B differs; diagnose'
assert results['B-exact__B-cpu']['relative_max'] < 2e-4, 'CPU screen failed; diagnose with unchanged threshold'
assert results['A-fast__B-fast']['bitwise_equal'], 'Fast A/B differs; review before timing'
run('stateful-driver', [sys.executable, ROOT / 'check_stateful.py', '--model', MODEL, '--engine', ENGINE,
                      '--out', OUT / 'stateful'], timeout=1800)
print('GGUF_FUNCTIONAL_COMPLETE_ROOT_REVIEW_REQUIRED', flush=True)
