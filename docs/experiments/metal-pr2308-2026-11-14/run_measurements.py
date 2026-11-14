"""Approved A/B blocks; run only after root has reviewed real-model correctness."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--format', choices=['checkpoint', 'gguf'], required=True)
p.add_argument('--model', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent
assert (root / (a.format + '-ready-for-measurements.json')).exists(), 'Functional review required'
model = str(Path(a.model).resolve())
out = root / ('measurements-' + a.format)
out.mkdir(exist_ok=True)
env = {'PATH': os.environ['PATH'], 'HOME': os.environ['HOME'], 'PYTHONUNBUFFERED': '1'}
for text in ['prose', 'code', 'chat']:
    for cohort, lengths in [('sequence', '64,512,64'), ('cold512', '512')]:
        name = text + '-' + cohort
        target = out / name
        assert not target.exists(), 'Do not overwrite/repeat a measurement: ' + str(target)
        assert shutil.disk_usage(root).free >= 15000000000
        command = [sys.executable, str(root / 'harness/bench'), 'run', str(target),
                   'A=' + str(root / 'binaries/A/qwen36-bench'),
                   'B=' + str(root / 'binaries/B/qwen36-bench'),
                   '--model', model, '--text', text, '--lengths', lengths,
                   '--gen', '32', '--runs', '4', '--rest', '60', '--env-only',
                   '--env', 'QWEN_EXPERT_CACHE_BYTES=536870912']
        start = time.time()
        record = {'command': command, 'start': start, 'format': a.format,
                  'cohort': cohort, 'text': text, 'status': 'running'}
        with (out / (name + '.log')).open('x') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            record['pid'] = process.pid
            (out / (name + '.json')).write_text(json.dumps(record, indent=2))
            state = json.loads((root / 'STATE.json').read_text())
            state.update(phase='measurements-' + a.format, current_block=name,
                         active_processes=[{'role': 'measurement-controller', 'pid': os.getpid()},
                                           {'role': name, 'pid': process.pid}])
            (root / 'STATE.json').write_text(json.dumps(state, indent=2) + '\n')
            print('START', name, process.pid, flush=True)
            record['returncode'] = process.wait()
        record.update(end=time.time(), status='completed' if record['returncode'] == 0 else 'failed')
        (out / (name + '.json')).write_text(json.dumps(record, indent=2))
        print('END', name, record['returncode'], flush=True)
        assert record['returncode'] == 0, 'Inspect preserved block before continuing'
        files = list(target.glob('v*-run*.csv'))
        assert len(files) == 8
        expected = list(map(int, lengths.split(',')))
        for f in files:
            rows = list(csv.DictReader(f.open()))
            assert [int(r['ctx_tokens']) for r in rows] == expected
            assert all(int(r['gen_tokens']) == 32 for r in rows)
            stderr = f.with_suffix('.err').read_text()
            assert '[bench] Metal graph ready' in stderr
            before = (target / (f.stem + '.thermal-before.txt')).read_text().splitlines()
            assert before[1] == '0', 'Non-nominal thermal start'
print('ALL_BLOCKS_COLLECTED_REVIEW_REQUIRED', flush=True)
