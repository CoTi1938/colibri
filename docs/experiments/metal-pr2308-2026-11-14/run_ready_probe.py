"""User-requested first A+B diagnostic, not an automatic eight-run repeat."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
base = root / 'measurements-gguf'
assert 'ALL_BLOCKS_COLLECTED_REVIEW_REQUIRED' in (root / 'measurements-gguf.log').read_text()
original = json.loads((base / 'code-cold512.json').read_text())
assert original['returncode'] == 0
name = 'code-cold512-probe1'
target = base / name
assert not target.exists(), 'Inspect existing probe; no overwrite'
command = original['command'][:]
assert Path(command[3]) == base / 'code-cold512'
command[3] = str(target)
command[command.index('--runs') + 1] = '1'
env = {'PATH': os.environ['PATH'], 'HOME': os.environ['HOME'], 'PYTHONUNBUFFERED': '1'}
awake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
record = {'command': command, 'start': time.time(), 'status': 'running',
          'reason': 'User requests first A+B inspection, no automatic full8. Diagnostic only; retain original4+4 as primary.'}
with (base / (name + '.log')).open('x') as log:
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    record['pid'] = process.pid
    (base / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    state = json.loads((root / 'STATE.json').read_text())
    state.update(phase='measurements-gguf', current_block=name,
                 active_processes=[{'role':'ready-probe-controller','pid':os.getpid()}, {'role':name,'pid':process.pid}],
                 awake_guard={'command':'caffeinate -i -w ' + str(os.getpid()),'pid':awake.pid})
    (root / 'STATE.json').write_text(json.dumps(state, indent=2) + '\n')
    record['returncode'] = process.wait()
record.update(end=time.time(), status='completed' if record['returncode'] == 0 else 'failed')
(base / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
assert record['returncode'] == 0
assert len(list(target.glob('v*-run*.csv'))) == 2
print('FIRST_AB_READY_PROBE_COMPLETE_REVIEW_BEFORE_ANY_MORE_RUNS', flush=True)
