"""One clean repetition of flagged blocks, with unchanged saved commands/settings."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
format_name, *blocks = sys.argv[1:]
assert format_name in ('checkpoint', 'gguf') and blocks
base = root / ('measurements-' + format_name)
assert 'ALL_BLOCKS_COLLECTED_REVIEW_REQUIRED' in (root / ('measurements-' + format_name + '.log')).read_text()
env = {'PATH': os.environ['PATH'], 'HOME': os.environ['HOME'], 'PYTHONUNBUFFERED': '1'}
awake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
for block in blocks:
    original = json.loads((base / (block + '.json')).read_text())
    assert original['returncode'] == 0
    name = block + '-repeat1'
    target = base / name
    assert not target.exists(), 'A repeat already exists; no blind retries'
    command = original['command'][:]
    assert command[2] == 'run' and Path(command[3]) == base / block
    command[3] = str(target)
    record = {'command': command, 'source_record': block + '.json', 'start': time.time(),
              'reason': 'Prespecified single repeat; original data retained', 'status': 'running'}
    with (base / (name + '.log')).open('x') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = process.pid
        (base / (name + '.json')).write_text(json.dumps(record, indent=2))
        state = json.loads((root / 'STATE.json').read_text())
        state.update(phase='measurements-' + format_name, current_block=name,
                     active_processes=[{'role': 'recheck-controller', 'pid': os.getpid()},
                                       {'role': name, 'pid': process.pid}],
                     awake_guard={'command': 'caffeinate -i -w ' + str(os.getpid()), 'pid': awake.pid})
        (root / 'STATE.json').write_text(json.dumps(state, indent=2) + '\n')
        print('START', name, process.pid, flush=True)
        record['returncode'] = process.wait()
    record.update(end=time.time(), status='completed' if record['returncode'] == 0 else 'failed')
    (base / (name + '.json')).write_text(json.dumps(record, indent=2))
    print('END', name, record['returncode'], flush=True)
    assert record['returncode'] == 0
    files = list(target.glob('v*-run*.csv'))
    expected = list(map(int, command[command.index('--lengths') + 1].split(',')))
    assert len(files) == 8
    for f in files:
        rows = list(csv.DictReader(f.open()))
        assert [int(r['ctx_tokens']) for r in rows] == expected
        assert all(int(r['gen_tokens']) == 32 for r in rows)
        assert '[bench] Metal graph ready' in f.with_suffix('.err').read_text()
        assert (target / (f.stem + '.thermal-before.txt')).read_text().splitlines()[1] == '0'
print('RECHECKS_COLLECTED_REVIEW_REQUIRED', flush=True)
