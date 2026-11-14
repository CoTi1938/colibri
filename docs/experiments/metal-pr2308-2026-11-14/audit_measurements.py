"""Audit completed process records and paired outputs; no timing or promotion."""
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
fmt = sys.argv[1]
assert fmt in ('checkpoint', 'gguf')
base = ROOT / ('measurements-' + fmt)
aggregate = json.loads((base / 'aggregate.json').read_text())
blocks = collections.defaultdict(list)
for sample in aggregate['samples']:
    blocks[sample['block']].append(sample)
audit = {'format': fmt, 'blocks': {}, 'scope': 'Checks process conditions and paired scalar fingerprints; not an independent full-logit comparison or a performance promotion.'}
for name, rows in sorted(blocks.items()):
    record = json.loads((base / (name + '.json')).read_text())
    assert record['status'] == 'completed' and record['returncode'] == 0, name
    processes = {(r['variant'], r['run']): r for r in rows}
    expected_processes = 2 if name == 'code-cold512-probe1' else 8
    assert len(processes) == expected_processes, (name, len(processes))
    expected_rows = 3 if 'sequence' in name else 1
    assert len(rows) == expected_processes * expected_rows
    paired = []
    for i in range(expected_rows):
        point = [r for r in rows if r['row_index'] == i]
        equal = {k: len({r[k] for r in point}) == 1 for k in ['fingerprint', 'tokens', 'nll', 'top1']}
        assert all(equal.values()), (name, i, equal)
        assert all(r['gen_tokens'] == 32 for r in point)
        assert all(r['ctx_tokens'] == ([64,512,64][i] if expected_rows == 3 else 512) for r in point)
        paired.append({'row_index': i, 'equal_across_all_runs': equal})
    conditions = []
    for (variant, run), r in sorted(processes.items()):
        conditions.append({'variant': variant, 'run': run,
                           'excluded_reason': r['excluded_reason'],
                           'thermal_nominal': r['thermal_before'] == r['thermal_after'] == '0',
                           'system_swapout_delta': r['swapout_pages_system_delta'],
                           'ac_before_and_after': all("AC Power" in (r['power_' + point] or '') for point in ['before','after'])})
    clean = [c for c in conditions if not c['excluded_reason']]
    assert all(c['thermal_nominal'] and c['system_swapout_delta'] == 0 and c['ac_before_and_after'] for c in clean), (name, conditions)
    audit['blocks'][name] = {'processes': expected_processes, 'csv_rows': len(rows), 'paired': paired, 'conditions': conditions,
                             'excluded_warmups': len(list((base / name).glob('warmup.csv')))}
audit['timed_processes'] = sum(b['processes'] for b in audit['blocks'].values())
audit['csv_rows'] = len(aggregate['samples'])
audit['aggregate_sha256'] = hashlib.sha256((base / 'aggregate.json').read_bytes()).hexdigest()
(base / 'measurement-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({'format': fmt, 'blocks': len(blocks), 'processes': audit['timed_processes'], 'rows': audit['csv_rows'], 'status': 'audit passed'}))
