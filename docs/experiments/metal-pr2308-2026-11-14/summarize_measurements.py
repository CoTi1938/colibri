"""Report raw samples, ranges and scoped counters without inventing missing values."""
import csv
import json
from pathlib import Path
import re
import statistics
import sys

root = Path(sys.argv[1]).resolve()
samples = []
flags_path = root / 'quality-flags.json'
flags = json.loads(flags_path.read_text()) if flags_path.exists() else {}
exclusions = {(f['block'], f['run']): f['reason'] for f in flags.get('exclude_from_primary', [])}
def match_number(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

for block in sorted(root.iterdir()):
    if not block.is_dir():
        continue
    for path in sorted(block.glob('v*-run*.csv')):
        variant, run = re.match(r'v(\d)-run(\d+)', path.stem).groups()
        stderr = path.with_suffix('.err').read_text()
        before = (block / (path.stem + '.thermal-before.txt')).read_text()
        after_path = block / (path.stem + '.thermal-after.txt')
        if not after_path.exists():
            continue  # never analyze a partially written running measurement
        driver_log = root / (block.name + '.log')
        if driver_log.exists() and path.stem + ': exit 0' not in driver_log.read_text():
            continue
        after = after_path.read_text()
        start_swap = match_number(r'swapouts=(\d+)', before)
        end_swap = match_number(r'swapouts=(\d+)', after)
        base = {'block': block.name, 'variant': 'AB'[int(variant)], 'run': int(run),
                'excluded_reason': exclusions.get((block.name, path.stem)),
                'model_ready_s': match_number(r'\[bench\] Metal graph ready in ([\d.]+) s', stderr),
                'max_rss_bytes': match_number(r'(\d+)\s+maximum resident set size', stderr),
                'peak_footprint_bytes': match_number(r'(\d+)\s+peak memory footprint', stderr),
                'swapout_pages_system_delta': end_swap-start_swap if None not in (start_swap, end_swap) else None,
                'thermal_before': before.splitlines()[1] if len(before.splitlines()) > 1 else None,
                'thermal_after': after.splitlines()[1] if len(after.splitlines()) > 1 else None}
        for point in ['before', 'after']:
            power = block / (path.stem + '.power-' + point + '.txt')
            base['power_' + point] = power.read_text().strip() if power.exists() else None
        for index, row in enumerate(csv.DictReader(path.open())):
            item = dict(base, row_index=index, fingerprint=row['fingerprint'], tokens=row['tokens'])
            for key, value in row.items():
                if key not in ('fingerprint', 'tokens'):
                    item[key] = float(value)
            item['second_token_from_ready_s'] = item['prefill_s'] + item['gen_first_ms'] / 1000
            samples.append(item)

metrics = ['model_ready_s', 'prefill_s', 'prefill_tps', 'gen_first_ms', 'second_token_from_ready_s',
           'gen_steady_tps', 'max_rss_bytes', 'peak_footprint_bytes', 'swapout_pages_system_delta',
           'expert_miss', 'disk_s', 'nll', 'top1']
summary = []
for block, index in sorted(set((s['block'], s['row_index']) for s in samples)):
    record = {'block': block, 'row_index': index, 'variants': {}, 'changes_percent': {}, 'repeat_candidates': []}
    for variant in 'AB':
        group = [s for s in samples if (s['block'], s['row_index'], s['variant']) == (block, index, variant)
                 and not s['excluded_reason']]
        record['variants'][variant] = {'count': len(group), 'metrics': {},
                                      'fingerprints': sorted(set(s['fingerprint'] for s in group)),
                                      'token_hashes': sorted(set(s['tokens'] for s in group))}
        for metric in metrics:
            values = [s[metric] for s in group if s.get(metric) is not None]
            record['variants'][variant]['metrics'][metric] = (
                {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'samples': values}
                if values else None)
    record['complete'] = all(record['variants'][v]['count'] == 4 for v in 'AB')
    for metric in metrics:
        av = record['variants']['A']['metrics'][metric]
        bv = record['variants']['B']['metrics'][metric]
        if av and bv and av['median']:
            change = (bv['median'] / av['median'] - 1) * 100
            record['changes_percent'][metric] = change
            if record['complete'] and metric in ('prefill_s', 'gen_first_ms', 'model_ready_s') and change > 5:
                record['repeat_candidates'].append(metric)
            if record['complete'] and metric == 'gen_steady_tps' and change < -5:
                record['repeat_candidates'].append(metric)
    summary.append(record)
result = {'samples': samples, 'summary': summary,
          'limits': ['Warm-up excluded. Missing values are null, never zero.',
                     'RSS/footprint and thermal/power/swap snapshots cover the whole process; sequence rows repeat these process-wide values.',
                     'Ready excludes tokenizer/tokenization. First decode predicts second token.',
                     'Second-token time is the per-run sum of timed prefill and first forward, excluding untimed bookkeeping between them; not end-to-end serving latency.',
                     'Steady covers forward steps2-32; last predicts unused33rd token.',
                     'Expert misses are logical cache misses; disk_s sums overlapping loader work.',
                     'Swap is system-wide; cold means engine/expert cache, not purged filesystem.',
                     'Repeat candidates require checking noise, power, thermal state and swap first.']}
(root / 'aggregate.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'samples': len(samples), 'cells': len(summary),
                  'repeat_candidates': [(x['block'], x['row_index'], x['repeat_candidates'])
                                        for x in summary if x['repeat_candidates']]}))
