#!/usr/bin/env python3
"""Apply one reviewed negative control to a disposable engine source file."""
import sys
from pathlib import Path

mutations = {
    'fallback_ignores_borrows': (
        'if (s->eid < 0 || s->borrows || (!allow_pinned && s->pinned)) continue;',
        'if (s->eid < 0 || (!allow_pinned && (s->borrows || s->pinned))) continue;'),
    'prefetch_ignores_borrows': (
        'int lru = expert_victim(lc, 0);',
        'int lru = -1; for (int i = 0; i < lc->n; i++) { '
        'if (lc->slots[i].eid < 0 || lc->slots[i].pinned) continue; '
        'if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i; }'),
    'prefetch_ignores_demand_claim': (
        'if (lc->demand_gathers) { m->is_queued[layer*c->n_experts+eid]=0; pthread_mutex_unlock(&g_pilot_mx); return; }',
        '/* Negative control: omit demand priority. */'),
    'release_only_once_per_distinct_slot': (
        'if (!slots[i]->borrows) abort();\n        slots[i]->borrows--;',
        'if (!slots[i]->borrows) abort();\n'
        '        for (int j = i + 1; j < n; j++) if (slots[j] == slots[i]) slots[j] = NULL;\n'
        '        slots[i]->borrows--;'),
}
if len(sys.argv) != 4 or sys.argv[1] not in mutations:
    raise SystemExit('usage: qwen36-mutate.py MUTATION PATCHED_SOURCE DISPOSABLE_OUTPUT')
source_path, target_path = Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
if source_path == target_path:
    raise SystemExit('Refusing to modify the checked-out source')
source = source_path.read_text(encoding='utf-8')
old, new = mutations[sys.argv[1]]
if source.count(old) != 1:
    raise SystemExit('Mutation anchor does not match exactly once')
# Explicit LF output keeps source bytes independent of host newline conventions.
target_path.write_bytes(source.replace(old, new).encode('utf-8'))
