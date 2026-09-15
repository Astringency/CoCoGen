"""Balance the unchanged 60 evaluation cells over two GPUs within 24 hours."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import time

from .calibrate import implementation_hash
from .common import STUDY, sha_file, write_json
from .inputs import records_for
from .schedule_24h import PDES, allocation, wait_success


def code_identity():
    root = Path(__file__).parent
    return {name: sha_file(root/name) for name in ['cell_schedule.py', 'cell_worker.py', 'cell24_job.sh']}


def assignment_hash(assignment):
    return hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest()


def cell_allocation(hours, records):
    """Exhaust all 16^4 per-PDE cell counts; each PDE has 15 equal-cost cells.

    Cost comes from full held-out sampling. Cells retain their original inputs
    and sampler. Prefer fewer model splits when projected makespans tie.
    """
    by_pde = {pde: sorted(r['cell'] for r in records if r['pde'] == pde) for pde in PDES}
    assert all(len(cells) == 15 and len(set(cells)) == 15 for cells in by_pde.values())
    assert len(records) == len({r['cell'] for r in records}) == 60
    assert all(hours[pde] > 0 for pde in PDES)
    best = None
    for counts in itertools.product(range(16), repeat=4):
        if sum(counts) in (0, 60):
            continue
        totals = [sum(hours[pde]*n/15 for pde, n in zip(PDES, counts)),
                  sum(hours[pde]*(15-n)/15 for pde, n in zip(PDES, counts))]
        key = (round(max(totals), 12), sum(0 < n < 15 for n in counts), abs(sum(counts)-30), counts)
        if best is None or key < best[0]:
            best = (key, counts, totals)
    _, counts, totals = best
    groups = [[], []]
    for pde, n in zip(PDES, counts):
        groups[0].extend(by_pde[pde][:n])
        groups[1].extend(by_pde[pde][n:])
    assert not set(groups[0]) & set(groups[1])
    assert set(groups[0]+groups[1]) == {r['cell'] for r in records}
    return groups, totals


def schedule(study, hours_limit=24., margin=1.15):
    study = Path(study)
    out = study/'protocol/budget_24h.json'
    write_json(out, dict(status='waiting_for_calibration', hours_limit=hours_limit, margin=margin,
        scheduling_unit='evaluation cell', first_phase_cells=60, samples_per_cell=1000,
        user_preference='约 1 天，优先效果与耗时的平衡'))
    wait_success(study/'logs'/f'calibrate_cost_{group}.exit' for group in ('ph', 'dn'))
    inputs = json.loads((study/'validation/inputs.json').read_text())
    assert inputs['status'] == 'passed' and inputs['count'] == 66
    assert json.loads((study/'validation/evaluator.json').read_text())['status'] == 'passed'
    checks = json.loads((study/'validation/cell_budget.json').read_text())
    assert checks['status'] == 'passed' and checks['code'] == code_identity()
    selected = {pde: json.loads((study/'protocol/selected'/f'{pde}.json').read_text()) for pde in PDES}
    for row in selected.values():
        assert row['status'] == 'validated' and row['implementation_sha256'] == implementation_hash()
        assert row['config']['steps']*(1+row['config']['repaint']) <= 500
    records = [r for pde in PDES for r in records_for(pde, study)]
    hours = {pde: row['estimated_main_gpu_hours'] for pde, row in selected.items()}
    cells, totals = cell_allocation(hours, records)
    whole_groups, whole_totals = allocation(hours)
    assignment = dict(gpu_cells=cells,
        selected_sha256={pde: sha_file(study/'protocol/selected'/f'{pde}.json') for pde in PDES},
        implementation_sha256=implementation_hash(), code=code_identity())
    projected = max(totals)*margin
    plan = dict(status='ready' if projected <= hours_limit else 'needs_lower_cost',
        hours_limit=hours_limit, margin=margin, estimated_hours_with_margin=projected,
        estimated_gpu_hours=hours, gpu_hours=totals, gpu_cell_counts=[len(x) for x in cells],
        first_phase_cells=60, samples_per_cell=1000, scheduling_unit='evaluation cell',
        user_preference='约 1 天，优先效果与耗时的平衡',
        whole_pde_comparison=dict(gpu_groups=whole_groups, gpu_hours=whole_totals,
            estimated_hours_with_margin=max(whole_totals)*margin),
        assignment=assignment, assignment_sha256=assignment_hash(assignment))
    write_json(out, plan)
    print(json.dumps(plan), flush=True)
    if projected > hours_limit:
        raise RuntimeError('Measured projection exceeds the user budget; formal sampling was not started')
    code = Path(__file__).resolve().parents[1]
    for gpu in (0, 1):
        name = f'cocogen_cell24_gpu{gpu}'
        assert subprocess.run(['tmux', 'has-session', '-t', name], capture_output=True).returncode != 0, name
        assert not (study/'logs'/f'cell24_gpu{gpu}.exit').exists(), 'Previous job exit needs review'
    started = time.time()
    plan.update(status='running', started_unix=started)
    write_json(out, plan)
    for gpu in (0, 1):
        command = shlex.join(['bash', str(code/'cocogen_eval/cell24_job.sh'), f'gpu{gpu}',
                             'cocogen_eval.cell_worker', '--worker-id', str(gpu), '--device', f'cuda:{gpu}'])
        subprocess.run(['tmux', 'new-session', '-d', '-s', f'cocogen_cell24_gpu{gpu}', command], check=True)
    wait_success(study/'logs'/f'cell24_gpu{gpu}.exit' for gpu in (0, 1))
    from .cell_worker import finish
    finish(study)
    plan.update(status='first60_complete', finished_unix=time.time(),
        actual_main_hours=(time.time()-started)/3600, completion_receipt=str(study/'reports/first60_complete.json'))
    write_json(out, plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=STUDY)
    parser.add_argument('--hours-limit', type=float, default=24.)
    parser.add_argument('--margin', type=float, default=1.15)
    args = parser.parse_args()
    schedule(args.study, args.hours_limit, args.margin)
