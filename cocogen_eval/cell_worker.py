"""Evaluate a disjoint assignment of original main cells with the frozen sampler."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .calibrate import implementation_hash
from .common import STUDY, load_network, sha_file, write_json
from .evaluate import aggregate, evaluate_cell
from .inputs import records_for
from .schedule_24h import PDES


def load_assignment(study):
    from .cell_schedule import assignment_hash, code_identity
    study = Path(study)
    plan = json.loads((study/'protocol/budget_24h.json').read_text())
    assert plan['status'] in ('running', 'first60_complete')
    assignment = plan['assignment']
    assert assignment_hash(assignment) == plan['assignment_sha256']
    assert assignment['code'] == code_identity()
    assert assignment['implementation_sha256'] == implementation_hash()
    records = [r for pde in PDES for r in records_for(pde, study)]
    left, right = map(set, assignment['gpu_cells'])
    assert len(assignment['gpu_cells'][0]) == len(left) and len(assignment['gpu_cells'][1]) == len(right)
    assert not left & right and left | right == {r['cell'] for r in records}
    assert len(records) == len(left | right) == 60
    for pde in PDES:
        assert sha_file(study/'protocol/selected'/f'{pde}.json') == assignment['selected_sha256'][pde]
    return plan, records


def worker(study, worker_id, device):
    assert worker_id in (0, 1)
    study = Path(study)
    plan, records = load_assignment(study)
    assigned = set(plan['assignment']['gpu_cells'][worker_id])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    free, _ = torch.cuda.mem_get_info(device)
    assert free >= 20*1024**3, 'Insufficient free GPU memory'
    completed = []
    for pde in PDES:
        subset = sorted([r for r in records if r['pde'] == pde and r['cell'] in assigned], key=lambda r:r['cell'])
        if not subset:
            continue
        selected = json.loads((study/'protocol/selected'/f'{pde}.json').read_text())
        assert selected['status'] == 'validated'
        network, manifest = load_network(pde, selected['stage'], device)
        assert manifest['checkpoint_sha256'] == selected['checkpoint_sha256']
        normalizer = json.loads((study/'inputs'/pde/'normalizer.json').read_text())
        for record in subset:
            result = evaluate_cell(pde, record, network, manifest, selected, normalizer, 32, device, study)
            completed.append(dict(cell=result['cell'], n=result['n'], summary=str(study/'main'/result['cell']/'summary.json')))
            write_json(study/'main/workers'/f'gpu{worker_id}_progress.json', dict(worker_id=worker_id,
                completed_cells=len(completed), assigned_cells=len(assigned), cells=completed,
                assignment_sha256=plan['assignment_sha256']))
        del network
        torch.cuda.empty_cache()
    assert {r['cell'] for r in completed} == assigned and len(completed) == len(assigned)
    write_json(study/'main/workers'/f'gpu{worker_id}_complete.json', dict(status='complete',
        worker_id=worker_id, cells=completed, samples=len(completed)*1000, assignment_sha256=plan['assignment_sha256']))


def completed_cells(study, plan):
    """Reject missing, duplicate, reassigned or partial cells before aggregation."""
    study = Path(study)
    entries = []
    for gpu in (0, 1):
        row = json.loads((study/'main/workers'/f'gpu{gpu}_complete.json').read_text())
        assigned = set(plan['assignment']['gpu_cells'][gpu])
        assert row['status'] == 'complete' and row['worker_id'] == gpu
        assert row['assignment_sha256'] == plan['assignment_sha256']
        assert row['samples'] == 1000*len(assigned)
        assert len(row['cells']) == len(assigned) and {r['cell'] for r in row['cells']} == assigned
        assert all(r['n'] == 1000 for r in row['cells'])
        entries.extend(row['cells'])
    assert len(entries) == len({r['cell'] for r in entries}) == 60
    return entries


def finish(study):
    study = Path(study)
    plan, records = load_assignment(study)
    entries = completed_cells(study, plan)
    for pde in PDES:
        expected = {r['cell'] for r in records if r['pde'] == pde}
        subset = sorted([r for r in entries if r['cell'] in expected], key=lambda r:r['cell'])
        assert len(subset) == len(expected) == 15
        selected = json.loads((study/'protocol/selected'/f'{pde}.json').read_text())
        for entry in subset:
            row = json.loads(Path(entry['summary']).read_text())
            assert row['status'] == 'complete' and row['n'] == 1000 and row['cell'] == entry['cell']
            assert row['ids'] == list(range(2000, 3000) if pde == 'nsnonbounded' else range(1000))
            assert row['selected'] == selected
        write_json(study/'main'/pde/'complete.json', dict(status='complete', pde=pde,
            cells=subset, samples=15000, selected=selected, assignment_sha256=plan['assignment_sha256']))
    aggregate(study)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=STUDY)
    parser.add_argument('--worker-id', type=int, required=True, choices=[0, 1])
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    worker(args.study, args.worker_id, torch.device(args.device))
