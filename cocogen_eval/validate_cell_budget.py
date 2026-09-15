"""CPU checks for cell assignment, dispatch, completion and budget enforcement."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from . import cell_schedule, cell_worker
from .calibrate import implementation_hash
from .common import STUDY, sha_file, write_json
from .schedule_24h import PDES, allocation


def validate(study):
    records = [dict(pde=pde, cell=f'{pde}/id/synthetic_{i:02d}') for pde in PDES for i in range(15)]
    hours = dict(zip(PDES, [10.46, 10.27, 7.4, 10.8]))
    groups, totals = cell_schedule.cell_allocation(hours, records)
    _, whole = allocation(hours)
    assert max(totals) < max(whole) and max(totals)*1.15 < 24 < max(whole)*1.15
    assert len(set(groups[0]+groups[1])) == 60 and not set(groups[0]) & set(groups[1])
    folder = Path(study)/'validation'
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cell_budget_', dir=folder) as tmp:
        root = Path(tmp)
        (root/'logs').mkdir()
        for name in ['ph', 'dn']:
            (root/'logs'/f'calibrate_cost_{name}.exit').write_text('0\n')
        write_json(root/'validation/inputs.json', dict(status='passed', count=66))
        write_json(root/'validation/evaluator.json', dict(status='passed'))
        write_json(root/'validation/cell_budget.json', dict(status='passed', code=cell_schedule.code_identity()))
        for pde in PDES:
            write_json(root/'protocol/selected'/f'{pde}.json', dict(status='validated', stage='base',
                config=dict(steps=100, repaint=4), estimated_main_gpu_hours=hours[pde],
                implementation_sha256=implementation_hash(), checkpoint_sha256=pde))
            write_json(root/'inputs'/pde/'normalizer.json', dict(mean=[0., 0.], std=[1., 1.], eps=1e-8))
        assignment = dict(gpu_cells=groups, implementation_sha256=implementation_hash(),
            code=cell_schedule.code_identity(),
            selected_sha256={pde: sha_file(root/'protocol/selected'/f'{pde}.json') for pde in PDES})
        plan = dict(status='running', assignment=assignment, assignment_sha256=cell_schedule.assignment_hash(assignment))
        write_json(root/'protocol/budget_24h.json', plan)
        called = []
        def fake_evaluate(pde, record, network, manifest, selected, normalizer, batch, device, study):
            assert batch == 32 and selected['checkpoint_sha256'] == pde
            called.append(record['cell'])
            result = dict(cell=record['cell'], status='complete', n=1000, selected=selected,
                ids=list(range(2000, 3000) if pde == 'nsnonbounded' else range(1000)))
            write_json(Path(study)/'main'/record['cell']/'summary.json', result)
            return result
        def fake_records(pde, study):
            return [r for r in records if r['pde'] == pde]
        with patch.object(cell_worker, 'records_for', side_effect=fake_records), \
             patch.object(cell_worker, 'load_network', side_effect=lambda pde, *args: (object(), dict(checkpoint_sha256=pde))), \
             patch.object(cell_worker, 'evaluate_cell', side_effect=fake_evaluate), \
             patch.object(torch.cuda, 'mem_get_info', return_value=(80*1024**3, 80*1024**3)), \
             patch.object(torch.cuda, 'empty_cache'):
            cell_worker.worker(root, 0, torch.device('cpu'))
            try:
                cell_worker.finish(root)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('Partial worker output was aggregated')
            assert not list((root/'main').glob('*/complete.json'))
            cell_worker.worker(root, 1, torch.device('cpu'))
            assert len(called) == len(set(called)) == 60
            assert set(called) == {r['cell'] for r in records}
            with patch.object(cell_worker, 'aggregate') as aggregate:
                cell_worker.finish(root)
                aggregate.assert_called_once_with(root)
            for pde in PDES:
                complete = json.loads((root/'main'/pde/'complete.json').read_text())
                assert complete['samples'] == 15000 and len(complete['cells']) == 15
            path = root/'main/workers/gpu0_complete.json'
            receipt = json.loads(path.read_text())
            receipt['cells'][1] = receipt['cells'][0]
            write_json(path, receipt)
            try:
                cell_worker.completed_cells(root, plan)
            except AssertionError:
                pass
            else:
                raise AssertionError('Duplicated cell was accepted')
            path = root/'protocol/selected/darcy.json'
            selected = json.loads(path.read_text()); selected['config']['steps'] = 500
            write_json(path, selected)
            try:
                cell_worker.load_assignment(root)
            except AssertionError:
                pass
            else:
                raise AssertionError('Changed sampler selection was accepted')
        for pde in PDES:
            path = root/'protocol/selected'/f'{pde}.json'
            selected = json.loads(path.read_text())
            selected.update(estimated_main_gpu_hours=20., config=dict(steps=100, repaint=4))
            write_json(path, selected)
        with patch.object(cell_schedule, 'records_for', side_effect=fake_records), \
             patch.object(cell_schedule.subprocess, 'run', side_effect=AssertionError('Over-budget job launch')), \
             contextlib.redirect_stdout(io.StringIO()):
            try:
                cell_schedule.schedule(root)
            except RuntimeError as error:
                assert 'exceeds the user budget' in str(error)
            else:
                raise AssertionError('Over-budget plan was accepted')
        rejected = json.loads((root/'protocol/budget_24h.json').read_text())
        assert rejected['status'] == 'needs_lower_cost' and rejected['estimated_hours_with_margin'] == 46.
    result = dict(status='passed', synthetic=True, code=cell_schedule.code_identity(),
        comparison_fixture=dict(cell_hours=max(totals), whole_pde_hours=max(whole)),
        checks=['all 60 original cells dispatched once across two workers', '15 cells and 15000 samples per PDE',
            'whole-PDE imbalance reduced without changing any sampler', 'partial and duplicate cell completion rejected',
            'frozen selection hash enforced', '15 percent margin and over-24-hour launch rejection'],
        limitation='GPU sampling is covered separately by evaluator and per-PDE tests; dispatch uses synthetic cell results')
    write_json(folder/'cell_budget.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=STUDY)
    validate(parser.parse_args().study)
