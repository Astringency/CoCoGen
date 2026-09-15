"""Check task allocation and that an excessive projection cannot launch jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from .calibrate import implementation_hash
from .common import STUDY, write_json
from .schedule_24h import PDES, allocation, schedule


def validate(study):
    weights=dict(zip(PDES,[10.,9.,3.,2.]))
    groups,hours=allocation(weights)
    assert max(hours)==12.
    assert sorted(groups[0]+groups[1])==sorted(PDES)
    assert not (set(groups[0])&set(groups[1]))
    folder=Path(study)/'validation';folder.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='budget_test_',dir=folder) as temporary:
        root=Path(temporary)
        (root/'logs').mkdir()
        for group in ['ph','dn']:
            (root/'logs'/f'calibrate_cost_{group}.exit').write_text('0\n')
        write_json(root/'validation/inputs.json',dict(status='passed',count=66))
        write_json(root/'validation/evaluator.json',dict(status='passed'))
        for pde in PDES:
            write_json(root/'protocol/selected'/f'{pde}.json',dict(status='validated',
                implementation_sha256=implementation_hash(),estimated_main_gpu_hours=20.,
                config=dict(steps=500,repaint=0),stage='base',validation_macro_mean=.3))
        with patch('cocogen_eval.schedule_24h.subprocess.run',side_effect=AssertionError('An over-budget plan launched a job')):
            try:
                schedule(root,24.,1.15)
            except RuntimeError as error:
                assert 'exceeds the user budget' in str(error)
            else:
                raise AssertionError('Over-budget projection was accepted')
        plan=json.loads((root/'protocol/budget_24h.json').read_text())
        assert plan['status']=='needs_lower_cost' and plan['estimated_hours_with_margin']==46.
    write_json(folder/'budget.json',dict(status='passed',synthetic=True,
        checks=['all four PDEs assigned exactly once','balanced two-GPU assignment',
                '15 percent margin included','over-24-hour projection cannot launch a GPU job']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=STUDY)
    validate(parser.parse_args().study)
