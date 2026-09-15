"""Audit all frozen main masks and raw-data channel/ID alignment on CPU."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import STUDY, write_json
from .inputs import historical_input, identity, physics_function, records_for, scores
from .prepare import read_fields


def validate(study):
    torch.set_num_threads(4)
    rows = []
    for pde in ['darcy', 'poisson', 'helmholtz', 'nsnonbounded', 'burger']:
        raw_checked = set()
        for record in records_for(pde, study):
            fields, masks, ids, params = historical_input(record)
            cfg = record['config']
            count = int(cfg['num_obs'])
            if cfg['sensor_mode'] == 'sensor_column':
                count = int(cfg['num_sensor_columns'])*fields.shape[-2]
            if pde == 'burger':
                expected = [count]
            else:
                expected = [count if record['task'] in ['forward','both'] else 0,
                            count if record['task'] in ['inverse','both'] else 0]
            observed = masks.flatten(2).sum(2)
            assert torch.equal(observed, torch.tensor(expected).expand_as(observed)), (record['cell'], observed)
            if record['dist'] not in raw_checked:
                first = ids[0]
                original = torch.from_numpy(read_fields(pde, cfg['data_path'], first, first+2))
                assert torch.equal(fields[:2], original), f'Raw data/archived channel or ID mismatch: {record["cell"]}'
                raw_checked.add(record['dist'])
            errors = scores(fields[:2], fields[:2], pde)
            assert all(v == [0., 0.] for v in errors.values())
            with torch.no_grad():
                residual = physics_function(pde, cfg, params)(fields[:2])
            assert torch.isfinite(residual).all(), record['cell']
            rows.append(dict(cell=record['cell'], n=len(ids), expected_observations=expected,
                identity=identity(fields, masks, ids), status='passed'))
            print(record['cell'], 'passed', flush=True)
            write_json(Path(study)/'validation/inputs.json', dict(status='in_progress', cells=rows))
    assert len(rows) == 66
    write_json(Path(study)/'validation/inputs.json', dict(status='passed', cells=rows, count=66))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=STUDY)
    validate(parser.parse_args().study)
