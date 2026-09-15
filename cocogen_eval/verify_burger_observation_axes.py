"""Check Burgers observation geometry against archived masks and dataset axes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io
import torch

from cocogen_eval.common import sha_file, sha_tensor, write_json
from cocogen_eval.verify_archived_predictions import ArchiveReader, require


def verify(root, original_study, generator):
    torch.set_num_threads(4)
    reader = ArchiveReader(root, original_study)
    catalog = reader.read_json('protocol/catalog.json')
    items = [item for item in catalog['cells'] if item['pde'] == 'burger']
    require(len(items) == 6, 'Expected all six frozen Burgers cells')
    datasets, cells = {}, []
    for item in items:
        record = reader.read_json(item['frozen_record'], item['frozen_sha256'])
        cfg = record['config']
        require(cfg['zeta_obs_a'] == 0 and cfg['zeta_obs_u'] > 0, 'Expected only solution observations')
        source = cfg['data_path']
        if source not in datasets:
            path = Path(source)
            before = path.stat()
            raw = scipy.io.loadmat(path, variable_names=['input', 'tspan', 'viscosity'])
            initial = np.asarray(raw['input'], dtype=np.float32)
            require(initial.shape == (10000, 128), 'Unexpected initial-field layout')
            times = np.asarray(raw['tspan'], dtype=np.float64).reshape(-1) if 'tspan' in raw else None
            viscosity = float(np.asarray(raw['viscosity']).reshape(-1)[0]) if 'viscosity' in raw else None
            if times is not None:
                require(times.shape == (128,) and np.allclose(times, np.linspace(0, 1, 128), rtol=0, atol=1e-12), 'Unexpected stored physical time grid')
            if viscosity is not None:
                require(viscosity == .01, 'Unexpected stored viscosity')
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Dataset changed during metadata read')
            datasets[source] = dict(initial=initial, bytes=after.st_size, mtime_ns=after.st_mtime_ns,
                initial_first1000_sha256=sha_tensor(torch.from_numpy(initial[:1000])),
                stored_tspan=None if times is None else times.tolist(), stored_viscosity=viscosity,
                missing_metadata=[key for key in ('tspan','viscosity') if key not in raw],
                scope='Initial fields read; tspan and viscosity checked only when stored; entire MAT file is not hashed')
        initial = datasets[source]['initial']
        ids, full_rows, full_columns, observed = [], set(), set(), set()
        column_sets = set()
        for batch in record['batches']:
            path = reader.check(batch['result_path'], batch['result_sha256'])
            payload = torch.load(path, map_location='cpu', weights_only=False)
            rows = [row for row in record['rows'] if row['result_path'] == batch['result_path']]
            indices = [row['result_row'] for row in rows]
            batch_ids = [row['sample_id'] for row in rows]
            truth = payload['coef_ground_truth'][indices]
            mask = payload['masks']['sol'][indices]
            require(truth.shape == mask.shape == (len(rows), 1, 128, 128), 'Unexpected archived field layout')
            require(torch.equal(truth, payload['sol_ground_truth'][indices]), 'Burgers truth aliases differ')
            require(torch.isfinite(truth).all(), 'Nonfinite archived truth')
            require(torch.equal(truth[:, 0, 0, :], torch.from_numpy(initial[batch_ids])), 'First time level does not match the original initial condition')
            require(torch.all((mask == 0) | (mask == 1)), 'Mask must be binary')
            mask = mask.bool()
            counts = mask.flatten(1).sum(1)
            rows_full = (mask.sum(-1) == 128).sum(-1).flatten()
            columns_full = (mask.sum(-2) == 128).sum(-1).flatten()
            if record['setting'] == 'sensor_column':
                require(torch.all(counts == 640) and torch.all(columns_full == 5), 'Expected five complete spatial columns')
                require(torch.equal(mask, mask[:, :, :1, :].expand_as(mask)), 'Spatial sensor positions change over time')
                require(torch.all(rows_full == 0), 'Complete time snapshots are not this protocol')
                column_sets.update(tuple(torch.where(m[0, 0])[0].tolist()) for m in mask)
            else:
                require(record['setting'] == 'random' and torch.all(counts == 500), 'Unexpected random-observation protocol')
            observed.update(counts.tolist()); full_rows.update(rows_full.tolist()); full_columns.update(columns_full.tolist())
            ids.extend(batch_ids)
        require(ids == record['sample_ids'] == list(range(1000)), 'Wrong or duplicated formal sample IDs')
        cells.append(dict(cell=item['cell'],n=len(ids),batches=len(record['batches']),
            observations_per_sample=sorted(observed),complete_time_levels_per_sample=sorted(full_rows),
            complete_spatial_columns_per_sample=sorted(full_columns),sensor_column_index_sets=[list(x) for x in sorted(column_sets)],
            first_time_level_matches_dataset_initial_condition=True,only_solution_mask_applied=True))
        print(f"Verified {item['cell']}: {len(ids)} archived masks", flush=True)
    return dict(status='passed',generated_at=datetime.now(timezone.utc).isoformat(),
        scope='All six frozen FM4PDE Burgers input cells; observation geometry and initial-level alignment only',
        final_study_complete=False,training_started_by_this_audit=False,training_cache_created=False,
        original_dataset_metadata_read=True,full_dataset_files_hashed=False,
        layout='B,1,T,X; axis -2 contains physical time, axis -1 contains periodic space',
        sensor_column_meaning='Five fixed spatial locations, each observed at all 128 time levels; 640 scalar values',
        physics_parameter_limit='Absent MAT metadata is not inferred as measured. Frozen FM4PDE defaults use T=1 and nu=0.01; this audit does not independently establish those constants or recompute residuals.',
        cells=cells,datasets={key:{k:v for k,v in value.items() if k!='initial'} for key,value in datasets.items()},
        generator=dict(path=str(generator),sha256=sha_file(generator),source_text=Path(generator).read_text()),
        archived_sources={key:{k:v for k,v in value.items() if k!='mtime_ns'} for key,value in reader.opened.items()},
        auditor_sha256=sha_file(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--original-study',type=Path,required=True)
    parser.add_argument('--generator',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve previous audit receipts')
    write_json(args.output,verify(args.root,args.original_study,args.generator))
