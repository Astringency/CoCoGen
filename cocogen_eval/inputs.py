"""Frozen FM4PDE input identities and physical-space scoring."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .common import CATALOG, STUDY, sha_file, sha_tensor, write_json
from .fm_physics.masks import make_pair_masks
from .fm_physics.pde_residuals import compute_pde_residual


PHYSICS_KEYS = ('enforce_boundary_conditions', 'boundary_condition_mode', 'bc_weight',
    'endpoint_bc_weight', 'boundary_residual_normalization', 'ns_operator_mode',
    'coef_positive_mode', 'coef_positive_floor', 'allow_unknown_boundary_conditions',
    'hermite_collocation_times', 'hermite_num_collocation',
    'hermite_include_integral_residual', 'hermite_integral_weight')


def freeze_catalog(study=STUDY):
    dest = Path(study)/'protocol/catalog.json'
    if dest.exists():
        return json.loads(dest.read_text())
    source = json.loads(CATALOG.read_text())
    records = []
    for item in source['cells']:
        path = Path(item['record_path'])
        assert sha_file(path) == item['record_sha256'], path
        record = json.loads(path.read_text())
        assert len(set(record['sample_ids'])) == 1000
        target = Path(study)/'protocol/cells'/f"{item['cell']}.json"
        write_json(target, record)
        records.append(dict(item, frozen_record=str(target), frozen_sha256=sha_file(target)))
    assert len(records) == 66
    result = dict(source_catalog=str(CATALOG), source_sha256=sha_file(CATALOG), cells=records,
        first_phase_cells=60, second_phase_cells=6, samples_per_cell=1000,
        sequence='complete four existing PDEs first, then train and evaluate Burgers')
    write_json(dest, result)
    return result


def records_for(pde, study=STUDY):
    catalog = freeze_catalog(study)
    result = []
    for item in catalog['cells']:
        if item['pde'] != pde:
            continue
        assert sha_file(item['frozen_record']) == item['frozen_sha256']
        result.append(json.loads(Path(item['frozen_record']).read_text()))
    return result


def physics_function(pde, config, params=None):
    merged = dict(params or {})
    # Main result files contain no hidden trajectories. Restrict parameters to
    # known equation constants/configuration, never accept dense truth arrays.
    for key, value in merged.items():
        if torch.is_tensor(value) and value.ndim > 1:
            raise ValueError(f'Dense physics parameter is not permitted: {key}')
    merged.update({key:config[key] for key in PHYSICS_KEYS if key in config})
    if pde == 'nsnonbounded':
        merged.setdefault('nu', merged.pop('viscosity', .001))
        merged.setdefault('T', 1.)
        merged.setdefault('solver_dt', .0001)
    def residual(fields):
        a, u = (fields, fields) if pde == 'burger' else (fields[:, :1], fields[:, 1:])
        return compute_pde_residual(pde, a, u, pde_params=merged,
            k=config.get('k', merged.get('k', 1)),
            residual_mode=config.get('residual_mode', 'auto')).residual
    return residual


def calibration_input(pde, record, indices, study=STUDY, device='cuda:0'):
    cache = torch.load(Path(study)/'inputs'/pde/'calibration.pt', map_location='cpu', weights_only=False)
    fields = cache['fields'][indices].to(device)
    ids = [cache['ids'][i] for i in indices]
    cfg = record['config']
    shape = fields[:, :1].shape
    masks = make_pair_masks(shape, shape, cfg['num_obs'], cfg['sensor_mode'],
        cfg['shared_mask'], cfg['mask_seed'], device=device,
        num_sensor_columns=cfg.get('num_sensor_columns'))
    if record['task'] == 'forward':
        masks.sol.zero_()
    elif record['task'] == 'inverse':
        masks.coef.zero_()
    mask = masks.sol if pde == 'burger' else torch.cat([masks.coef, masks.sol], 1)
    return fields, mask.bool(), ids, cache['pde_params']


def historical_input(record):
    """Load every archived batch once; concatenate in the frozen ID order."""
    fields, masks, ids = [], [], []
    for batch in record['batches']:
        path = batch['result_path']
        assert sha_file(path) == batch['result_sha256'], path
        payload = torch.load(path, map_location='cpu', weights_only=False)
        rows = [r for r in record['rows'] if r['result_path'] == path]
        index = [r['result_row'] for r in rows]
        if record['pde'] == 'nsnonbounded':
            x, mask = payload['truths'][index], payload['masks'][index]
        else:
            assert not payload['ground_truth_metadata'].get('synthetic', False)
            assert not payload.get('pde_params'), 'New parameters need explicit review/slicing'
            if record['pde'] == 'burger':
                x, mask = payload['coef_ground_truth'][index], payload['masks']['sol'][index]
            else:
                x = torch.cat([payload['coef_ground_truth'][index], payload['sol_ground_truth'][index]], 1)
                mask = torch.cat([payload['masks']['coef'][index], payload['masks']['sol'][index]], 1)
        assert torch.isfinite(x).all() and torch.all((mask == 0) | (mask == 1))
        fields.append(x.float()); masks.append(mask.bool())
        ids.extend(r['sample_id'] for r in rows)
    order = [ids.index(i) for i in record['sample_ids']]
    fields, masks = torch.cat(fields)[order], torch.cat(masks)[order]
    ids = [ids[i] for i in order]
    assert len(set(ids)) == 1000 and ids == record['sample_ids']
    return fields, masks, ids, {}


def scores(prediction, truth, pde):
    # CPU float64 matches recomputed FM4PDE per-example physical relative L2.
    prediction, truth = prediction.detach().cpu().double(), truth.detach().cpu().double()
    result = {}
    for name, channel in ([('u', 0)] if pde == 'burger' else [('a', 0), ('u', 1)]):
        p, q = prediction[:, channel].flatten(1), truth[:, channel].flatten(1)
        denom = q.norm(dim=1)
        if (denom == 0).any():
            raise ValueError('Zero relative-error denominator')
        result[name] = ((p-q).norm(dim=1)/denom).tolist()
    return result


def target_score(errors, task):
    fields = ['u'] if task == 'forward' or 'a' not in errors else (['a'] if task == 'inverse' else ['a', 'u'])
    return sum(sum(errors[f])/len(errors[f]) for f in fields)/len(fields)


def identity(fields, masks, ids):
    return dict(ids=ids, truth_sha256=sha_tensor(fields), mask_sha256=sha_tensor(masks))
