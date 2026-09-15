"""Audit saved formal predictions on CPU against the archived FM4PDE inputs.

Run after the first batches of a new PDE have been saved. This does not sample,
load a model, or establish complete-cell/PDE quality; collect_metrics does the
complete-cell checks. Imports must resolve to the frozen sampling source tree.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from cocogen_eval.calibrate import implementation_hash
from cocogen_eval.common import sha_file, write_json
from cocogen_eval.inputs import identity
from cocogen_eval.main_inputs import historical_input


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(study, cell, batches):
    study = Path(study).resolve()
    require(batches > 0, 'At least one saved batch is required')
    catalog_path = study / 'protocol/catalog.json'
    catalog = json.loads(catalog_path.read_text())
    items = [item for item in catalog['cells'] if item['cell'] == cell]
    require(len(items) == 1, 'Cell must have one frozen catalog entry')
    item = items[0]
    record_path = Path(item['frozen_record'])
    require(sha_file(record_path) == item['frozen_sha256'], 'Changed frozen record')
    record = json.loads(record_path.read_text())
    pde = record['pde']
    selected_path = study / 'protocol/selected' / f'{pde}.json'
    selected = json.loads(selected_path.read_text())
    require(selected['status'] == 'validated', 'Sampling configuration was not validated')
    require(selected['implementation_sha256'] == implementation_hash(), 'Wrong sampling source tree')
    normalizer = json.loads((study / 'inputs' / pde / 'normalizer.json').read_text())
    truth, masks, ids, _ = historical_input(record)
    expected_ids = list(range(2000, 3000)) if pde == 'nsnonbounded' else list(range(1000))
    require(ids == expected_ids, 'Wrong formal sample IDs')
    require(truth.device.type == masks.device.type == 'cpu', 'Audit must use CPU inputs')
    if pde != 'burger' and record['task'] != 'both':
        inactive = 1 if record['task'] == 'forward' else 0
        require(not masks[:, inactive].any().item(), 'Inactive field has observations')
    paths = sorted((study / 'main' / cell / 'batches').glob('*/receipt.json'))
    require(len(paths) >= batches, 'Required batches have not been saved yet')
    nfe = selected['config']['steps'] * (1 + selected['config']['repaint'])
    checked, offset = [], 0
    for path in paths[:batches]:
        receipt = json.loads(path.read_text())
        request, sampling = receipt['request'], receipt['sampling']
        require(receipt['status'] == 'complete' and request['cell'] == cell, 'Wrong batch identity')
        require(request['offset'] == offset, 'Missing or out-of-order batch')
        require(request['selected'] == selected, 'Changed selected configuration')
        require(request['normalizer'] == normalizer, 'Changed training normalizer')
        require(request['implementation_sha256'] == selected['implementation_sha256'], 'Changed implementation')
        require(request['checkpoint']['checkpoint_sha256'] == selected['checkpoint_sha256'], 'Wrong checkpoint')
        require(request['checkpoint']['stage'] == selected['stage'], 'Wrong model stage')
        require(request['physics_config'] == record['config'], 'Wrong physics configuration')
        require(sampling['config'] == selected['config'] and sampling['nfe'] == nfe
                and sampling['final_time'] == 0, 'Wrong sampling schedule')
        prediction_path = Path(receipt['prediction_path'])
        require(prediction_path == path.parent / 'prediction.pt', 'Wrong prediction path')
        require(sha_file(prediction_path) == receipt['prediction_sha256'], 'Changed prediction file')
        payload = torch.load(prediction_path, map_location='cpu', weights_only=False)
        require(payload['request'] == request and payload['sampling'] == sampling
                and payload['errors'] == receipt['errors'], 'Payload and receipt differ')
        n = len(payload['ids'])
        require(n == min(request['batch_size'], 1000 - offset) and n > 0, 'Wrong batch size')
        x, mask = truth[offset:offset+n], masks[offset:offset+n]
        require(payload['ids'] == ids[offset:offset+n], 'Wrong batch sample IDs')
        require(request['input'] == identity(x, mask, payload['ids']), 'Wrong input fields or masks')
        prediction = payload['prediction']
        require(prediction.shape == x.shape and torch.isfinite(prediction).all().item(), 'Invalid prediction')
        require(torch.equal(prediction[mask], x[mask]), 'Observed values were changed')
        channels = [('u', 0)] if pde == 'burger' else [('a', 0), ('u', 1)]
        observations = {}
        for field, channel in channels:
            counts = mask[:, channel].flatten(1).sum(1)
            observed_delta = (prediction[:, channel] - x[:, channel])[mask[:, channel]]
            observations[field] = dict(min_points=int(counts.min()), max_points=int(counts.max()),
                max_absolute_error=float(observed_delta.abs().max()) if observed_delta.numel() else None)
            target = x[:, channel].double().flatten(1)
            delta = prediction[:, channel].double().flatten(1) - target
            denominator = torch.linalg.vector_norm(target, dim=1)
            require((denominator > 0).all().item(), 'Zero relative-error denominator')
            errors = (torch.linalg.vector_norm(delta, dim=1) / denominator).tolist()
            saved = receipt['errors'][field]
            require(len(saved) == n and all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
                    for a, b in zip(errors, saved)), 'Saved errors disagree with CPU float64 recomputation')
        checked.append(dict(receipt=str(path), receipt_sha256=sha_file(path),
            prediction_sha256=receipt['prediction_sha256'], offset=offset, n=n,
            sampler_seconds=sampling['seconds'], peak_bytes=sampling['peak_bytes'],
            nfe=nfe, final_time=sampling['final_time'], observations=observations,
            ids_and_masks_match=True, observed_values_exact=True,
            errors_recomputed_cpu_float64=True, code_commit=request['git_commit']))
        offset += n
    return dict(status='passed', generated_at=datetime.now(timezone.utc).isoformat(),
        scope='Saved formal batches only; CPU input, prediction and metric audit; no complete-cell result or residual recomputation',
        cell=cell, samples=offset, batches=checked,
        implementation_sha256=selected['implementation_sha256'],
        selected_sha256=sha_file(selected_path), catalog_sha256=sha_file(catalog_path),
        record_sha256=sha_file(record_path), auditor_sha256=sha_file(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--batches', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = validate(args.study, args.cell, args.batches)
    require(not args.output.exists(), 'Use a new output path to preserve previous audit evidence')
    write_json(args.output, result)
    print(json.dumps(dict(status=result['status'], cell=result['cell'], samples=result['samples'],
                          output=str(args.output))))
