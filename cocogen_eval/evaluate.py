"""Restart-safe CoCoGen evaluation on the exact 60/66 FM4PDE main cells."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .calibrate import implementation_hash
from .common import STUDY, load_network, save_torch, sha_file, write_json
from .inputs import historical_input, identity, physics_function, records_for, scores
from .sampler import SamplerConfig, sample


def statistics(values):
    a = np.asarray(values, dtype=np.float64)
    assert len(a) > 0 and np.isfinite(a).all()
    return dict(mean=float(a.mean()), std=float(a.std(ddof=0)), median=float(np.median(a)),
        p90=float(np.quantile(a, .9)), p95=float(np.quantile(a, .95)), max=float(a.max()),
        mean_standard_error=float(a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else None)


def evaluate_cell(pde, record, network, manifest, selected, normalizer, batch_size, device, study):
    out = Path(study)/'main'/record['cell']
    started = time.monotonic()
    truth, masks, ids, params = historical_input(record)
    assert len(ids) == 1000
    physical = physics_function(pde, record['config'], params)
    config = SamplerConfig(**selected['config'])
    core_request = dict(cell=record['cell'], checkpoint=manifest,
        normalizer=normalizer, selected=selected, batch_size=batch_size,
        input=identity(truth, masks, ids), physics_config=record['config'],
        implementation_sha256=implementation_hash(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        torch_version=torch.__version__, precision='float32', tf32=True)
    receipts = []
    for offset in range(0, len(ids), batch_size):
        chunk = out/'batches'/f'{offset:04d}'
        x, m, sample_ids = truth[offset:offset+batch_size], masks[offset:offset+batch_size], ids[offset:offset+batch_size]
        request = dict(core_request, input=identity(x, m, sample_ids), offset=offset)
        receipt_path = chunk/'receipt.json'
        if receipt_path.exists():
            row = json.loads(receipt_path.read_text())
            # The code commit may include only new reporting/launch files; actual
            # sampling implementation hashes must still match exactly.
            previous = dict(row['request']); current = dict(request)
            previous.pop('git_commit'); current.pop('git_commit')
            assert previous == current, f'Existing evaluation request changed: {chunk}'
            assert sha_file(row['prediction_path']) == row['prediction_sha256']
            receipts.append(row)
            continue
        x_gpu, mask_gpu = x.to(device), m.to(device)
        prediction, sampling = sample(network, torch.where(mask_gpu, x_gpu, 0.), mask_gpu,
            sample_ids, normalizer, manifest['sde'], manifest['conditioning_label'], config,
            residual_fn=physical, namespace=pde,
            progress=lambda r: print(json.dumps(dict(cell=record['cell'], offset=offset, **r)), flush=True))
        assert torch.equal(prediction[mask_gpu], x_gpu[mask_gpu])
        with torch.no_grad():
            predicted_residual = physical(prediction).flatten(1).square().mean(1).cpu().double().tolist()
            truth_residual = physical(x_gpu).flatten(1).square().mean(1).cpu().double().tolist()
        errors = scores(prediction, x_gpu, pde)
        path = chunk/'prediction.pt'
        save_torch(path, dict(prediction=prediction.cpu(), ids=sample_ids, request=request,
                             sampling=sampling, errors=errors))
        row = dict(status='complete', request=request, sampling=sampling, errors=errors,
            predicted_residual_mse=predicted_residual, truth_residual_mse=truth_residual,
            prediction_path=str(path), prediction_sha256=sha_file(path))
        write_json(receipt_path, row)
        receipts.append(row)
        print(json.dumps(dict(cell=record['cell'], completed=min(offset+batch_size, 1000),
            seconds=sampling['seconds'], mean_errors={k:sum(v)/len(v) for k,v in errors.items()})), flush=True)
        del prediction, x_gpu, mask_gpu
    completed_ids = sum([r['request']['input']['ids'] for r in receipts], [])
    assert completed_ids == ids and len(set(completed_ids)) == 1000
    fields = ['u'] if pde == 'burger' else ['a', 'u']
    errors = {k:sum([r['errors'][k] for r in receipts], []) for k in fields}
    historical_by_id = {r['sample_id']:r for r in record['rows']}
    comparison = {}
    for field in fields:
        historical = [historical_by_id[i][f'error_{field}'] for i in ids]
        delta = [a-b for a,b in zip(errors[field], historical)]
        comparison[field] = dict(cocogen=statistics(errors[field]), fm4pde=statistics(historical),
            paired_difference=statistics(delta), smaller_error_fraction=float(np.mean(np.asarray(delta)<0)))
    result = dict(status='complete', cell=record['cell'], pde=pde, dist=record['dist'],
        setting=record['setting'], task=record['task'], n=1000, ids=ids,
        comparison=comparison, errors=errors,
        residual=dict(prediction=statistics(sum([r['predicted_residual_mse'] for r in receipts], [])),
            truth=statistics(sum([r['truth_residual_mse'] for r in receipts], [])),
            mode=record['config']['residual_mode'],
            status='approximate endpoint residual' if pde=='nsnonbounded' else 'static/full-time-space residual'),
        sampler_seconds=sum(r['sampling']['seconds'] for r in receipts),
        wall_seconds_this_invocation=time.monotonic()-started,
        score_evaluations_per_sample=config.steps*(1+config.repaint),
        sample_weighted_NFE=1000*config.steps*(1+config.repaint),
        selected=selected, batches=[dict(receipt=str(out/'batches'/f'{r["request"]["offset"]:04d}'/'receipt.json'),
            prediction_path=r['prediction_path'], prediction_sha256=r['prediction_sha256']) for r in receipts],
        input=core_request['input'])
    write_json(out/'summary.json', result)
    return result


def evaluate(pde, device, study, batch_size):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    selected = json.loads((Path(study)/'protocol/selected'/f'{pde}.json').read_text())
    assert selected['status'] == 'validated', 'Calibration and held-out validation must finish first'
    assert selected['implementation_sha256'] == implementation_hash()
    network, manifest = load_network(pde, selected['stage'], device)
    assert manifest['checkpoint_sha256'] == selected['checkpoint_sha256']
    normalizer = json.loads((Path(study)/'inputs'/pde/'normalizer.json').read_text())
    results = []
    for record in sorted(records_for(pde, study), key=lambda r:r['cell']):
        result = evaluate_cell(pde, record, network, manifest, selected, normalizer, batch_size, device, study)
        results.append(dict(cell=result['cell'], n=result['n'], summary=str(Path(study)/'main'/result['cell']/'summary.json')))
        write_json(Path(study)/'main'/pde/'progress.json', dict(pde=pde, completed_cells=len(results), cells=results))
    expected = 6 if pde=='burger' else 15
    assert len(results) == expected
    write_json(Path(study)/'main'/pde/'complete.json', dict(status='complete', pde=pde, cells=results,
        samples=expected*1000, selected=selected))


def aggregate(study, include_burger=False):
    pdes = ['darcy', 'poisson', 'helmholtz', 'nsnonbounded'] + (['burger'] if include_burger else [])
    cells = []
    for pde in pdes:
        complete = json.loads((Path(study)/'main'/pde/'complete.json').read_text())
        assert complete['status'] == 'complete'
        for cell in complete['cells']:
            result = json.loads(Path(cell['summary']).read_text())
            assert result['status'] == 'complete' and result['n'] == 1000 and len(set(result['ids'])) == 1000
            for batch in result['batches']:
                assert sha_file(batch['prediction_path']) == batch['prediction_sha256']
            cells.append(result)
    assert len(cells) == (66 if include_burger else 60)
    out = Path(study)/'reports'; out.mkdir(exist_ok=True)
    path = out/('all66.csv' if include_burger else 'first60.csv')
    columns = ['cell','pde','dist','setting','task','n','cocogen_a','fm4pde_a','cocogen_u','fm4pde_u',
               'pde_residual_mse','ground_truth_residual_mse','NFE','seconds']
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for cell in cells:
            row = {k:cell[k] for k in columns[:6]}
            for field in ('a', 'u'):
                for model in ('cocogen','fm4pde'):
                    row[f'{model}_{field}'] = cell['comparison'].get(field, {}).get(model, {}).get('mean', '')
            row.update(pde_residual_mse=cell['residual']['prediction']['mean'],
                ground_truth_residual_mse=cell['residual']['truth']['mean'],
                NFE=cell['score_evaluations_per_sample'], seconds=cell['sampler_seconds'])
            writer.writerow(row)
    write_json(out/('all66_complete.json' if include_burger else 'first60_complete.json'),
        dict(status='complete', cells=len(cells), samples=sum(x['n'] for x in cells),
             csv=str(path), csv_sha256=sha_file(path), predictions_hash_verified=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pde')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--include-burger', action='store_true')
    parser.add_argument('--study', type=Path, default=STUDY)
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args.study, args.include_burger)
    else:
        if args.pde is None:
            parser.error('--pde is required unless --aggregate is set')
        evaluate(args.pde, torch.device(args.device), args.study, args.batch_size)
