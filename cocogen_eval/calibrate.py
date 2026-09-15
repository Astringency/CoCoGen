"""Select sampling settings on disjoint ID calibration samples only."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .common import STUDY, load_network, save_torch, sha_file, write_json
from .inputs import PHYSICS_KEYS, calibration_input, identity, physics_function, records_for, scores, target_score
from .sampler import SamplerConfig, sample


def implementation_hash():
    root = Path(__file__).parent
    files = [root/name for name in ['sampler.py', 'network.py', 'common.py', 'inputs.py']]
    files += sorted((root/'fm_physics').glob('*.py'))
    return hashlib.sha256(json.dumps({str(p.relative_to(root)):sha_file(p) for p in files}, sort_keys=True).encode()).hexdigest()


def trial(pde, stage, network, manifest, records, normalizer, config, start, stop, batch_size, study, device):
    signature = implementation_hash()
    label = f'{stage}_s{config.steps}_r{config.repaint}_p{config.max_physical_update:g}_N{config.physics_steps}_M{config.physics_post}_{config.forward_method}'
    out = Path(study)/'calibration'/pde/signature[:10]/f'{start}-{stop}'/label
    groups = [calibration_input(pde, r, list(range(start, stop)), study, device) for r in records]
    fields = torch.cat([x[0] for x in groups])
    mask = torch.cat([x[1] for x in groups])
    ids = sum([x[2] for x in groups], [])
    physics_settings = [{k:r['config'].get(k) for k in (*PHYSICS_KEYS, 'residual_mode', 'k')} for r in records]
    assert all(x == physics_settings[0] for x in physics_settings)
    assert all(x[3] == groups[0][3] for x in groups)
    residual = physics_function(pde, records[0]['config'], groups[0][3])
    request = dict(stage=stage, checkpoint_sha256=manifest['checkpoint_sha256'],
        normalizer=normalizer, config=asdict(config), implementation_sha256=signature,
        batch_size=batch_size, settings=[r['setting'] for r in records],
        input=identity(fields, mask, ids), physics=physics_settings[0], params=groups[0][3])
    receipt_path = out/'receipt.json'
    if receipt_path.exists():
        cached = json.loads(receipt_path.read_text())
        assert cached['request'] == request, f'Calibration cache request mismatch: {out}'
        assert sha_file(cached['prediction_path']) == cached['prediction_sha256']
        return cached
    predictions, receipts = [], []
    for offset in range(0, len(fields), batch_size):
        x, m, sample_ids = fields[offset:offset+batch_size], mask[offset:offset+batch_size], ids[offset:offset+batch_size]
        prediction, receipt = sample(network, torch.where(m, x, 0.), m, sample_ids,
            normalizer, manifest['sde'], manifest['conditioning_label'], config,
            residual_fn=residual, namespace=pde,
            progress=lambda row: print(json.dumps(dict(pde=pde, trial=label, offset=offset, **row)), flush=True))
        predictions.append(prediction.cpu()); receipts.append(receipt)
    prediction = torch.cat(predictions)
    field_errors = scores(prediction, fields, pde)
    residual_values = []
    with torch.no_grad():
        for offset in range(0, len(fields), batch_size):
            r = residual(prediction[offset:offset+batch_size].to(device))
            residual_values.extend(r.flatten(1).square().mean(1).cpu().double().tolist())
    n = stop-start
    settings = {}
    for i, record in enumerate(records):
        errors = {k:v[i*n:(i+1)*n] for k,v in field_errors.items()}
        settings[record['setting']] = dict(errors=errors,
            target_mean=target_score(errors, record['task']),
            residual_mse=residual_values[i*n:(i+1)*n])
    path = out/'predictions.pt'
    save_torch(path, dict(prediction=prediction, ids=ids, settings=[r['setting'] for r in records], request=request))
    result = dict(request=request, label=label, settings=settings,
        macro_mean=sum(x['target_mean'] for x in settings.values())/len(settings),
        seconds=sum(x['seconds'] for x in receipts),
        batch_receipts=receipts, prediction_path=str(path), prediction_sha256=sha_file(path))
    write_json(receipt_path, result)
    print(json.dumps(dict(pde=pde, trial=label, stage='trial_complete', n_per_setting=n,
        macro_mean=result['macro_mean'], means={k:v['target_mean'] for k,v in settings.items()},
        seconds=result['seconds'])), flush=True)
    return result


def calibrate(pde, device, study, batch_size):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    records = sorted([r for r in records_for(pde, study) if r['dist'] == 'id'], key=lambda r:r['setting'])
    assert len(records) == (2 if pde == 'burger' else 5)
    normalizer = json.loads((Path(study)/'inputs'/pde/'normalizer.json').read_text())
    # Prespecified coarse grid: compare checkpoint stage, integration resolution
    # and extra RePaint visits. No formal-main errors enter this selection.
    base = SamplerConfig()
    candidates = [
        ('base', replace(base, steps=100, repaint=0, physics_steps=0, physics_post=0)),
        ('base', replace(base, steps=500, repaint=0)),
        ('base', replace(base, repaint=0)),
        ('base', replace(base, repaint=1)),
        ('base', replace(base, repaint=3)),
        ('control', replace(base, repaint=1)),
        ('control', replace(base, repaint=3)),
    ]
    cached_stage = None
    network = manifest = None
    def run(stage, config, start, stop):
        nonlocal network, manifest, cached_stage
        if cached_stage != stage:
            del network
            torch.cuda.empty_cache()
            network, manifest = load_network(pde, stage, device)
            cached_stage = stage
        return trial(pde, stage, network, manifest, records, normalizer, config,
                     start, stop, batch_size, study, device)
    coarse = [run(stage, config, 0, 4) for stage, config in candidates]
    best = min(coarse, key=lambda x:x['macro_mean'])
    best_config = SamplerConfig(**best['request']['config'])
    best_stage = best['request']['stage']
    # Check whether residual correction helps the reconstruction, and whether the
    # native implementation's larger physical update is appropriate on this PDE.
    for cfg in (replace(best_config, physics_steps=0, physics_post=0),
                replace(best_config, physics_steps=50, physics_post=10, max_physical_update=2e-3)):
        coarse.append(run(best_stage, cfg, 0, 4))
    unique = {json.dumps([x['request']['stage'], x['request']['config']], sort_keys=True):x for x in coarse}
    finalists = sorted(unique.values(), key=lambda x:x['macro_mean'])[:2]
    refined = [run(x['request']['stage'], SamplerConfig(**x['request']['config']), 0, 32) for x in finalists]
    winner = min(refined, key=lambda x:x['macro_mean'])
    selected = dict(pde=pde, stage=winner['request']['stage'], config=winner['request']['config'],
        checkpoint_sha256=winner['request']['checkpoint_sha256'],
        implementation_sha256=implementation_hash(), batch_size=batch_size,
        selection_rule='minimum macro physical relative L2 over five ID observation settings on calibration indices 1000..1031',
        validation_rule='frozen selection evaluated on disjoint 1032..1095, without retuning',
        formal_main_ids='2000..2999' if pde == 'nsnonbounded' else '0..999',
        coarse=[dict(label=x['label'], macro_mean=x['macro_mean'], seconds=x['seconds']) for x in coarse],
        refined=[dict(label=x['label'], macro_mean=x['macro_mean'], seconds=x['seconds']) for x in refined],
        selected_before_validation=True, status='selected')
    selected_path = Path(study)/'protocol/selected'/f'{pde}.json'
    write_json(selected_path, selected)
    validation = run(selected['stage'], SamplerConfig(**selected['config']), 32, 96)
    selected.update(validation_macro_mean=validation['macro_mean'],
        validation_settings={k:v['target_mean'] for k,v in validation['settings'].items()},
        validation_prediction_path=validation['prediction_path'], status='validated')
    total_samples = len(records)*64
    selected['estimated_main_gpu_hours'] = validation['seconds']/total_samples*15000/3600
    write_json(selected_path, selected)
    print(json.dumps(dict(pde=pde, stage='calibration_complete', **selected)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pde', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--study', type=Path, default=STUDY)
    args = parser.parse_args()
    calibrate(args.pde, torch.device(args.device), args.study, args.batch_size)
