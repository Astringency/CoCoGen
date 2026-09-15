"""Small, separate Darcy sampling-budget diagnostic; never changes main results."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import torch

from cocogen_eval.calibrate import implementation_hash
from cocogen_eval.common import RUNS, REMOTE_REPO, load_network, save_torch, sha_file, write_json
from cocogen_eval.inputs import calibration_input, identity, physics_function, records_for, scores, target_score
from cocogen_eval.sampler import SamplerConfig, sample

SETTINGS = ('sparse_forward', 'sparse_inverse', 'sparse_joint')
CONFIG_NAMES = ('s100_r4', 's2000_r4', 's2000_r10')
ASSIGNMENTS = (
    [('s100_r4', setting) for setting in SETTINGS] + [('s2000_r10', 'sparse_forward')],
    [('s2000_r4', setting) for setting in SETTINGS],
    [('s2000_r10', 'sparse_inverse')],
    [('s2000_r10', 'sparse_joint')],
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def prepare(root, study):
    require(not (root / 'manifest.json').exists(), 'Diagnostic assets already exist')
    selected_path = study / 'protocol/selected/darcy.json'
    selected = json.loads(selected_path.read_text())
    require(selected['status'] == 'validated' and selected['stage'] == 'control', 'Wrong frozen selection')
    require(selected['implementation_sha256'] == implementation_hash(), 'Sampling source differs')
    base = SamplerConfig(**selected['config'])
    require((base.steps, base.repaint) == (100, 4), 'Wrong short baseline')
    run = REMOTE_REPO / 'output/withcontrol' / RUNS['darcy'][1]
    sources = {'checkpoint.ckpt': run / 'checkpoints/last.ckpt',
               'model.yaml': next((run / 'configs').glob('*-model.yaml')),
               'normalizer.json': study / 'inputs/darcy/normalizer.json'}
    require(sha_file(sources['checkpoint.ckpt']) == selected['checkpoint_sha256'], 'Changed checkpoint')
    assets = root / 'assets'; assets.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, source in sources.items():
        dest = assets / name
        require(not dest.exists(), f'Existing asset: {dest}')
        shutil.copyfile(source, dest)
        digest = sha_file(source)
        require(sha_file(dest) == digest, 'Asset copy differs')
        provenance[name] = dict(source=str(source), sha256=digest, bytes=dest.stat().st_size)
    records = {r['setting']: r for r in records_for('darcy', study) if r['dist'] == 'id'}
    groups = {}
    for setting in SETTINGS:
        record = records[setting]
        fields, masks, ids, params = calibration_input('darcy', record, list(range(8)), study, 'cpu')
        require(ids == list(range(1000, 1008)), 'Diagnostic must use calibration IDs 1000..1007')
        groups[setting] = dict(fields=fields.cpu(), masks=masks.cpu(), ids=ids, params=params,
                               config=record['config'], task=record['task'])
    save_torch(assets / 'inputs.pt', groups)
    configs = dict(zip(CONFIG_NAMES, [asdict(base), asdict(replace(base, steps=2000)),
                                      asdict(replace(base, steps=2000, repaint=10))]))
    write_json(root / 'manifest.json', dict(status='prepared', pde='darcy', stage='control',
        created_at=datetime.now(timezone.utc).isoformat(), source_study=str(study),
        source_selected_sha256=sha_file(selected_path), assets=provenance,
        source_calibration_sha256=sha_file(study / 'inputs/darcy/calibration.pt'),
        inputs_sha256=sha_file(assets / 'inputs.pt'), implementation_sha256=implementation_hash(),
        diagnostic_source_sha256=sha_file(__file__), configs=configs, assignments=ASSIGNMENTS,
        ids=list(range(1000, 1008)), batch_size=8, settings=SETTINGS, seed=base.seed,
        precision='float32', tf32=True, cudnn_benchmark=False,
        scope='Exploratory ID calibration diagnostic, 8 cases x 3 sparse tasks x 3 configurations. '
              'IDs were used in prior calibration, never in formal main evaluation. No main retuning.',
        comparison_note='Compare all configurations on the same A800 runtime and batch size. '
                        'Do not substitute the A100/batch32 formal score for this diagnostic baseline.'))
    print(json.dumps(dict(status='prepared', root=str(root), manifest_sha256=sha_file(root/'manifest.json'))), flush=True)


def inputs(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['implementation_sha256'] == implementation_hash(), 'Wrong sampling implementation')
    require(manifest['diagnostic_source_sha256'] == sha_file(__file__), 'Changed diagnostic source')
    for name, asset in manifest['assets'].items():
        require(sha_file(root / 'assets' / name) == asset['sha256'], f'Changed asset: {name}')
    require(sha_file(root/'assets/inputs.pt') == manifest['inputs_sha256'], 'Changed diagnostic inputs')
    groups = torch.load(root/'assets/inputs.pt', map_location='cpu', weights_only=False)
    require(set(groups) == set(SETTINGS), 'Wrong diagnostic task coverage')
    for setting, group in groups.items():
        require(group['ids'] == manifest['ids'], 'Wrong diagnostic IDs')
        counts = group['masks'].flatten(2).sum(2)
        expected = [500 if setting != 'sparse_inverse' else 0,
                    500 if setting != 'sparse_forward' else 0]
        require(torch.equal(counts, torch.tensor(expected).expand(8, 2)), 'Wrong effective observations')
    return manifest, groups, json.loads((root/'assets/normalizer.json').read_text())


def runtime(root, device):
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    require(free >= 50 * 1024**3, 'Diagnostic requires 50 GiB free before loading its model')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest, groups, norm = inputs(root)
    network, checkpoint = load_network('darcy', 'control', device,
        checkpoint=root/'assets/checkpoint.ckpt', config_path=root/'assets/model.yaml')
    require(checkpoint['checkpoint_sha256'] == manifest['assets']['checkpoint.ckpt']['sha256'], 'Wrong model')
    environment = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), device=str(device), gpu=torch.cuda.get_device_name(device),
        total_bytes=total, free_bytes_before_load=free, precision='float32', tf32=True, cudnn_benchmark=False,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())
    return manifest, groups, norm, network, checkpoint, environment


def benchmark(root, device):
    manifest, groups, norm, network, checkpoint, environment = runtime(root, device)
    group = groups['sparse_joint']; physical = physics_function('darcy', group['config'], group['params'])
    cfg = replace(SamplerConfig(**manifest['configs']['s100_r4']), steps=4, repaint=0,
                  physics_steps=2, physics_post=1)
    measurements = []
    for n in (1, 4, 8):
        x, mask = group['fields'][:n].to(device), group['masks'][:n].to(device)
        pred, receipt = sample(network, torch.where(mask, x, 0.), mask, group['ids'][:n], norm,
            checkpoint['sde'], checkpoint['conditioning_label'], cfg, residual_fn=physical, namespace='darcy')
        require(torch.equal(pred[mask], x[mask]), 'Benchmark observation mismatch')
        measurements.append(dict(batch=n, peak_bytes=receipt['peak_bytes'], seconds=receipt['seconds']))
        del pred
    with torch.no_grad():
        t = torch.full((8,), .5, device=device); c = torch.zeros((8, network.cond_size), device=device)
        torch.cuda.synchronize(device); start = time.monotonic()
        for _ in range(30):
            output = network(x, c, t, c)
        torch.cuda.synchronize(device)
        seconds = (time.monotonic() - start) / 30
        require(torch.isfinite(output).all().item(), 'Invalid benchmark network output')
    predicted = [sum(manifest['configs'][name]['steps']*(1+manifest['configs'][name]['repaint'])
                     for name, _ in assignment) * seconds for assignment in ASSIGNMENTS]
    result = dict(status='passed', manifest_sha256=sha_file(root/'manifest.json'), environment=environment,
        memory_measurements=measurements, seconds_per_batch8_score=seconds,
        estimated_worker_network_seconds=predicted,
        estimate_note='Network-only projection; excludes physics corrections, noise generation, loading and I/O. '
                      'Four-step probes verify memory and finiteness, not long-run reconstruction quality.')
    write_json(root/'benchmark.json', result)
    print(json.dumps(result), flush=True)


def worker(root, device, worker_id):
    benchmark_result = json.loads((root/'benchmark.json').read_text())
    require(benchmark_result['status'] == 'passed' and
            benchmark_result['manifest_sha256'] == sha_file(root/'manifest.json'), 'Missing matching memory benchmark')
    manifest, groups, norm, network, checkpoint, environment = runtime(root, device)
    jobs = manifest['assignments'][worker_id]
    for name, setting in jobs:
        group = groups[setting]; cfg = SamplerConfig(**manifest['configs'][name])
        out = root/'runs'/name/setting
        request = dict(manifest_sha256=sha_file(root/'manifest.json'), config=asdict(cfg), setting=setting,
            task=group['task'], input=identity(group['fields'], group['masks'], group['ids']),
            checkpoint_sha256=checkpoint['checkpoint_sha256'], implementation_sha256=implementation_hash(),
            environment=environment, worker_id=worker_id, batch_size=8, namespace='darcy')
        receipt_path = out/'receipt.json'
        if receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            old_request = dict(old['request']); current = dict(request)
            # Available GPU memory and a reporting-only Git revision may change after a restart.
            for value in (old_request, current):
                value['environment'] = {k:v for k,v in value['environment'].items()
                                        if k not in ('free_bytes_before_load', 'git_commit')}
            require(old_request == current and sha_file(out/'prediction.pt') == old['prediction_sha256'],
                    'Existing diagnostic result differs')
            continue
        x, mask = group['fields'].to(device), group['masks'].to(device)
        physical = physics_function('darcy', group['config'], group['params'])
        def progress(row):
            event = dict(worker=worker_id, config=name, setting=setting, **row)
            write_json(root/'workers'/f'{worker_id}_progress.json', event)
            print(json.dumps(event), flush=True)
        progress(dict(event='started', total_nfe=cfg.steps*(1+cfg.repaint)))
        prediction, sampling = sample(network, torch.where(mask, x, 0.), mask, group['ids'], norm,
            checkpoint['sde'], checkpoint['conditioning_label'], cfg, residual_fn=physical,
            namespace='darcy', progress=progress)
        require(torch.equal(prediction[mask], x[mask]), 'Changed observations')
        errors = scores(prediction, x, 'darcy')
        residual = physical(prediction).flatten(1).square().mean(1).cpu().double().tolist()
        truth_residual = physical(x).flatten(1).square().mean(1).cpu().double().tolist()
        path = out/'prediction.pt'
        save_torch(path, dict(prediction=prediction.cpu(), ids=group['ids'], request=request,
                             sampling=sampling, errors=errors))
        result = dict(status='complete', request=request, sampling=sampling, errors=errors,
            target_mean=target_score(errors, group['task']), predicted_residual_mse=residual,
            truth_residual_mse=truth_residual, observed_max_absolute_error=0.0,
            prediction_file='prediction.pt', prediction_sha256=sha_file(path))
        write_json(receipt_path, result)
        progress(dict(event='complete', target_mean=result['target_mean'], seconds=sampling['seconds']))
        del prediction, x, mask
    write_json(root/'workers'/f'{worker_id}_complete.json', dict(status='complete', worker=worker_id,
        jobs=jobs, manifest_sha256=sha_file(root/'manifest.json')))


def collect(root):
    manifest, groups, _ = inputs(root)
    results = []
    for name in CONFIG_NAMES:
        for setting in SETTINGS:
            out = root/'runs'/name/setting
            r = json.loads((out/'receipt.json').read_text()); group = groups[setting]
            require(r['status']=='complete' and r['request']['config']==manifest['configs'][name], 'Incomplete result')
            require(r['request']['setting']==setting and r['request']['task']==group['task'], 'Wrong task')
            require(r['request']['manifest_sha256']==sha_file(root/'manifest.json'), 'Wrong diagnostic manifest')
            require(r['request']['checkpoint_sha256']==manifest['assets']['checkpoint.ckpt']['sha256']
                    and r['request']['implementation_sha256']==manifest['implementation_sha256'], 'Wrong model/source')
            require(sha_file(out/'prediction.pt')==r['prediction_sha256'], 'Changed prediction')
            payload = torch.load(out/'prediction.pt', map_location='cpu', weights_only=False)
            require(payload['request']==r['request'] and payload['sampling']==r['sampling'], 'Payload differs')
            require(payload['ids']==manifest['ids'], 'Wrong diagnostic sample IDs')
            require(r['request']['input']==identity(group['fields'],group['masks'],group['ids']), 'Changed input')
            require(torch.equal(payload['prediction'][group['masks']],group['fields'][group['masks']]), 'Changed observations')
            actual = scores(payload['prediction'],group['fields'],'darcy')
            for field in ('a','u'):
                require(len(r['errors'][field])==8 and all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
                    for a,b in zip(actual[field],r['errors'][field])), 'Wrong saved errors')
            nfe = manifest['configs'][name]['steps']*(1+manifest['configs'][name]['repaint'])
            require(r['sampling']['config']==manifest['configs'][name] and
                    r['sampling']['nfe']==nfe and r['sampling']['final_time']==0, 'Wrong schedule/NFE/final time')
            require(math.isclose(target_score(actual,group['task']),r['target_mean'],rel_tol=1e-12), 'Wrong target mean')
            results.append(dict(config=name,setting=setting,task=group['task'],n=8,nfe=nfe,
                target_mean=r['target_mean'],mean_errors={k:statistics.fmean(v) for k,v in actual.items()},
                errors=actual,seconds=r['sampling']['seconds'],peak_bytes=r['sampling']['peak_bytes'],
                initial_noise_sha256=r['sampling']['initial_noise_sha256'],
                prediction_sha256=r['prediction_sha256'],receipt_sha256=sha_file(out/'receipt.json')))
    comparisons=[]
    for setting in SETTINGS:
        baseline=next(r for r in results if r['config']=='s100_r4' and r['setting']==setting)
        for r in [r for r in results if r['setting']==setting]:
            require(r['initial_noise_sha256']==baseline['initial_noise_sha256'], 'Initial random fields differ')
            comparisons.append(dict(setting=setting,config=r['config'],target_mean=r['target_mean'],
                baseline_target_mean=baseline['target_mean'],difference=r['target_mean']-baseline['target_mean'],
                relative_error_reduction=1-r['target_mean']/baseline['target_mean']))
    result=dict(status='complete_diagnostic_only',manifest_sha256=sha_file(root/'manifest.json'),
        scope=manifest['scope'],results=results,comparisons=comparisons,
        config_macros={name:statistics.fmean(r['target_mean'] for r in results if r['config']==name)
                       for name in CONFIG_NAMES},
        limitations='Eight calibration cases, one seed, ID only. Not a formal-main estimate or a multi-seed study. '
                    'Changing repaint also changes the number of physical-correction visits; this tests the full schedule.',
        collector_sha256=sha_file(__file__))
    write_json(root/'comparison.json',result)
    print(json.dumps(dict(status=result['status'],config_macros=result['config_macros'],comparisons=comparisons)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--study',type=Path,required=True)
    bench=sub.add_parser('benchmark');bench.add_argument('--device',required=True)
    work=sub.add_parser('worker');work.add_argument('--device',required=True);work.add_argument('--worker-id',type=int,choices=range(4),required=True)
    sub.add_parser('collect')
    args=parser.parse_args();root=args.root.resolve();torch.set_num_threads(4)
    if args.command=='prepare':prepare(root,args.study.resolve())
    elif args.command=='benchmark':benchmark(root,torch.device(args.device))
    elif args.command=='worker':worker(root,torch.device(args.device),args.worker_id)
    else:collect(root)
