"""CoCoGen calibration with an explicit score-evaluation cost ceiling."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .calibrate import implementation_hash, trial
from .common import STUDY, load_network, write_json
from .inputs import records_for
from .sampler import SamplerConfig


def configs(max_nfe):
    base=SamplerConfig()
    result=[replace(base,steps=100,repaint=0,physics_steps=0,physics_post=0)]
    for steps,repaint in [(100,0),(100,1),(100,3),(100,4),(250,0),(250,1),(250,3),
                          (500,0),(500,1),(500,3),(1000,0),(1000,1),(2000,0)]:
        if steps*(repaint+1)<=max_nfe:
            result.append(replace(base,steps=steps,repaint=repaint))
    return result


def calibrate(pde,device,study,batch_size,max_nfe):
    assert max_nfe>=100
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=True
    records=sorted([r for r in records_for(pde,study) if r['dist']=='id'],key=lambda r:r['setting'])
    assert len(records)==5
    normalizer=json.loads((Path(study)/'inputs'/pde/'normalizer.json').read_text())
    network=manifest=None
    cached_stage=None
    def run(stage,config,start,stop):
        nonlocal network,manifest,cached_stage
        if stage!=cached_stage:
            del network
            torch.cuda.empty_cache()
            network,manifest=load_network(pde,stage,device)
            cached_stage=stage
        return trial(pde,stage,network,manifest,records,normalizer,config,start,stop,batch_size,study,device)
    protocol=Path(study)/'protocol/cost_selection'/f'{pde}.json'
    grid=configs(max_nfe)
    write_json(protocol,dict(pde=pde,status='coarse_running',max_nfe=max_nfe,
        candidates=[dict(stage=stage,config=asdict(c)) for stage in ('base','control') for c in grid],
        selection='minimum macro relative L2 within NFE ceiling; 32 calibration and 64 separate validation samples per setting'))
    coarse=[run(stage,c,0,4) for stage in ('base','control') for c in grid]
    best=min(coarse,key=lambda r:r['macro_mean'])
    best_cfg=SamplerConfig(**best['request']['config'])
    best_stage=best['request']['stage']
    for cfg in (replace(best_cfg,physics_steps=0,physics_post=0),
                replace(best_cfg,physics_steps=50,physics_post=10,max_physical_update=2e-3)):
        coarse.append(run(best_stage,cfg,0,4))
    unique={json.dumps([r['request']['stage'],r['request']['config']],sort_keys=True):r for r in coarse}
    finalists=sorted(unique.values(),key=lambda r:r['macro_mean'])[:2]
    refined=[run(r['request']['stage'],SamplerConfig(**r['request']['config']),0,32) for r in finalists]
    winner=min(refined,key=lambda r:r['macro_mean'])
    long_reference=[]
    for path in (Path(study)/'calibration'/pde/implementation_hash()[:10]/'0-4').glob('*/receipt.json'):
        row=json.loads(path.read_text())
        cfg=row['request']['config']
        if cfg['steps']*(1+cfg['repaint'])>max_nfe:
            long_reference.append(dict(label=row['label'],macro_mean=row['macro_mean'],seconds=row['seconds'],
                n_per_setting=4,nfe=cfg['steps']*(1+cfg['repaint'])))
    selected=dict(pde=pde,stage=winner['request']['stage'],config=winner['request']['config'],
        checkpoint_sha256=winner['request']['checkpoint_sha256'],implementation_sha256=implementation_hash(),
        batch_size=batch_size,max_nfe=max_nfe,
        selection_rule='minimum macro physical relative L2 across five ID tasks on 1000..1031, subject to the NFE ceiling',
        validation_rule='frozen winner checked on 1032..1095 without retuning',
        formal_main_ids='2000..2999' if pde=='nsnonbounded' else '0..999',
        coarse=[dict(label=r['label'],macro_mean=r['macro_mean'],seconds=r['seconds']) for r in coarse],
        refined=[dict(label=r['label'],macro_mean=r['macro_mean'],seconds=r['seconds']) for r in refined],
        longer_reference_coarse_only=long_reference,selected_before_validation=True,status='selected')
    target=Path(study)/'protocol/selected'/f'{pde}.json'
    write_json(target,selected)
    validation=run(selected['stage'],SamplerConfig(**selected['config']),32,96)
    selected.update(status='validated',validation_macro_mean=validation['macro_mean'],
        validation_settings={k:v['target_mean'] for k,v in validation['settings'].items()},
        validation_prediction_path=validation['prediction_path'],
        estimated_main_gpu_hours=validation['seconds']/(len(records)*64)*15000/3600)
    write_json(target,selected)
    write_json(protocol,dict(status='complete',selected=selected))
    print(json.dumps(dict(event='cost_calibration_complete',**selected)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pde',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--max-nfe',type=int,default=500)
    parser.add_argument('--study',type=Path,default=STUDY)
    a=parser.parse_args()
    calibrate(a.pde,torch.device(a.device),a.study,a.batch_size,a.max_nfe)
