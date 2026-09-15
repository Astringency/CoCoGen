"""Checkpoint selection, short sampler calibration and six Burgers main cells."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from cocogen_eval.calibrate import implementation_hash,trial
from cocogen_eval.common import STUDY,load_network,sha_file,write_json
from cocogen_eval.cost_calibrate import configs
from cocogen_eval.evaluate import aggregate,evaluate_cell
from cocogen_eval.inputs import records_for
from cocogen_eval.sampler import SamplerConfig
from .data import require_first60


def trained_model(study,kind,device):
    root=Path(study)/'training/burger'
    complete=json.loads((root/'complete.json').read_text())
    assert complete['status']=='trained' and kind in {'best','last'}
    path=complete[f'{kind}_checkpoint']
    assert sha_file(path)==complete[f'{kind}_sha256']
    return load_network('burger',stage=kind,device=device,checkpoint=path,config_path=root/'model.yaml')


def burger_records(study):
    records=sorted(records_for('burger',study),key=lambda r:r['cell'])
    assert len(records)==6
    for row in records:
        config=row['config']
        # The two aliases have distinct candidate masks, but FM4PDE's coefficient
        # observation weight is zero. Only the archived solution mask is active.
        assert row['task']=='both' and config['zeta_obs_a']==0 and config['zeta_obs_u']>0
        assert config['guidance_components']=='obs_pde' and config['noise_level']==0
        assert config['num_obs']==500
        if row['setting']=='sensor_column':
            assert config['num_sensor_columns']==5
    return records


def setup(device):
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    free,_=torch.cuda.mem_get_info(device)
    if free<20*1024**3:
        raise RuntimeError('Burgers sampling requires 20 GiB of free headroom')
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=True


def calibrate(study,device,batch_size=32,max_nfe=500):
    require_first60(study)
    setup(device)
    records=[r for r in burger_records(study) if r['dist']=='id']
    assert len(records)==2
    normalizer=json.loads((Path(study)/'inputs/burger/normalizer.json').read_text())
    network=manifest=None
    loaded_kind=None
    def run(kind,config,start,stop):
        nonlocal network,manifest,loaded_kind
        if kind!=loaded_kind:
            del network
            torch.cuda.empty_cache()
            network,manifest=trained_model(study,kind,device)
            loaded_kind=kind
        return trial('burger',kind,network,manifest,records,normalizer,config,
                     start,stop,batch_size,study,device)
    coarse=[run(kind,config,0,8) for kind in ['best','last'] for config in configs(max_nfe)]
    best=min(coarse,key=lambda r:r['macro_mean'])
    cfg=SamplerConfig(**best['request']['config'])
    kind=best['request']['stage']
    for candidate in [replace(cfg,physics_steps=0,physics_post=0),
                      replace(cfg,physics_steps=50,physics_post=10,max_physical_update=.002),
                      replace(cfg,physics_steps=50,physics_post=10,max_physical_update=.02)]:
        coarse.append(run(kind,candidate,0,8))
    unique={json.dumps([r['request']['stage'],r['request']['config']],sort_keys=True):r for r in coarse}
    finalists=sorted(unique.values(),key=lambda r:r['macro_mean'])[:2]
    refined=[run(r['request']['stage'],SamplerConfig(**r['request']['config']),0,32) for r in finalists]
    winner=min(refined,key=lambda r:r['macro_mean'])
    selected=dict(pde='burger',stage=winner['request']['stage'],config=winner['request']['config'],
        checkpoint_sha256=winner['request']['checkpoint_sha256'],implementation_sha256=implementation_hash(),
        batch_size=batch_size,max_nfe=max_nfe,status='selected',selected_before_validation=True,
        selection_rule='minimum macro physical relative L2 for random500 and five sensor columns on ID 1000..1031',
        validation_rule='frozen winner tested on separate 1032..1095 without retuning',
        formal_main_ids='0..999',training_checkpoint_panel='1096..1351',
        coarse=[dict(label=r['label'],macro_mean=r['macro_mean'],seconds=r['seconds']) for r in coarse],
        refined=[dict(label=r['label'],macro_mean=r['macro_mean'],seconds=r['seconds']) for r in refined])
    target=Path(study)/'protocol/selected/burger.json'
    write_json(target,selected)
    validation=run(selected['stage'],SamplerConfig(**selected['config']),32,96)
    selected.update(status='validated',validation_macro_mean=validation['macro_mean'],
        validation_settings={k:v['target_mean'] for k,v in validation['settings'].items()},
        validation_prediction_path=validation['prediction_path'],
        estimated_main_gpu_hours=validation['seconds']/128*6000/3600)
    write_json(target,selected)
    print(json.dumps(dict(event='burger_calibration_complete',**selected)),flush=True)


def worker(study,device,worker_id,batch_size=32):
    assert worker_id in (0,1)
    require_first60(study)
    setup(device)
    selected=json.loads((Path(study)/'protocol/selected/burger.json').read_text())
    assert selected['status']=='validated' and selected['implementation_sha256']==implementation_hash()
    network,manifest=trained_model(study,selected['stage'],device)
    assert manifest['checkpoint_sha256']==selected['checkpoint_sha256']
    normalizer=json.loads((Path(study)/'inputs/burger/normalizer.json').read_text())
    records=burger_records(study)[worker_id::2]
    result=[]
    for record in records:
        row=evaluate_cell('burger',record,network,manifest,selected,normalizer,batch_size,device,study)
        result.append(dict(cell=row['cell'],n=row['n'],summary=str(Path(study)/'main'/row['cell']/'summary.json')))
        write_json(Path(study)/'main/burger/workers'/f'{worker_id}_progress.json',
            dict(status='running',worker_id=worker_id,cells=result))
    assert len(result)==3
    write_json(Path(study)/'main/burger/workers'/f'{worker_id}_complete.json',
        dict(status='complete',worker_id=worker_id,cells=result))


def finish(study):
    cells=[]
    for worker_id in [0,1]:
        record=json.loads((Path(study)/'main/burger/workers'/f'{worker_id}_complete.json').read_text())
        assert record['status']=='complete' and record['worker_id']==worker_id
        cells.extend(record['cells'])
    expected={r['cell'] for r in burger_records(study)}
    assert len(cells)==6 and {r['cell'] for r in cells}==expected
    for entry in cells:
        record=json.loads(Path(entry['summary']).read_text())
        assert record['status']=='complete' and record['n']==1000 and record['ids']==list(range(1000))
        for batch in record['batches']:
            assert sha_file(batch['prediction_path'])==batch['prediction_sha256']
    selected=json.loads((Path(study)/'protocol/selected/burger.json').read_text())
    write_json(Path(study)/'main/burger/complete.json',dict(status='complete',pde='burger',
        cells=cells,samples=6000,selected=selected))
    aggregate(study,include_burger=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=STUDY)
    parser.add_argument('--mode',choices=['calibrate','worker','finish'],required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--worker-id',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    if args.mode=='calibrate':
        calibrate(args.study,torch.device(args.device),args.batch_size)
    elif args.mode=='worker':
        worker(args.study,torch.device(args.device),args.worker_id,args.batch_size)
    else:
        finish(args.study)
