"""Verify completed training states on CUDA/DDP without restarting training.

This restores model, optimizer and each rank's RNG, then runs one forward
batch. It never performs backward or an optimizer step and never rewrites
training metadata or checkpoint files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def verify_terminal(study):
    from .external_stop import read, sha, verify_completion

    study=Path(study).resolve()
    complete=read(study/'training/burger/complete.json')
    if complete.get('reason')=='user_authorized_validation_plateau':
        return verify_completion(study)
    request=read(study/'training/burger/request.json')['request']
    if complete['status']!='trained' or complete['request']!=request:
        raise ValueError('Training completion differs from its saved request')
    if (study/'logs/cocogen_burger_train.exit').read_text().strip()!='0':
        raise ValueError('Natural completion requires the original successful exit')
    cfg=request['config']
    if complete['reason']=='validation_plateau':
        if complete['epochs_completed']<cfg['min_epochs']:
            raise ValueError('Natural plateau completion precedes its minimum epoch')
    elif complete['reason']=='max_epochs':
        if complete['epochs_completed']!=cfg['max_epochs']:
            raise ValueError('Maximum-epoch completion has the wrong epoch')
    else:
        raise ValueError('Unknown training completion reason')
    for kind in ('best','last'):
        path=Path(complete[f'{kind}_checkpoint'])
        if not path.resolve().is_relative_to(study) or sha(path)!=complete[f'{kind}_sha256']:
            raise ValueError('Terminal checkpoint location or hash differs')
    return complete


def restore(root,original_study,output):
    import builtins
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from cocogen_eval.network import UNET1
    from .archive_paths import mapped_study_reads
    from .train import TrainConfig, code_identity, restore_rng, score_loss

    root,original_study,output=Path(root).resolve(strict=True),Path(original_study),Path(output)
    rank=int(os.environ['RANK']); local_rank=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    assert world==2 and not output.exists()
    device=torch.device('cuda',local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    assert torch.cuda.mem_get_info(device)[0]>=30*1024**3
    torch.cuda.reset_peak_memory_stats(device)
    dist.init_process_group('nccl')
    try:
        normalizer=json.loads((root/'inputs/burger/normalizer.json').read_text())
        blocked=[row['path'] for row in normalizer['sources']]
        raw_open=builtins.open
        with mapped_study_reads(root,original_study,blocked) as mapping:
            for path in (original_study/'protocol/catalog.json',Path(blocked[0])):
                try:
                    raw_open(path,'rb')
                except PermissionError:
                    pass
                else:
                    raise AssertionError('Original-path denial failed')
            complete=verify_terminal(original_study)
            request=complete['request']
            assert request['world_size']==world and request['code']==code_identity()
            cfg=TrainConfig(**request['config'])
            cache=json.loads((root/'inputs/burger/training_cache.json').read_text())
            validation=torch.load(cache['validation_path'],map_location='cpu',weights_only=False)
            assert validation['ids']==list(range(1096,1352))
            records=[]
            for kind in ('best','last'):
                checkpoint=torch.load(complete[f'{kind}_checkpoint'],map_location='cpu',weights_only=False)
                assert checkpoint['request']==request and checkpoint['normalizer']==normalizer
                if kind=='last':
                    assert checkpoint['epoch']+1==complete['epochs_completed']
                    if complete['reason']=='validation_plateau':
                        assert checkpoint['stop_condition_met']
                network=UNET1(**checkpoint['model_config']['model']['params']['unet_config']['params']).to(device)
                state={k.removeprefix('unet.'):v for k,v in checkpoint['state_dict'].items()}
                network.load_state_dict(state,strict=True)
                optimizer=torch.optim.AdamW(network.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
                optimizer.load_state_dict(checkpoint['optimizer'])
                wrapped=DistributedDataParallel(network,device_ids=[local_rank],broadcast_buffers=True)
                assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in network.state_dict().items())
                saved=checkpoint['optimizer']
                saved_ids=[p for group in saved['param_groups'] for p in group['params']]
                for parameter,identifier in zip(network.parameters(),saved_ids,strict=True):
                    actual=optimizer.state[parameter]
                    expected=saved['state'][identifier]
                    assert actual.keys()==expected.keys()
                    assert all(torch.equal(actual[k].cpu(),expected[k]) for k in actual)
                    assert actual['step'].item()==checkpoint['global_step']
                rng=checkpoint['rank_rng'][rank]
                restore_rng(rng,device)
                assert random.getstate()==rng['python']
                np_state=np.random.get_state()
                assert np_state[0]==rng['numpy'][0] and np.array_equal(np_state[1],rng['numpy'][1])
                assert np_state[2:]==rng['numpy'][2:]
                assert torch.equal(torch.get_rng_state(),rng['cpu'])
                assert torch.equal(torch.cuda.get_rng_state(device),rng['cuda'])
                wrapped.eval()
                with torch.no_grad():
                    fields=validation['fields'][rank:rank+1].to(device)
                    loss=score_loss(wrapped,fields,torch.full((1,),.5,device=device),torch.zeros_like(fields),cfg)
                    assert torch.isfinite(loss).all()
                    dist.all_reduce(loss)
                    assert torch.isfinite(loss).all()
                records.append(dict(kind=kind,epoch=checkpoint['epoch']+1,global_step=checkpoint['global_step'],
                    checkpoint_sha256=complete[f'{kind}_sha256'],model_and_optimizer_exact=True,
                    python_numpy_cpu_cuda_rng_exact=True,ddp_forward_and_collective_finite=True,
                    original_stop_condition_met=checkpoint['stop_condition_met']))
                del wrapped,network,optimizer,checkpoint,state,fields,loss,actual,expected,saved
                torch.cuda.empty_cache()
            assert len(mapping['direct_original_access_attempts'])==2
            local=dict(rank=rank,checkpoints=records,mapped_opens=mapping['mapped_opens'],
                       blocked_canary_reads=2,unexpected_original_access_attempts=0,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
            ranks=[None]*world
            dist.all_gather_object(ranks,local)
        if rank==0:
            with output.open('x') as stream:
                json.dump(dict(status='passed',scope='CUDA/DDP state restore and forward; no resumed optimization',
                    optimization_steps=0,training_files_rewritten=False,ranks=ranks,
                    helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),stream,indent=2)
                stream.write('\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--original-study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    restore(args.root,args.original_study,args.output)
