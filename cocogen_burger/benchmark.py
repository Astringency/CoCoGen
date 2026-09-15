"""Measure safe Burgers training batch size only after first60 is complete."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from cocogen_eval.common import STUDY, write_json
from cocogen_eval.network import UNET1
from .data import require_first60
from .train import TrainConfig, code_identity, model_config, score_loss


def benchmark(study,device):
    require_first60(study)
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info(device)
    if free<50*1024**3:
        raise RuntimeError('Training benchmark requires at least 50 GiB free memory')
    cfg=TrainConfig()
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=True
    results=[]
    peak_per_sample=None
    for batch in [1,4,8,16,32]:
        free,_=torch.cuda.mem_get_info(device)
        if peak_per_sample and peak_per_sample*batch>.60*free:
            break
        network=UNET1(**model_config(cfg)['model']['params']['unet_config']['params']).to(device).train()
        optimizer=torch.optim.AdamW(network.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        x=torch.randn(batch,1,128,128,device=device)
        def step():
            times=torch.rand(batch,device=device)*(1-cfg.eps)+cfg.eps
            noise=torch.randn_like(x)
            optimizer.zero_grad(set_to_none=True)
            loss=score_loss(network,x,times,noise,cfg).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),float('inf'),error_if_nonfinite=True)
            optimizer.step()
            return loss
        for _ in range(3): step()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started=time.monotonic()
        for _ in range(10): loss=step()
        torch.cuda.synchronize(device)
        seconds=(time.monotonic()-started)/10
        peak=torch.cuda.max_memory_allocated(device)
        if batch==1: peak_per_sample=peak
        assert torch.isfinite(loss)
        row=dict(batch=batch,seconds_per_train_step=seconds,peak_bytes=peak,
            estimated_two_gpu_seconds_per_epoch=((50000+2*batch-1)//(2*batch))*seconds,
            estimate_note='excludes DDP communication, validation and checkpoint writing; update from actual epochs')
        results.append(row)
        print(json.dumps(row),flush=True)
        del network,optimizer,x,loss
        torch.cuda.empty_cache()
    assert results and max(r['batch'] for r in results)>=8, 'No safe batch of at least 8 was demonstrated'
    chosen=max(r['batch'] for r in results)
    result=dict(status='passed',n_feat=cfg.n_feat,resolution=cfg.resolution,batch_size_per_rank=chosen,
        training_code_sha256=code_identity()['cocogen_burger/train.py'],measurements=results,
        device=str(device),gpu=torch.cuda.get_device_name(device),total_bytes=total,torch=str(torch.__version__))
    write_json(Path(study)/'validation/burger_training_gpu.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=STUDY)
    parser.add_argument('--device',default='cuda:0')
    a=parser.parse_args()
    benchmark(a.study,torch.device(a.device))
