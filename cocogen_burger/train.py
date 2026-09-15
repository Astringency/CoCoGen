"""Native CoCoGen score matching with resumable epoch-boundary DDP training.

The model is the archived UNET1 adapted to one full time-space channel. Loss,
VP schedule and AdamW follow the server implementation. No physics or observed
test values enter optimization. Training is blocked until first60 is verified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from cocogen_eval.common import STUDY, save_torch, sha_file, write_json
from cocogen_eval.network import UNET1
from .data import CachedDataset, require_first60


@dataclass(frozen=True)
class TrainConfig:
    batch_size_per_rank: int = 32
    max_epochs: int = 1200
    min_epochs: int = 300
    validation_every: int = 10
    patience_checks: int = 20
    relative_min_delta: float = .002
    lr: float = 1e-4
    weight_decay: float = .01
    seed: int = 101
    beta_min: float = 1e-4
    beta_max: float = 10.
    eps: float = 1e-5
    n_feat: int = 256
    resolution: int = 128
    pool_size: int = 4

    def __post_init__(self):
        assert 0 < self.min_epochs <= self.max_epochs
        assert min(self.batch_size_per_rank,self.validation_every,self.patience_checks)>0
        assert 0 <= self.relative_min_delta < 1 and 0 < self.eps < 1
        assert 0 < self.beta_min <= self.beta_max and self.lr>0
        assert self.resolution%(4*self.pool_size)==0 and self.n_feat%8==0


def model_config(cfg):
    return dict(model=dict(target='models.score_matching.DiffusionSDE', params=dict(
        lr=cfg.lr, eps=cfg.eps, elbo_weight=0.,
        unet_config=dict(target='cocogen_eval.network.UNET1', params=dict(
            in_channels=1, n_feat=cfg.n_feat, pool_size=cfg.pool_size,
            data_size=cfg.resolution, cond_size=1, controlnet=dict(use=False))),
        sde_config=dict(target='sdes.forward.VP', params=dict(beta_min=cfg.beta_min,beta_max=cfg.beta_max)))))


def score_loss(network, fields, times, noise, cfg):
    integral = cfg.beta_min*times+.5*(cfg.beta_max-cfg.beta_min)*times.square()
    sigma = torch.sqrt(-torch.expm1(-integral))[:,None,None,None]
    mean = torch.exp(-.5*integral)[:,None,None,None]*fields
    c = torch.full((len(fields),1), 4., device=fields.device)
    score = network(mean+sigma*noise, c, times, torch.zeros_like(c))
    return (score*sigma+noise).square().flatten(1).sum(1)


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if device.type=='cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def make_validation(fields, rank, world, device, seed):
    generator = torch.Generator().manual_seed(seed+100000)
    noise = torch.randn(fields.shape, generator=generator)
    times = torch.rand((len(fields),), generator=generator)*(1-1e-5)+1e-5
    index = slice(rank, len(fields), world)
    return fields[index].to(device), noise[index].to(device), times[index].to(device)


@torch.no_grad()
def validate(network, panel, cfg, distributed=False):
    network.eval()
    fields, noise, times = panel
    result = {}
    for label, fixed_t in [('uniform',None),('t001',.001),('t01',.01),('t1',.1),('t5',.5),('t10',1.)]:
        total = torch.zeros(2, dtype=torch.float64, device=fields.device)
        for offset in range(0,len(fields),cfg.batch_size_per_rank):
            x, z, t = (v[offset:offset+cfg.batch_size_per_rank] for v in (fields,noise,times))
            if fixed_t is not None:
                t = torch.full_like(t, fixed_t)
            loss = score_loss(network,x,t,z,cfg)
            total[0] += loss.double().sum(); total[1] += len(x)
        if distributed:
            dist.all_reduce(total)
        value = (total[0]/total[1]).item()
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite validation loss')
        result[label] = dict(loss_sum=value, loss_per_pixel=value/(cfg.resolution**2))
    return result


def code_identity():
    root = Path(__file__).resolve().parents[1]
    names = ['cocogen_burger/train.py','cocogen_burger/data.py','cocogen_eval/network.py']
    return {name:sha_file(root/name) for name in names}


def train(study, cfg, *, resume=True):
    study = Path(study)
    rank = int(os.environ.get('RANK',0)); local_rank = int(os.environ.get('LOCAL_RANK',0))
    world = int(os.environ.get('WORLD_SIZE',1))
    if not torch.cuda.is_available():
        raise RuntimeError('Real Burgers training requires CUDA; use the synthetic validation program for CPU checks')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    if world>1:
        dist.init_process_group('nccl')
    try:
        # All ranks check the gate structure; rank 0 also rehashes predictions.
        gate = require_first60(study, verify_predictions=(rank==0))
        if world>1: dist.barrier()
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < 30*1024**3:
            raise RuntimeError('Less than 30 GiB of free GPU memory before training')
        benchmark = json.loads((study/'validation/burger_training_gpu.json').read_text())
        assert benchmark['status']=='passed' and benchmark['batch_size_per_rank']>=cfg.batch_size_per_rank
        assert benchmark['n_feat']==cfg.n_feat and benchmark['resolution']==cfg.resolution
        assert benchmark['training_code_sha256']==code_identity()['cocogen_burger/train.py']
        torch.backends.cudnn.benchmark=True
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cuda.matmul.allow_tf32=True
        random.seed(cfg.seed+rank); np.random.seed(cfg.seed+rank); torch.manual_seed(cfg.seed+rank)
        out = study/'training/burger'; out.mkdir(parents=True,exist_ok=True)
        cache = json.loads((study/'inputs/burger/training_cache.json').read_text())
        if rank==0:
            assert sha_file(cache['training_path'])==cache['training_sha256']
            assert sha_file(cache['validation_path'])==cache['validation_sha256']
        normalizer = json.loads((study/'inputs/burger/normalizer.json').read_text())
        assert sha_file(study/'inputs/burger/normalizer.json')==cache['normalizer_sha256']
        request = dict(config=asdict(cfg), world_size=world, cache_sha256=sha_file(study/'inputs/burger/training_cache.json'),
            normalizer_sha256=cache['normalizer_sha256'], code=code_identity(), torch_version=str(torch.__version__),
            precision='float32', tf32=True, checkpoint_panel='1096..1351', sampler_panel='1000..1095', main_ids='0..999')
        model_cfg = model_config(cfg)
        network = UNET1(**model_cfg['model']['params']['unet_config']['params']).to(device)
        optimizer = torch.optim.AdamW(network.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        last = out/'checkpoints/last.ckpt'
        start_epoch, global_step = 0, 0
        best, plateau_reference, stale = float('inf'), float('inf'), 0
        saved_rng = None
        latest_validation = None
        stopped = False
        if last.exists():
            if not resume:
                raise FileExistsError(f'Existing training checkpoint: {last}')
            checkpoint = torch.load(last,map_location='cpu',weights_only=False)
            assert checkpoint['request']==request, 'Training resume configuration, data or code changed'
            network.load_state_dict({k.removeprefix('unet.'):v for k,v in checkpoint['state_dict'].items()},strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch, global_step = checkpoint['epoch']+1, checkpoint['global_step']
            best, plateau_reference, stale = checkpoint['best'], checkpoint['plateau_reference'], checkpoint['stale_checks']
            saved_rng = checkpoint['rank_rng'][rank]
            latest_validation = checkpoint['latest_validation']
            stopped = checkpoint['stop_condition_met']
        if rank==0:
            import yaml
            (out/'model.yaml').write_text(yaml.safe_dump(model_cfg,sort_keys=False))
            write_json(out/'request.json',dict(request=request,first60_gate=gate,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        wrapped = DistributedDataParallel(network,device_ids=[local_rank],broadcast_buffers=True) if world>1 else network
        dataset = CachedDataset(cache['training_path'])
        assert len(dataset)==50000
        sampler = DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=cfg.seed,drop_last=False)
        loader = DataLoader(dataset,batch_size=cfg.batch_size_per_rank,sampler=sampler,num_workers=0,pin_memory=True)
        validation = torch.load(cache['validation_path'],map_location='cpu',weights_only=False)
        assert validation['ids']==list(range(1096,1352))
        panel = make_validation(validation['fields'],rank,world,device,cfg.seed)
        if saved_rng is not None:
            restore_rng(saved_rng,device)
        if world>1: dist.barrier()
        run_start = time.monotonic()
        completed_epoch = start_epoch-1
        for epoch in range(start_epoch,cfg.max_epochs):
            if stopped:
                break
            epoch_start = time.monotonic()
            sampler.set_epoch(epoch); wrapped.train()
            total = torch.zeros(3,dtype=torch.float64,device=device)
            for fields in loader:
                fields = fields.to(device,non_blocking=True)
                t = torch.rand(len(fields),device=device)*(1-cfg.eps)+cfg.eps
                noise = torch.randn_like(fields)
                optimizer.zero_grad(set_to_none=True)
                losses = score_loss(wrapped,fields,t,noise,cfg)
                loss = losses.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss, epoch={epoch}, step={global_step}')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(),float('inf'),error_if_nonfinite=True)
                optimizer.step(); global_step += 1
                total[0] += losses.detach().double().sum(); total[1] += len(fields); total[2] += grad_norm.detach().double()
            if world>1:
                dist.all_reduce(total)
                # DDP broadcasts buffers before a train forward. Match rank 0
                # explicitly before independent validation calls and checkpointing.
                for value in network.buffers():
                    dist.broadcast(value,0)
            improved = False
            if (epoch+1)%cfg.validation_every==0 or epoch==0 or epoch+1==cfg.max_epochs:
                latest_validation = validate(network,panel,cfg,distributed=world>1)
                current = latest_validation['uniform']['loss_per_pixel']
                improved = current<best
                best = min(best,current)
                if current < plateau_reference*(1-cfg.relative_min_delta):
                    plateau_reference, stale = current, 0
                else:
                    stale += 1
            stopped = epoch+1>=cfg.min_epochs and stale>=cfg.patience_checks
            states = [None]*world
            own_rng = rng_state(device)
            if world>1:
                dist.all_gather_object(states,own_rng)
            else:
                states[0]=own_rng
            completed_epoch = epoch
            if rank==0:
                payload = dict(state_dict={f'unet.{k}':v for k,v in network.state_dict().items()},
                    optimizer=optimizer.state_dict(), epoch=epoch, global_step=global_step,
                    rank_rng=states, best=best, plateau_reference=plateau_reference, stale_checks=stale,
                    request=request, model_config=model_cfg, normalizer=normalizer,
                    latest_validation=latest_validation, stop_condition_met=stopped)
                # Record every epoch's receipt and retain 100-epoch snapshots;
                # `last` is the atomic resume point, and `best` follows validation.
                if improved:
                    save_torch(out/'checkpoints/best.ckpt',payload)
                save_torch(last,payload)
                if (epoch+1)%100==0:
                    save_torch(out/'checkpoints'/f'epoch_{epoch+1:04d}.ckpt',payload)
                seconds = time.monotonic()-epoch_start
                row = dict(epoch=epoch,epochs_completed=epoch+1,global_step=global_step,
                    train_loss_sum=(total[0]/total[1]).item(),
                    train_loss_per_pixel=(total[0]/total[1]).item()/cfg.resolution**2,
                    mean_gradient_norm=(total[2]/(len(loader)*world)).item(),
                    validation=latest_validation,validation_was_run=((epoch+1)%cfg.validation_every==0 or epoch==0 or epoch+1==cfg.max_epochs),
                    best_validation_per_pixel=best,stale_checks=stale,seconds=seconds,
                    seconds_since_resume=time.monotonic()-run_start,stop_condition_met=stopped,
                    last_sha256=sha_file(last),peak_bytes=torch.cuda.max_memory_allocated(device))
                write_json(out/'epochs'/f'{epoch+1:04d}.json',row)
                write_json(out/'progress.json',row)
                print(json.dumps(row),flush=True)
            if world>1: dist.barrier()
            if stopped:
                break
        if rank==0:
            assert completed_epoch+1>=cfg.min_epochs
            write_json(out/'complete.json',dict(status='trained',epochs_completed=completed_epoch+1,
                reason='validation_plateau' if stopped else 'max_epochs',request=request,
                best_checkpoint=str(out/'checkpoints/best.ckpt'),best_sha256=sha_file(out/'checkpoints/best.ckpt'),
                last_checkpoint=str(last),last_sha256=sha_file(last),
                next='select/check sampling configuration and complete six Burgers main cells'))
    finally:
        if world>1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=STUDY)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    config=TrainConfig(**json.loads(args.config.read_text())) if args.config else TrainConfig()
    train(args.study,config)
