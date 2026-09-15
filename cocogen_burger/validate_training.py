"""CPU synthetic checks: no real Burgers data is optimized by these tests."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import torch

from cocogen_eval.common import STUDY,load_network,save_torch,sha_file,write_json
from cocogen_eval.network import UNET1
from .data import CachedDataset,require_first60
from .train import TrainConfig,code_identity,make_validation,model_config,restore_rng,rng_state,score_loss,validate


def checks(study):
    torch.set_num_threads(1)
    torch.manual_seed(73)
    device=torch.device('cpu')
    cfg=replace(TrainConfig(),n_feat=8,resolution=32,batch_size_per_rank=2)
    fields=torch.randn(4,1,32,32)
    times=torch.tensor([.001,.01,.2,1.])
    noise=torch.randn_like(fields)
    class Oracle:
        def __call__(self,x,c,t,context):
            integral=cfg.beta_min*t+.5*(cfg.beta_max-cfg.beta_min)*t.square()
            sigma=torch.sqrt(-torch.expm1(-integral))[:,None,None,None]
            return -noise/sigma
    assert score_loss(Oracle(),fields,times,noise,cfg).max()<1e-10
    class Zero:
        def __call__(self,x,c,t,context): return torch.zeros_like(x)
    assert torch.equal(score_loss(Zero(),fields,times,torch.ones_like(fields),cfg),torch.full((4,),1024.))
    def create():
        net=UNET1(**model_config(cfg)['model']['params']['unet_config']['params'])
        return net,torch.optim.AdamW(net.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    def step(net,opt):
        net.train(); opt.zero_grad(set_to_none=True)
        t=torch.rand(len(fields))*(1-cfg.eps)+cfg.eps
        loss=score_loss(net,fields,t,torch.randn_like(fields),cfg).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),float('inf'),error_if_nonfinite=True)
        opt.step()
        return loss.detach().clone()
    folder=Path(study)/'validation';folder.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='burger_cpu_test_',dir=folder) as tmp:
        path=Path(tmp)
        try:
            require_first60(path)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Missing first60 gate was accepted')
        np.save(path/'cache.npy',fields.numpy())
        cached=CachedDataset(path/'cache.npy')
        item=cached[0]; item.zero_()
        assert torch.equal(cached[0],fields[0]), 'Dataset item writes leaked into cache'
        net,opt=create()
        step(net,opt);step(net,opt)
        checkpoint=path/'resume.pt'
        save_torch(checkpoint,dict(network=net.state_dict(),optimizer=opt.state_dict(),rng=rng_state(device)))
        expected_loss=step(net,opt)
        expected={k:v.clone() for k,v in net.state_dict().items()}
        resumed,resumed_opt=create()
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        resumed.load_state_dict(payload['network'],strict=True);resumed_opt.load_state_dict(payload['optimizer'])
        restore_rng(payload['rng'],device)
        actual_loss=step(resumed,resumed_opt)
        assert torch.equal(actual_loss,expected_loss)
        assert all(torch.equal(v,expected[k]) for k,v in resumed.state_dict().items())
        first=validate(resumed,make_validation(fields,0,1,device,73),cfg)
        second=validate(resumed,make_validation(fields,0,1,device,73),cfg)
        assert first==second
        ranks=[make_validation(fields,r,2,device,73) for r in [0,1]]
        whole=make_validation(fields,0,1,device,73)
        for col in range(3):
            reconstructed=torch.empty_like(whole[col])
            reconstructed[::2]=ranks[0][col];reconstructed[1::2]=ranks[1][col]
            assert torch.equal(reconstructed,whole[col])
        import yaml
        model_path=path/'model.yaml';model_path.write_text(yaml.safe_dump(model_config(cfg)))
        weights_path=path/'sample.ckpt'
        save_torch(weights_path,dict(state_dict={f'unet.{k}':v for k,v in resumed.state_dict().items()},epoch=2,global_step=3))
        loaded,manifest=load_network('burger',stage='best',device=device,checkpoint=weights_path,config_path=model_path)
        with torch.no_grad():
            context=torch.full((len(fields),1),4.)
            a=resumed(fields,context,times,torch.zeros_like(context))
            b=loaded(fields,context,times,torch.zeros_like(context))
            assert torch.equal(a,b) and loaded.in_channels==1 and manifest['conditioning_label']==4
        stopped_resume_check(path/'stopped_resume',resumed,resumed_opt,cfg)
    write_json(folder/'burger_training_cpu.json',dict(status='passed',synthetic=True,
        training_code=code_identity(),
        checks=['first60 prerequisite rejects missing completion','cache items do not modify stored data',
            'score matching noise target and summed-loss scale','strict model/optimizer/RNG resume reproduces next update',
            'fixed validation is repeatable','rank partition preserves validation fields, times and noise',
            'one-channel checkpoint loads into the sampling network without output changes',
            'actual training entry recovers a stopped checkpoint without reading training data or updating weights'],
        limitation='Real two-GPU training and batch-size benchmark must wait until first60 is complete'))


def stopped_resume_check(study,network,optimizer,cfg):
    """Exercise the actual train entry using CPU stand-ins for its CUDA gate.

    No real training data or CUDA kernels are used. Dataset access fails if the
    resumed entry accidentally attempts another epoch after the recorded stop.
    """
    import importlib
    import json
    from types import SimpleNamespace
    from unittest.mock import patch
    training=importlib.import_module('cocogen_burger.train')
    cfg=replace(cfg,min_epochs=300,max_epochs=301)
    inputs=study/'inputs/burger';inputs.mkdir(parents=True)
    normalizer=dict(mean=[0.],std=[1.],eps=1e-8)
    write_json(inputs/'normalizer.json',normalizer)
    data=inputs/'synthetic.npy';np.save(data,np.zeros((1,1,cfg.resolution,cfg.resolution),dtype=np.float32))
    validation=inputs/'validation.pt'
    save_torch(validation,dict(fields=torch.zeros(256,1,cfg.resolution,cfg.resolution),ids=list(range(1096,1352))))
    norm_sha=sha_file(inputs/'normalizer.json')
    write_json(inputs/'training_cache.json',dict(training_path=str(data),training_sha256=sha_file(data),
        validation_path=str(validation),validation_sha256=sha_file(validation),normalizer_sha256=norm_sha))
    from dataclasses import asdict
    request=dict(config=asdict(cfg),world_size=1,cache_sha256=sha_file(inputs/'training_cache.json'),
        normalizer_sha256=norm_sha,code=code_identity(),torch_version=str(torch.__version__),precision='float32',
        tf32=True,checkpoint_panel='1096..1351',sampler_panel='1000..1095',main_ids='0..999')
    write_json(study/'validation/burger_training_gpu.json',dict(status='passed',batch_size_per_rank=cfg.batch_size_per_rank,
        n_feat=cfg.n_feat,resolution=cfg.resolution,training_code_sha256=code_identity()['cocogen_burger/train.py']))
    out=study/'training/burger'
    latest=dict(uniform=dict(loss_per_pixel=.25,loss_sum=.25*cfg.resolution**2))
    payload=dict(state_dict={f'unet.{k}':v for k,v in network.state_dict().items()},optimizer=optimizer.state_dict(),
        epoch=299,global_step=300,best=.25,plateau_reference=.25,stale_checks=cfg.patience_checks,
        rank_rng=[rng_state(torch.device('cpu'))],request=request,latest_validation=latest,stop_condition_met=True)
    last=out/'checkpoints/last.ckpt';best=out/'checkpoints/best.ckpt'
    save_torch(last,payload);save_torch(best,payload)
    before=(sha_file(last),sha_file(best))
    class ForbiddenTrainingDataset:
        def __init__(self,path): pass
        def __len__(self): return 50000
        def __getitem__(self,index):
            raise AssertionError('Stopped checkpoint caused another training epoch')
    class CpuRuntime:
        cuda=SimpleNamespace(is_available=lambda:True,set_device=lambda device:None,
                             mem_get_info=lambda device:(80*1024**3,80*1024**3))
        device=staticmethod(lambda *args,**kwargs:torch.device('cpu'))
        def __getattr__(self,name): return getattr(torch,name)
    with patch.object(training,'torch',CpuRuntime()), \
         patch.object(training,'CachedDataset',ForbiddenTrainingDataset), \
         patch.object(training,'require_first60',return_value=dict(synthetic_gate=True)), \
         patch.dict('os.environ',{'RANK':'0','LOCAL_RANK':'0','WORLD_SIZE':'1'}):
        training.train(study,cfg)
    complete=json.loads((out/'complete.json').read_text())
    assert complete['status']=='trained' and complete['epochs_completed']==300
    assert complete['reason']=='validation_plateau'
    assert (sha_file(last),sha_file(best))==before
    assert not list((out/'epochs').glob('*.json')), 'A new epoch receipt was emitted'


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=STUDY)
    checks(parser.parse_args().study)
