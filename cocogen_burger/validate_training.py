"""CPU synthetic checks: no real Burgers data is optimized by these tests."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import torch

from cocogen_eval.common import STUDY,load_network,save_torch,write_json
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
    write_json(folder/'burger_training_cpu.json',dict(status='passed',synthetic=True,
        training_code=code_identity(),
        checks=['first60 prerequisite rejects missing completion','cache items do not modify stored data',
            'score matching noise target and summed-loss scale','strict model/optimizer/RNG resume reproduces next update',
            'fixed validation is repeatable','rank partition preserves validation fields, times and noise',
            'one-channel checkpoint loads into the sampling network without output changes'],
        limitation='Real two-GPU training and batch-size benchmark must wait until first60 is complete'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=STUDY)
    checks(parser.parse_args().study)
