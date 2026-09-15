"""Meaningful sampler regressions, native-network parity and GPU throughput."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from .common import REMOTE_REPO, STUDY, load_network, write_json
from .sampler import Normalizer, PerSampleNoise, SamplerConfig, physical_correction, sample


def regressions():
    class Oracle:
        cond_size = 1
        times = []
        def __call__(self, x, c, t, mask):
            assert (c == 3).all()
            self.times.extend(t.tolist())
            return -x
    norm = dict(mean=[2., 5.], std=[3., .1], eps=1e-8)
    observed = torch.randn(2, 2, 8, 8)
    mask = torch.zeros_like(observed, dtype=torch.bool); mask[:, :, ::3, ::3] = True
    cfg = SamplerConfig(steps=5, repaint=2, physics_steps=0, physics_post=0)
    network = Oracle()
    args = (network, observed, mask, [1, 2], norm, dict(beta_min=.01, beta_max=1.), 3, cfg)
    out, receipt = sample(*args)
    assert torch.equal(out[mask], observed[mask])
    assert min(network.times) > 0 and max(network.times) == 1
    assert receipt['nfe'] == 15 and receipt['final_time'] == 0
    changed = observed.clone(); changed[~mask] = float('nan')
    again, _ = sample(network, changed, *args[2:])
    assert torch.equal(out, again), 'Unobserved truth leaked'
    noise = PerSampleNoise([1, 2], (2, 4, 4), 'cpu', seed=7)
    singleton = PerSampleNoise([2], (2, 4, 4), 'cpu', seed=7)
    for _ in range(19):
        assert torch.equal(noise.draw()[1], singleton.draw()[0])
    normalizer = Normalizer(norm, 'cpu')
    x = torch.randn_like(observed)
    def residual(p):
        u = p[:, 1:]
        return u[..., 1:]-2*u[..., :-1]
    with torch.no_grad():
        updated, _ = physical_correction(x, mask, normalizer, residual, replace(cfg, max_physical_update=.01))
    assert torch.equal(updated[:, 0], x[:, 0]), 'Coefficient channel changed'
    assert torch.equal(updated[mask], x[mask]), 'Observed entry changed'
    assert (residual(normalizer.decode(updated)).flatten(1).square().mean(1) <=
            residual(normalizer.decode(x)).flatten(1).square().mean(1)).all()
    z = torch.randn(1, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    loss = residual(z).square().sum()
    grad, = torch.autograd.grad(loss, z)
    plus, minus = z.detach().clone(), z.detach().clone()
    plus[0, 1, 1, 1] += 1e-5; minus[0, 1, 1, 1] -= 1e-5
    fd = (residual(plus).square().sum()-residual(minus).square().sum())/2e-5
    assert torch.allclose(fd, grad[0, 1, 1, 1], rtol=1e-7, atol=1e-7)
    return dict(status='passed', checks=['positive score times', 'terminal t=0', 'NFE count',
        'observations exact', 'unobserved truth ignored', 'batch-independent RNG',
        'solution-only correction', 'global residual descent', 'neighbor finite-difference gradient'])


def gpu_audit(pde, device, study):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    result = dict(regressions=regressions(), pde=pde, device=str(device), torch=torch.__version__,
                  tf32=True, precision='float32', measurements=[])
    sys.path.insert(0, str(REMOTE_REPO))
    spec = importlib.util.spec_from_file_location('original_server_unet', REMOTE_REPO/'models/unets.py')
    native_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(native_module)
    for stage in ('base', 'control'):
        network, manifest = load_network(pde, stage, device)
        raw = yaml.safe_load(Path(manifest['config_path']).read_text())
        params = raw.get('model', raw)['params']['unet_config']['params']
        native = native_module.UNET1(**OmegaConf.create(params)).to(device).eval()
        native.load_state_dict(network.state_dict(), strict=True)
        with torch.no_grad():
            x = torch.randn(1, 2, 128, 128, device=device)
            c = torch.full((1, 1), float(manifest['conditioning_label']), device=device)
            for t in [.05, .5, 1.]:
                times = torch.full((1,), t, device=device)
                a = network(x, c, times, torch.zeros_like(c))
                b = native(x, c, times, torch.zeros_like(c))
                assert torch.equal(a, b), (stage, t, (a-b).abs().max().item())
        del native, x, a, b
        torch.cuda.empty_cache()
        per_sample_peak = None
        for batch in (1, 8, 16, 32, 64):
            free, _ = torch.cuda.mem_get_info(device)
            if per_sample_peak and per_sample_peak*batch > .55*free:
                break
            with torch.no_grad():
                x = torch.randn(batch, 2, 128, 128, device=device)
                c = torch.full((batch, 1), float(manifest['conditioning_label']), device=device)
                t = torch.full((batch,), .5, device=device)
                context = torch.zeros_like(c)
                for _ in range(3):
                    network(x, c, t, context)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.monotonic()
                for _ in range(10):
                    value = network(x, c, t, context)
                torch.cuda.synchronize(device)
                seconds = (time.monotonic()-start)/10
                peak = torch.cuda.max_memory_allocated(device)
                if batch == 1:
                    per_sample_peak = peak
                assert torch.isfinite(value).all()
                row = dict(stage=stage, batch=batch, seconds_per_NFE=seconds,
                    peak_bytes=peak, samples_per_gpu_hour_4000NFE=batch*3600/(4000*seconds))
                result['measurements'].append(row)
                print(json.dumps(row), flush=True)
                del x, c, t, context, value
            write_json(Path(study)/'validation'/f'{pde}.json', result)
        del network
        torch.cuda.empty_cache()
    result['status'] = 'passed'
    write_json(Path(study)/'validation'/f'{pde}.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pde', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--study', type=Path, default=STUDY)
    args = p.parse_args()
    gpu_audit(args.pde, torch.device(args.device), args.study)
