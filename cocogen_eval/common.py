from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


REMOTE_REPO = Path('/research_data/users/zhangxifeng/C01Python/CoCoGen')
FM_REPO = Path('/research_data/users/zhangxifeng/C01Python/FM4PDE')
STUDY = FM_REPO / 'outputs/cocogen_main_20260915'
DATA_ROOT = Path('/large_storage/zhangxf/PDEdata')
CATALOG = FM_REPO / 'outputs/pretrained/resume_main_20260911/evaluation_inputs/catalog.json'
LABELS = dict(darcy=0, poisson=1, helmholtz=2, nsnonbounded=3, burger=4)
RUNS = {
    'darcy': ('2025-11-15T17-57-32_cocogen4darcy', '2025-11-27T21-17-46_cocogen4darcy'),
    'poisson': ('2025-11-14T18-14-05_cocogen4poisson', '2025-11-27T21-31-52_cocogen4poisson'),
    'helmholtz': ('2025-11-15T18-01-40_cocogen4helmholtz', '2025-11-28T09-24-03_cocogen4helmholtz'),
    'nsnonbounded': ('2025-11-15T18-08-44_cocogen4nsnonbounded', '2025-11-27T21-32-06_cocogen4nsnonbounded'),
}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def save_torch(path, data):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def load_network(pde, stage='base', device='cpu', checkpoint=None, config_path=None):
    import torch
    import yaml
    from .network import UNET1
    if checkpoint is None:
        name = RUNS[pde][0 if stage == 'base' else 1]
        run = REMOTE_REPO / 'output' / ('' if stage == 'base' else 'withcontrol') / name
        checkpoint = run / 'checkpoints/last.ckpt'
        config_path = next((run / 'configs').glob('*-model.yaml'))
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg = cfg.get('model', cfg)['params']
    network = UNET1(**cfg['unet_config']['params'])
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = payload['state_dict']
    state = {k.removeprefix('unet.'): v for k, v in state.items() if k.startswith('unet.')}
    network.load_state_dict(state, strict=True)
    for param in network.parameters():
        param.requires_grad_(False)
    network.to(device).eval()
    return network, dict(
        pde=pde, stage=stage, checkpoint=str(checkpoint), checkpoint_sha256=sha_file(checkpoint),
        config_path=str(config_path), config_sha256=sha_file(config_path),
        epoch=payload.get('epoch'), global_step=payload.get('global_step'),
        sde=cfg['sde_config']['params'], conditioning_label=LABELS[pde],
        parameters=sum(p.numel() for p in network.parameters()),
    )
