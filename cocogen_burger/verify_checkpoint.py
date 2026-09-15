"""Inspect a real training checkpoint on CPU without changing the live run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from cocogen_eval.common import sha_file, write_json
from cocogen_eval.network import UNET1
from .train import TrainConfig, code_identity, model_config


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(study, source):
    torch.set_num_threads(4)
    study, source = study.resolve(), source.resolve()
    require(source.is_relative_to(study), 'Checkpoint must belong to this study')
    require(source.name == 'last.ckpt', 'This check binds the resume checkpoint to its epoch receipt')
    # Training saves by atomic rename. Keep one opened inode for both its hash
    # and torch.load so a later live checkpoint replacement cannot mix epochs.
    with source.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location='cpu', weights_only=False)
    checkpoint_sha = digest.hexdigest()
    request = json.loads((study/'training/burger/request.json').read_text())['request']
    require(checkpoint['request'] == request, 'Checkpoint request differs from active training')
    require(request['code'] == code_identity(), 'Training implementation differs')
    cfg = TrainConfig(**request['config'])
    cache_path = study/'inputs/burger/training_cache.json'
    cache = json.loads(cache_path.read_text())
    require(sha_file(cache_path) == request['cache_sha256'], 'Cache manifest differs')
    normalizer_path = study/'inputs/burger/normalizer.json'
    normalizer = json.loads(normalizer_path.read_text())
    require(sha_file(normalizer_path) == request['normalizer_sha256'], 'Normalizer differs')
    require(checkpoint['normalizer'] == normalizer, 'Embedded normalizer differs')
    require(checkpoint['model_config'] == model_config(cfg), 'Model configuration differs')
    require(cache['samples'] == 50000 and cache['validation_ids'] == list(range(1096, 1352)), 'Wrong training/validation scope')
    world = request['world_size']
    require(world == 2 and len(checkpoint['rank_rng']) == world, 'Missing rank RNG states')
    steps_per_epoch = math.ceil(math.ceil(cache['samples']/world)/cfg.batch_size_per_rank)
    require(checkpoint['global_step'] == (checkpoint['epoch']+1)*steps_per_epoch, 'Wrong epoch/step relationship')
    receipt_path = study/'training/burger/epochs'/f"{checkpoint['epoch']+1:04d}.json"
    deadline = time.monotonic()+5
    while not receipt_path.exists() and time.monotonic() < deadline:
        time.sleep(.1)
    receipt = json.loads(receipt_path.read_text())
    require(receipt['last_sha256'] == checkpoint_sha, 'Checkpoint differs from its epoch receipt')
    require(receipt['global_step'] == checkpoint['global_step'], 'Epoch receipt step differs')
    network = UNET1(**checkpoint['model_config']['model']['params']['unet_config']['params'])
    network.load_state_dict({k.removeprefix('unet.'):v for k,v in checkpoint['state_dict'].items()}, strict=True)
    for name, value in network.state_dict().items():
        require(torch.isfinite(value).all(), f'Nonfinite model value: {name}')
    optimizer = torch.optim.AdamW(network.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer'])
    states = 0
    for parameter in network.parameters():
        state = optimizer.state[parameter]
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Missing or unexpected optimizer state')
        require(state['step'].item() == checkpoint['global_step'], 'Optimizer step differs')
        for name in ('exp_avg', 'exp_avg_sq'):
            require(state[name].shape == parameter.shape and torch.isfinite(state[name]).all(), 'Invalid optimizer moment')
        states += 1
    for state in checkpoint['rank_rng']:
        require(set(state) == {'python', 'numpy', 'cpu', 'cuda'}, 'Missing RNG kind')
        random.Random().setstate(state['python'])
        np.random.RandomState().set_state(state['numpy'])
        torch.Generator(device='cpu').set_state(state['cpu'])
        require(state['cuda'].dtype == torch.uint8 and state['cuda'].ndim == 1 and state['cuda'].numel() > 0, 'Missing CUDA RNG bytes')
    return dict(status='passed', generated_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU restore and completeness checks of one real saved checkpoint; no training updates, relocation, CUDA resume or convergence check',
        final_study_complete=False, training_complete=False,
        checkpoint=str(source), checkpoint_sha256=checkpoint_sha, epoch=checkpoint['epoch'],
        global_step=checkpoint['global_step'], model_tensors=len(network.state_dict()),
        parameter_count=sum(p.numel() for p in network.parameters()), optimizer_parameter_states=states,
        strict_model_restore=True, finite_model_and_optimizer=True, rank_rng_states=world,
        cpu_rng_formats_restored=True, cuda_rng_restored=False,
        cache_manifest_sha256=sha_file(cache_path), actual_training_cache_rehashed=False,
        epoch_receipt_sha256=sha_file(receipt_path), code=request['code'], auditor_sha256=sha_file(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve previous audit receipts')
    write_json(args.output, verify(args.study, args.checkpoint))
