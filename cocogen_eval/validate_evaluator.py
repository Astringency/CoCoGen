"""Exercise prediction receipts, exact resume and tamper detection end to end."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import tempfile

import torch

from . import evaluate
from .common import STUDY, write_json
from .sampler import SamplerConfig


def validate(study):
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(19)
    truth = torch.randn(1000, 2, 4, 4, generator=gen)
    masks = torch.zeros_like(truth, dtype=torch.bool); masks[..., 0, :] = True
    ids = list(range(1000))
    record = dict(cell='poisson/id/synthetic_receipt_test', pde='poisson', dist='id',
        setting='synthetic_receipt_test', task='both', config=dict(residual_mode='auto'),
        rows=[dict(sample_id=i, error_a=.5, error_u=.5) for i in ids])
    manifest = dict(checkpoint_sha256='synthetic', sde=dict(beta_min=.01, beta_max=1.), conditioning_label=0)
    selected = dict(config=asdict(SamplerConfig(steps=2, repaint=1, physics_steps=0, physics_post=0)))
    normalizer = dict(mean=[0., 0.], std=[1., 1.], eps=1e-8)
    class Network:
        cond_size = 1
        fail = False
        def __call__(self, x, c, t, context):
            assert not self.fail, 'Resume called the network again'
            return -x
    network = Network()
    original_input = evaluate.historical_input
    original_physics = evaluate.physics_function
    evaluate.historical_input = lambda _: (truth, masks, ids, {})
    evaluate.physics_function = lambda *args: lambda x:x[:, 1:]-x[:, :1]
    folder = Path(study)/'validation'; folder.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='evaluator_', dir=folder) as temporary:
            args = ('poisson', record, network, manifest, selected, normalizer, 32, torch.device('cpu'), temporary)
            first = evaluate.evaluate_cell(*args)
            assert first['n'] == 1000 and len(first['batches']) == 32
            network.fail = True
            resumed = evaluate.evaluate_cell(*args)
            assert first['errors'] == resumed['errors']
            assert first['comparison'] == resumed['comparison']
            target = Path(first['batches'][0]['prediction_path'])
            with target.open('ab') as f:
                f.write(b'corruption')
            try:
                evaluate.evaluate_cell(*args)
            except AssertionError:
                pass
            else:
                raise AssertionError('Corrupted prediction was accepted on resume')
    finally:
        evaluate.historical_input = original_input
        evaluate.physics_function = original_physics
    write_json(folder/'evaluator.json', dict(status='passed', synthetic=True,
        checks=['1000 IDs and partial final batch', 'physical per-sample metrics',
                'resume without new network calls', 'prediction tamper detection']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=STUDY)
    validate(parser.parse_args().study)
