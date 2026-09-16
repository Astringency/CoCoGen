"""Read-only decision for the user's revised Burgers plateau stopping rule.

This checks immutable epoch receipts. It does not stop processes, alter a
checkpoint, claim training completion, or launch evaluation. A positive result
requires a separately recorded checkpoint/coordination handoff.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(rows, *, minimum_epochs=100, patience_checks=6,
           relative_min_delta=.002, validation_every=10, max_epochs=1200):
    if not rows or minimum_epochs < 1 or patience_checks < 1:
        raise ValueError('Nonempty history and positive stopping thresholds required')
    if not 0 <= relative_min_delta < 1:
        raise ValueError('Invalid relative improvement threshold')
    best = reference = math.inf
    stale = 0
    checks = []
    best_epoch = None
    previous_validation = None
    for expected, row in enumerate(rows, 1):
        if row['epochs_completed'] != expected or row['epoch'] != expected-1:
            raise ValueError('Epoch history is not contiguous')
        scheduled = expected == 1 or expected % validation_every == 0 or expected == max_epochs
        if row['validation_was_run'] is not scheduled:
            raise ValueError('Validation marker differs from the frozen schedule')
        if scheduled:
            value = float(row['validation']['uniform']['loss_per_pixel'])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid validation loss')
            if value < best:
                best, best_epoch = value, expected
            if value < reference*(1-relative_min_delta):
                reference, stale = value, 0
            else:
                stale += 1
            previous_validation = row['validation']
            checks.append(dict(epoch=expected, loss_per_pixel=value, stale_checks=stale))
        elif row['validation'] != previous_validation:
            raise ValueError('Nonvalidation epoch changed the carried validation result')
        if row['stale_checks'] != stale or row['best_validation_per_pixel'] != best:
            raise ValueError('Stored plateau state differs from independent recomputation')
    eligible = rows[-1]['epochs_completed'] >= minimum_epochs and stale >= patience_checks
    return dict(status='eligible_for_early_stop' if eligible else 'watching_plateau',
                eligible=eligible, epochs_completed=rows[-1]['epochs_completed'],
                best_validation_epoch=best_epoch, best_validation_per_pixel=best,
                plateau_reference=reference, stale_checks=stale,
                minimum_epochs=minimum_epochs, patience_checks=patience_checks,
                relative_min_delta=relative_min_delta, validation_checks=checks,
                stop_applied=False, training_complete=False,
                requires_checkpoint_and_evaluation_handoff=eligible)


def decide(study, policy_path):
    study, policy_path = Path(study).resolve(), Path(policy_path).resolve()
    policy = json.loads(policy_path.read_text())
    if policy['status'] != 'authorized' or policy['study'] != str(study):
        raise ValueError('Policy does not authorize this study')
    request_path = study/'training/burger/request.json'
    if digest(request_path) != policy['training_request_sha256']:
        raise ValueError('Training request differs from the authorized policy')
    request = json.loads(request_path.read_text())['request']
    cfg = request['config']
    if cfg['relative_min_delta'] != policy['relative_min_delta']:
        raise ValueError('Improvement definition must remain the frozen definition')
    progress = json.loads((study/'training/burger/progress.json').read_text())
    end = progress['epochs_completed']
    rows, sources = [], []
    expected_steps = math.ceil(math.ceil(50000/request['world_size'])/cfg['batch_size_per_rank'])
    for epoch in range(1, end+1):
        path = study/f'training/burger/epochs/{epoch:04d}.json'
        raw = path.read_bytes()
        row = json.loads(raw)
        if row['global_step'] != epoch*expected_steps:
            raise ValueError('Wrong epoch/optimization-step relationship')
        rows.append(row)
        sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
    if rows[-1] != progress:
        raise ValueError('Progress does not match its immutable epoch receipt')
    decision = assess(rows, minimum_epochs=policy['minimum_epochs'],
                      patience_checks=policy['patience_checks'],
                      relative_min_delta=policy['relative_min_delta'],
                      validation_every=cfg['validation_every'], max_epochs=cfg['max_epochs'])
    decision.update(generated_at=datetime.now(timezone.utc).isoformat(),
                    policy_sha256=digest(policy_path), training_request_sha256=digest(request_path),
                    original_min_epochs=cfg['min_epochs'], original_patience_checks=cfg['patience_checks'],
                    global_step=progress['global_step'], sources=sources,
                    checker_sha256=digest(__file__),
                    scope='Read-only external early-stop decision; live training settings unchanged')
    return decision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = decide(args.study, args.policy)
    if args.output:
        # Keep each observed decision separate from its input epoch receipts.
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'sources'}, ensure_ascii=False))
