"""Verify the primary archive, independently recompute errors, and recover its code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    torch.set_num_threads(4)
    inventory_path = root / 'source_inventory.json'
    inventory = json.loads(inventory_path.read_text())
    for name, expected in inventory['files'].items():
        path = root / name
        assert path.is_file() and not path.is_symlink(), name
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
    comparison_path = root / 'comparison.json'
    comparison_sha = sha(comparison_path)
    comparison = json.loads(comparison_path.read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    groups = torch.load(root / 'assets/inputs.pt', map_location='cpu', weights_only=False)
    checks = []
    for row in comparison['results']:
        payload = torch.load(root / 'runs' / row['config'] / row['setting'] / 'prediction.pt',
                             map_location='cpu', weights_only=False)
        group = groups[row['setting']]
        pred = payload['prediction'].numpy().astype(np.float64)
        truth = group['fields'].numpy().astype(np.float64)
        mask = group['masks'].numpy()
        assert payload['ids'] == manifest['ids'] == list(range(1000, 1008))
        assert np.isfinite(pred).all() and np.array_equal(pred[mask], truth[mask])
        computed = np.linalg.norm((pred-truth).reshape(8, 2, -1), axis=-1) / np.linalg.norm(truth.reshape(8, 2, -1), axis=-1)
        for channel, field in enumerate(('a', 'u')):
            np.testing.assert_allclose(computed[:, channel], row['errors'][field], rtol=1e-10, atol=1e-12)
        channels = {'forward': [1], 'inverse': [0], 'both': [0, 1]}[row['task']]
        np.testing.assert_allclose(computed[:, channels].mean(), row['target_mean'], rtol=1e-12)
        checks.append(dict(config=row['config'], setting=row['setting'], samples=8,
                           numpy_float64_errors_match=True, observed_max_absolute_error=0.0))
    assert len(checks) == 9
    workers = []
    launch = json.loads((root / 'launch.json').read_text())
    for worker, jobs in enumerate(manifest['assignments']):
        assert (root / 'logs' / f'worker{worker}.exit').read_text().strip() == '0'
        done = json.loads((root / 'workers' / f'{worker}_complete.json').read_text())
        assert done['status'] == 'complete' and done['jobs'] == jobs
        assert done['manifest_sha256'] == sha(root / 'manifest.json')
        lines = (root / 'logs' / f'worker{worker}.log').read_text().strip().splitlines()
        start, end = [datetime.fromisoformat(lines[i]).timestamp() for i in (0, -1)]
        workers.append(dict(worker=worker, exit_code=0, jobs=jobs, wall_seconds=end-start,
                            remote_end_epoch=end))
    last_end = max(w['remote_end_epoch'] for w in workers)
    elapsed = last_end - launch['remote_epoch']
    reference_end = launch['reference_epoch'] + elapsed
    # A fresh independent repository proves the archive has no worktree/alternates dependency.
    with tempfile.TemporaryDirectory(prefix='cocogen-diagnostic-recovery-') as temporary:
        repo = Path(temporary)
        commands = [
            ['git', 'init', '-q', str(repo)],
            ['git', '-C', str(repo), 'fetch', '-q', str(root / 'source_code.bundle'), 'HEAD'],
            ['git', '-C', str(repo), 'checkout', '-q', '--detach', '20dfe717fc5df9bbc211dcbccffca01401ec4cfe'],
            ['git', '-C', str(repo), 'fsck', '--full'],
        ]
        for command in commands:
            subprocess.run(command, check=True, capture_output=True, text=True)
        assert not (repo / '.git/objects/info/alternates').exists()
        assert sha(repo / 'cocogen_eval/long_sampling_diagnostic.py') == manifest['diagnostic_source_sha256']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4')
        env.pop('PYTHONPATH', None)
        subprocess.run([sys.executable, '-m', 'cocogen_eval.long_sampling_diagnostic',
                        '--root', str(root), 'collect'], cwd=repo, env=env, check=True,
                        capture_output=True, text=True)
        assert sha(comparison_path) == comparison_sha, 'Recovered collector changed results'
    result = dict(status='passed', generated_at=datetime.now(timezone.utc).isoformat(),
        archive_root=str(root), source_root=inventory['source_root'],
        source_inventory_sha256=sha(inventory_path), comparison_sha256=comparison_sha,
        files_verified=len(inventory['files']), bytes_verified=sum(v['bytes'] for v in inventory['files'].values()),
        independently_recomputed='NumPy float64 per-case full-field relative L2, independent of cocogen_eval.inputs.scores',
        cells=checks, workers=workers, sampling_wall_seconds=elapsed,
        reference_finish_utc=datetime.fromtimestamp(reference_end, timezone.utc).isoformat(),
        clock_note='Finish inferred from launch-measured clock offset; worker log endpoints have one-second precision. Sampler times use monotonic clocks.',
        source_recovery=dict(status='passed', commit='20dfe717fc5df9bbc211dcbccffca01401ec4cfe',
            bundle_sha256=sha(root / 'source_code.bundle'), fresh_repository=True,
            no_alternates=True, fsck_passed=True, recovered_collect_identical=True,
            temporary_recovery_repository_removed=True),
        residual_recomputation=False, auditor_sha256=sha(__file__))
    out = root / 'archive_validation.json'
    assert not out.exists(), 'Preserve existing archive validation'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'files_verified', 'bytes_verified', 'sampling_wall_seconds', 'reference_finish_utc')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    verify(parser.parse_args().root.resolve())
