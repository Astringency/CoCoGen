"""Read-only check of the actual Burgers training loader's time/space axes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time

import numpy as np
import scipy.io

from .common import sha_file, write_json
from .prepare import read_fields, train_files


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    started = time.monotonic()
    sources = []
    for path in train_files('burger'):
        before = path.stat()
        metadata = scipy.io.loadmat(path, variable_names=['input', 'tspan', 'viscosity'])
        initial = np.asarray(metadata['input'], dtype=np.float32)
        fields = read_fields('burger', path)
        require(initial.shape == (10000, 128), f'Unexpected initial layout: {path}')
        require(fields.shape == (10000, 1, 128, 128), f'Unexpected training layout: {path}')
        require(np.isfinite(initial).all(), f'Nonfinite initial conditions: {path}')
        require(np.array_equal(fields[:, 0, 0, :], initial), f'First time layer differs from input: {path}')
        ambiguous = np.all(fields[:, 0, :, 0] == initial, axis=1)
        require(not ambiguous.any(), f'Sample cannot distinguish the two axes: {path}')
        times = np.asarray(metadata['tspan'], dtype=np.float64).reshape(-1) if 'tspan' in metadata else None
        viscosity = float(np.asarray(metadata['viscosity']).reshape(-1)[0]) if 'viscosity' in metadata else None
        if times is not None:
            require(times.shape == (128,) and np.allclose(times, np.linspace(0, 1, 128), rtol=0, atol=1e-12), f'Unexpected stored times: {path}')
        if viscosity is not None:
            require(viscosity == .01, f'Unexpected stored viscosity: {path}')
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f'File changed while reading: {path}')
        sources.append(dict(path=str(path), bytes=after.st_size, mtime_ns=after.st_mtime_ns,
            samples=len(initial), loader_shape=list(fields.shape),
            first_time_layer_matches_initial_exactly=True, transposed_first_layer_matching_samples=int(ambiguous.sum()),
            initial_float32_sha256=hashlib.sha256(np.ascontiguousarray(initial).tobytes()).hexdigest(),
            stored_tspan=None if times is None else times.tolist(), stored_viscosity=viscosity,
            missing_metadata=[key for key in ('tspan', 'viscosity') if key not in metadata]))
        del fields, initial, metadata
        print(f'Verified training axes: {path.name}, 10000 samples', flush=True)
    return dict(status='passed', generated_at=datetime.now(timezone.utc).isoformat(),
        scope='All five original training shards through read_fields; first time layer versus original initial conditions',
        samples=sum(row['samples'] for row in sources), layout='B,1,T,X',
        training_started=False, training_cache_created=False, final_study_complete=False,
        full_dataset_files_hashed=False, full_fields_finiteness_checked=False,
        physics_parameter_limit='Missing metadata does not independently verify T=1 or viscosity=0.01; no PDE residuals are recomputed',
        sources=sources, loader_sha256=sha_file(Path(__file__).with_name('prepare.py')),
        auditor_sha256=sha_file(__file__), seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve previous receipts')
    write_json(args.output, verify())
