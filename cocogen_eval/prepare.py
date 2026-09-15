"""Restore training statistics and freeze disjoint calibration inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch

from .common import DATA_ROOT, STUDY, write_json, save_torch


FIELDS = dict(darcy=('thresh_a_data', 'thresh_p_data'), poisson=('f_data', 'phi_data'),
              helmholtz=('f_data', 'psi_data'), nsnonbounded=('w0', 'w'), burger=('output',))


def train_files(pde):
    folder = 'burgers' if pde == 'burger' else pde
    tail = '10000-128-128-10' if pde == 'nsnonbounded' else '10000-128-128'
    suffix = '_new' if pde == 'nsnonbounded' else ''
    return [DATA_ROOT / folder / f'{pde}_{tail}_{i}{suffix}.mat' for i in range(1, 6)]


def read_fields(pde, path, start=0, stop=None):
    """Return original loader's float32 channels without normalization."""
    names = FIELDS[pde]
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            if pde == 'darcy':
                arrays = [np.asarray(f[k][:, :, start:stop], dtype=np.float32).transpose(2, 0, 1) for k in names]
            elif pde == 'nsnonbounded':
                arrays = [np.asarray(f['w0'][start:stop], dtype=np.float32),
                          np.asarray(f['w'][start:stop, :, :, -1], dtype=np.float32)]
            else:
                raise ValueError(pde)
    else:
        raw = scipy.io.loadmat(path, variable_names=list(names))
        arrays = [np.asarray(raw[k][start:stop], dtype=np.float32) for k in names]
    return np.stack(arrays, axis=1)


def restore(pde, study=STUDY):
    out = Path(study) / 'inputs' / pde
    if (out / 'normalizer.json').exists() and (out / 'calibration.pt').exists():
        return json.loads((out / 'normalizer.json').read_text())
    start_time = time.monotonic()
    totals = np.zeros(len(FIELDS[pde]), dtype=np.float64)
    squares = totals.copy()
    count = 0
    sample_count = 0
    sources = []
    for path in train_files(pde):
        # MAT v5 cannot slice compressed variables; load each shard only once.
        field = read_fields(pde, path)
        assert field.shape == (10000, len(FIELDS[pde]), 128, 128), (path, field.shape)
        digest = hashlib.sha256()
        for offset in range(0, len(field), 128):
            chunk = np.ascontiguousarray(field[offset:offset+128])
            if not np.isfinite(chunk).all():
                raise ValueError(f'Nonfinite training values: {path} offset {offset}')
            digest.update(chunk.tobytes())
            totals += chunk.sum(axis=(0, 2, 3), dtype=np.float64)
            squares += np.square(chunk, dtype=np.float64).sum(axis=(0, 2, 3))
            count += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
        sample_count += len(field)
        sources.append(dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
                            selected_float32_bchw_sha256=digest.hexdigest(),samples=len(field)))
        del field
        print(json.dumps(dict(pde=pde,stage='training_statistics',completed_files=len(sources),seconds=time.monotonic()-start_time)),flush=True)
    mean = totals / count
    # Old PDEtransform uses torch.std correction=1 and divides by (sd + 1e-8).
    variance = np.maximum((squares - totals * mean) / (count - 1), 0)
    std = np.sqrt(variance)
    record = dict(pde=pde,mean=mean.astype(np.float32).tolist(),std=std.astype(np.float32).tolist(),eps=1e-8,
        samples=sample_count,elements_per_channel=count,correction=1,channel_names=list(FIELDS[pde]),
        algorithm='float32 fields, float64 streaming moments; old training sample std restored, reduction roundoff can differ',
        sources=sources,seconds=time.monotonic()-start_time)
    write_json(out / 'normalizer.json', record)
    folder = 'burgers' if pde == 'burger' else pde
    tail = '10000-128-128-10' if pde == 'nsnonbounded' else '10000-128-128'
    path = DATA_ROOT / folder / f'{pde}_test_{tail}_id.mat'
    ids = list(range(1000,1096))
    fields = torch.from_numpy(read_fields(pde,path,1000,1096))
    assert len(fields)==96
    params = {}
    if pde=='nsnonbounded':
        with h5py.File(path,'r') as f:
            params={k:float(f.attrs[k]) for k in ['viscosity','T']}
    elif pde=='helmholtz':
        raw=scipy.io.loadmat(path,variable_names=['k'])
        params={'k':float(np.asarray(raw.get('k',1)).reshape(-1)[0])}
    save_torch(out / 'calibration.pt',dict(pde=pde,fields=fields,ids=ids,pde_params=params,source=str(path),
        role='held-out calibration subset of ID test file, disjoint from all formal main IDs; not used for model training',
        formal_ids='2000..2999' if pde=='nsnonbounded' else '0..999'))
    print(json.dumps(dict(pde=pde,stage='complete',seconds=time.monotonic()-start_time)),flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pde',required=True,choices=list(FIELDS))
    parser.add_argument('--study',type=Path,default=STUDY)
    args=parser.parse_args()
    restore(args.pde,args.study)
