"""Burgers training cache and independent checkpoint-selection inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from cocogen_eval.common import STUDY, DATA_ROOT, save_torch, sha_file, write_json
from cocogen_eval.prepare import read_fields, restore, train_files


def require_first60(study=STUDY, *, verify_predictions=True):
    """Check the actual first-phase products before training can begin."""
    study = Path(study)
    gate = json.loads((study/'reports/first60_complete.json').read_text())
    assert gate['status'] == 'complete' and gate['cells'] == 60 and gate['samples'] == 60000
    assert sha_file(gate['csv']) == gate['csv_sha256']
    catalog = json.loads((study/'protocol/catalog.json').read_text())
    expected = {r['cell'] for r in catalog['cells'] if r['pde'] != 'burger'}
    assert len(expected) == 60
    seen = set()
    for pde in ('darcy', 'poisson', 'helmholtz', 'nsnonbounded'):
        complete = json.loads((study/'main'/pde/'complete.json').read_text())
        assert complete['status'] == 'complete' and len(complete['cells']) == 15
        for entry in complete['cells']:
            row = json.loads(Path(entry['summary']).read_text())
            assert row['status'] == 'complete' and row['n'] == 1000
            assert row['cell'] not in seen
            seen.add(row['cell'])
            expected_ids = list(range(2000, 3000)) if pde == 'nsnonbounded' else list(range(1000))
            assert row['ids'] == expected_ids
            ids = []
            for batch in row['batches']:
                receipt = json.loads(Path(batch['receipt']).read_text())
                assert receipt['status'] == 'complete'
                assert receipt['prediction_path'] == batch['prediction_path']
                assert receipt['prediction_sha256'] == batch['prediction_sha256']
                if verify_predictions:
                    assert sha_file(batch['prediction_path']) == batch['prediction_sha256']
                ids.extend(receipt['request']['input']['ids'])
            assert ids == expected_ids
    assert seen == expected
    return dict(first60_receipt_sha256=sha_file(study/'reports/first60_complete.json'),
                csv_sha256=gate['csv_sha256'], cells=60, samples=60000,
                predictions_hash_verified=verify_predictions)


def array_sha(array):
    digest = hashlib.sha256()
    for offset in range(0, len(array), 128):
        digest.update(np.ascontiguousarray(array[offset:offset+128]).tobytes())
    return digest.hexdigest()


def prepare(study=STUDY):
    study = Path(study)
    gate = require_first60(study)
    out = study/'inputs/burger'
    manifest_path = out/'training_cache.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        assert sha_file(existing['training_path']) == existing['training_sha256']
        assert sha_file(existing['validation_path']) == existing['validation_sha256']
        assert sha_file(out/'normalizer.json') == existing['normalizer_sha256']
        return existing
    normalizer = restore('burger', study)
    mean, std = float(normalizer['mean'][0]), float(normalizer['std'][0])+normalizer['eps']
    path = out/'training_standardized.npy'
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(50000,1,128,128))
    offset = 0
    for source, receipt in zip(train_files('burger'), normalizer['sources']):
        fields = read_fields('burger', source)
        assert fields.shape == (10000,1,128,128)
        assert array_sha(fields) == receipt['selected_float32_bchw_sha256']
        output[offset:offset+len(fields)] = (fields-mean)/std
        offset += len(fields)
        print(json.dumps(dict(stage='burger_training_cache', samples=offset)), flush=True)
    assert offset == 50000
    output.flush(); del output
    temporary.replace(path)
    # This panel selects checkpoints, so it is disjoint from both sampling
    # calibration (1000..1095) and all formal-main IDs (0..999).
    source = DATA_ROOT/'burgers/burger_test_10000-128-128_id.mat'
    raw = torch.from_numpy(read_fields('burger', source, 1096, 1352))
    assert raw.shape == (256,1,128,128)
    validation = dict(fields=(raw-mean)/std, ids=list(range(1096,1352)), source=str(source),
        role='fixed held-out checkpoint-selection panel; no model optimization; separate from sampling calibration and main',
        channel_order='B,1,time,space', normalizer=normalizer)
    validation_path = out/'training_validation.pt'
    save_torch(validation_path, validation)
    manifest = dict(pde='burger', samples=50000, channels=1, resolution=128,
        channel_order='B,1,time,space', dtype='float32', training_path=str(path),
        training_sha256=sha_file(path), normalizer_sha256=sha_file(out/'normalizer.json'),
        validation_path=str(validation_path), validation_sha256=sha_file(validation_path),
        validation_ids=validation['ids'], sampler_calibration_ids=list(range(1000,1096)),
        formal_main_ids=list(range(1000)), first60_gate=gate)
    write_json(manifest_path, manifest)
    return manifest


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.array = np.load(path, mmap_mode='r')

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        # Copy one field so torch never wraps a read-only memory mapping.
        return torch.from_numpy(np.array(self.array[index], copy=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=STUDY)
    prepare(parser.parse_args().study)
