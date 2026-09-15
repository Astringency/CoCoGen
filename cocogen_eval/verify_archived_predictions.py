"""Recompute saved metrics using only files inside a possibly relocated archive.

This audits saved predictions, not model training, residuals, or study closeout.
Absolute paths in original JSON records are translated without rewriting them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from cocogen_eval.calibrate import implementation_hash
from cocogen_eval.common import sha_file, sha_tensor, write_json
from cocogen_eval import main_inputs


def require(ok, message):
    if not ok:
        raise ValueError(message)


class ArchiveReader:
    def __init__(self, root, original_study):
        self.root = Path(root).resolve()
        self.original = Path(original_study)
        require(self.original.is_absolute(), 'The original study path must be absolute')
        self.dependencies = {}
        self.opened = {}
        manifest = self.read_json('archive/dependencies/manifest.json')
        require(manifest['status'] == 'verified', 'Missing dependency archive')
        for entry in manifest['files']:
            source = entry['original_path']
            require(source not in self.dependencies, 'Duplicate dependency mapping')
            self.dependencies[source] = entry

    def resolve(self, source):
        path = Path(source)
        if path.is_absolute():
            if path.is_relative_to(self.original):
                relative = path.relative_to(self.original)
            else:
                require(str(path) in self.dependencies, f'No archived mapping: {source}')
                relative = Path(self.dependencies[str(path)]['archive_path'])
        else:
            relative = path
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid archive-relative path')
        candidate = self.root / relative
        require(candidate.resolve().is_relative_to(self.root), 'Archive reference leaves its root')
        require(candidate.is_file() and not candidate.is_symlink(), f'Missing regular archive file: {relative}')
        return candidate

    def check(self, source, expected=None):
        path = self.resolve(source)
        key = str(path.relative_to(self.root))
        stat = path.stat()
        if key in self.opened:
            row = self.opened[key]
            require((row['bytes'], row['mtime_ns']) == (stat.st_size, stat.st_mtime_ns), f'File changed during audit: {key}')
        else:
            row = dict(sha256=sha_file(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            self.opened[key] = row
        if str(source) in self.dependencies:
            dependency = self.dependencies[str(source)]
            require(row['sha256'] == dependency['sha256'] and row['bytes'] == dependency['bytes'], f'Changed dependency: {key}')
        require(expected is None or row['sha256'] == expected, f'Hash mismatch: {key}')
        return path

    def read_json(self, source, expected=None):
        return json.loads(self.check(source, expected).read_text())


def close(actual, expected, description):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), description)


def verify(root, original_study, metrics_path, requested_cells, allow_partial):
    torch.set_num_threads(4)
    reader = ArchiveReader(root, original_study)
    metrics = reader.read_json(metrics_path)
    require(metrics['status'] in ('partial', 'complete'), 'Unknown metrics status')
    require(allow_partial or metrics['status'] == 'complete', 'A partial snapshot requires --allow-partial')
    catalog_path = 'protocol/catalog.json'
    catalog = reader.read_json(catalog_path)
    require(len(catalog['cells']) == 66 and len({c['cell'] for c in catalog['cells']}) == 66, 'Wrong study catalog')
    reader.check(catalog['source_catalog'], catalog['source_sha256'])
    original_catalog = str(reader.original / catalog_path)
    reader.check(catalog_path, metrics['evidence'][original_catalog])
    available = {c['cell']: c for c in metrics['cells']}
    require(len(available) == len(metrics['cells']) == metrics['completed_cells'], 'Duplicate or missing completed cells')
    chosen = set(requested_cells) if requested_cells else set(available)
    require(chosen and chosen <= set(available), 'Requested cells are absent from this snapshot')
    require(allow_partial or chosen == set(available), 'A subset requires --allow-partial')
    items = {c['cell']: c for c in catalog['cells']}
    field_rows = {(f['cell'], f['field']): f for f in metrics['fields']}
    implementation = implementation_hash()
    checked = []
    for cell in sorted(chosen):
        item = items[cell]
        record = reader.read_json(item['frozen_record'], item['frozen_sha256'])
        require(record['cell'] == cell, 'Wrong frozen cell')
        pde = record['pde']
        expected_ids = list(range(2000, 3000)) if pde == 'nsnonbounded' else list(range(1000))
        selected_path = f'protocol/selected/{pde}.json'
        selected = reader.read_json(selected_path, metrics['evidence'][str(reader.original / selected_path)])
        require(selected['status'] == 'validated' and selected['implementation_sha256'] == implementation, 'Wrong frozen sampler')
        summary_path = str(reader.original / 'main' / cell / 'summary.json')
        summary = reader.read_json(summary_path, metrics['evidence'][summary_path])
        require(summary['status'] == 'complete' and summary['n'] == 1000 and summary['ids'] == expected_ids, 'Incomplete cell')
        require(summary['selected'] == selected, 'Changed sampling selection')
        normalizer = reader.read_json(f'inputs/{pde}/normalizer.json')
        # Rebase only the in-memory record passed to the existing input loader.
        # Both batch references and per-example membership references must move.
        rebased = dict(record)
        locations = {b['result_path']: str(reader.check(b['result_path'], b['result_sha256'])) for b in record['batches']}
        rebased['batches'] = [dict(b, result_path=locations[b['result_path']]) for b in record['batches']]
        rebased['rows'] = [dict(r, result_path=locations[r['result_path']]) for r in record['rows']]
        truth, masks, ids, _ = main_inputs.historical_input(rebased)
        require(ids == expected_ids, 'Archived FM4PDE IDs differ')
        channels = [('u', 0)] if pde == 'burger' else [('a', 0), ('u', 1)]
        targets = ('u',) if pde == 'burger' or record['task'] == 'forward' else (('a',) if record['task'] == 'inverse' else ('a', 'u'))
        errors = {field: [] for field, _ in channels}
        offset, seconds = 0, 0.0
        for batch in summary['batches']:
            receipt_path = batch['receipt']
            receipt = reader.read_json(receipt_path, metrics['evidence'][receipt_path])
            request, sampling = receipt['request'], receipt['sampling']
            require(receipt['status'] == 'complete' and request['cell'] == cell and request['offset'] == offset, 'Wrong batch membership')
            require(request['selected'] == selected and request['normalizer'] == normalizer, 'Wrong batch configuration')
            require(request['implementation_sha256'] == implementation and request['physics_config'] == record['config'], 'Wrong implementation or physics')
            model = request['checkpoint']
            require(model['checkpoint_sha256'] == selected['checkpoint_sha256'], 'Wrong checkpoint identity')
            reader.check(model['checkpoint'], model['checkpoint_sha256'])
            reader.check(model['config_path'], model['config_sha256'])
            require(set(request['evaluation_code_sha256']) == {'evaluate.py', 'main_inputs.py'}, 'Incomplete evaluation source identity')
            for name, digest in request['evaluation_code_sha256'].items():
                require(sha_file(Path(main_inputs.__file__).parent / name) == digest, 'Wrong evaluation source')
            require(sampling['config'] == selected['config'] and sampling['nfe'] == selected['config']['steps'] * (1 + selected['config']['repaint']) and sampling['final_time'] == 0, 'Wrong sampler schedule')
            require(receipt['prediction_path'] == batch['prediction_path'], 'Prediction path differs')
            pred_path = reader.check(batch['prediction_path'], batch['prediction_sha256'])
            require(batch['prediction_sha256'] == receipt['prediction_sha256'] == metrics['evidence'][batch['prediction_path']], 'Prediction digest differs')
            payload = torch.load(pred_path, map_location='cpu', weights_only=False)
            require(payload['request'] == request and payload['sampling'] == sampling and payload['errors'] == receipt['errors'], 'Prediction payload differs')
            n = len(payload['ids'])
            require(n == min(request['batch_size'], 1000-offset) and n > 0 and payload['ids'] == ids[offset:offset+n], 'Wrong batch IDs')
            x, mask = truth[offset:offset+n], masks[offset:offset+n]
            require(request['input'] == dict(ids=payload['ids'], truth_sha256=sha_tensor(x), mask_sha256=sha_tensor(mask)), 'Archived inputs differ from sampled inputs')
            prediction = payload['prediction']
            require(prediction.shape == x.shape and torch.isfinite(prediction).all().item(), 'Invalid prediction')
            require(torch.equal(prediction[mask], x[mask]), 'Observed values differ')
            for field, channel in channels:
                target = x[:, channel].numpy().astype(np.float64).reshape(n, -1)
                delta = prediction[:, channel].numpy().astype(np.float64).reshape(n, -1) - target
                denominator = np.linalg.norm(target, axis=1)
                require((denominator > 0).all(), 'Zero error denominator')
                actual = (np.linalg.norm(delta, axis=1) / denominator).tolist()
                np.testing.assert_allclose(actual, receipt['errors'][field], rtol=1e-10, atol=1e-12)
                errors[field].extend(actual)
            offset += n
            seconds += sampling['seconds']
        require(offset == 1000, 'Incomplete prediction coverage')
        comparisons = {}
        historical = {row['sample_id']: row for row in record['rows']}
        for field in targets:
            np.testing.assert_allclose(errors[field], summary['errors'][field], rtol=1e-10, atol=1e-12)
            reference = [historical[i][f'error_{field}'] for i in expected_ids]
            row = field_rows[cell, field]
            actual_mean, reference_mean = statistics.fmean(errors[field]), statistics.fmean(reference)
            close(actual_mean, row['cocogen'], 'Recomputed CoCoGen metric differs')
            close(reference_mean, row['fm4pde'], 'FM4PDE metric differs')
            close(statistics.fmean(a-b for a, b in zip(errors[field], reference)), row['paired_difference'], 'Paired difference differs')
            if reference_mean > 0:
                close(actual_mean / reference_mean, row['ratio'], 'Error ratio differs')
            close(sum(a < b for a, b in zip(errors[field], reference)) / 1000, row['smaller_error_fraction'], 'Paired win fraction differs')
            comparisons[field] = dict(cocogen=actual_mean, fm4pde=reference_mean)
        close(statistics.fmean(v['cocogen'] for v in comparisons.values()), available[cell]['cocogen'], 'Task weighting differs')
        close(seconds, available[cell]['sampler_seconds'], 'Sampling time differs')
        checked.append(dict(cell=cell, n=1000, batches=len(summary['batches']), target_fields=targets,
            observed_max_absolute_error=0.0, errors_recomputed_numpy_float64=True, comparisons=comparisons))
        print(json.dumps(dict(event='archive_cell_verified', cell=cell, done=len(checked), total=len(chosen))), flush=True)
    return dict(status='passed', generated_at=datetime.now(timezone.utc).isoformat(),
        scope='Saved predictions and their archived inputs; no training, residual recomputation, or study closeout',
        final_study_complete=False, archive_root=str(reader.root), original_study=str(reader.original),
        original_external_data_read=False, metrics_phase=metrics['phase'], metrics_status=metrics['status'],
        expected_cells_in_phase=metrics['expected_cells'], verified_cells=len(checked), verified_samples=1000*len(checked),
        entire_phase_verified=metrics['status'] == 'complete' and len(checked) == metrics['expected_cells'],
        metrics_sha256=sha_file(reader.resolve(metrics_path)), cells=checked,
        files={p: {k: v for k, v in row.items() if k != 'mtime_ns'} for p, row in sorted(reader.opened.items())},
        implementation_sha256=implementation, auditor_sha256=sha_file(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--original-study', type=Path, required=True)
    parser.add_argument('--metrics', required=True, help='Path relative to the archive root')
    parser.add_argument('--cell', action='append', default=[])
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing audit receipts; choose a new output path')
    result = verify(args.root, args.original_study, args.metrics, args.cell, args.allow_partial)
    write_json(args.output, result)
    print(json.dumps(dict(status=result['status'], verified_cells=result['verified_cells'], output=str(args.output))), flush=True)
