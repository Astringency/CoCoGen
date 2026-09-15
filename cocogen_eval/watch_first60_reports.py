"""Collect and independently audit new first-phase cells as sampling finishes.

This CPU-only observer never starts, stops, or changes sampling or training.
Its completion means first60 reporting is verified, not that all66 is complete.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from cocogen_eval.collect_metrics import collect
from cocogen_eval.common import sha_file, write_json
from cocogen_eval.verify_archived_predictions import require, verify


def read(path):
    return json.loads(Path(path).read_text())


def coverage(study, metrics_path, audits):
    metrics = read(metrics_path)
    available = {row['cell']: row for row in metrics['cells']}
    fields = {(row['cell'], row['field']): row for row in metrics['fields']}
    covered, predictions = {}, {}
    for relative, digest in audits.items():
        path = study / relative
        require(path.resolve().is_relative_to(study), 'Audit reference leaves study')
        require(sha_file(path) == digest, 'Changed audit receipt')
        audit = read(path)
        require(audit['status'] == 'passed' and not audit['original_external_data_read'], 'Invalid archive audit')
        for cell in audit['cells']:
            name = cell['cell']
            require(name in available and name not in covered, 'Duplicate or out-of-scope audit cell')
            require(cell['n'] == 1000 and cell['errors_recomputed_numpy_float64'] and cell['observed_max_absolute_error'] == 0, 'Incomplete numerical audit')
            require(set(cell['comparisons']) == set(available[name]['target_fields']), 'Wrong target fields')
            for field, values in cell['comparisons'].items():
                for model in ('cocogen', 'fm4pde'):
                    require(math.isclose(values[model], fields[name, field][model], rel_tol=1e-10, abs_tol=1e-12), 'Audit and snapshot metrics differ')
            prefix = str(study / 'main' / name) + '/'
            current = {p: h for p, h in metrics['evidence'].items() if p.startswith(prefix) and p.endswith('/prediction.pt')}
            require(len(current) == 32, 'Expected 32 prediction batches per cell')
            for source, digest in current.items():
                relative_prediction = str(Path(source).relative_to(study))
                require(audit['files'][relative_prediction]['sha256'] == digest, 'Audited prediction differs from current snapshot')
                predictions[relative_prediction] = digest
            covered[name] = relative
    require(set(covered) == set(available), 'Audit coverage is incomplete')
    return dict(status='passed', generated_at=datetime.now(timezone.utc).isoformat(),
        scope='Union of archived prediction audits, bound to this metrics snapshot; no training or study closeout',
        final_study_complete=False, metrics_sha256=sha_file(metrics_path), verified_cells=len(covered),
        verified_samples=1000*len(covered), prediction_hashes_bound_to_latest_snapshot=len(predictions),
        source_audits=audits, cells=covered)


def publish(study, metrics):
    name = 'metrics_first60' + ('_partial' if metrics['status'] == 'partial' else '')
    folder = study / 'reports'
    csv_path = folder / f'{name}_targets.csv'
    temporary = csv_path.with_suffix('.csv.watch.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics['fields'][0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(metrics['fields'])
    temporary.replace(csv_path)
    write_json(folder / f'{name}.json', metrics)
    return dict(json=str((folder / f'{name}.json').relative_to(study)),
                json_sha256=sha_file(folder / f'{name}.json'),
                csv=str(csv_path.relative_to(study)), csv_sha256=sha_file(csv_path))


def watch(study, output, baseline_metrics, baseline_coverage, interval, once):
    study = study.resolve()
    output = output.resolve()
    require(output.is_relative_to(study), 'Store reporting receipts inside the study')
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'state.json'
    if state_path.exists():
        state = read(state_path)
        metrics_path = study / state['metrics']
        require(sha_file(metrics_path) == state['metrics_sha256'], 'Changed committed snapshot')
        audits = state['audits']
    else:
        metrics_path = study / baseline_metrics
        base = read(study / baseline_coverage)
        require(base['status'] == 'passed' and base['metrics_sha256'] == sha_file(metrics_path), 'Baseline coverage does not bind the snapshot')
        audits = {'validation/' + name.removeprefix('validation_'): digest for name, digest in base['source_audits'].items()}
    verified = coverage(study, metrics_path, audits)
    metrics = read(metrics_path)
    known = set(verified['cells'])
    catalog = read(study / 'protocol/catalog.json')
    expected = {row['cell'] for row in catalog['cells'] if row['pde'] != 'burger'}
    require(len(expected) == 60 and known <= expected, 'Wrong first-phase scope')

    def commit(verification):
        coverage_path = output / f'coverage_{len(known):04d}_{time.time_ns()}.json'
        write_json(coverage_path, verification)
        published = publish(study, metrics)
        state = dict(status='first60_reports_verified' if len(known) == 60 else 'waiting_for_new_cells',
            updated_at=datetime.now(timezone.utc).isoformat(), final_study_complete=False,
            verified_cells=len(known), metrics=str(metrics_path.relative_to(study)),
            metrics_sha256=sha_file(metrics_path), coverage=str(coverage_path.relative_to(study)),
            coverage_sha256=sha_file(coverage_path), audits=audits, published=published,
            observer_sha256=sha_file(__file__))
        write_json(state_path, state)
        print(json.dumps({key: state[key] for key in ('status', 'updated_at', 'verified_cells', 'metrics')}), flush=True)

    commit(verified)
    while len(known) < 60:
        observed = {cell for cell in expected if (study / 'main' / cell / 'summary.json').exists()}
        require(known <= observed, 'A previously completed summary disappeared')
        ready = len(observed) < 60 or (study / 'reports/first60_complete.json').exists()
        if observed != known and ready:
            metrics = collect(study, allow_partial=len(observed) < 60)
            current = {row['cell'] for row in metrics['cells']}
            require(known < current <= expected, 'Invalid completion transition')
            metrics_path = output / f'metrics_{len(current):04d}_{time.time_ns()}.json'
            write_json(metrics_path, metrics)
            audit_path = output / f'audit_{len(current):04d}_{time.time_ns()}.json'
            result = verify(study, study, str(metrics_path.relative_to(study)), sorted(current-known), True)
            write_json(audit_path, result)
            audits = {**audits, str(audit_path.relative_to(study)): sha_file(audit_path)}
            verified = coverage(study, metrics_path, audits)
            known = current
            commit(verified)
        if once or len(known) == 60:
            break
        # Worker failures stop only this observer. It never restarts jobs.
        for name in ('cell24_gpu0', 'cell24_gpu1', 'cell24_schedule'):
            path = study / 'logs' / f'{name}.exit'
            if path.exists():
                value = path.read_text().strip()
                require(not value or value == '0', f'Inspect failed source job: {name}, exit {value}')
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-metrics', required=True)
    parser.add_argument('--baseline-coverage', required=True)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    require(5 <= args.interval <= 60, 'Use a bounded observation interval')
    watch(args.study, args.output, args.baseline_metrics, args.baseline_coverage, args.interval, args.once)
