"""Collect verified task-target metrics without loading a model or changing sampling.

Complete mode requires every expected cell and the aggregate completion receipt.
--allow-partial emits an explicitly partial snapshot, with no partial-PDE macro.
Only Python's standard library is needed, so collection uses no GPU resources.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


PDES = ('darcy', 'poisson', 'helmholtz', 'nsnonbounded')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def target_fields(pde, task):
    require(task in ('forward', 'inverse', 'both'), f'Unknown task: {task}')
    if pde == 'burger':
        require(task == 'both', 'Burgers main evaluation must use its solution-only joint task')
        return ('u',)
    return {'forward': ('u',), 'inverse': ('a',), 'both': ('a', 'u')}[task]


def field_metrics(errors, historical, saved):
    require(len(errors) == len(historical) == 1000, 'Expected 1000 paired errors')
    require(all(math.isfinite(v) and v >= 0 for v in errors + historical), 'Invalid errors')
    cocogen = statistics.fmean(errors)
    reference = statistics.fmean(historical)
    difference = statistics.fmean(a - b for a, b in zip(errors, historical))
    for actual, recorded in (
        (cocogen, saved['cocogen']['mean']),
        (reference, saved['fm4pde']['mean']),
        (difference, saved['paired_difference']['mean']),
    ):
        require(math.isclose(actual, recorded, rel_tol=1e-10, abs_tol=1e-12), 'Summary metric mismatch')
    return dict(cocogen=cocogen, fm4pde=reference, paired_difference=difference,
                ratio=cocogen / reference if reference > 0 else None,
                smaller_error_fraction=sum(a < b for a, b in zip(errors, historical)) / 1000)


def collect(study, include_burger=False, allow_partial=False):
    study = Path(study).resolve()
    evidence = {}

    def read(path):
        path = Path(path)
        data = path.read_bytes()
        evidence[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    pdes = PDES + (('burger',) if include_burger else ())
    expected_count = 66 if include_burger else 60
    phase = 'all66' if include_burger else 'first60'
    catalog = read(study / 'protocol/catalog.json')
    items = [item for item in catalog['cells'] if item['pde'] in pdes]
    require(len(items) == len({item['cell'] for item in items}) == expected_count, 'Invalid catalog scope')
    fields, cells, pending = [], [], []
    for item in sorted(items, key=lambda row: row['cell']):
        cell = item['cell']
        record = read(item['frozen_record'])
        require(evidence[item['frozen_record']] == item['frozen_sha256'], f'Changed reference record: {cell}')
        require(record['cell'] == cell and record['pde'] == item['pde'], 'Catalog identity mismatch')
        summary_path = study / 'main' / cell / 'summary.json'
        if not summary_path.exists():
            pending.append(cell)
            continue
        summary = read(summary_path)
        for key in ('cell', 'pde', 'dist', 'setting', 'task'):
            require(summary[key] == record[key], f'Summary identity mismatch: {cell}/{key}')
        expected_ids = list(range(2000, 3000)) if record['pde'] == 'nsnonbounded' else list(range(1000))
        require(summary['status'] == 'complete' and summary['n'] == 1000 and summary['ids'] == expected_ids,
                f'Incomplete sample membership: {cell}')
        historical_by_id = {row['sample_id']: row for row in record['rows']}
        require(len(record['rows']) == len(historical_by_id) == 1000 and
                sorted(historical_by_id) == expected_ids, f'Invalid reference membership: {cell}')
        selected = read(study / 'protocol/selected' / f'{record["pde"]}.json')
        require(summary['selected'] == selected and selected['status'] == 'validated', 'Changed selection')
        nfe = selected['config']['steps'] * (1 + selected['config']['repaint'])
        target = target_fields(record['pde'], record['task'])
        errors = {field: [] for field in target}
        completed_ids = []
        sampling_seconds = []
        for batch in summary['batches']:
            receipt = read(batch['receipt'])
            require(receipt['status'] == 'complete' and receipt['request']['cell'] == cell,
                    f'Wrong batch: {cell}')
            require(receipt['request']['selected'] == selected and
                    receipt['request']['implementation_sha256'] == selected['implementation_sha256'] and
                    receipt['request']['checkpoint']['checkpoint_sha256'] == selected['checkpoint_sha256'],
                    'Batch sampling provenance mismatch')
            require(receipt['sampling']['config'] == selected['config'] and
                    receipt['sampling']['nfe'] == nfe and receipt['sampling']['final_time'] == 0,
                    'Batch sampling schedule mismatch')
            require(receipt['prediction_path'] == batch['prediction_path'], 'Prediction path mismatch')
            digest = sha_file(batch['prediction_path'])
            require(digest == batch['prediction_sha256'] == receipt['prediction_sha256'], 'Changed prediction')
            evidence[batch['prediction_path']] = digest
            completed_ids.extend(receipt['request']['input']['ids'])
            sampling_seconds.append(receipt['sampling']['seconds'])
            for field in target:
                errors[field].extend(receipt['errors'][field])
        require(completed_ids == expected_ids, f'Incomplete/duplicate batch IDs: {cell}')
        cell_fields = []
        for field in target:
            require(errors[field] == summary['errors'][field], f'Changed batch errors: {cell}/{field}')
            historical = [historical_by_id[i][f'error_{field}'] for i in expected_ids]
            row = {key: record[key] for key in ('cell', 'pde', 'dist', 'setting', 'task')}
            row.update(field=field, n=1000,
                       **field_metrics(errors[field], historical, summary['comparison'][field]))
            fields.append(row)
            cell_fields.append(row)
        require(summary['score_evaluations_per_sample'] == nfe, 'NFE mismatch')
        require(math.isclose(statistics.fmean(sampling_seconds) * len(sampling_seconds),
                             summary['sampler_seconds'], rel_tol=1e-10, abs_tol=1e-9), 'Sampling time mismatch')
        cells.append(dict(cell=cell, pde=record['pde'], dist=record['dist'], setting=record['setting'],
                          task=record['task'], n=1000, target_fields=target, NFE=nfe,
                          cocogen=statistics.fmean(row['cocogen'] for row in cell_fields),
                          fm4pde=statistics.fmean(row['fm4pde'] for row in cell_fields),
                          sampler_seconds=summary['sampler_seconds']))
    require(allow_partial or not pending, f'{len(pending)} evaluation cells are still missing')
    complete_path = study / 'reports' / f'{phase}_complete.json'
    if not pending:
        complete = read(complete_path)
        require(complete['status'] == 'complete' and complete['cells'] == expected_count and
                complete['samples'] == expected_count * 1000 and complete['predictions_hash_verified'],
                'Invalid aggregate completion receipt')
        require(sha_file(complete['csv']) == complete['csv_sha256'], 'Changed aggregate CSV')
        evidence[complete['csv']] = complete['csv_sha256']
    pde_metrics = []
    for pde in pdes:
        subset = [row for row in cells if row['pde'] == pde]
        expected = 6 if pde == 'burger' else 15
        require(sum(item['pde'] == pde for item in items) == expected, 'Incorrect PDE coverage')
        complete = len(subset) == expected
        pde_metrics.append(dict(pde=pde, completed_cells=len(subset), expected_cells=expected,
                                status='complete' if complete else 'partial',
                                cocogen_cell_macro=statistics.fmean(row['cocogen'] for row in subset) if complete else None,
                                fm4pde_cell_macro=statistics.fmean(row['fm4pde'] for row in subset) if complete else None))
    macro = None if pending else {
        model: statistics.fmean(row[f'{model}_cell_macro'] for row in pde_metrics)
        for model in ('cocogen', 'fm4pde')}
    return dict(status='partial' if pending else 'complete', phase=phase,
                generated_at=datetime.now(timezone.utc).isoformat(),
                completed_cells=len(cells), expected_cells=expected_count, pending_cells=pending,
                metric='Mean physical per-sample relative L2 on the full target field, including its observed locations',
                weighting='Within each cell, equal target-field weight; within each complete PDE, equal cell weight; overall, equal PDE weight',
                comparison='Same archived FM4PDE sample IDs and effective observations; each model uses its own frozen sampling configuration',
                scope_note='Partial PDEs have no macro score. This is one fixed sampling seed, not a multi-seed uncertainty estimate.',
                fields=fields, cells=cells, pdes=pde_metrics, overall_pde_macro=macro,
                evidence=evidence, collector_sha256=sha_file(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--include-burger', action='store_true')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = collect(args.study, args.include_burger, args.allow_partial)
    name = f'metrics_{result["phase"]}' + ('_partial' if result['status'] == 'partial' else '')
    folder = args.study / 'reports'
    folder.mkdir(exist_ok=True)
    path = folder / f'{name}.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)
    if result['fields']:
        with (folder / f'{name}_targets.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result['fields'][0]))
            writer.writeheader()
            writer.writerows(result['fields'])
    print(json.dumps(dict(status=result['status'], completed_cells=result['completed_cells'],
                          expected_cells=result['expected_cells'], output=str(path))))


if __name__ == '__main__':
    main()
