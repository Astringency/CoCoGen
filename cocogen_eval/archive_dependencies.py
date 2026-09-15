"""Preserve fixed FM4PDE inputs and used pretrained models without using GPUs.

This prepares dependencies during evaluation. It is not a final-study completion
receipt: Burgers, final metrics, code recovery and cleanup have separate gates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .common import REMOTE_REPO, RUNS, sha_file, write_json


def inventory(study):
    catalog_path = study / 'protocol/catalog.json'
    catalog = json.loads(catalog_path.read_text())
    assert len(catalog['cells']) == 66
    assert len({c['cell'] for c in catalog['cells']}) == 66
    files = {}

    def add(path, expected, role):
        path = Path(path)
        assert path.is_absolute() and path.is_file(), path
        item = dict(original_path=str(path), sha256=expected,
                    bytes=path.stat().st_size, roles=[role])
        if str(path) in files:
            previous = files[str(path)]
            assert previous['sha256'] == expected and previous['bytes'] == item['bytes'], path
            previous['roles'] = sorted(set(previous['roles'] + [role]))
        else:
            files[str(path)] = item

    add(catalog['source_catalog'], catalog['source_sha256'], 'fm4pde_catalog')
    for cell in catalog['cells']:
        path = Path(cell['frozen_record'])
        assert sha_file(path) == cell['frozen_sha256'], path
        record = json.loads(path.read_text())
        assert record['cell'] == cell['cell']
        add(cell['record_path'], cell['record_sha256'], 'fm4pde_record')
        for batch in record['batches']:
            add(batch['result_path'], batch['result_sha256'], 'fm4pde_result')

    used_models = {}
    for path in sorted((study / 'calibration').glob('*/*/*/*/receipt.json')):
        pde = path.relative_to(study / 'calibration').parts[0]
        if pde not in RUNS:
            continue
        request = json.loads(path.read_text())['request']
        key = (pde, request['stage'])
        expected = request['checkpoint_sha256']
        assert key not in used_models or used_models[key] == expected, key
        used_models[key] = expected
    assert set(used_models) == {(pde, stage) for pde in RUNS for stage in ('base', 'control')}
    selected_hashes = {}
    for pde in RUNS:
        path = study / 'protocol/selected' / f'{pde}.json'
        selected = json.loads(path.read_text())
        assert selected['status'] == 'validated'
        assert used_models[pde, selected['stage']] == selected['checkpoint_sha256']
        selected_hashes[pde] = sha_file(path)
    for (pde, stage), expected in sorted(used_models.items()):
        run = REMOTE_REPO / 'output' / ('' if stage == 'base' else 'withcontrol') / RUNS[pde][stage == 'control']
        add(run / 'checkpoints/last.ckpt', expected, f'pretrained_checkpoint:{pde}:{stage}')
        configs = sorted((run / 'configs').glob('*-model.yaml'))
        assert len(configs) == 1, (run, configs)
        add(configs[0], sha_file(configs[0]), f'pretrained_config:{pde}:{stage}')
    return sorted(files.values(), key=lambda x: x['original_path']), dict(
        catalog_sha256=sha_file(catalog_path), selected_sha256=selected_hashes)


def preserve(study):
    study = Path(study).resolve()
    folder = study / 'archive/dependencies'
    files, identity = inventory(study)
    objects = {}
    for row in files:
        expected = row['sha256']
        assert len(expected) == 64 and all(c in '0123456789abcdef' for c in expected)
        target = folder / 'objects' / expected[:2] / expected
        row['archive_path'] = str(target.relative_to(study))
        if expected in objects:
            assert objects[expected]['bytes'] == row['bytes']
        objects[expected] = row
    missing_bytes = sum(r['bytes'] for r in objects.values() if not (study / r['archive_path']).exists())
    assert shutil.disk_usage(study).free >= missing_bytes + 5 * 1024**3, 'Insufficient archive disk space'
    print(json.dumps(dict(stage='copying_dependencies', source_files=len(files),
                          unique_objects=len(objects), missing_bytes=missing_bytes)), flush=True)
    for index, (expected, row) in enumerate(sorted(objects.items()), 1):
        target = study / row['archive_path']
        if target.exists():
            assert target.stat().st_size == row['bytes'] and sha_file(target) == expected, target
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
            try:
                digest = hashlib.sha256()
                with Path(row['original_path']).open('rb') as source, temporary.open('xb') as dest:
                    while chunk := source.read(8 * 1024 * 1024):
                        digest.update(chunk)
                        dest.write(chunk)
                    dest.flush()
                    os.fsync(dest.fileno())
                assert temporary.stat().st_size == row['bytes'] and digest.hexdigest() == expected, row['original_path']
                assert sha_file(temporary) == expected, temporary
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        if index % 100 == 0 or index == len(objects):
            print(json.dumps(dict(stage='verified_dependency_objects', done=index, total=len(objects))), flush=True)
    result = dict(status='verified', scope='Frozen FM4PDE inputs for 66 cells and eight used existing pretrained models',
        final_study_complete=False,
        pending='Burgers training/evaluation, final metrics, final code archive recovery check and task cleanup',
        generated_at=datetime.now(timezone.utc).isoformat(), identity=identity,
        source_files=len(files), unique_objects=len(objects), unique_bytes=sum(r['bytes'] for r in objects.values()),
        roles=dict(Counter(role for row in files for role in row['roles'])), files=files,
        archive_script_sha256=sha_file(__file__),
        git_commit=subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]),
                                            'rev-parse', 'HEAD'], text=True).strip())
    manifest = folder / 'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        for key in ['identity', 'source_files', 'unique_objects', 'unique_bytes', 'roles', 'files', 'archive_script_sha256']:
            assert previous[key] == result[key], f'Existing archive identity differs: {key}'
    else:
        write_json(manifest, result)
    print(json.dumps(dict(stage='dependencies_verified', manifest=str(manifest),
                          manifest_sha256=sha_file(manifest), final_study_complete=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    if args.inventory_only:
        files, identity = inventory(args.study.resolve())
        print(json.dumps(dict(source_files=len(files), source_bytes=sum(r['bytes'] for r in files),
                              roles=dict(Counter(role for r in files for role in r['roles'])), identity=identity)))
    else:
        preserve(args.study)
