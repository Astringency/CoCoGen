"""Copy a bounded audit to a new directory and prohibit original-data reads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(root, probe_path, bundle_path, commit, output):
    root = root.resolve()
    probe = json.loads(probe_path.read_text())
    assert probe['status'] == 'passed' and probe['archive_root'] == str(root)
    assert not output.exists(), 'Preserve previous recovery evidence'
    assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
    copied_bytes = sum(row['bytes'] for row in probe['files'].values())
    assert shutil.disk_usage(tempfile.gettempdir()).free > copied_bytes + 5 * 1024**3
    dependencies = json.loads((root / 'archive/dependencies/manifest.json').read_text())
    old_files = [entry['original_path'] for entry in dependencies['files']]
    metrics = next(name for name, row in probe['files'].items()
                   if row['sha256'] == probe['metrics_sha256'] and name.startswith('reports/'))
    with tempfile.TemporaryDirectory(prefix='cocogen-archive-relocation-') as temporary:
        temporary = Path(temporary)
        data, code = temporary / 'data', temporary / 'code'
        for relative, row in probe['files'].items():
            relative = Path(relative)
            assert not relative.is_absolute() and '..' not in relative.parts
            source, target = root / relative, data / relative
            assert source.resolve().is_relative_to(root) and not source.is_symlink()
            assert source.stat().st_size == row['bytes'] and sha(source) == row['sha256']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            assert not target.is_symlink()
            assert (source.stat().st_dev, source.stat().st_ino) != (target.stat().st_dev, target.stat().st_ino)
            assert sha(target) == row['sha256']
        for command in (
            ['git', 'init', '-q', str(code)],
            ['git', '-C', str(code), 'fetch', '-q', str(bundle_path.resolve()), 'HEAD'],
            ['git', '-C', str(code), 'checkout', '-q', '--detach', commit],
            ['git', '-C', str(code), 'fsck', '--full'],
        ):
            subprocess.run(command, check=True, capture_output=True, text=True)
        assert not (code / '.git/objects/info/alternates').exists()
        assert sha(code / 'cocogen_eval/verify_archived_predictions.py') == probe['auditor_sha256']
        child_output = temporary / 'relocated_audit.json'
        guard_output = temporary / 'read_guard.json'
        argv = ['verify_archived_predictions', '--root', str(data), '--original-study',
                probe['original_study'], '--metrics', metrics, '--allow-partial', '--output', str(child_output)]
        for cell in probe['cells']:
            argv.extend(['--cell', cell['cell']])
        # The child can use the restored repository, copied data, Python runtime,
        # and system libraries. Every original dependency filename and the entire
        # old study directory are denied by a Python audit hook during recovery.
        bootstrap = '''import json, os, runpy, sys
old_root = %r
old_files = set(%r)
denied = []
def guard(event, args):
    if event != 'open' or not args or isinstance(args[0], int):
        return
    value = os.path.abspath(os.fsdecode(args[0]))
    if value == old_root or value.startswith(old_root + os.sep) or value in old_files:
        denied.append(value)
        raise PermissionError('Recovery may not open an original data path: ' + value)
sys.addaudithook(guard)
canaries = [old_root + '/protocol/catalog.json', sorted(old_files)[0]]
for path in canaries:
    try:
        open(path, 'rb')
    except PermissionError:
        pass
    else:
        raise AssertionError('The original-path read guard did not fire')
sys.argv = %r
runpy.run_module('cocogen_eval.verify_archived_predictions', run_name='__main__')
assert len(denied) == len(canaries), 'Unexpected attempted access to original files'
with open(%r, 'w') as stream:
    json.dump({'status': 'passed', 'mechanism': 'Python open audit hook', 'blocked_canary_reads': len(canaries),
               'unexpected_original_path_attempts': 0}, stream)
''' % (probe['original_study'], old_files, argv, str(guard_output))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        env.pop('PYTHONPATH', None)
        result = subprocess.run([sys.executable, '-u', '-'], input=bootstrap, cwd=code,
                                env=env, text=True, capture_output=True)
        print(result.stdout, end='', flush=True)
        if result.stderr:
            print(result.stderr, end='', file=sys.stderr, flush=True)
        result.check_returncode()
        recovered = json.loads(child_output.read_text())
        guard = json.loads(guard_output.read_text())
        assert recovered['status'] == 'passed' and recovered['cells'] == probe['cells']
        assert recovered['files'] == probe['files'] and recovered['metrics_sha256'] == probe['metrics_sha256']
        assert guard['status'] == 'passed'
        relocated_root = str(data)
    assert not Path(relocated_root).exists()
    receipt = dict(status='passed', generated_at=datetime.now(timezone.utc).isoformat(),
        scope='Bounded relocation and original-path denial test; not whole-study completion',
        final_study_complete=False, original_archive_root=str(root), relocated_root=relocated_root,
        probe_sha256=sha(probe_path), copied_files=len(probe['files']), copied_bytes=copied_bytes,
        independent_regular_copies=True, verified_cells=probe['verified_cells'],
        verified_samples=probe['verified_samples'], cells=[row['cell'] for row in probe['cells']],
        original_path_guard=guard, restored_metrics_identical=True, restored_file_hashes_identical=True,
        recovered_source=dict(commit=commit, bundle_sha256=sha(bundle_path), fsck_passed=True,
                              no_alternates=True, auditor_sha256=probe['auditor_sha256']),
        temporary_data_and_code_removed=True, helper_sha256=sha(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status='passed', copied_files=receipt['copied_files'],
                          copied_bytes=copied_bytes, output=str(output))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.probe, args.bundle, args.commit, args.output)
