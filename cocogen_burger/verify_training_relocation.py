"""Exercise real training-asset reads after copying them outside the study.

This CPU check restores model/optimizer state and revalidates actual data and
the first60 gate. It does not run CUDA resume or any optimization step.
"""
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
import time

from cocogen_eval.common import sha_file, write_json


def child(root, original, output):
    import builtins
    import torch
    from .archive_paths import mapped_study_reads
    from .data import CachedDataset, require_first60
    from .verify_checkpoint import verify

    torch.set_num_threads(4)
    normalizer = json.loads((root/'inputs/burger/normalizer.json').read_text())
    original_files = [row['path'] for row in normalizer['sources']]
    raw_open = builtins.open
    with mapped_study_reads(root, original, original_files) as mapping:
        for path in (original/'protocol/catalog.json', Path(original_files[0])):
            try:
                raw_open(path, 'rb')
            except PermissionError:
                pass
            else:
                raise AssertionError('Original-data denial did not work')
        gate = require_first60(root)
        cache = json.loads((root/'inputs/burger/training_cache.json').read_text())
        assert sha_file(cache['training_path']) == cache['training_sha256']
        assert sha_file(cache['validation_path']) == cache['validation_sha256']
        data = CachedDataset(cache['training_path'])
        assert len(data) == 50000 and data.array.shape == (50000, 1, 128, 128)
        samples = {}
        for index in (0, 24999, 49999):
            value = data[index]
            assert torch.isfinite(value).all()
            samples[str(index)] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        validation = torch.load(cache['validation_path'], map_location='cpu', weights_only=False)
        assert validation['ids'] == list(range(1096, 1352))
        assert validation['fields'].shape == (256, 1, 128, 128) and torch.isfinite(validation['fields']).all()
        checkpoint = verify(root, root/'training/burger/checkpoints/last.ckpt')
        assert len(mapping['direct_original_access_attempts']) == 2
        result = dict(status='passed', first60_gate=gate, training_cache_sha256=cache['training_sha256'],
            validation_sha256=cache['validation_sha256'], samples_read=samples,
            checkpoint=checkpoint, mapped_files=len(mapping['mapped_files']), mapped_opens=mapping['mapped_opens'],
            blocked_canary_reads=2, unexpected_original_access_attempts=0,
            mapping_sha256=sha_file(Path(__file__).with_name('archive_paths.py')))
    write_json(output, result)


def run(study, bundle, commit, output):
    import torch

    study = study.resolve()
    assert not output.exists() and len(commit) == 40
    assert not list(set(commit)-set('0123456789abcdef'))
    # Full cache plus first60 predictions and two checkpoints fit well below
    # this conservative scratch-space prerequisite in the current study.
    assert shutil.disk_usage(tempfile.gettempdir()).free > 32*1024**3
    source_files = {}
    with tempfile.TemporaryDirectory(prefix='cocogen-training-relocation-') as temporary:
        temporary = Path(temporary)
        data, code = temporary/'data', temporary/'code'

        def relative(path):
            path = Path(path)
            if path.is_absolute():
                assert path.is_relative_to(study)
                path = path.relative_to(study)
            assert '..' not in path.parts
            return path

        def copy(path, expected=None, replace=False):
            name = relative(path)
            if str(name) in source_files and not replace:
                if expected: assert source_files[str(name)]['sha256'] == expected
                return data/name
            source, target = study/name, data/name
            assert source.resolve().is_relative_to(study) and not source.is_symlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source.open('rb') as incoming, target.open('wb' if replace else 'xb') as outgoing:
                before = os.fstat(incoming.fileno())
                for block in iter(lambda: incoming.read(8*1024*1024), b''):
                    outgoing.write(block); digest.update(block)
                after = os.fstat(incoming.fileno())
            assert (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns)
            value = digest.hexdigest()
            assert expected is None or value == expected
            assert sha_file(target) == value
            assert (target.stat().st_dev,target.stat().st_ino) != (before.st_dev,before.st_ino)
            source_files[str(name)] = dict(sha256=value, bytes=before.st_size)
            return target

        def read(path):
            return json.loads(copy(path).read_text())

        gate = read('reports/first60_complete.json')
        copy(gate['csv'],gate['csv_sha256'])
        read('protocol/catalog.json')
        for pde in ('darcy','poisson','helmholtz','nsnonbounded'):
            complete = read(f'main/{pde}/complete.json')
            for cell in complete['cells']:
                summary = read(cell['summary'])
                for batch in summary['batches']:
                    read(batch['receipt'])
                    copy(batch['prediction_path'],batch['prediction_sha256'])
            print(f'Copied training prerequisite: {pde}',flush=True)
        cache = read('inputs/burger/training_cache.json')
        copy(cache['training_path'],cache['training_sha256'])
        copy(cache['validation_path'],cache['validation_sha256'])
        copy('inputs/burger/normalizer.json',cache['normalizer_sha256'])
        for name in ('training/burger/request.json','training/burger/train_config.json',
                     'training/burger/model.yaml','validation/burger_training_gpu.json'):
            read(name) if name.endswith('.json') else copy(name)
        # Capture coherent last/best states while the live trainer atomically
        # replaces its rolling checkpoints. Never modify those source files.
        for attempt in range(4):
            last_path = copy('training/burger/checkpoints/last.ckpt',replace=attempt>0)
            best_path = copy('training/burger/checkpoints/best.ckpt',replace=attempt>0)
            last = torch.load(last_path,map_location='cpu',weights_only=False)
            best = torch.load(best_path,map_location='cpu',weights_only=False)
            coherent = best['epoch'] <= last['epoch'] and best['best'] == last['best'] and best['request'] == last['request']
            epochs = sorted({last['epoch']+1,best['epoch']+1})
            checkpoint_epoch = last['epoch']+1
            del last,best
            if coherent: break
            time.sleep(.5)
        assert coherent, 'Could not capture coherent rolling checkpoints; inspect before retrying'
        for epoch in epochs:
            name = f'training/burger/epochs/{epoch:04d}.json'
            deadline = time.monotonic()+5
            while not (study/name).exists() and time.monotonic()<deadline: time.sleep(.1)
            read(name)
        for command in (
            ['git','init','-q',str(code)],
            ['git','-C',str(code),'fetch','-q',str(bundle.resolve()),commit],
            ['git','-C',str(code),'checkout','-q','--detach',commit],
            ['git','-C',str(code),'fsck','--full'],
        ):
            subprocess.run(command,check=True,capture_output=True,text=True)
        assert not (code/'.git/objects/info/alternates').exists()
        assert sha_file(code/'cocogen_burger/verify_training_relocation.py') == sha_file(__file__)
        child_output = temporary/'child.json'
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        env.pop('PYTHONPATH',None)
        subprocess.run([sys.executable,'-u','-m','cocogen_burger.verify_training_relocation',
            '--child','--study',str(data),'--original-study',str(study),'--output',str(child_output)],
            cwd=code,env=env,check=True)
        recovered = json.loads(child_output.read_text())
        assert recovered['status'] == 'passed' and recovered['checkpoint']['epoch']+1 == checkpoint_epoch
        assert recovered['checkpoint']['checkpoint_sha256'] == source_files['training/burger/checkpoints/last.ckpt']['sha256']
        # Include post-read integrity checks so the probe cannot hide accidental
        # mutations of copied metadata, predictions, caches or checkpoints.
        assert all(sha_file(data/name)==row['sha256'] for name,row in source_files.items())
        temporary_root = str(data)
    assert not Path(temporary_root).exists()
    result = dict(status='passed',generated_at=datetime.now(timezone.utc).isoformat(),
        scope='Real training-asset relocation and CPU restore; no CUDA resume, training updates, convergence or complete-study claim',
        final_study_complete=False, training_complete=False, cuda_resume_exercised=False,
        original_study=str(study),copied_files=len(source_files),copied_bytes=sum(r['bytes'] for r in source_files.values()),
        independent_regular_copies=True, copied_files_unchanged=True, original_data_fallback=False,
        checkpoint_epoch=checkpoint_epoch, recovered=recovered, files=source_files,
        recovered_source=dict(commit=commit,bundle_sha256=sha_file(bundle),fsck_passed=True,no_alternates=True),
        runtime_note='Uses the installed Python and system libraries; not a binary environment recovery or OS sandbox',
        temporary_data_and_code_removed=True, helper_sha256=sha_file(__file__))
    write_json(output,result)
    print(json.dumps(dict(status='passed',checkpoint_epoch=checkpoint_epoch,copied_files=result['copied_files'],output=str(output))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bundle',type=Path)
    parser.add_argument('--commit')
    parser.add_argument('--original-study',type=Path)
    parser.add_argument('--child',action='store_true')
    args = parser.parse_args()
    if args.child:
        assert args.original_study is not None
        child(args.study,args.original_study,args.output)
    else:
        assert args.bundle is not None and args.commit is not None
        run(args.study,args.bundle,args.commit,args.output)
