"""Resume frozen training from a moved study without rewriting its metadata.

This is an explicit CUDA/DDP entry point. CPU relocation checks do not imply
that real CUDA resume has already been exercised.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .archive_paths import mapped_study_reads


def resume(root, original_study):
    from .train import TrainConfig, train

    root = root.resolve(strict=True)
    if not (root/'training/burger/checkpoints/last.ckpt').is_file():
        raise FileNotFoundError('A real resume checkpoint is required')
    request = json.loads((root/'training/burger/request.json').read_text())['request']
    if int(os.environ.get('WORLD_SIZE', 1)) != request['world_size']:
        raise ValueError('Launch with the same DDP world size as the saved request')
    config = TrainConfig(**json.loads((root/'training/burger/train_config.json').read_text()))
    normalizer = json.loads((root/'inputs/burger/normalizer.json').read_text())
    blocked = [row['path'] for row in normalizer['sources']]
    with mapped_study_reads(root, original_study, blocked) as mapping:
        train(root, config, resume=True)
        print(json.dumps(dict(archive_resume_returned=True, mapped_files=len(mapping['mapped_files']),
            mapped_opens=mapping['mapped_opens'], direct_original_access_attempts=mapping['direct_original_access_attempts'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--original-study', type=Path, required=True)
    args = parser.parse_args()
    resume(args.root, args.original_study)
