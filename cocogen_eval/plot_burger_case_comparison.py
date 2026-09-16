"""Plot fixed Burgers examples from archived physical fields, without sampling.

Reference-only mode checks actual FM4PDE inputs while CoCoGen is training.
Comparison mode requires all six completed Burgers cells; no stand-in prediction
is created when a result is absent. Both modes choose sample 0 before seeing its
quality, use the solution observation mask, and preserve the [time, space] axes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

from cocogen_eval.common import sha_file, write_json
from cocogen_eval.plot_main_case_comparison import error, reference_error
from cocogen_eval.verify_archived_predictions import ArchiveReader, close, require


DISTS = ('id', 'smooth', 'rough')
SETTINGS = ('random', 'sensor_column')
LABELS = dict(random='500 random space-time points',
              sensor_column='5 fixed spatial sensors / 128 time levels')
CELLS = tuple(f'burger/{dist}/{setting}' for dist in DISTS for setting in SETTINGS)


def reference_example(reader, item):
    record = reader.read_json(item['frozen_record'], item['frozen_sha256'])
    require(record['cell'] == item['cell'] and record['pde'] == 'burger', 'Wrong Burgers cell')
    require(record['sample_ids'] == list(range(1000)), 'Expected the frozen formal IDs')
    cfg = record['config']
    require(record['task'] == 'both' and cfg['zeta_obs_a'] == 0 and cfg['zeta_obs_u'] > 0,
            'Only solution observations should be active')
    row = next(row for row in record['rows'] if row['sample_id'] == 0)
    digest = next(batch['result_sha256'] for batch in record['batches']
                  if batch['result_path'] == row['result_path'])
    payload = torch.load(reader.check(row['result_path'], digest), map_location='cpu', weights_only=False)
    require(not payload['ground_truth_metadata'].get('synthetic', False), 'Synthetic truth is excluded')
    index = row['result_row']
    truth = payload['sol_ground_truth'][index].float()
    require(torch.equal(truth, payload['coef_ground_truth'][index]), 'Burgers truth aliases differ')
    fm = payload['sol_final'][index].float()
    mask = payload['masks']['sol'][index]
    require(torch.all((mask == 0) | (mask == 1)), 'Observation mask is not binary')
    mask = mask.bool()
    require(truth.shape == fm.shape == mask.shape == (1, 128, 128), 'Expected [1,time,space] fields')
    require(torch.isfinite(truth).all() and torch.isfinite(fm).all(), 'Nonfinite physical fields')
    columns = []
    if record['setting'] == 'sensor_column':
        columns = torch.where(mask[0, 0])[0].tolist()
        require(int(mask.sum()) == 640 and columns == [20, 22, 37, 103, 121], 'Unexpected sensors')
        require(torch.equal(mask, mask[:, :1].expand_as(mask)), 'Sensors must stay fixed in space')
    else:
        require(record['setting'] == 'random' and int(mask.sum()) == 500, 'Expected 500 random points')
    fm_error = reference_error(fm[0], truth[0])
    close(fm_error, row['error_u'], 'FM4PDE example differs from its frozen error')
    return dict(cell=record['cell'], dist=record['dist'], setting=record['setting'], sample_id=0,
                truth=truth[0].numpy(), mask=mask[0].numpy(), fm4pde=fm[0].numpy(),
                fm4pde_error=fm_error, observations=int(mask.sum()), spatial_sensor_indices=columns,
                reference_sha256=digest)


def add_cocogen(reader, metrics, example):
    name = example['cell']
    source = str(reader.original / 'main' / name / 'summary.json')
    summary = reader.read_json(source, metrics['evidence'][source])
    require(summary['status'] == 'complete' and summary['n'] == 1000
            and summary['ids'] == list(range(1000)), 'Incomplete Burgers cell')
    selected_source = str(reader.original / 'protocol/selected/burger.json')
    selected = reader.read_json(selected_source, metrics['evidence'][selected_source])
    require(selected['status'] == 'validated' and summary['selected'] == selected, 'Changed selection')
    batch = summary['batches'][0]
    receipt = reader.read_json(batch['receipt'], metrics['evidence'][batch['receipt']])
    require(receipt['status'] == 'complete' and receipt['request']['cell'] == name
            and receipt['request']['offset'] == 0 and receipt['request']['selected'] == selected,
            'Wrong CoCoGen batch')
    digest = batch['prediction_sha256']
    require(digest == receipt['prediction_sha256'] == metrics['evidence'][batch['prediction_path']],
            'Wrong prediction hash')
    payload = torch.load(reader.check(batch['prediction_path'], digest), map_location='cpu', weights_only=False)
    require(payload['request'] == receipt['request'] and payload['sampling'] == receipt['sampling']
            and payload['errors'] == receipt['errors'], 'Prediction and receipt differ')
    require(payload['ids'][0] == 0 and payload['ids'] == list(range(len(payload['ids']))), 'Wrong example IDs')
    prediction = payload['prediction'][0]
    require(prediction.shape == (1, 128, 128) and torch.isfinite(prediction).all(), 'Invalid prediction')
    prediction = prediction[0].numpy()
    require(np.array_equal(prediction[example['mask']], example['truth'][example['mask']]),
            'Observed values differ')
    score = error(prediction, example['truth'])
    close(score, payload['errors']['u'][0], 'CoCoGen example differs from saved error')
    sampling = payload['sampling']
    nfe = selected['config']['steps'] * (1 + selected['config']['repaint'])
    require(sampling['config'] == selected['config'] and sampling['nfe'] == nfe
            and summary['score_evaluations_per_sample'] == nfe and sampling['final_time'] == 0,
            'Displayed sampling budget differs')
    example.update(cocogen=prediction, cocogen_error=score, prediction_sha256=digest, nfe=nfe)


def render(root, original_study, metrics_path, out, reference_only=False):
    torch.set_num_threads(4)
    require(not out.exists(), 'Choose a new output directory to preserve previous figures')
    reader = ArchiveReader(root, original_study)
    catalog = reader.read_json('protocol/catalog.json')
    items = {item['cell']: item for item in catalog['cells'] if item['pde'] == 'burger'}
    require(set(items) == set(CELLS), 'Expected all six Burgers cells')
    metrics = None
    if not reference_only:
        require(metrics_path is not None, 'Comparison mode requires an actual metrics snapshot')
        metrics = reader.read_json(metrics_path)
        require(metrics['phase'] == 'all66' and metrics['status'] == 'complete'
                and metrics['completed_cells'] == 66, 'Comparison requires completed all66 metrics')
        require({row['cell'] for row in metrics['cells'] if row['pde'] == 'burger'} == set(CELLS),
                'Missing Burgers metrics')
    examples = []
    for name in CELLS:
        example = reference_example(reader, items[name])
        if metrics is not None:
            add_cocogen(reader, metrics, example)
        examples.append(example)
    models = ('fm4pde',) if reference_only else ('fm4pde', 'cocogen')
    bound = max(float(np.abs(example[key]).max()) for example in examples for key in ('truth', *models))
    require(np.isfinite(bound) and bound > 0, 'Invalid physical color range')
    out.mkdir(parents=True)
    stem = 'burger_reference_examples' if reference_only else 'burger_main_comparisons'
    pdf_path = out / f'{stem}.pdf'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.facecolor': 'white'})
    files, provenance = {}, []
    with PdfPages(pdf_path, metadata={'Title': 'Burgers: first formal examples'}) as pdf:
        for example in examples:
            arrays = [np.ma.array(example['truth'], mask=~example['mask']), example['truth']]
            titles = [f"Observed values / {example['observations']} points", 'Ground truth']
            for model in models:
                arrays.append(example[model])
                name = 'FM4PDE (archived)' if model == 'fm4pde' else f"CoCoGen ({example['nfe']} NFE)"
                titles.append(f"{name}\nrelative L2: {example[model+'_error']:.2%}")
            fig, axes = plt.subplots(1, len(arrays), figsize=(3.65*len(arrays), 4.9), layout='constrained')
            fig.suptitle(f"Burgers / {example['dist'].upper()} / {LABELS[example['setting']]} / sample 0", fontsize=13)
            cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('white')
            for ax, array, title in zip(axes, arrays, titles):
                im = ax.imshow(array, origin='lower', interpolation='nearest', cmap=cmap, vmin=-bound, vmax=bound)
                ax.set_title(title, fontsize=10, pad=10)
                ax.set_xticks([0, 64, 127]); ax.set_yticks([0, 64, 127])
                ax.set_xlabel('Space index'); ax.set_ylabel('Time index')
            fig.colorbar(im, ax=axes.tolist(), fraction=.025, pad=.02, label='u, physical values')
            scope = 'FM4PDE reference only; CoCoGen comparison is pending.' if reference_only else 'One fixed example per cell; not the 1,000-case mean.'
            fig.supxlabel(scope + '\nSample 0 selected by index; only solution observations. Shared color scale across all six pages; no interpolation.', fontsize=9)
            pdf.savefig(fig, dpi=180)
            path = out / (example['cell'].replace('/', '_') + '_sample_0.png')
            fig.savefig(path, dpi=150); plt.close(fig)
            files[path.name] = sha_file(path)
            provenance.append({k: v for k, v in example.items() if k not in ('truth', 'mask', *models)})
    files[pdf_path.name] = sha_file(pdf_path)
    write_json(out / 'provenance.json', dict(status='rendered_pending_visual_inspection',
        mode='fm4pde_reference_only' if reference_only else 'cocogen_fm4pde_comparison',
        final_study_complete=False, cocogen_predictions_loaded=not reference_only,
        metrics_sha256=None if metrics is None else sha_file(reader.resolve(metrics_path)),
        axes='Horizontal: space index; vertical: time index, increasing upward',
        selection='First formal sample ID 0 in each of six cells; no quality-based selection',
        pdf_pages=6, color_limits=[-bound, bound], cells=provenance,
        metric_precision=dict(fm4pde='Float32 subtraction and float64 norms', cocogen='Float64 subtraction and norms'),
        sources=reader.opened, outputs=files, renderer_sha256=sha_file(__file__)))
    print(f'Rendered {len(examples)} source-checked pages to {pdf_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--original-study', type=Path, required=True)
    parser.add_argument('--metrics')
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    render(args.root, args.original_study, args.metrics, args.out, args.reference_only)
