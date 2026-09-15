"""Plot the first formal example for each completed ID task of existing models."""
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
from cocogen_eval.verify_archived_predictions import ArchiveReader, close, require


PDES = ('darcy', 'poisson', 'helmholtz', 'nsnonbounded')
PDE_NAMES = dict(darcy='Darcy', poisson='Poisson', helmholtz='Helmholtz', nsnonbounded='Navier-Stokes')
SETTINGS = ('full_forward', 'full_inverse', 'sparse_forward', 'sparse_inverse', 'sparse_joint')
LABELS = dict(full_forward='Full-observation forward', full_inverse='Full-observation inverse',
              sparse_forward='Sparse forward', sparse_inverse='Sparse inverse',
              sparse_joint='Sparse joint reconstruction')
MEANINGS = dict(darcy=('permeability', 'pressure'), poisson=('source', 'solution'),
                helmholtz=('source', 'solution'), nsnonbounded=('initial vorticity', 'final vorticity'))


def error(prediction, truth):
    delta = prediction.astype(np.float64).ravel() - truth.astype(np.float64).ravel()
    denominator = np.linalg.norm(truth.astype(np.float64).ravel())
    require(denominator > 0, 'Zero relative-error denominator')
    return float(np.linalg.norm(delta) / denominator)


def render(root, original_study, metrics_path, out):
    torch.set_num_threads(4)
    reader = ArchiveReader(root, original_study)
    metrics = reader.read_json(metrics_path)
    require(metrics['status'] in ('partial', 'complete'), 'Invalid metrics snapshot')
    catalog = reader.read_json('protocol/catalog.json')
    items = {r['cell']: r for r in catalog['cells']}
    selected = [r for r in metrics['cells'] if r['pde'] in PDES and r['dist'] == 'id']
    selected.sort(key=lambda r: (PDES.index(r['pde']), SETTINGS.index(r['setting'])))
    require(selected, 'No complete ID cells are available')
    examples, limits = [], {}
    for cell in selected:
        name, pde = cell['cell'], cell['pde']
        item = items[name]
        record = reader.read_json(item['frozen_record'], item['frozen_sha256'])
        summary_source = str(reader.original / 'main' / name / 'summary.json')
        summary = reader.read_json(summary_source, metrics['evidence'][summary_source])
        require(summary['status'] == 'complete' and summary['n'] == 1000, 'Only complete cells may be illustrated')
        require(summary['score_evaluations_per_sample'] == 500, 'The displayed CoCoGen budget must match the result')
        sample_id = 2000 if pde == 'nsnonbounded' else 0
        require(sample_id == min(summary['ids']), 'Use the first formal ID without selecting by quality')
        row = next(r for r in record['rows'] if r['sample_id'] == sample_id)
        reference_source = row['result_path']
        reference_hash = next(b['result_sha256'] for b in record['batches'] if b['result_path'] == reference_source)
        reference_path = reader.check(reference_source, reference_hash)
        reference = torch.load(reference_path, map_location='cpu', weights_only=False)
        index = row['result_row']
        if pde == 'nsnonbounded':
            truth = reference['truths'][index].float()
            fm = reference['predictions'][index].float()
            mask = reference['masks'][index].bool().clone()
        else:
            require(not reference['ground_truth_metadata'].get('synthetic', False), 'Synthetic truth is excluded')
            truth = torch.cat([reference['coef_ground_truth'][index], reference['sol_ground_truth'][index]]).float()
            fm = torch.cat([reference['coef_final'][index], reference['sol_final'][index]]).float()
            mask = torch.cat([reference['masks']['coef'][index], reference['masks']['sol'][index]]).bool()
        if record['task'] == 'forward':
            mask[1] = False
            targets = ('u',)
        elif record['task'] == 'inverse':
            mask[0] = False
            targets = ('a',)
        else:
            require(record['task'] == 'both', 'Unknown task')
            targets = ('a', 'u')
        first = summary['batches'][0]
        receipt = reader.read_json(first['receipt'], metrics['evidence'][first['receipt']])
        prediction_path = reader.check(first['prediction_path'], first['prediction_sha256'])
        require(first['prediction_sha256'] == receipt['prediction_sha256'] == metrics['evidence'][first['prediction_path']], 'Wrong CoCoGen prediction hash')
        payload = torch.load(prediction_path, map_location='cpu', weights_only=False)
        require(payload['request'] == receipt['request'] and payload['errors'] == receipt['errors'] and payload['sampling'] == receipt['sampling'], 'Wrong CoCoGen payload')
        index = payload['ids'].index(sample_id)
        prediction = payload['prediction'][index]
        require(truth.shape == fm.shape == prediction.shape == mask.shape == (2, 128, 128), 'Wrong image shape')
        require(torch.isfinite(truth).all() and torch.isfinite(fm).all() and torch.isfinite(prediction).all(), 'Nonfinite image')
        require(torch.equal(prediction[mask], truth[mask]), 'CoCoGen observations do not match this example')
        arrays = dict(truth=truth.numpy(), fm4pde=fm.numpy(), cocogen=prediction.numpy())
        errors = {}
        for channel, field in enumerate(('a', 'u')):
            e_fm = error(arrays['fm4pde'][channel], arrays['truth'][channel])
            e_coco = error(arrays['cocogen'][channel], arrays['truth'][channel])
            close(e_fm, row[f'error_{field}'], 'FM4PDE image error differs from frozen reference')
            close(e_coco, payload['errors'][field][index], 'CoCoGen image error differs from saved prediction')
            errors[field] = dict(fm4pde=e_fm, cocogen=e_coco)
            low, high = limits.get((pde, channel), (float('inf'), -float('inf')))
            limits[pde, channel] = (min(low, *(float(a[channel].min()) for a in arrays.values())),
                                   max(high, *(float(a[channel].max()) for a in arrays.values())))
        examples.append(dict(cell=name, pde=pde, setting=cell['setting'], sample_id=sample_id,
            arrays=arrays, mask=mask.numpy(), target_fields=targets, errors=errors,
            reference_sha256=reference_hash, prediction_sha256=first['prediction_sha256']))
    for key, (low, high) in list(limits.items()):
        if low < 0 < high:
            bound = max(abs(low), abs(high))
            limits[key] = (-bound, bound)
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = out / 'first_id_main_comparisons.pdf'
    require(not pdf_path.exists(), 'Preserve previous figure outputs')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    files, provenance = {}, []
    with PdfPages(pdf_path, metadata={'Title': 'CoCoGen and FM4PDE: first formal ID examples'}) as pdf:
        for item in examples:
            pde = item['pde']
            fig, axes = plt.subplots(2, 4, figsize=(13, 7), layout='constrained')
            fig.suptitle(f"{PDE_NAMES[pde]} / ID / {LABELS[item['setting']]} / sample {item['sample_id']}", fontsize=15)
            for channel, field in enumerate(('a', 'u')):
                mask = item['mask'][channel]
                truth = item['arrays']['truth'][channel]
                pictures = [np.ma.array(truth, mask=~mask), truth,
                            item['arrays']['fm4pde'][channel], item['arrays']['cocogen'][channel]]
                titles = [f'Observed values\n{int(mask.sum())} grid points', 'Ground truth']
                for model, title in [('fm4pde', 'FM4PDE (archived)'), ('cocogen', 'CoCoGen (500 NFE)')]:
                    detail = f"relative L2: {100*item['errors'][field][model]:.2f}%" if field in item['target_fields'] else 'auxiliary field; excluded from task score'
                    titles.append(title + '\n' + detail)
                low, high = limits[pde, channel]
                cmap = plt.get_cmap('RdBu_r' if low < 0 < high else 'cividis').copy()
                cmap.set_bad('white')
                for column, (array, title) in enumerate(zip(pictures, titles)):
                    ax = axes[channel, column]
                    im = ax.imshow(array, origin='lower', interpolation='nearest', cmap=cmap, vmin=low, vmax=high)
                    ax.set_title(title, fontsize=9, pad=9)
                    ax.set_xticks([0, 64, 127]); ax.set_yticks([0, 64, 127]); ax.set_xlabel('column index')
                    if column == 0:
                        ax.set_ylabel(f'{field}: {MEANINGS[pde][channel]}\nrow index')
                        if not mask.any():
                            ax.text(.5, .5, 'No observations', transform=ax.transAxes, ha='center', va='center', color='#4b5563')
                fig.colorbar(im, ax=axes[channel, :].tolist(), fraction=.025, pad=.015, label=f'{field}, physical values')
            fig.supxlabel('First formal ID, chosen by index. Each panel is one example, not the 1,000-case mean.\n'
                          'Same ground truth and effective observations. Field color scales shared across each PDE\'s pages; no image interpolation.', fontsize=9)
            pdf.savefig(fig, dpi=180)
            path = out / (item['cell'].replace('/', '_') + f"_sample_{item['sample_id']}.png")
            fig.savefig(path, dpi=150); plt.close(fig)
            files[path.name] = sha_file(path)
            provenance.append({k: v for k, v in item.items() if k not in ('arrays', 'mask')})
    files[pdf_path.name] = sha_file(pdf_path)
    write_json(out / 'provenance.json', dict(status='rendered_pending_visual_inspection',
        scope='First formal sample for completed ID tasks of the four existing models; qualitative examples only',
        metrics_sha256=sha_file(reader.resolve(metrics_path)), metrics_completed_cells=metrics['completed_cells'],
        cells=provenance, pdf_pages=len(examples), color_limits={f'{pde}/{channel}': list(v) for (pde, channel), v in limits.items()},
        sources=reader.opened, outputs=files, renderer_sha256=sha_file(__file__)))
    print(f'Rendered {len(examples)} source-checked pages to {pdf_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--original-study', type=Path, required=True)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    render(args.root, args.original_study, args.metrics, args.out)
