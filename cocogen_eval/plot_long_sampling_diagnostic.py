"""Export source-checked scientific figures for the separate Darcy diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

from cocogen_eval.common import sha_file, write_json
from cocogen_eval.long_sampling_diagnostic import CONFIG_NAMES, SETTINGS, inputs, require


def render(root):
    torch.set_num_threads(4)
    manifest, groups, _ = inputs(root)
    comparison_path = root/'comparison.json'
    comparison = json.loads(comparison_path.read_text())
    require(comparison['status']=='complete_diagnostic_only' and
            comparison['manifest_sha256']==sha_file(root/'manifest.json'), 'A verified full diagnostic is required')
    rows = {(r['config'],r['setting']):r for r in comparison['results']}
    require(len(rows)==9, 'Expected nine complete diagnostic cells')
    predictions, provenance = {}, {}
    limits = {0:[], 1:[]}
    for setting in SETTINGS:
        for channel in (0,1):
            value = groups[setting]['fields'][:,channel].numpy()
            limits[channel].extend([float(value.min()),float(value.max())])
        for name in CONFIG_NAMES:
            path = root/'runs'/name/setting/'prediction.pt'
            require(sha_file(path)==rows[name,setting]['prediction_sha256'], 'Changed prediction')
            payload = torch.load(path,map_location='cpu',weights_only=False)
            require(payload['ids']==manifest['ids'], 'Wrong figure IDs')
            array = payload['prediction'].numpy()
            require(array.shape==(8,2,128,128) and np.isfinite(array).all(), 'Invalid figure fields')
            predictions[name,setting] = array
            provenance[f'{name}/{setting}'] = sha_file(path)
            for channel in (0,1):
                limits[channel].extend([float(array[:,channel].min()),float(array[:,channel].max())])
    limits = {channel:(min(values),max(values)) for channel,values in limits.items()}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,
                         'axes.labelsize':10,'figure.facecolor':'white','savefig.facecolor':'white'})
    cmap = plt.get_cmap('cividis').copy(); cmap.set_bad('#ffffff')
    out = root/'figures'; out.mkdir(exist_ok=True)
    pdf_path = out/'darcy_long_sampling_all8.pdf'
    require(not pdf_path.exists(), 'Preserve the existing figure artifact')
    exported=[]; pages=0
    labels={'s100_r4':'100 steps, r=4\n500 NFE',
            's2000_r4':'2,000 steps, r=4\n10,000 NFE',
            's2000_r10':'2,000 steps, r=10\n22,000 NFE'}
    task_names={'sparse_forward':'Sparse forward','sparse_inverse':'Sparse inverse',
                'sparse_joint':'Sparse joint reconstruction'}
    with PdfPages(pdf_path,metadata={'Title':'Darcy sampling-budget diagnostic: all eight calibration cases',
                                    'Author':'CoCoGen experiment'}) as pdf:
        for setting in SETTINGS:
            group=groups[setting]
            targets={'forward':('u',),'inverse':('a',),'both':('a','u')}[group['task']]
            for case,sample_id in enumerate(manifest['ids']):
                fig,axes=plt.subplots(2,5,figsize=(15,7.0),layout='constrained')
                fig.suptitle(f'Darcy / ID calibration / {task_names[setting]} / sample {sample_id}',fontsize=15)
                for channel,field,meaning in [(0,'a','permeability'),(1,'u','pressure')]:
                    truth=group['fields'][case,channel].numpy()
                    mask=group['masks'][case,channel].numpy()
                    arrays=[np.ma.array(truth,mask=~mask),truth]
                    arrays += [predictions[name,setting][case,channel] for name in CONFIG_NAMES]
                    titles=[f'Observed values\n{int(mask.sum())} grid points','Ground truth']
                    for name in CONFIG_NAMES:
                        error=rows[name,setting]['errors'][field][case]*100
                        detail=f'relative L2: {error:.2f}%' if field in targets else 'auxiliary field; excluded from task score'
                        titles.append(labels[name]+'\n'+detail)
                    for column,(array,title) in enumerate(zip(arrays,titles)):
                        ax=axes[channel,column]
                        im=ax.imshow(array,origin='lower',interpolation='nearest',cmap=cmap,
                                     vmin=limits[channel][0],vmax=limits[channel][1])
                        ax.set_title(title,pad=9,fontsize=9 if column>=2 else 10)
                        ax.set_xticks([0,64,127]);ax.set_yticks([0,64,127])
                        ax.set_xlabel('column index')
                        if column==0:ax.set_ylabel(f'{field}: {meaning}\nrow index')
                        if column==0 and not mask.any():
                            ax.text(.5,.5,'No observations',transform=ax.transAxes,ha='center',va='center',color='#4b5563')
                    fig.colorbar(im,ax=axes[channel,:].tolist(),fraction=.02,pad=.012,label=f'{field}, physical values')
                fig.supxlabel('Same 8 calibration cases, seed and observations across configurations. '
                              'One shared color scale per field across all pages; no image interpolation.\n'
                              'RePaint r is the number of extra visits. This is an exploratory diagnostic, not a formal-main result.',fontsize=9)
                pdf.savefig(fig,dpi=180)
                pages+=1
                if case==0:
                    path=out/f'{setting}_sample_{sample_id}.png'
                    fig.savefig(path,dpi=150)
                    exported.append(path)
                plt.close(fig)
    require(pages==24, 'Expected three tasks times eight cases')
    exported.append(pdf_path)
    write_json(out/'provenance.json',dict(status='rendered_pending_visual_inspection',
        manifest_sha256=sha_file(root/'manifest.json'),comparison_sha256=sha_file(comparison_path),
        prediction_sha256=provenance,ids=manifest['ids'],settings=SETTINGS,pdf_pages=pages,
        color_limits={str(k):list(v) for k,v in limits.items()},renderer_sha256=sha_file(__file__),
        outputs={path.name:sha_file(path) for path in exported},
        selection='Every calibration case appears in the PDF; PNG previews use the first ID, chosen before results.'))
    print(json.dumps(dict(pdf=str(pdf_path),pages=pages,previews=[str(p) for p in exported if p.suffix=='.png'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();render(args.root.resolve())
