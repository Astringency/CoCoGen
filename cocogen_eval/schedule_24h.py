"""Launch all 60 main cells only after measured throughput fits the user budget."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .calibrate import implementation_hash
from .common import STUDY, sha_file, write_json
from .evaluate import aggregate


PDES=('darcy','poisson','helmholtz','nsnonbounded')


def allocation(hours):
    """Exact two-GPU partition for four whole-PDE jobs; no duplicated cells."""
    choices=[]
    for bits in itertools.product([0,1],repeat=3):
        groups=[[],[]]
        groups[0].append(PDES[0])
        for pde,gpu in zip(PDES[1:],bits):
            groups[gpu].append(pde)
        if not groups[1]:
            continue
        totals=[sum(hours[pde] for pde in group) for group in groups]
        choices.append((max(totals),abs(totals[0]-totals[1]),groups,totals))
    _,_,groups,totals=min(choices)
    return groups,totals


def wait_success(paths):
    paths=[Path(p) for p in paths]
    while True:
        for path in paths:
            if path.exists() and path.read_text().strip()!='0':
                raise RuntimeError(f'Prerequisite job exited unsuccessfully: {path}')
        if all(path.exists() for path in paths):
            return
        time.sleep(10)


def schedule(study,hours_limit=24.,margin=1.15):
    study=Path(study)
    out=study/'protocol/budget_24h.json'
    write_json(out,dict(status='waiting_for_calibration',hours_limit=hours_limit,margin=margin,
        user_preference='约 1 天，优先效果与耗时的平衡',first_phase_cells=60,samples_per_cell=1000,
        burgers='train only after all 60 cells complete and prediction hashes are verified'))
    wait_success(study/'logs'/f'calibrate_cost_{group}.exit' for group in ('ph','dn'))
    inputs=json.loads((study/'validation/inputs.json').read_text())
    assert inputs['status']=='passed' and inputs['count']==66
    assert json.loads((study/'validation/evaluator.json').read_text())['status']=='passed'
    selected={pde:json.loads((study/'protocol/selected'/f'{pde}.json').read_text()) for pde in PDES}
    for pde,row in selected.items():
        assert row['status']=='validated' and row['implementation_sha256']==implementation_hash()
        assert row['config']['steps']*(1+row['config']['repaint'])<=500
        assert row['estimated_main_gpu_hours']>0
    hours={pde:row['estimated_main_gpu_hours'] for pde,row in selected.items()}
    groups,totals=allocation(hours)
    projected=max(totals)*margin
    plan=dict(status='ready' if projected<=hours_limit else 'needs_lower_cost',hours_limit=hours_limit,
        user_preference='约 1 天，优先效果与耗时的平衡',first_phase_cells=60,samples_per_cell=1000,
        margin=margin,estimated_hours_with_margin=projected,estimated_gpu_hours=hours,
        gpu_groups=groups,gpu_hours=totals,
        selected_sha256={pde:sha_file(study/'protocol/selected'/f'{pde}.json') for pde in PDES},
        selected={pde:dict(stage=row['stage'],config=row['config'],validation_macro_mean=row['validation_macro_mean'])
                  for pde,row in selected.items()})
    write_json(out,plan)
    print(json.dumps(plan),flush=True)
    if projected>hours_limit:
        raise RuntimeError('Measured projection exceeds the user budget; formal sampling was not started')
    started=time.time()
    code=Path(__file__).resolve().parents[1]
    for gpu in range(2):
        session=f'cocogen_main24_gpu{gpu}'
        existing=subprocess.run(['tmux','has-session','-t',session],capture_output=True)
        if existing.returncode==0:
            raise RuntimeError(f'An existing live session must be inspected before scheduling: {session}')
        exit_path=study/'logs'/f'main24_gpu{gpu}.exit'
        if exit_path.exists():
            raise RuntimeError(f'Previous main24 exit needs explicit review before resuming: {exit_path}')
    for gpu,pdes in enumerate(groups):
        session=f'cocogen_main24_gpu{gpu}'
        command=shlex.join(['bash',str(code/'cocogen_eval/main24_group.sh'),str(gpu),*pdes])
        subprocess.run(['tmux','new-session','-d','-s',session,command],check=True,cwd=code)
    plan.update(status='running',started_unix=started)
    write_json(out,plan)
    wait_success(study/'logs'/f'main24_gpu{gpu}.exit' for gpu in (0,1))
    aggregate(study)
    plan.update(status='first60_complete',finished_unix=time.time(),actual_main_hours=(time.time()-started)/3600,
        completion_receipt=str(study/'reports/first60_complete.json'))
    write_json(out,plan)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=STUDY)
    parser.add_argument('--hours-limit',type=float,default=24.)
    parser.add_argument('--margin',type=float,default=1.15)
    args=parser.parse_args()
    schedule(args.study,args.hours_limit,args.margin)
