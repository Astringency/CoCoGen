"""Complete the deferred Burgers phase after first60, with separate tmux jobs."""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import shlex
import subprocess
import time

from cocogen_eval.common import STUDY,sha_file,write_json
from .data import require_first60
from .train import TrainConfig,code_identity


def has_session(name):
    return subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0


def launch(code,study,name,module,args=()):
    command=shlex.join(['bash',str(code/'cocogen_burger/job.sh'),name,module,*map(str,args)])
    if has_session(name):
        actual=subprocess.check_output(['tmux','display-message','-p','-t',name,'#{pane_start_command}'],text=True).strip()
        if actual!=command:
            raise RuntimeError(f'Existing session command differs: {name}: {actual}')
        return
    exit_path=study/'logs'/f'{name}.exit'
    if exit_path.exists():
        if exit_path.read_text().strip()=='0':
            return
        raise RuntimeError(f'Failed job must be inspected before resuming: {exit_path}')
    subprocess.run(['tmux','new-session','-d','-s',name,command],check=True,cwd=code)


def wait_job(study,name,artifact):
    exit_path=study/'logs'/f'{name}.exit'
    while not exit_path.exists():
        if not has_session(name):
            raise RuntimeError(f'Job has no live session and no exit receipt: {name}')
        time.sleep(10)
    if exit_path.read_text().strip()!='0':
        raise RuntimeError(f'Burgers job failed: {name}')
    if not Path(artifact).exists():
        raise RuntimeError(f'Job exited without required artifact: {artifact}')


def free_memory_gate():
    while True:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows.strip().splitlines()}
        if all(memory.get(i,0)>=50*1024 for i in (0,1)):
            return memory
        print(json.dumps(dict(stage='waiting_for_gpu_memory',free_mib=memory)),flush=True)
        time.sleep(30)


def pipeline(study):
    study=Path(study)
    code=Path(__file__).resolve().parents[1]
    status_path=study/'protocol/burger_pipeline.json'
    def status(stage,**extra):
        write_json(status_path,dict(stage=stage,code=str(code),updated_unix=time.time(),**extra))
        print(json.dumps(dict(stage=stage,**extra)),flush=True)
    status('waiting_for_first60')
    while not (study/'reports/first60_complete.json').exists():
        time.sleep(20)
    gate=require_first60(study)
    checks=json.loads((study/'validation/burger_training_cpu.json').read_text())
    assert checks['status']=='passed'
    assert checks['training_code']==code_identity(), 'CPU training checks do not bind this source version'
    status('preparing_burger_data',first60_gate=gate)
    launch(code,study,'cocogen_burger_prepare','cocogen_burger.data')
    wait_job(study,'cocogen_burger_prepare',study/'inputs/burger/training_cache.json')
    status('measuring_training_memory',free_mib=free_memory_gate())
    launch(code,study,'cocogen_burger_benchmark','cocogen_burger.benchmark',['--device','cuda:0'])
    wait_job(study,'cocogen_burger_benchmark',study/'validation/burger_training_gpu.json')
    benchmark=json.loads((study/'validation/burger_training_gpu.json').read_text())
    assert benchmark['status']=='passed'
    cfg=replace(TrainConfig(),batch_size_per_rank=benchmark['batch_size_per_rank'])
    config_path=study/'training/burger/train_config.json'
    write_json(config_path,asdict(cfg))
    status('training',batch_size_per_rank=cfg.batch_size_per_rank,free_mib=free_memory_gate())
    launch(code,study,'cocogen_burger_train','torch.distributed.run',
        ['--standalone','--nnodes=1','--nproc-per-node=2','--module','cocogen_burger.train','--config',config_path])
    wait_job(study,'cocogen_burger_train',study/'training/burger/complete.json')
    trained=json.loads((study/'training/burger/complete.json').read_text())
    assert trained['status']=='trained'
    for kind in ['best','last']:
        assert sha_file(trained[f'{kind}_checkpoint'])==trained[f'{kind}_sha256']
    status('calibrating_sampler',epochs_completed=trained['epochs_completed'])
    launch(code,study,'cocogen_burger_calibrate','cocogen_burger.evaluate',['--mode','calibrate','--device','cuda:0'])
    wait_job(study,'cocogen_burger_calibrate',study/'protocol/selected/burger.json')
    selected=json.loads((study/'protocol/selected/burger.json').read_text())
    assert selected['status']=='validated'
    status('evaluating_six_cells',estimated_gpu_hours=selected['estimated_main_gpu_hours'])
    for i in [0,1]:
        launch(code,study,f'cocogen_burger_main{i}','cocogen_burger.evaluate',
            ['--mode','worker','--worker-id',i,'--device',f'cuda:{i}'])
    for i in [0,1]:
        wait_job(study,f'cocogen_burger_main{i}',study/'main/burger/workers'/f'{i}_complete.json')
    launch(code,study,'cocogen_burger_finish','cocogen_burger.evaluate',['--mode','finish'])
    wait_job(study,'cocogen_burger_finish',study/'reports/all66_complete.json')
    status('all66_complete',completion_receipt=str(study/'reports/all66_complete.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=STUDY)
    pipeline(parser.parse_args().study)
