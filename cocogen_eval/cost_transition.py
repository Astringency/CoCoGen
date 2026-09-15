"""Stop superseded calibration workers immediately after a completed trial."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

from .common import STUDY, write_json


def transition(study, workers):
    pending = dict(workers)
    finished = []
    while pending:
        for pde, pid in list(pending.items()):
            cmdline_path = Path(f'/proc/{pid}/cmdline')
            if not cmdline_path.exists():
                finished.append(dict(pde=pde,pid=pid,status='already_exited'))
                del pending[pde]
                continue
            cmd = cmdline_path.read_bytes().replace(b'\0',b' ').decode()
            assert 'cocogen_eval.calibrate ' in cmd and f'--pde {pde} ' in cmd, (pid,cmd)
            paths = list((Path(study)/'calibration'/pde).glob('*/0-4/control_s2000_r1_p0.0002_N50_M10_euler/receipt.json'))
            if not paths:
                continue
            receipt = json.loads(paths[0].read_text())
            assert receipt['request']['stage']=='control' and receipt['request']['config']['steps']==2000
            os.kill(pid,signal.SIGINT)
            finished.append(dict(pde=pde,pid=pid,status='interrupted_after_trial',completed_receipt=str(paths[0]),
                reason='User questioned 5–7 day cost; expand short-step calibration before formal sampling'))
            del pending[pde]
        write_json(Path(study)/'protocol/cost_transition.json',dict(pending=pending,workers=finished,
            main_sampling='automatic continuation paused; no formal samples generated'))
        if pending:
            time.sleep(2)
    print(json.dumps(dict(status='complete',workers=finished)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=STUDY)
    parser.add_argument('--poisson-pid',type=int,required=True)
    parser.add_argument('--darcy-pid',type=int,required=True)
    args=parser.parse_args()
    transition(args.study,dict(poisson=args.poisson_pid,darcy=args.darcy_pid))
