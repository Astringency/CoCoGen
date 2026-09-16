"""Checkpoint-preserving handoff for the user's revised plateau rule.

Default mode is read-only inspection. Applying the handoff is explicit and is
rejected unless the recorded policy is eligible and all bound processes match.
The original launcher exits nonzero after SIGTERM; its receipt is never changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import tempfile
import time

from .early_stop_policy import decide


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Never overwrite a previous intent, completion or provenance receipt."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
        try:
            json.dump(value,stream,indent=2,ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
            # Atomic publication with an exclusive final name. The temporary
            # link is removed below, leaving an independent regular file.
            os.link(temporary,path)
        finally:
            temporary.unlink(missing_ok=True)


def process_identity(pid):
    root=Path(f'/proc/{pid}')
    try:
        raw=(root/'stat').read_text()
        fields=raw[raw.rfind(')')+2:].split()
        if fields[0]=='Z':
            return None
        return dict(pid=int(pid),ppid=int(fields[1]),start_ticks=int(fields[19]),
                    cwd=str((root/'cwd').resolve(strict=True)),
                    argv=(root/'cmdline').read_bytes().decode().split('\0')[:-1])
    except (FileNotFoundError,ProcessLookupError):
        return None


def bound_alive(expected):
    return process_identity(expected['pid'])==expected


def send_bound_term(expected):
    # pidfd keeps the target stable even if a PID is recycled between checks.
    # The training conda Python lacks these compiled bindings; the server's
    # system Python exposes both. Use the same identity guard in that helper.
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        helper=subprocess.run(['/usr/bin/python3','-c',
            'import json,sys,os,signal; '
            'assert hasattr(os,"pidfd_open") and hasattr(signal,"pidfd_send_signal"), "System Python lacks pidfd"; '
            'from cocogen_burger.external_stop import send_bound_term; send_bound_term(json.load(sys.stdin))'],
            input=json.dumps(expected),text=True,capture_output=True,
            cwd=Path(__file__).resolve().parents[1])
        if helper.returncode:
            raise ValueError('Bound signal helper rejected the action: '+helper.stderr.strip())
        return
    fd=os.pidfd_open(expected['pid'])
    try:
        if not bound_alive(expected):
            raise ValueError('Process identity changed; no signal sent')
        signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
    finally:
        os.close(fd)


def load_binding(study, path):
    study=Path(study).resolve()
    binding=read(path)
    if binding['study']!=str(study) or binding['training_request_sha256']!=sha(study/'training/burger/request.json'):
        raise ValueError('Process binding belongs to a different study or training request')
    code=Path(binding['code_root'])
    if not code.is_relative_to(study):
        raise ValueError('Training worktree must be inside this study')
    targets=binding['processes']
    if set(targets)!={'launcher','rank0','rank1','coordinator'}:
        raise ValueError('Exactly this launcher, two ranks and coordinator must be bound')
    if len({p['pid'] for p in targets.values()})!=4:
        raise ValueError('Process identities are not distinct')
    modules={'launcher':'torch.distributed.run','rank0':'cocogen_burger.train',
             'rank1':'cocogen_burger.train','coordinator':'cocogen_burger.pipeline'}
    for role,p in targets.items():
        args=p['argv']
        if p['cwd']!=str(code) or args[args.index('-m')+1]!=modules[role]:
            raise ValueError(f'Unexpected module or working directory: {role}')
        if args[args.index('--study')+1]!=str(study):
            raise ValueError('Process does not name this study')
    args=targets['launcher']['argv']
    if args[args.index('--module')+1]!='cocogen_burger.train' or '--nproc-per-node=2' not in args:
        raise ValueError('Launcher is not the two-rank Burgers training job')
    if any(targets[r]['ppid']!=targets['launcher']['pid'] for r in ('rank0','rank1')):
        raise ValueError('Training ranks do not belong to the bound launcher')
    if binding['jobs']!={'training':'cocogen_burger_train','coordinator':'burger_pipeline'}:
        raise ValueError('Unexpected original exit-receipt names')
    request=read(study/'training/burger/request.json')['request']
    for rel,expected in request['code'].items():
        if sha(code/rel)!=expected:
            raise ValueError('Frozen running training source differs')
    return binding


def checkpoint_pair(study,directory=None):
    """Inspect real tensors on CPU, leaving both serialized files untouched."""
    import torch
    from cocogen_eval.network import UNET1
    from .train import TrainConfig,model_config
    from .verify_checkpoint import verify

    study=Path(study)
    directory=study/'training/burger/checkpoints' if directory is None else Path(directory)
    if not directory.resolve().is_relative_to(study.resolve()):
        raise ValueError('Checkpoint directory must belong to this study')
    last=verify(study,directory/'last.ckpt')
    path=directory/'best.ckpt'
    with path.open('rb') as stream:
        checksum=hashlib.file_digest(stream,'sha256').hexdigest()
        stream.seek(0)
        best=torch.load(stream,map_location='cpu',weights_only=False)
    request=read(study/'training/burger/request.json')['request']
    cfg=TrainConfig(**request['config'])
    if best['request']!=request or best['model_config']!=model_config(cfg):
        raise ValueError('Best checkpoint does not match the frozen training request/model')
    if best['normalizer']!=read(study/'inputs/burger/normalizer.json'):
        raise ValueError('Best checkpoint normalizer changed')
    if best['epoch']>last['epoch'] or best['best']!=last['best_validation_per_pixel']:
        raise ValueError('Best/last pair advanced during inspection; retry without stopping')
    row=read(study/f"training/burger/epochs/{best['epoch']+1:04d}.json")
    if not row['validation_was_run'] or row['validation']!=best['latest_validation']:
        raise ValueError('Best checkpoint validation does not match its epoch receipt')
    if best['global_step']!=row['global_step'] or best['best']!=row['validation']['uniform']['loss_per_pixel']:
        raise ValueError('Best checkpoint step/loss differs from its epoch receipt')
    if len(best['rank_rng'])!=request['world_size']:
        raise ValueError('Best checkpoint lacks rank RNG states')
    network=UNET1(**best['model_config']['model']['params']['unet_config']['params'])
    network.load_state_dict({k.removeprefix('unet.'):v for k,v in best['state_dict'].items()},strict=True)
    if not all(torch.isfinite(value).all() for value in network.state_dict().values()):
        raise ValueError('Best checkpoint has nonfinite model tensors')
    optimizer=torch.optim.AdamW(network.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    optimizer.load_state_dict(best['optimizer'])
    states=0
    for parameter in network.parameters():
        state=optimizer.state[parameter]
        if set(state)!={'step','exp_avg','exp_avg_sq'} or state['step'].item()!=best['global_step']:
            raise ValueError('Best optimizer state is incomplete or at the wrong step')
        for name in ('exp_avg','exp_avg_sq'):
            if state[name].shape!=parameter.shape or not torch.isfinite(state[name]).all():
                raise ValueError('Best optimizer moments are invalid')
        states+=1
    return dict(last=last,best=dict(checkpoint=str(path),checkpoint_sha256=checksum,
                epoch=best['epoch'],global_step=best['global_step'],
                best_validation_per_pixel=best['best'],model_tensors=len(network.state_dict()),
                optimizer_parameter_states=states,strict_cpu_model_and_optimizer_restore=True,
                stop_condition_met=best['stop_condition_met']))


def snapshot_pair(study,pair):
    """Freeze the verified cutoff before sending any process signal."""
    epoch=pair['last']['epoch']+1
    directory=Path(study)/f'training/burger/early_stop/checkpoints_epoch_{epoch:04d}'
    directory.mkdir(parents=True,exist_ok=True)
    for kind in ('best','last'):
        source=Path(study)/f'training/burger/checkpoints/{kind}.ckpt'
        target=directory/f'{kind}.ckpt'
        expected=pair[kind]['checkpoint_sha256']
        if target.exists():
            if sha(target)!=expected:
                raise ValueError('An existing cutoff snapshot differs')
            continue
        with source.open('rb') as stream, tempfile.NamedTemporaryFile(dir=directory,delete=False) as out:
            temporary=Path(out.name)
            try:
                shutil.copyfileobj(stream,out,length=8*1024**2)
                out.flush(); os.fsync(out.fileno())
                if sha(temporary)!=expected:
                    raise ValueError('Checkpoint changed before snapshot; no signal sent')
                os.link(temporary,target)
            finally:
                temporary.unlink(missing_ok=True)
    return directory


def inspect(study,policy_path,binding_path,*,checkpoints=False):
    study=Path(study).resolve()
    binding=load_binding(study,binding_path)
    decision=decide(study,policy_path)
    result=dict(decision=decision,processes_alive={k:bound_alive(p) for k,p in binding['processes'].items()},
                policy_path=str(Path(policy_path).resolve()),policy_sha256=sha(policy_path),
                binding_path=str(Path(binding_path).resolve()),binding_sha256=sha(binding_path),
                stop_applied=False)
    if checkpoints:
        result['checkpoints']=checkpoint_pair(study)
    return result


def exit_records(study,binding):
    return {role:int((Path(study)/f'logs/{name}.exit').read_text().strip())
            for role,name in binding['jobs'].items()}


def wait_terminal(study,binding,timeout=90):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        alive={k:bound_alive(p) for k,p in binding['processes'].items()}
        if not any(alive.values()) and all((Path(study)/f'logs/{j}.exit').exists() for j in binding['jobs'].values()):
            codes=exit_records(study,binding)
            if any(code==0 for code in codes.values()):
                raise ValueError('Unexpected natural completion; inspect before handoff')
            return codes
        time.sleep(1)
    raise TimeoutError('Original processes did not finish; no completion was fabricated')


def finalize(study,policy_path,binding_path):
    study=Path(study).resolve()
    out=study/'training/burger/early_stop'
    if (study/'training/burger/complete.json').exists():
        return verify_completion(study)
    intent=read(out/'intent.json')
    if intent['policy_sha256']!=sha(policy_path) or intent['binding_sha256']!=sha(binding_path):
        raise ValueError('Stopping intent differs from current policy/binding')
    binding=load_binding(study,binding_path)
    if any(bound_alive(p) for p in binding['processes'].values()):
        raise ValueError('Original training/coordinator still alive')
    codes=exit_records(study,binding)
    if any(code==0 for code in codes.values()):
        raise ValueError('Expected preserved nonzero exits after the intentional stop')
    cutoff=intent['decision']['epochs_completed']
    directory=out/f'checkpoints_epoch_{cutoff:04d}'
    if intent['checkpoint_directory']!=str(directory):
        raise ValueError('Unexpected checkpoint cutoff directory')
    decision=decide(study,policy_path,epochs_completed=cutoff)
    if not decision['eligible']:
        raise ValueError('Latest complete checkpoint no longer meets the stop policy')
    if decision['sources']!=intent['decision']['sources']:
        raise ValueError('Stopping-decision source epoch receipts changed')
    pair=checkpoint_pair(study,directory)
    if pair['last']['epoch']+1!=decision['epochs_completed']:
        raise ValueError('Final checkpoint and epoch receipt differ')
    if any(pair[k]['checkpoint_sha256']!=intent['checkpoints'][k]['checkpoint_sha256'] for k in ('best','last')):
        raise ValueError('Selected checkpoint changed after stopping intent')
    if (study/'training/burger/complete.json').exists():
        raise FileExistsError('Preserve the existing training completion record')
    receipt=dict(status='stopped_for_validation_plateau',generated_at=datetime.now(timezone.utc).isoformat(),
                 intent_sha256=sha(out/'intent.json'),policy_path=str(Path(policy_path).resolve()),
                 policy_sha256=sha(policy_path),binding_path=str(Path(binding_path).resolve()),
                 binding_sha256=sha(binding_path),decision=decision,checkpoints=pair,
                 checkpoint_directory=str(directory),
                 original_exit_codes=codes,checkpoint_files_rewritten=False,
                 original_progress_sha256=sha(study/'training/burger/progress.json'),
                 epochs_completed_before_process_exit=read(study/'training/burger/progress.json')['epochs_completed'],
                 original_checkpoint_stop_condition_met=pair['last']['stop_condition_met'],
                 executor_sha256=sha(__file__),
                 git_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip())
    if (out/'receipt.json').exists():
        saved=read(out/'receipt.json')
        if any(saved[k]!=receipt[k] for k in ('status','intent_sha256','policy_sha256','binding_sha256',
                'original_exit_codes','original_progress_sha256','checkpoint_directory',
                'epochs_completed_before_process_exit','original_checkpoint_stop_condition_met')):
            raise ValueError('Existing stopping receipt conflicts with this finalization')
        if saved['decision']['sources']!=decision['sources'] or any(
                saved['checkpoints'][k]['checkpoint_sha256']!=pair[k]['checkpoint_sha256'] for k in ('best','last')):
            raise ValueError('Final state changed after the earlier stopping receipt')
    else:
        write_new(out/'receipt.json',receipt)
    complete=dict(status='trained',reason='user_authorized_validation_plateau',
                  epochs_completed=decision['epochs_completed'],request=read(study/'training/burger/request.json')['request'],
                  epochs_completed_before_process_exit=receipt['epochs_completed_before_process_exit'],
                  early_stop_receipt=str(out/'receipt.json'),early_stop_receipt_sha256=sha(out/'receipt.json'),
                  next='calibrate best/last on held-out samples, then complete six Burgers main cells')
    for kind in ('best','last'):
        complete[f'{kind}_checkpoint']=str(directory/f'{kind}.ckpt')
        complete[f'{kind}_sha256']=pair[kind]['checkpoint_sha256']
    write_new(study/'training/burger/complete.json',complete)
    return complete


def verify_completion(study):
    study=Path(study).resolve()
    complete=read(study/'training/burger/complete.json')
    if complete['status']!='trained' or complete['reason']!='user_authorized_validation_plateau':
        raise ValueError('Not an externally authorized plateau completion')
    path=Path(complete['early_stop_receipt'])
    if not path.resolve().is_relative_to(study) or sha(path)!=complete['early_stop_receipt_sha256']:
        raise ValueError('Completion receipt path/hash differs')
    receipt=read(path)
    if receipt['status']!='stopped_for_validation_plateau' or receipt['checkpoint_files_rewritten']:
        raise ValueError('Invalid checkpoint-preserving handoff receipt')
    if sha(study/'training/burger/early_stop/intent.json')!=receipt['intent_sha256']:
        raise ValueError('Original stopping intent changed')
    if sha(receipt['policy_path'])!=receipt['policy_sha256'] or sha(receipt['binding_path'])!=receipt['binding_sha256']:
        raise ValueError('Policy or process binding changed')
    binding=load_binding(study,receipt['binding_path'])
    if any(bound_alive(p) for p in binding['processes'].values()):
        raise ValueError('Original training or coordinator still running')
    if exit_records(study,binding)!=receipt['original_exit_codes'] or any(v==0 for v in receipt['original_exit_codes'].values()):
        raise ValueError('Original exit receipts changed')
    intent=read(study/'training/burger/early_stop/intent.json')
    cutoff=intent['decision']['epochs_completed']
    directory=study/f'training/burger/early_stop/checkpoints_epoch_{cutoff:04d}'
    if intent['checkpoint_directory']!=str(directory) or receipt['checkpoint_directory']!=str(directory):
        raise ValueError('Cutoff checkpoint directory changed')
    decision=decide(study,receipt['policy_path'],epochs_completed=cutoff)
    if not decision['eligible'] or decision['epochs_completed']!=complete['epochs_completed']:
        raise ValueError('Training completion does not satisfy the revised policy')
    if decision['sources']!=receipt['decision']['sources'] or decision['sources']!=intent['decision']['sources']:
        raise ValueError('Decision source epoch receipts changed')
    if sha(study/'training/burger/progress.json')!=receipt['original_progress_sha256']:
        raise ValueError('Original post-stop progress changed')
    if complete['request']!=read(study/'training/burger/request.json')['request']:
        raise ValueError('Original training request changed')
    for kind in ('best','last'):
        expected=directory/f'{kind}.ckpt'
        if Path(complete[f'{kind}_checkpoint'])!=expected or sha(expected)!=complete[f'{kind}_sha256']:
            raise ValueError('Completed checkpoint path/hash differs')
        if complete[f'{kind}_sha256']!=receipt['checkpoints'][kind]['checkpoint_sha256']:
            raise ValueError('Checkpoint differs from its verified stopping receipt')
        if complete[f'{kind}_sha256']!=intent['checkpoints'][kind]['checkpoint_sha256']:
            raise ValueError('Checkpoint differs from its stopping intent')
    return complete


def apply(study,policy_path,binding_path):
    study=Path(study).resolve()
    first=inspect(study,policy_path,binding_path)
    if not first['decision']['eligible']:
        raise ValueError('Policy is not eligible; no process was stopped')
    if not all(first['processes_alive'].values()):
        raise ValueError('Original process identity missing; inspect before stopping')
    if (study/'training/burger/complete.json').exists():
        raise FileExistsError('Training already has a completion record')
    binding=load_binding(study,binding_path)
    if any((study/f'logs/{j}.exit').exists() for j in binding['jobs'].values()):
        raise ValueError('An original job already has an exit receipt')
    pair=checkpoint_pair(study)
    directory=snapshot_pair(study,pair)
    current=inspect(study,policy_path,binding_path)
    if not current['decision']['eligible'] or not all(current['processes_alive'].values()):
        raise ValueError('Stop eligibility or live process identity changed')
    if pair['last']['epoch']+1!=current['decision']['epochs_completed']:
        raise ValueError('Checkpoint advanced during inspection; retry without stopping')
    for kind in ('best','last'):
        if sha(study/f'training/burger/checkpoints/{kind}.ckpt')!=pair[kind]['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during inspection; no signal sent')
    intent={**current,'checkpoints':pair,'checkpoint_directory':str(directory),
            'requested_signal':'SIGTERM to the bound torchrun launcher only',
            'created_at':datetime.now(timezone.utc).isoformat(),'executor_sha256':sha(__file__)}
    intent_path=study/'training/burger/early_stop/intent.json'
    if intent_path.exists():
        saved=read(intent_path)
        if saved['policy_sha256']!=current['policy_sha256'] or saved['binding_sha256']!=current['binding_sha256'] or not saved['decision']['eligible']:
            raise ValueError('An incompatible stopping intent already exists')
        if saved['checkpoint_directory']!=str(directory):
            raise ValueError('Earlier stopping intent selected a different epoch; inspect before retrying')
    else:
        write_new(intent_path,intent)
    send_bound_term(binding['processes']['launcher'])
    wait_terminal(study,binding)
    return finalize(study,policy_path,binding_path)


def start_evaluation(study):
    from .pipeline import launch
    verify_completion(study)
    code=Path(__file__).resolve().parents[1]
    launch(code,Path(study),'cocogen_burger_evaluation_after_stop','cocogen_burger.pipeline',
           ['--evaluation-only'],clear_cuda_mask=True)


def watch(study,policy_path,binding_path):
    """Watch this one bound run; perform a handoff only when eligible."""
    previous=None
    while True:
        complete_path=Path(study)/'training/burger/complete.json'
        if complete_path.exists():
            complete=read(complete_path)
            if complete.get('reason')=='user_authorized_validation_plateau':
                return verify_completion(study)
            raise RuntimeError('Original trainer completed naturally; leave its coordinator in control')
        state=inspect(study,policy_path,binding_path)
        decision=state['decision']
        if not all(state['processes_alive'].values()):
            raise RuntimeError('A bound process ended or changed; no automatic restart or signal')
        if decision['epochs_completed']!=previous:
            previous=decision['epochs_completed']
            print(json.dumps(dict(event='watching_plateau',epochs_completed=previous,
                    stale_checks=decision['stale_checks'],eligible=decision['eligible']),ensure_ascii=False),flush=True)
        if decision['eligible']:
            return apply(study,policy_path,binding_path)
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--processes',type=Path,required=True)
    parser.add_argument('--mode',choices=['inspect','apply','finalize','watch'],default='inspect')
    parser.add_argument('--check-checkpoints',action='store_true')
    parser.add_argument('--start-evaluation',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='inspect':
        if args.start_evaluation:
            parser.error('Inspection cannot start evaluation')
        result=inspect(args.study,args.policy,args.processes,checkpoints=args.check_checkpoints)
    else:
        operation={'apply':apply,'finalize':finalize,'watch':watch}[args.mode]
        result=operation(args.study,args.policy,args.processes)
        if args.start_evaluation:
            start_evaluation(args.study)
    if args.output:
        write_new(args.output,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
