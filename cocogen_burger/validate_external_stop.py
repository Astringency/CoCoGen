"""Handoff tests: synthetic study files; signals target only a test child."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import external_stop as stop
from .early_stop_policy import decide
from .validate_early_stop_policy import history


class Handoff(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.policy=self.root/'protocol/policy.json'
        self.binding_path=self.root/'protocol/processes.json'
        code=self.root/'code'
        code.mkdir()
        (code/'frozen.py').write_text('unchanged training source\n')
        request=dict(config=dict(relative_min_delta=.002,batch_size_per_rank=32,
                     validation_every=10,max_epochs=1200,min_epochs=300,patience_checks=20),
                     world_size=2,code={'frozen.py':stop.sha(code/'frozen.py')})
        self.put('training/burger/request.json',dict(request=request))
        checksum=stop.sha(self.root/'training/burger/request.json')
        self.put('protocol/policy.json',dict(status='authorized',study=str(self.root),
                 training_request_sha256=checksum,relative_min_delta=.002,minimum_epochs=100,patience_checks=6))
        identities={}
        for i,(role,module) in enumerate([('launcher','torch.distributed.run'),
                  ('rank0','cocogen_burger.train'),('rank1','cocogen_burger.train'),
                  ('coordinator','cocogen_burger.pipeline')]):
            args=['python','-m',module,'--study',str(self.root)]
            if role=='launcher':
                args+=['--module','cocogen_burger.train','--nproc-per-node=2']
            identities[role]=dict(pid=2_000_000_000+i,ppid=2_000_000_000 if i in (1,2) else 1,
                                  start_ticks=10,cwd=str(code),argv=args)
        self.binding=dict(study=str(self.root),training_request_sha256=checksum,code_root=str(code),
                          processes=identities,jobs=dict(training='cocogen_burger_train',coordinator='burger_pipeline'))
        self.put('protocol/processes.json',self.binding)
        self.rows(100)
        self.pair={}
        for kind,epoch in [('best',39),('last',99)]:
            path=self.root/f'training/burger/checkpoints/{kind}.ckpt'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((kind+' immutable fake checkpoint').encode())
            self.pair[kind]=dict(checkpoint=str(path),checkpoint_sha256=stop.sha(path),epoch=epoch,
                                 stop_condition_met=False)

    def put(self,rel,value):
        path=self.root/rel
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def rows(self,end,values=None):
        for row in history(end,values):
            row['global_step']=row['epochs_completed']*782
            self.put(f"training/burger/epochs/{row['epochs_completed']:04d}.json",row)
        self.put('training/burger/progress.json',row)

    def prepare_intent(self):
        with patch.object(stop,'bound_alive',return_value=True):
            inspection=stop.inspect(self.root,self.policy,self.binding_path)
        directory=stop.snapshot_pair(self.root,self.pair)
        intent={**inspection,'checkpoints':self.pair,'checkpoint_directory':str(directory)}
        self.put('training/burger/early_stop/intent.json',intent)
        self.put_exit('cocogen_burger_train',1)
        self.put_exit('burger_pipeline',1)
        selected=copy.deepcopy(self.pair)
        for kind in selected:
            selected[kind]['checkpoint']=str(directory/f'{kind}.ckpt')
        return selected

    def put_exit(self,job,code):
        path=self.root/f'logs/{job}.exit'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(str(code)+'\n')

    def finalized(self):
        pair=self.prepare_intent()
        with patch.object(stop,'checkpoint_pair',return_value=pair):
            return stop.finalize(self.root,self.policy,self.binding_path)

    def test_inspect_is_current_and_read_only(self):
        self.rows(79)
        result=stop.inspect(self.root,self.policy,self.binding_path)
        self.assertFalse(result['decision']['eligible'])
        self.assertEqual(result['decision']['stale_checks'],3)
        self.assertFalse((self.root/'training/burger/early_stop').exists())

    def test_historical_cutoff_does_not_override_live_improvement(self):
        self.rows(110,{110:.1})
        self.assertFalse(decide(self.root,self.policy)['eligible'])
        self.assertTrue(decide(self.root,self.policy,epochs_completed=100)['eligible'])
        with self.assertRaises(ValueError):
            decide(self.root,self.policy,epochs_completed=111)

    def test_ineligible_and_missing_process_never_signal(self):
        for end in [99,100]:
            self.rows(end)
            with patch.object(stop,'send_bound_term') as signal_mock, self.assertRaises(ValueError):
                stop.apply(self.root,self.policy,self.binding_path)
            signal_mock.assert_not_called()
        self.assertFalse((self.root/'training/burger/complete.json').exists())

    def test_missing_checkpoint_never_signals(self):
        with patch.object(stop,'bound_alive',return_value=True), \
             patch.object(stop,'checkpoint_pair',side_effect=FileNotFoundError), \
             patch.object(stop,'send_bound_term') as signal_mock, self.assertRaises(FileNotFoundError):
            stop.apply(self.root,self.policy,self.binding_path)
        signal_mock.assert_not_called()

    def test_snapshot_is_independent_and_detects_source_change(self):
        directory=stop.snapshot_pair(self.root,self.pair)
        original=self.root/'training/burger/checkpoints/last.ckpt'
        self.assertNotEqual(original.stat().st_ino,(directory/'last.ckpt').stat().st_ino)
        original.write_bytes(b'next epoch')
        self.assertEqual(stop.sha(directory/'last.ckpt'),self.pair['last']['checkpoint_sha256'])
        (directory/'last.ckpt').unlink()
        with self.assertRaises(ValueError):
            stop.snapshot_pair(self.root,self.pair)

    def test_finalize_preserves_originals_and_internal_stop_false(self):
        originals={p:p.read_bytes() for p in (self.root/'training/burger/checkpoints').iterdir()}
        complete=self.finalized()
        self.assertEqual(complete['epochs_completed'],100)
        self.assertIn('checkpoints_epoch_0100',complete['last_checkpoint'])
        receipt=stop.read(self.root/'training/burger/early_stop/receipt.json')
        self.assertFalse(receipt['original_checkpoint_stop_condition_met'])
        self.assertEqual(receipt['original_exit_codes'],dict(training=1,coordinator=1))
        self.assertEqual(stop.verify_completion(self.root),complete)
        for p,data in originals.items():
            self.assertEqual(p.read_bytes(),data)

    def test_inflight_epoch_does_not_change_selected_cutoff(self):
        pair=self.prepare_intent()
        self.rows(101)
        with patch.object(stop,'checkpoint_pair',return_value=pair):
            complete=stop.finalize(self.root,self.policy,self.binding_path)
        self.assertEqual(complete['epochs_completed'],100)
        self.assertEqual(complete['epochs_completed_before_process_exit'],101)
        stop.verify_completion(self.root)

    def test_finalize_rejects_live_process_or_natural_exit(self):
        self.prepare_intent()
        with patch.object(stop,'bound_alive',return_value=True), self.assertRaises(ValueError):
            stop.finalize(self.root,self.policy,self.binding_path)
        self.put_exit('cocogen_burger_train',0)
        with self.assertRaises(ValueError):
            stop.finalize(self.root,self.policy,self.binding_path)
        self.assertFalse((self.root/'training/burger/complete.json').exists())

    def test_resume_after_receipt_before_complete(self):
        pair=self.prepare_intent()
        original=stop.write_new
        def fail_complete(path,value):
            if Path(path).name=='complete.json':
                raise OSError('synthetic interrupted publication')
            return original(path,value)
        with patch.object(stop,'checkpoint_pair',return_value=pair):
            with patch.object(stop,'write_new',side_effect=fail_complete), self.assertRaises(OSError):
                stop.finalize(self.root,self.policy,self.binding_path)
            receipt_hash=stop.sha(self.root/'training/burger/early_stop/receipt.json')
            stop.finalize(self.root,self.policy,self.binding_path)
        self.assertEqual(stop.sha(self.root/'training/burger/early_stop/receipt.json'),receipt_hash)
        stop.verify_completion(self.root)

    def test_tampered_checkpoint_exit_policy_or_receipt_rejected(self):
        complete=self.finalized()
        paths=[Path(complete['last_checkpoint']),self.root/'logs/cocogen_burger_train.exit',
               self.policy,Path(complete['early_stop_receipt'])]
        for path in paths:
            original=path.read_bytes()
            path.write_bytes(b'0\n' if path.suffix=='.exit' else original+b' ')
            with self.assertRaises(ValueError):
                stop.verify_completion(self.root)
            path.write_bytes(original)

    def test_write_new_never_overwrites(self):
        path=self.root/'exclusive.json'
        stop.write_new(path,dict(first=True))
        with self.assertRaises(FileExistsError):
            stop.write_new(path,dict(second=True))
        self.assertEqual(stop.read(path),dict(first=True))

    def test_watch_waits_for_eligibility_and_applies_once(self):
        states=[dict(decision=dict(epochs_completed=99,stale_checks=5,eligible=False),
                     processes_alive=dict(launcher=True)),
                dict(decision=dict(epochs_completed=100,stale_checks=6,eligible=True),
                     processes_alive=dict(launcher=True))]
        with patch.object(stop,'inspect',side_effect=states), patch.object(stop.time,'sleep') as sleep, \
             patch.object(stop,'apply',return_value={'handed_off':True}) as apply:
            self.assertEqual(stop.watch(self.root,self.policy,self.binding_path),{'handed_off':True})
        sleep.assert_called_once_with(30)
        apply.assert_called_once_with(self.root,self.policy,self.binding_path)

    def test_completed_stop_metadata_survives_relocation_without_original(self):
        from .archive_paths import mapped_study_reads
        from .terminal_restore import verify_terminal
        complete=self.finalized()
        with tempfile.TemporaryDirectory() as temporary:
            moved=Path(temporary)/'archive'
            shutil.copytree(self.root,moved)
            self.root.rename(Path(temporary)/'inaccessible_original')
            with mapped_study_reads(moved,self.root) as mapping:
                self.assertEqual(verify_terminal(self.root),complete)
                self.assertEqual(mapping['direct_original_access_attempts'],[])
                self.assertGreater(mapping['mapped_opens'],100)
            selected=moved/Path(complete['last_checkpoint']).relative_to(self.root)
            selected.unlink()
            with mapped_study_reads(moved,self.root), self.assertRaises(FileNotFoundError):
                verify_terminal(self.root)

    def test_legacy_resume_cannot_restart_authorized_completed_run(self):
        from .resume_archive import resume
        self.finalized()
        with self.assertRaisesRegex(ValueError,'intentionally stopped'):
            resume(self.root,self.root.parent/'old_study')

    def test_bound_signal_targets_only_own_child(self):
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
        try:
            identity=stop.process_identity(child.pid)
            wrong={**identity,'start_ticks':identity['start_ticks']+1}
            with self.assertRaises(ValueError):
                stop.send_bound_term(wrong)
            self.assertIsNone(child.poll())
            stop.send_bound_term(identity)
            self.assertEqual(child.wait(timeout=5),-15)
        finally:
            if child.poll() is None:
                child.terminate(); child.wait(timeout=5)


class EvaluationPhase(unittest.TestCase):
    def test_only_evaluation_jobs_and_gpu_mask_reset(self):
        try:
            from . import pipeline
        except ModuleNotFoundError as error:
            if error.name in {'torch','numpy','yaml'}:
                self.skipTest('Full dependency environment required')
            raise
        with tempfile.TemporaryDirectory() as temporary:
            study=Path(temporary)
            (study/'training/burger').mkdir(parents=True)
            (study/'protocol/selected').mkdir(parents=True)
            trained=dict(status='trained',reason='user_authorized_validation_plateau',epochs_completed=100,
                         best_checkpoint='best',last_checkpoint='last',best_sha256='x',last_sha256='x')
            (study/'training/burger/complete.json').write_text(json.dumps(trained))
            (study/'protocol/selected/burger.json').write_text(json.dumps(dict(status='validated',estimated_main_gpu_hours=1)))
            with patch.object(stop,'verify_completion',return_value=trained) as verified, \
                 patch.object(pipeline,'sha_file',return_value='x'), \
                 patch.object(pipeline,'free_memory_gate',return_value={0:80000,1:80000}), \
                 patch.object(pipeline,'launch') as launched, patch.object(pipeline,'wait_job'):
                pipeline.evaluate_trained(study)
            verified.assert_called_once_with(study)
            self.assertEqual([c.args[2] for c in launched.call_args_list],
                  ['cocogen_burger_calibrate','cocogen_burger_main0','cocogen_burger_main1','cocogen_burger_finish'])
            self.assertTrue(all(c.kwargs['clear_cuda_mask'] for c in launched.call_args_list[:3]))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,help='Accepted for the remote job wrapper; fixtures use private temporary files')
    parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromNames([
        'cocogen_burger.validate_early_stop_policy','cocogen_burger.validate_external_stop'])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(tests=result.testsRun,skipped=len(result.skipped),
                          failures=len(result.failures),errors=len(result.errors),
                          status='passed' if result.wasSuccessful() else 'failed')),flush=True)
    sys.exit(0 if result.wasSuccessful() else 1)
