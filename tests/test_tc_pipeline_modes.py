"""TC mode discovery/routing tests: CPU checkpoints and mocked stage commands."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from pc_conv import PCConvReLU6
from launch_scripts.find_tc_pretrain import find_pretrain
from test_tc_pipeline import TCPipelineTests, ROOT


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'run_config.json').write_text(json.dumps(dict(input_quant_bits=None, center_student_input=False)))
        self.name = 'TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_1REP'
        self.request = dict(inp=[3,16], out=[16,32], pool=[1,0], task='cifar10', img_type='rgb', block='ODEXInitFFFB')
        self.checkpoint = dict(epoch=300, acc=.96, net_type='PCNetNoBatchNorm',
            init_args=dict(model_args=dict(inp_channels=[3,16], out_channels=[16,32], max_pool=[1,0],
                num_classes=10, stride=[1,1], kernel_size=[3,3], avg_pooling=True, first_bn=False,
                pc_conv_layer=PCConvReLU6), kwargs=dict(bias=False, tie_weights=False, tie_bp=False, bypass=False)),
            net={'linear.weight':torch.zeros(10,32), 'PcConvs.0.FFconv.weight':torch.zeros(16,3,3,3),
                 'PcConvs.1.FFconv.weight':torch.zeros(32,16,3,3)})
        self.path = self.write(self.name)

    def write(self, name, checkpoint=None):
        folder=self.root/name; folder.mkdir(exist_ok=True)
        path=folder/f'{name}_last_ckpt.pth'
        torch.save(checkpoint or self.checkpoint,path)
        return path

    def test_match_and_rejections(self):
        self.assertEqual(find_pretrain(self.root, **self.request), self.path)
        for change in (dict(task='cifar100'),dict(img_type='CiFAIR'),dict(block='S2NoisyIYAsXZAs0'),
                       dict(pool=[0,1]),dict(out=[16,64]),dict(bits=12),dict(center=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                find_pretrain(self.root, **dict(self.request, **change))
        other=self.root/'wrong_experiment';other.mkdir()
        (other/'run_config.json').write_text((self.root/'run_config.json').read_text())
        with self.assertRaises(ValueError): find_pretrain(other, **self.request)
        d=copy.deepcopy(self.checkpoint);d['epoch']=299;self.write(self.name,d)
        with self.assertRaises(ValueError): find_pretrain(self.root, **self.request)
        self.write(self.name)
        self.write(self.name.replace('1REP','2REP'))
        with self.assertRaisesRegex(ValueError,'found 2'): find_pretrain(self.root, **self.request)

    def test_toggle_pretrain_discovery(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint['init_args']['model_args']['inp_channels'] = [4, 16]
        checkpoint['net']['PcConvs.0.FFconv.weight'] = torch.zeros(16, 4, 3, 3)
        name = self.name.replace('_ODEXInitFFFB_', '_ToggleODEXInitFFFB_').replace(
            '_1REP', '_CiFAIR_1REP')
        path = self.write(name, checkpoint)
        request = dict(self.request, inp=[4, 16], img_type='CiFAIR',
                       block='ToggleODEXInitFFFB')
        self.assertEqual(find_pretrain(self.root, **request), path)

    def test_toggle_ft_and_eval_routes_from_verified_checkpoint(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint['init_args']['model_args']['inp_channels'] = [4, 16]
        checkpoint['net']['PcConvs.0.FFconv.weight'] = torch.zeros(16, 4, 3, 3)
        name = self.name.replace('_ODEXInitFFFB_', '_ToggleODEXInitFFFB_').replace(
            '_1REP', '_CiFAIR_1REP')
        last = self.write(name, checkpoint)
        torch.save(checkpoint, last.with_name(f'{name}_best_ckpt.pth'))

        worker = (ROOT/'launch_scripts/run_kdcrd_then_ft.sbatch').read_text()
        guard = worker[worker.index('mode="${mode:-default}"'):worker.index('# ---- Conda activation')]
        guard_result = subprocess.run(['bash', '-c', guard], cwd=ROOT,
            env=dict(os.environ, mode='ft_and_eval', TOGGLE_MODE='odexinit',
                     SWITCH_INF='false', TC_NONIDEALITIES='false'),
            text=True, capture_output=True)
        self.assertEqual(guard_result.returncode, 0, guard_result.stderr)

        phases = worker[worker.index('echo "==== PHASE 1:'):worker.index('# Merge summaries')]
        preamble = '''
set -o pipefail
COMBS=($'combo\\t0\\t4 16\\t16 32\\t1 0')
STRIDE=(1); KERNEL=(3)
run_one_combo() { echo CALL:pretrain; return 99; }
extract_model_name() { sed -n 's/.*Model Name: \\(.*\\) -----.*/\\1/p' "$1"; }
finetune_one_combo_model() { echo "CALL:ft:$2"; mkdir -p "$LOGDIR/$1"; echo 'Train finished, Model Name: fixture_ft -----' > "$LOGDIR/$1/finetune_${FT_ODE_BLOCK}.log"; }
eval_one_combo_model() { echo "CALL:eval:$2"; }
'''
        env = dict(os.environ, PATH=str(Path(sys.executable).parent)+':'+os.environ['PATH'],
            mode='ft_and_eval', OUTPUT_SAVE_PATH=str(self.root), PRETRAIN_SAVE_PATH=str(self.root),
            FT_OUTPUT_SAVE_PATH=str(self.root), LOGDIR=str(self.root/'toggle_logs'),
            TRAIN_ODE_BLOCK='ToggleODEXInitFFFB', FT_ODE_BLOCK='ToggleODEXInitFFFB',
            INF_ODE_BLOCK='TogglePulseODEXInitFFFB', FINAL_EVAL_ONLY='false',
            TASK='cifar10', IMG_TYPE='CiFAIR', PCN='PCNetNoBatchNorm',
            INPUT_QUANT_BITS='none', CENTER_STUDENT_INPUT='false')
        result = subprocess.run(['bash', '-c', preamble+phases], cwd=ROOT,
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
        self.assertNotIn('CALL:pretrain', result.stdout)
        self.assertIn('CALL:ft:'+name, result.stdout)
        self.assertIn('CALL:eval:fixture_ft', result.stdout)

    def test_real_pcn_phase_routing(self):
        worker=(ROOT/'launch_scripts/run_kdcrd_then_ft.sbatch').read_text()
        phases=worker[worker.index('echo "==== PHASE 1:'):worker.index('# Merge summaries')]
        preamble='''
set -o pipefail
COMBS=($'combo\\t0\\t3 16\\t16 32\\t1 0')
STRIDE=(1); KERNEL=(3)
run_one_combo() { mkdir -p "$LOGDIR/combo"; echo "Train finished, Model Name: $FIXTURE -----" > "$LOGDIR/combo/train_${TRAIN_ODE_BLOCK}.log"; echo CALL:pretrain; }
extract_model_name() { sed -n 's/.*Model Name: \\(.*\\) -----.*/\\1/p' "$1"; }
finetune_one_combo_model() { echo "CALL:ft:$2"; [[ "${FAIL_FT:-false}" != true ]] || return 9; mkdir -p "$LOGDIR/$1"; echo 'Train finished, Model Name: fixture_ft -----' > "$LOGDIR/$1/finetune_${FT_ODE_BLOCK}.log"; }
eval_one_combo_model() { echo "CALL:eval:$2"; }
'''
        env=dict(os.environ, PATH=str(Path(sys.executable).parent)+':'+os.environ['PATH'],
            OUTPUT_SAVE_PATH=str(self.root), PRETRAIN_SAVE_PATH=str(self.root),
            FT_OUTPUT_SAVE_PATH=str(self.root), LOGDIR=str(self.root/'logs'), FIXTURE=self.name,
            EXP='test',SLURM_JOB_ID='fixture',TRAIN_ODE_BLOCK='ODEXInitFFFB',FT_ODE_BLOCK='ODEXInitFFFB',
            INF_ODE_BLOCK='ODEXInitFFFB',FINAL_EVAL_ONLY='true',TASK='cifar10',IMG_TYPE='rgb',
            PCN='PCNetNoBatchNorm',INPUT_QUANT_BITS='none',CENTER_STUDENT_INPUT='false')
        expected={'default':['pretrain','ft','eval'],'pretrain_only':['pretrain'],
                  'ft_only':['ft'],'ft_and_eval':['ft','eval']}
        for mode, stages in expected.items():
            result=subprocess.run(['bash','-c',preamble+phases],cwd=ROOT,env=dict(env,mode=mode),text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            calls=[line.split(':')[1] for line in result.stdout.splitlines() if line.startswith('CALL:')]
            self.assertEqual(calls,stages)
            if 'ft' in stages: self.assertIn('CALL:ft:'+self.name,result.stdout)
        result=subprocess.run(['bash','-c',preamble+phases],cwd=ROOT,
            env=dict(env,mode='ft_and_eval',FAIL_FT='true'),text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn('CALL:eval',result.stdout)


class CNNModeTests(unittest.TestCase):
    def test_modes_keep_identical_stage_arguments(self):
        harness=TCPipelineTests()
        with tempfile.TemporaryDirectory() as td:
            result, baseline=harness.run_pipeline(td)
            self.assertEqual(result.returncode,0,result.stderr)
            # Reuse fake Python/checkpoint fixtures, bypass only sbatch/conda.
            env=dict(os.environ, PATH=str(Path(td)/'bin')+':'+os.environ['PATH'],TC_FEEDFORWARD='true',
                REPO_ROOT=str(ROOT),MODEL_NAME='wrn_28_2_cifar_nobn_no_bias_avgpool',TASK='cifar100',IMG_TYPE='rgb',
                PRETRAIN_OUTPUT_DIR=td+'/pre',FT_OUTPUT_DIR=td+'/ft',RESULT_PATH=td+'/results',PIPELINE_TRACE=td+'/trace.jsonl')
            # Same submission defaults as the original captured run.
            shell='''conda() { :; }; export -f conda
sbatch() { bash "${@: -1}"; }
source launch_scripts/slurm_search_feedforward_config.sh
'''
            for mode, indices in [('pretrain_only',[0]),('ft_only',[1]),('ft_and_eval',[1,2])]:
                Path(env['PIPELINE_TRACE']).write_text('')
                result=subprocess.run(['bash','-c',shell],cwd=ROOT,env=dict(env,mode=mode),text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr+result.stdout)
                rows=[json.loads(line) for line in Path(env['PIPELINE_TRACE']).read_text().splitlines()]
                self.assertEqual([r['argv'] for r in rows],[baseline[i]['argv'] for i in indices])


if __name__=='__main__': unittest.main()
