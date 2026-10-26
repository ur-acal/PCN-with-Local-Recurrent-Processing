"""Bounded real TC FT/save/eval rehearsal, never a production accuracy run.

Waits for the specified existing pipeline to exit before using the GPU.
Reuses the opt-in rehearsal hooks: four training batches, batch four,
then stop after the first actual CSR expansion. No running process is modified.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT / 'results'):
        raise ValueError('Rehearsal artifacts must be isolated under results/.')
    output.mkdir(parents=True, exist_ok=False)
    identity = process_identity(args.wait_pid) if args.wait_pid else None
    if identity is not None:
        print(f'WAITING for existing pipeline PID {args.wait_pid}; no GPU allocation.', flush=True)
        while process_identity(args.wait_pid) == identity:
            time.sleep(20)
    # Do not compete with a subsequently started GPU job either.
    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                  '--format=csv,noheader'], text=True).strip():
        time.sleep(20)
    env = dict(os.environ,
        PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'],
        TC_DRY_RUN='true', TASK='cifar100', IMG_TYPE='rgb', CKPT='last',
        NORMALIZE_STUDENT_INPUT='false', CENTER_STUDENT_INPUT='false',
        FT_TIMM_AUG_LEVEL='rgb_mid', ACTIVATION_CORNER_MODE='random_per_forward',
        ACTIVATION_RANDOM_CURVE_SHARING='per_spin',
        TEACHER_CKPT='./checkpoint/efficientnet_v2_l_cifar100_rgb_OldNoTimm_MatchDistill_NoNorm.pth',
        TEACHER_ARCH='efficientnet_v2_l', TEACHER_ARCH_SOURCE='torchvision',
        NONLINEAR_R_CURVE_SAMPLING='empirical_with_replacement',
        TC_EMPIRICAL_CURVE_BANK=str(ROOT/'hardware_data/mc_45_corners/pseudo_res_PVT_scanned_vprog'),
        R_VAL='13.3e3', R_MAX='200e3',
        REUSE_ACCEPTED_STEP_TRAINING='true', CHECKPOINT_ODE_RHS_TRAINING='true',
        CHECKPOINT_ODE_RHS_PORTION='1', FUSE_MEASURED_ACTIVATION='true',
        EXPANDED_GAUSSIAN_CURVE_CPU_OFFLOAD='false',
        REHEARSAL_ONLY='true', REHEARSAL_BATCHES='4',
        PYTHONPATH=str(ROOT/'diagnostic_scripts/rehearsal_hooks'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TEST_BS='4', N_TRIALS='1')
    roots = (
        'tc_rgb_cifar100_state1_pcn_resnet_depth_study_student_teacher_no_norm',
        'tc_rgb_cifar100_state2_pcn_22L7l6l6_pipeline_student_teacher_no_norm_random_per_forward_per_spin_relu_rgb_mid')
    reports = []
    for state, parent in enumerate(roots, 1):
        candidates = [p for p in (ROOT/'saved_ckpt_runs'/parent).glob('*/*_last_ckpt.pth')
                      if not p.parent.name.startswith('TIMMQAT')]
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous/missing pretrained checkpoint: {parent}')
        stage_root = output/f'state{state}'
        env.update(TC_STATE=str(state), MODEL_NAME=candidates[0].parent.name,
                   MODEL_DIR=str(candidates[0].parent.parent), OUTPUT_DIR=str(stage_root),
                   REHEARSAL_ARTIFACT_ROOT=str(stage_root))
        def command(stage):
            printed = subprocess.check_output(['bash', 'launch_scripts/run_tc_nonidealities.sh', stage],
                                               cwd=ROOT, env=env, text=True)
            cmd = shlex.split(printed)
            cmd[0] = sys.executable
            return cmd
        cmd = command('ft') + ['--batch_size', '4', '--num_workers', '0']
        with (output/f'state{state}_ft.log').open('w') as log:
            status = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if status:
            raise RuntimeError(f'FT rehearsal failed, state {state}: {status}')
        checkpoints = list(stage_root.glob('*/*_last_ckpt.pth'))
        checkpoints = [p for p in checkpoints if not p.name.endswith('_full_param_last_ckpt.pth')]
        if len(checkpoints) != 1:
            raise RuntimeError(f'Expected one baked FT checkpoint: {checkpoints}')
        env.update(MODEL_NAME=checkpoints[0].parent.name, MODEL_DIR=str(stage_root))
        cmd = command('eval') + ['--expanded_w_dir', str(stage_root/'expanded'),
                                 '--hw_val_path', str(stage_root/'eval')]
        with (output/f'state{state}_eval.log').open('w') as log:
            status = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if status != 85:
            raise RuntimeError(f'Eval did not stop at first expansion, state {state}: {status}')
        reports.append(dict(state=state, batches=4, batch_size=4,
            checkpoint=str(checkpoints[0]), stopped_after_first_expansion=True))
        (output/'report.json').write_text(json.dumps(reports, indent=2)+'\n')
        print(f'PASS state {state}: FT, save, eval initialization, first expansion.', flush=True)


if __name__ == '__main__':
    main()
