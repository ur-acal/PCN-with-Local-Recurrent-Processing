#!/bin/bash -l
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --nodelist=bhgrb4x0081,bhgrb4x0082
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/tc_ft_gap_%j.out

# Diagnose the state-1/state-2 CIFAR-10 TC fine-tuning performance gap while
# reusing the production launcher's own finetune_one_combo_model function.
set -Eeuo pipefail

REPO=/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing
cd "$REPO"
mkdir -p logs/slurm_jobs results/tc_ft_slurm_gap

TC_STATE="${TC_STATE:-1}"
FT_TIMM_AUG_LEVEL="${FT_TIMM_AUG_LEVEL:-none}"
DIAG_TAG="${DIAG_TAG:-$(date +%Y%m%d_%H%M%S)}"
case "$TC_STATE" in
  1) DIAG_BLOCK=ODEXInitFFFB ;;
  2) DIAG_BLOCK=S2NoisyIYAsXZAs0 ;;
  *) echo "TC_STATE must be 1 or 2" >&2; exit 2 ;;
esac
case "$FT_TIMM_AUG_LEVEL" in
  none|no_aug) ;;
  *) echo "FT_TIMM_AUG_LEVEL must be none or no_aug" >&2; exit 2 ;;
esac

run_name="tc_rgb_cifar10_state${TC_STATE}_pcn_22L6l7l6_ft_optimized_${FT_TIMM_AUG_LEVEL}"
pretrain_root="./saved_ckpt_runs/tc_rgb_cifar10_state${TC_STATE}_pcn_resnet_depth_study"
DIAG_ROOT="$REPO/results/tc_ft_slurm_gap/${DIAG_TAG}_job${SLURM_JOB_ID}_state${TC_STATE}_${FT_TIMM_AUG_LEVEL}"
mkdir -p "$DIAG_ROOT" "$DIAG_ROOT/timing" "$DIAG_ROOT/profile" "$DIAG_ROOT/system"

# These are the exact n06/n17/n26 architecture arrays selected by the failed
# 22-layer pipeline task (COMB_SEL_SET=2).
INP_STR="3 16 16 16 16 16 16 16 32 32 32 32 32 32 32 32 64 64 64 64 64 64"
OUT_STR="16 16 16 16 16 16 16 32 32 32 32 32 32 32 32 64 64 64 64 64 64 64"
POOL_STR="0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0"
COMB_TAG=N22_C16_n06_n17_n26
COMB_LIST="${COMB_TAG}"$'\t'"diagnostic"$'\t'"${INP_STR}"$'\t'"${OUT_STR}"$'\t'"${POOL_STR}"

export mode=ft_only
export TC_NONIDEALITIES=true
export TC_STATE TOGGLE_MODE=none SWITCH_INF=false
export TASK=cifar10 IMG_TYPE=rgb
export EXP_PREFIX="$run_name"
export PRETRAIN_SAVE_PATH="$pretrain_root"
export FT_OUTPUT_SAVE_PATH="$DIAG_ROOT/no_checkpoint_output"
export OUTPUT_SAVE_PATH="$DIAG_ROOT/no_checkpoint_output"
export TEACHER_CKPT=./checkpoint/efficientnet_v2_l_cifar10_rgb_OldNoTimm_MatchDistill.pth
export TEACHER_ARCH=efficientnet_v2_l
export TEACHER_ARCH_SOURCE=torchvision
export DISTILL_METHOD=srrl
export TIMM_RE_PROB=0.0
export FT_TIMM_AUG_LEVEL
export FT_LEARNING_RATE=0.005 FT_NUM_EPOCHS=140 WARMUP_FT=0
export FINAL_EVAL_ONLY=true VALIDATION_MODE=false
export PCN_CHAN_0_LIST=16 PCN_NUM_LAYERS_LIST=22
export NUM_COMB_PER_NUM_LAYER=3 COMB_SEL_SET=2 MAX_TASKS_PER_GPU=1
export REUSE_ACCEPTED_STEP_TRAINING=true CHECKPOINT_ODE_RHS_TRAINING=true
export ODE_BLOCK="$DIAG_BLOCK" BLOCKS_LIST="$DIAG_BLOCK"
export COMB_LIST

# Source only the setup and function definitions from the real production
# worker.  The phase marker prevents any train/eval phase from launching.
set +u
source <(sed '/^# PHASE 1: TRAIN/,$d' launch_scripts/run_kdcrd_then_ft.sbatch)
set -u
REAL_PYTHON="$(command -v python)"
LOGDIR="$DIAG_ROOT/production_logs"
mkdir -p "$LOGDIR"

# The production prefix installs a process-group trap; replace it with a local
# sampler cleanup so an interrupted diagnostic does not kill its Slurm shell.
GPU_SAMPLER_PID=""
cleanup() {
  if [[ -n "$GPU_SAMPLER_PID" ]]; then
    kill "$GPU_SAMPLER_PID" 2>/dev/null || true
    wait "$GPU_SAMPLER_PID" 2>/dev/null || true
  fi
}
finalize() {
  status=$?
  trap - EXIT
  cleanup
  printf '%s\n' "$status" > "$DIAG_ROOT/system/worker_exit_status.txt"
  tar -C "$(dirname "$DIAG_ROOT")" -czf "${DIAG_ROOT}.tar.gz" "$(basename "$DIAG_ROOT")" || true
  echo "DIAGNOSTIC_DIRECTORY=$DIAG_ROOT"
  echo "DIAGNOSTIC_ARCHIVE=${DIAG_ROOT}.tar.gz"
  echo "SLURM_STDOUT=$REPO/logs/slurm_jobs/tc_ft_gap_${SLURM_JOB_ID}.out"
  exit "$status"
}
trap finalize EXIT
trap 'exit 130' TERM INT

{
  echo "date=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "pwd=$PWD"
  echo "python=$REAL_PYTHON"
  echo "job_id=$SLURM_JOB_ID"
  echo "state=$TC_STATE"
  echo "augmentation=$FT_TIMM_AUG_LEVEL"
  echo "block=$DIAG_BLOCK"
  echo "pretrain_root=$pretrain_root"
  echo "diagnostic_root=$DIAG_ROOT"
  env | sort | grep -E '^(SLURM|CUDA|NVIDIA|CONDA|OMP|MKL|OPENBLAS|PYTORCH|TORCH|TRITON|TC_|REUSE_|CHECKPOINT_)' || true
} > "$DIAG_ROOT/system/launch_environment.txt"

git rev-parse HEAD > "$DIAG_ROOT/system/git_head.txt"
git branch --show-current > "$DIAG_ROOT/system/git_branch.txt"
git status --short > "$DIAG_ROOT/system/git_status.txt"
git diff --stat > "$DIAG_ROOT/system/git_diff_stat.txt"
git diff --binary > "$DIAG_ROOT/system/git_diff.patch"
git diff --cached --binary > "$DIAG_ROOT/system/git_diff_cached.patch"
git log -1 --decorate --stat > "$DIAG_ROOT/system/git_last_commit.txt"
sha256sum train_ode_cifar.py trainer_timm.py ode_pc.py tc_shared_correction.py \
  TorchDiffEqPack/odesolver/ode_solver.py \
  TorchDiffEqPack/odesolver/adaptive_grid_solver.py \
  TorchDiffEqPack/odesolver/base.py \
  launch_scripts/run_kdcrd_then_ft.sbatch \
  diagnostic_scripts/profile_tc_ft_slurm_gap.py \
  launch_scripts/run_tc_ft_slurm_diagnostic.sbatch \
  > "$DIAG_ROOT/system/code_sha256.txt"
mkdir -p "$DIAG_ROOT/system/diagnostic_sources"
cp diagnostic_scripts/profile_tc_ft_slurm_gap.py \
  launch_scripts/run_tc_ft_slurm_diagnostic.sbatch \
  launch_scripts/submit_tc_ft_slurm_diagnostic.sh \
  "$DIAG_ROOT/system/diagnostic_sources/"
lscpu > "$DIAG_ROOT/system/lscpu.txt" 2>&1 || true
taskset -pc $$ > "$DIAG_ROOT/system/taskset.txt" 2>&1 || true
numactl --show > "$DIAG_ROOT/system/numactl.txt" 2>&1 || true
nvidia-smi -q > "$DIAG_ROOT/system/nvidia_smi_q.txt" 2>&1 || true
nvidia-smi topo -m > "$DIAG_ROOT/system/nvidia_smi_topology.txt" 2>&1 || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv > "$DIAG_ROOT/system/gpu_processes_at_start.csv" 2>&1 || true
scontrol show job "$SLURM_JOB_ID" -dd > "$DIAG_ROOT/system/slurm_job.txt" 2>&1 || true
scontrol show node "$(hostname)" > "$DIAG_ROOT/system/slurm_node.txt" 2>&1 || true
squeue --nodes="$(hostname)" -o '%.18i %.9P %.24j %.8u %.2t %.10M %.6D %R' \
  > "$DIAG_ROOT/system/slurm_node_jobs.txt" 2>&1 || true
ps -eo pid,ppid,psr,pcpu,pmem,stat,comm,args --sort=-pcpu \
  > "$DIAG_ROOT/system/processes_at_start.txt" 2>&1 || true
"$REAL_PYTHON" -m pip freeze > "$DIAG_ROOT/system/pip_freeze.txt" 2>&1 || true

nvidia-smi --query-gpu=timestamp,index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu \
  --format=csv -l 1 > "$DIAG_ROOT/system/gpu_samples.csv" 2>&1 &
GPU_SAMPLER_PID=$!

selection=last
MODEL_NAME="$($REAL_PYTHON launch_scripts/find_tc_pretrain.py \
  --root "$pretrain_root" --inp "$INP_STR" --out "$OUT_STR" --pool "$POOL_STR" \
  --stride "${STRIDE[*]}" --kernel "${KERNEL[*]}" \
  --task "$TASK" --img_type "$IMG_TYPE" --block "$FT_ODE_BLOCK" --pcn "$PCN" \
  --bits "$INPUT_QUANT_BITS" --center "$CENTER_STUDENT_INPUT" --selection "$selection" \
  2> >(tee "$DIAG_ROOT/system/checkpoint_discovery_stderr.txt" >&2))"
SOURCE_CKPT="$pretrain_root/$MODEL_NAME/${MODEL_NAME}_${selection}_ckpt.pth"
printf '%s\n' "$MODEL_NAME" > "$DIAG_ROOT/system/model_name.txt"
printf '%s\n' "$SOURCE_CKPT" > "$DIAG_ROOT/system/source_checkpoint.txt"
sha256sum "$SOURCE_CKPT" "$TEACHER_CKPT" > "$DIAG_ROOT/system/checkpoint_sha256.txt"
stat "$SOURCE_CKPT" "$TEACHER_CKPT" > "$DIAG_ROOT/system/checkpoint_stat.txt"

# Intercept only the production train_ode_cifar.py process.  All arguments
# remain exactly those assembled by finetune_one_combo_model.
python() {
  if [[ "${1:-}" == -u && "${2:-}" == train_ode_cifar.py ]]; then
    shift 2
    "$REAL_PYTHON" -u diagnostic_scripts/profile_tc_ft_slurm_gap.py "$@"
  else
    "$REAL_PYTHON" "$@"
  fi
}

export TC_FT_DIAG_MODE=timing
export TC_FT_DIAG_BATCHES="${TC_FT_DIAG_TIMING_BATCHES:-20}"
export TC_FT_DIAG_WARMUP_BATCHES="${TC_FT_DIAG_WARMUP_BATCHES:-2}"
export TC_FT_DIAG_OUTPUT="$DIAG_ROOT/timing"
finetune_one_combo_model "${COMB_TAG}_timing" "$MODEL_NAME" "$FT_ODE_BLOCK"

export TC_FT_DIAG_MODE=profile
export TC_FT_DIAG_BATCHES="${TC_FT_DIAG_PROFILE_BATCHES:-1}"
export TC_FT_DIAG_WARMUP_BATCHES=0
export TC_FT_DIAG_OUTPUT="$DIAG_ROOT/profile"
finetune_one_combo_model "${COMB_TAG}_profile" "$MODEL_NAME" "$FT_ODE_BLOCK"

cleanup
GPU_SAMPLER_PID=""
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv > "$DIAG_ROOT/system/gpu_processes_at_end.csv" 2>&1 || true
squeue --nodes="$(hostname)" -o '%.18i %.9P %.24j %.8u %.2t %.10M %.6D %R' \
  > "$DIAG_ROOT/system/slurm_node_jobs_at_end.txt" 2>&1 || true
ps -eo pid,ppid,psr,pcpu,pmem,stat,comm,args --sort=-pcpu \
  > "$DIAG_ROOT/system/processes_at_end.txt" 2>&1 || true
sstat -j "${SLURM_JOB_ID}.batch" \
  --format=JobID,AveCPU,AveCPUFreq,AveRSS,MaxRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite \
  > "$DIAG_ROOT/system/slurm_step_stats.txt" 2>&1 || true
sacct -j "$SLURM_JOB_ID" \
  --format=JobID,JobName,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode,NodeList \
  > "$DIAG_ROOT/system/slurm_accounting.txt" 2>&1 || true
