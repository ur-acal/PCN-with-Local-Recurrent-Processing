#!/bin/bash -l
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --nodelist=bhgrb4x0081,bhgrb4x0082
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

# Diagnose the state-1/state-2 CIFAR-10 TC fine-tuning performance gap while
# reusing the production launcher's own finetune_one_combo_model function.
set -o pipefail

REPO=/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing
cd "$REPO"
mkdir -p logs/slurm_jobs results/tc_ft_slurm_gap

TC_STATE="${TC_STATE:-1}"
FT_TIMM_AUG_LEVEL="${FT_TIMM_AUG_LEVEL:-none}"
DIAG_TAG="${DIAG_TAG:-$(date +%Y%m%d_%H%M%S)}"
TC_FT_DIAG_PLACEMENTS="${TC_FT_DIAG_PLACEMENTS:-default gpu_local gpu_remote}"
TC_FT_DIAG_TIMING_REPEATS="${TC_FT_DIAG_TIMING_REPEATS:-2}"
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
mkdir -p "$DIAG_ROOT" "$DIAG_ROOT/placements" "$DIAG_ROOT/system"

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

# Resolve the assigned GPU's physical NUMA node.  Binding the diagnostic Python
# process before it starts also binds its data-loader children and CUDA control
# thread, making this a production-realistic placement experiment.
GPU_PCI_BUS_ID="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader,nounits \
  | sed -n '1p' | tr -d '[:space:]')"
case "$GPU_PCI_BUS_ID" in
  00000000:*) GPU_SYSFS_ID="0000:${GPU_PCI_BUS_ID#00000000:}" ;;
  *) GPU_SYSFS_ID="$GPU_PCI_BUS_ID" ;;
esac
GPU_NUMA_FILE="/sys/bus/pci/devices/${GPU_SYSFS_ID}/numa_node"
[[ -r "$GPU_NUMA_FILE" ]] || {
  echo "Cannot determine GPU NUMA node from $GPU_NUMA_FILE" >&2
  exit 2
}
read -r GPU_NUMA_NODE < "$GPU_NUMA_FILE"
[[ "$GPU_NUMA_NODE" =~ ^[0-9]+$ ]] || {
  echo "GPU has no usable NUMA node: $GPU_NUMA_NODE" >&2
  exit 2
}
GPU_REMOTE_NUMA_NODE=""
for node_path in /sys/devices/system/node/node[0-9]*; do
  node_id="${node_path##*node}"
  if [[ "$node_id" != "$GPU_NUMA_NODE" ]]; then
    GPU_REMOTE_NUMA_NODE="$node_id"
    break
  fi
done
read -r -a DIAG_PLACEMENT_LIST <<< "$TC_FT_DIAG_PLACEMENTS"
for placement in "${DIAG_PLACEMENT_LIST[@]}"; do
  case "$placement" in
    default|gpu_local) ;;
    gpu_remote)
      [[ -n "$GPU_REMOTE_NUMA_NODE" ]] || {
        echo "gpu_remote requested on a single-NUMA host" >&2
        exit 2
      }
      ;;
    *) echo "Unknown diagnostic placement: $placement" >&2; exit 2 ;;
  esac
done
[[ "$TC_FT_DIAG_TIMING_REPEATS" =~ ^[1-9][0-9]*$ ]] || {
  echo "TC_FT_DIAG_TIMING_REPEATS must be a positive integer" >&2
  exit 2
}

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
  echo "SLURM_STDOUT=$REPO/logs/slurm_jobs/slurm_${SLURM_JOB_ID}.out"
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
  echo "gpu_pci_bus_id=$GPU_PCI_BUS_ID"
  echo "gpu_sysfs_id=$GPU_SYSFS_ID"
  echo "gpu_numa_node=$GPU_NUMA_NODE"
  echo "gpu_remote_numa_node=$GPU_REMOTE_NUMA_NODE"
  echo "diagnostic_placements=$TC_FT_DIAG_PLACEMENTS"
  echo "diagnostic_timing_repeats=$TC_FT_DIAG_TIMING_REPEATS"
  echo "pretrain_root=$pretrain_root"
  echo "diagnostic_root=$DIAG_ROOT"
  env | sort | grep -E '^(SLURM|CUDA|NVIDIA|CONDA|OMP|MKL|OPENBLAS|PYTORCH|TORCH|TRITON|TC_|REUSE_|CHECKPOINT_)' || true
} > "$DIAG_ROOT/system/launch_environment.txt"

if command -v git >/dev/null 2>&1; then
  git rev-parse HEAD > "$DIAG_ROOT/system/git_head.txt" 2>&1 || true
  git branch --show-current > "$DIAG_ROOT/system/git_branch.txt" 2>&1 || true
  git status --short > "$DIAG_ROOT/system/git_status.txt" 2>&1 || true
  git diff --stat > "$DIAG_ROOT/system/git_diff_stat.txt" 2>&1 || true
  git diff --binary > "$DIAG_ROOT/system/git_diff.patch" 2>&1 || true
  git diff --cached --binary > "$DIAG_ROOT/system/git_diff_cached.patch" 2>&1 || true
  git log -1 --decorate --stat > "$DIAG_ROOT/system/git_last_commit.txt" 2>&1 || true
else
  printf '%s\n' 'git is unavailable on this compute node; use code_sha256.txt.' \
    > "$DIAG_ROOT/system/git_unavailable.txt"
fi
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
numactl --hardware > "$DIAG_ROOT/system/numactl_hardware.txt" 2>&1 || true
cp "$GPU_NUMA_FILE" "$DIAG_ROOT/system/gpu_numa_node.txt"
if [[ -r "/sys/bus/pci/devices/${GPU_SYSFS_ID}/local_cpulist" ]]; then
  cp "/sys/bus/pci/devices/${GPU_SYSFS_ID}/local_cpulist" \
    "$DIAG_ROOT/system/gpu_local_cpulist.txt"
fi
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
  2> >(tee "$DIAG_ROOT/system/checkpoint_discovery_stderr.txt" >&2))" || exit 2
SOURCE_CKPT="$pretrain_root/$MODEL_NAME/${MODEL_NAME}_${selection}_ckpt.pth"
[[ -s "$SOURCE_CKPT" ]] || { echo "Missing source checkpoint: $SOURCE_CKPT" >&2; exit 2; }
[[ -s "$TEACHER_CKPT" ]] || { echo "Missing teacher checkpoint: $TEACHER_CKPT" >&2; exit 2; }
printf '%s\n' "$MODEL_NAME" > "$DIAG_ROOT/system/model_name.txt"
printf '%s\n' "$SOURCE_CKPT" > "$DIAG_ROOT/system/source_checkpoint.txt"
sha256sum "$SOURCE_CKPT" "$TEACHER_CKPT" > "$DIAG_ROOT/system/checkpoint_sha256.txt"
stat "$SOURCE_CKPT" "$TEACHER_CKPT" > "$DIAG_ROOT/system/checkpoint_stat.txt"

# Intercept only the production train_ode_cifar.py process.  All training
# arguments remain exactly those assembled by finetune_one_combo_model.  The
# placement prefix is the sole experimental change.
run_diagnostic_python() {
  local placement="$1"
  shift
  case "$placement" in
    default)
      "$REAL_PYTHON" "$@"
      ;;
    gpu_local)
      numactl --cpunodebind="$GPU_NUMA_NODE" --membind="$GPU_NUMA_NODE" \
        "$REAL_PYTHON" "$@"
      ;;
    gpu_remote)
      numactl --cpunodebind="$GPU_REMOTE_NUMA_NODE" \
        --membind="$GPU_REMOTE_NUMA_NODE" "$REAL_PYTHON" "$@"
      ;;
  esac
}

python() {
  if [[ "${1:-}" == -u && "${2:-}" == train_ode_cifar.py ]]; then
    shift 2
    run_diagnostic_python "${TC_FT_DIAG_PLACEMENT:-default}" \
      -u diagnostic_scripts/profile_tc_ft_slurm_gap.py "$@"
  else
    "$REAL_PYTHON" "$@"
  fi
}

for (( repeat=1; repeat<=TC_FT_DIAG_TIMING_REPEATS; repeat++ )); do
  for placement in "${DIAG_PLACEMENT_LIST[@]}"; do
    export TC_FT_DIAG_PLACEMENT="$placement"
    export TC_FT_DIAG_RUN_LABEL="timing_${placement}_repeat${repeat}"
    export TC_FT_DIAG_MODE=timing
    export TC_FT_DIAG_BATCHES="${TC_FT_DIAG_TIMING_BATCHES:-20}"
    export TC_FT_DIAG_WARMUP_BATCHES="${TC_FT_DIAG_WARMUP_BATCHES:-2}"
    export TC_FT_DIAG_OUTPUT="$DIAG_ROOT/placements/$placement/timing_repeat${repeat}"
    finetune_one_combo_model \
      "${COMB_TAG}_${placement}_timing_repeat${repeat}" \
      "$MODEL_NAME" "$FT_ODE_BLOCK" || exit $?
  done
done

for placement in "${DIAG_PLACEMENT_LIST[@]}"; do
  export TC_FT_DIAG_PLACEMENT="$placement"
  export TC_FT_DIAG_RUN_LABEL="profile_${placement}"
  export TC_FT_DIAG_MODE=profile
  export TC_FT_DIAG_BATCHES="${TC_FT_DIAG_PROFILE_BATCHES:-1}"
  export TC_FT_DIAG_WARMUP_BATCHES=0
  export TC_FT_DIAG_OUTPUT="$DIAG_ROOT/placements/$placement/profile"
  finetune_one_combo_model "${COMB_TAG}_${placement}_profile" \
    "$MODEL_NAME" "$FT_ODE_BLOCK" || exit $?
done

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
