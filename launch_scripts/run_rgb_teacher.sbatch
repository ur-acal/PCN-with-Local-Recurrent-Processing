#!/bin/bash -l
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 90:10:00

set -eo pipefail
# Same initialization as the working PCN worker; no module swap here.
if [[ "${TEACHER_DRY_RUN:-false}" != true ]]; then
  source activate base
  conda activate scanbase
fi
cd "${REPO_ROOT:-/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT:-/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing}/.cache/torch}"
export IMG_TYPE=rgb
export RGB_DATA_ROOT="${RGB_DATA_ROOT:-../data}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export INPUT_QUANT_BITS="" INIT_CHECKPOINT=""
export MATCH_DISTILL_AUG_ORDER=false USE_DIRECT_RESIZE_FOR_TIMM_AUGS=false
case "${RGB_TEACHER_RECIPE:-legacy}" in
  legacy)
    export USE_TIMM=false MATCH_DISTILL_PREPROCESS=true ARCH_SOURCE=torchvision
    recipe_label=OldNoTimm_MatchDistill
    ;;
  timm_oldaugs)
    export USE_TIMM=true USE_OLD_AUGS_FOR_TIMM=true TIMM_ARCH_SOURCE=torchvision
    export TIMM_PRETRAINED=true TIMM_EPOCHS=100 TIMM_LR=0.002 TIMM_WEIGHT_DECAY=1e-6
    export TIMM_WARMUP_EPOCHS=0 TIMM_OPT=sgd TIMM_SCHED=multistep
    export TIMM_LR_REDUCE_ON=10,20,30,40,50,60,70,80,90 TIMM_DECAY_RATE=0.1
    export TIMM_FIRST_EVAL_EPOCH=10 EVAL_EVERY=5 TIMM_BATCH_SIZE=32 TIMM_RE_PROB=0
    recipe_label=torchvision_oldaugs_legacyopt
    ;;
  *) echo 'Unknown RGB_TEACHER_RECIPE' >&2; exit 2 ;;
esac
export TEACHER_LOG_DIR="${TEACHER_LOG_DIR:-./logs/teacher_rgb}"
output="${TEACHER_OUTPUT_DIR:-./checkpoint}"
if [[ "${TEACHER_DRY_RUN:-false}" != true ]]; then
  mkdir -p "$TEACHER_LOG_DIR" "$output"
  # Validate local inputs before the long run.  Torchvision retains its normal
  # cache-then-download behavior for ImageNet initialization; TORCH_HOME points
  # at shared scratch storage so later jobs reuse the same download.
  python -u - <<'PY'
import os, sys
from pathlib import Path
from urllib.parse import urlparse
import torch, torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from train_teacher import _get_efficientnet_v2_weights
print('Python:', sys.executable, 'torch:', torch.__version__, 'torchvision:', torchvision.__version__, flush=True)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable in the worker environment.')
for dataset in (CIFAR10, CIFAR100):
    for train in (True, False):
        dataset(os.environ['RGB_DATA_ROOT'], train=train, download=False)
cached = Path(torch.hub.get_dir())/'checkpoints'/Path(urlparse(_get_efficientnet_v2_weights('efficientnet_v2_l').url).path).name
print('Validated both RGB datasets. ImageNet initialization cache:', cached,
      'present' if cached.is_file() else 'absent; torchvision will download it',
      flush=True)
PY
fi
for dataset in cifar10 cifar100; do
  export DATASET_NAME="$dataset"
  export TEACHER_CHECKPOINT="${output}/efficientnet_v2_l_${dataset}_rgb_${recipe_label}.pth"
  export TEACHER_LOG="${TEACHER_LOG_DIR}/${dataset}_rgb_${recipe_label}.log"
  echo "Teacher recipe=${RGB_TEACHER_RECIPE:-legacy} dataset=$dataset checkpoint=$TEACHER_CHECKPOINT"
  if [[ "${TEACHER_DRY_RUN:-false}" == true ]]; then
    # Print the actual entry-point argv; do not load data/models or run training.
    ( python() { printf '%q ' python "$@"; printf '\n'; }
      export -f python
      # The timm launcher uses tee; prevent diagnostic file writes.
      tee() { cat; }; export -f tee
      mkdir() { :; }; export -f mkdir
      bash ./launch_scripts/run_teacher.sh )
  else
    if [[ -e "$TEACHER_CHECKPOINT" ]]; then
      echo "Refusing to overwrite existing teacher: $TEACHER_CHECKPOINT" >&2
      exit 2
    fi
    if [[ "$USE_TIMM" == true ]]; then
      bash ./launch_scripts/run_teacher.sh
    else
      bash ./launch_scripts/run_teacher.sh 2>&1 | tee "$TEACHER_LOG"
    fi
    echo "Completed $dataset successfully."
  fi
done
