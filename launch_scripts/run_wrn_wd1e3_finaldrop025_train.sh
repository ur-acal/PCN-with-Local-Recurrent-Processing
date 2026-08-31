#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
DATA_DIR="${DATA_DIR:-$ROOT/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/wrn_wd1e3_finaldrop025}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoint/baselines_wd1e3_finaldrop025}"
REFERENCE_LOG_ROOT="${REFERENCE_LOG_ROOT:-$ROOT/logs/wrn16_2_wd1e3_pilot/train_logs}"
SEED="${SEED:-4096}"
ACCURACY_TOLERANCE="${ACCURACY_TOLERANCE:-0.005}"
SKIP_ACCURACY_GATE="${SKIP_ACCURACY_GATE:-false}"
OVERRIDE="num_epochs=300,eval_every=5,weight_decay=0.001,dropout_rate=0.0,final_dropout_rate=0.25"
CURRENT_STAGE="initializing"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing scanbase Python: $PYTHON_BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/train_logs" "$CHECKPOINT_ROOT"

write_state() {
  local status="$1" stage="$2" detail="${3:-}"
  printf '{\n  "status": "%s",\n  "stage": "%s",\n  "detail": "%s",\n  "controller_pid": %d,\n  "updated_epoch_seconds": %d\n}\n' \
    "$status" "$stage" "$detail" "$$" "$(date +%s)" >"$OUTPUT_ROOT/state.json.tmp"
  mv "$OUTPUT_ROOT/state.json.tmp" "$OUTPUT_ROOT/state.json"
}

on_error() {
  local code=$?
  write_state failed "$CURRENT_STAGE" "exit_code_$code"
  exit "$code"
}
trap on_error ERR

model_name() {
  local architecture="$1"
  printf 'wrn_%s_cifar\n' "${architecture#WRN_}" | tr '[:upper:]' '[:lower:]'
}

checkpoint_path() {
  local dataset="$1" architecture="$2" model run
  model="$(model_name "$architecture")"
  run="custom_noresize_${dataset}_${model}"
  printf '%s/%s/custom_noresize/%s/%s/%s_best_ckpt.pth\n' \
    "$CHECKPOINT_ROOT" "$dataset" "$model" "$run" "$run"
}

training_complete() {
  local dataset="$1" architecture="$2" log ckpt
  log="$OUTPUT_ROOT/train_logs/${dataset}_${architecture}.log"
  ckpt="$(checkpoint_path "$dataset" "$architecture")"
  [[ -s "$log" ]] && grep -q -- 'Train finished' "$log" && [[ -s "$ckpt" ]]
}

train_one() {
  local dataset="$1" architecture="$2" model log ckpt
  model="$(model_name "$architecture")"
  log="$OUTPUT_ROOT/train_logs/${dataset}_${architecture}.log"
  ckpt="$(checkpoint_path "$dataset" "$architecture")"

  if training_complete "$dataset" "$architecture"; then
    echo "Skipping completed training: $dataset $architecture"
    return 0
  fi
  if [[ -e "$log" || -e "$ckpt" ]]; then
    echo "Refusing to overwrite partial training artifacts for $dataset $architecture" >&2
    return 2
  fi

  "$PYTHON_BIN" -u baseline/train_baseline_cifar.py \
    --model_name "$model" \
    --dataset "$dataset" \
    --data_dir "$DATA_DIR" \
    --output_dir "$CHECKPOINT_ROOT" \
    --case custom_noresize \
    --pretrained false \
    --seed "$SEED" \
    --override "$OVERRIDE" >"$log" 2>&1

  training_complete "$dataset" "$architecture" || {
    echo "Training finished without complete checkpoint artifacts: $dataset $architecture" >&2
    return 3
  }
}

best_accuracy() {
  local log="$1"
  sed -nE 's/.*Best (top1|acc): ([0-9]+([.][0-9]+)?).*/\2/p' "$log" | tail -n 1
}

check_gate() {
  local dataset="$1" reference_log="$2" candidate_log="$3" reference candidate
  reference="$(best_accuracy "$reference_log")"
  candidate="$(best_accuracy "$candidate_log")"
  [[ -n "$reference" && -n "$candidate" ]] || {
    echo "Could not parse clean accuracy for $dataset" >&2
    return 2
  }
  "$PYTHON_BIN" - "$dataset" "$reference" "$candidate" "$ACCURACY_TOLERANCE" "$OUTPUT_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

dataset = sys.argv[1]
reference, candidate, tolerance = map(float, sys.argv[2:5])
passed = candidate >= reference - tolerance
path = Path(sys.argv[5]) / "pilot_gate_rows"
path.mkdir(parents=True, exist_ok=True)
with (path / f"{dataset}.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["dataset", "reference_accuracy", "candidate_accuracy", "tolerance", "passed"])
    writer.writerow([dataset, reference, candidate, tolerance, str(passed).lower()])
raise SystemExit(0 if passed else 10)
PY
}

cat >"$OUTPUT_ROOT/recipe_manifest.json" <<EOF
{
  "architecture": "standard WRN with BatchNorm and learned shortcuts",
  "case": "custom_noresize",
  "seed": $SEED,
  "override": "$OVERRIDE",
  "final_feature_order": ["batchnorm", "dropout_0.25", "relu", "global_average_pool"],
  "pilot_gate_tolerance_fraction": $ACCURACY_TOLERANCE,
  "reference_logs": [
    "$REFERENCE_LOG_ROOT/standard_cifar10.log",
    "$REFERENCE_LOG_ROOT/standard_cifar100.log"
  ]
}
EOF

CURRENT_STAGE="pilot_training"
write_state running "$CURRENT_STAGE" "cifar10_and_cifar100_WRN_16_2"
pids=()
labels=()
for dataset in cifar10 cifar100; do
  train_one "$dataset" WRN_16_2 &
  pids+=("$!")
  labels+=("$dataset")
done
failed=()
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    failed+=("${labels[$index]}")
  fi
done
if (("${#failed[@]}" > 0)); then
  write_state failed "$CURRENT_STAGE" "${failed[*]}"
  exit 1
fi

CURRENT_STAGE="pilot_accuracy_gate"
write_state running "$CURRENT_STAGE" "tolerance_$ACCURACY_TOLERANCE"
gate_failed=()
for dataset in cifar10 cifar100; do
  if ! check_gate \
    "$dataset" \
    "$REFERENCE_LOG_ROOT/standard_${dataset}.log" \
    "$OUTPUT_ROOT/train_logs/${dataset}_WRN_16_2.log"; then
    gate_failed+=("$dataset")
  fi
done
if (("${#gate_failed[@]}" > 0)); then
  if [[ "$SKIP_ACCURACY_GATE" == "true" ]]; then
    write_state running "$CURRENT_STAGE" "clean_accuracy_gate_bypassed_${gate_failed[*]}"
  else
    write_state stopped "$CURRENT_STAGE" "clean_accuracy_gate_failed_${gate_failed[*]}"
    exit 10
  fi
fi

CURRENT_STAGE="remaining_training"
write_state running "$CURRENT_STAGE" "six_jobs"
datasets=(cifar10 cifar100 cifar10 cifar100 cifar10 cifar100)
architectures=(WRN_16_4 WRN_16_4 WRN_28_2 WRN_28_2 WRN_28_4 WRN_28_4)
pids=()
labels=()
for index in "${!architectures[@]}"; do
  train_one "${datasets[$index]}" "${architectures[$index]}" &
  pids+=("$!")
  labels+=("${datasets[$index]}_${architectures[$index]}")
done
failed=()
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    failed+=("${labels[$index]}")
  fi
done
if (("${#failed[@]}" > 0)); then
  write_state failed "$CURRENT_STAGE" "${failed[*]}"
  exit 1
fi

CURRENT_STAGE="completed"
write_state completed "$CURRENT_STAGE" "eight_checkpoints_ready"

