#!/bin/bash -l
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 72:10:00
#SBATCH -o logs/slurm_jobs/slurm_%j.out

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}" || exit 1

IS_SLURM="${IS_SLURM:-1}"
if [[ "${IS_SLURM}" == 1 ]]; then
  source activate base
  conda activate scanbase || exit 1
fi

args=(--rows "${ROW:?ROW is required}" --datasets "${DATASETS:-cifar10,cifar100}"
      --sizes "${SIZES:?SIZES is required}" --stage "${STAGE:-train}"
      --parallelism "${PARALLELISM:-4}" --seed "${SEED:-4096}"
      --eval-parallelism "${EVAL_PARALLELISM:-8}"
      --mismatch-seed "${BASE_SEED:-123}"
      --conditions "${CONDITIONS:-max_additive,multiplicative}"
      --data-dir "${DATA_DIR:-${REPO_ROOT}/../data}"
      --output-root "${OUTPUT_ROOT:-${REPO_ROOT}/logs/wrn_controls_local}"
      --phased)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  args+=(--dry-run)
fi
if [[ "${SIMULATE:-0}" == 1 ]]; then
  args+=(--simulate --simulate-failure "${SIMULATE_FAILURE:-}")
fi

# Match the RGB worker: call Python in this activated shell, without another Bash.
python -u -m baseline.run_wrn_controls "${args[@]}"
