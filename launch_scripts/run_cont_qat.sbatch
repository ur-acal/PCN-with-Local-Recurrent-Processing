#!/bin/bash -l
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 90:10:00
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

# Submit with the existing recipe environment and COMB_LIST. Continue latent
# QAT weights with a fresh optimizer/schedule; this is not exact-state recovery.
set -o pipefail
mode="${mode:-ft_and_eval}"
case "$mode" in
  ft_only|ft_and_eval) ;;
  *) echo 'QAT continuation requires mode=ft_only or ft_and_eval.' >&2; exit 2 ;;
esac

if [[ -z "${PRETRAIN_SAVE_PATH:-}" || -z "${CONTINUATION_SOURCE_MODEL:-}" ||
      -z "${FT_OUTPUT_SAVE_PATH:-}" ]]; then
  echo 'Set PRETRAIN_SAVE_PATH, CONTINUATION_SOURCE_MODEL and FT_OUTPUT_SAVE_PATH.' >&2
  exit 2
fi
if [[ "$CONTINUATION_SOURCE_MODEL" != *QAT* || "$CONTINUATION_SOURCE_MODEL" == */* ]]; then
  echo 'CONTINUATION_SOURCE_MODEL must be a QAT model directory name, not a path.' >&2
  exit 2
fi
CONTINUATION_SOURCE_SELECTION="${CONTINUATION_SOURCE_SELECTION:-full_param_last}"
case "$CONTINUATION_SOURCE_SELECTION" in
  full_param_last|latest) ;;
  *) echo 'QAT continuation requires latent weights: full_param_last or latest.' >&2; exit 2 ;;
esac
source_checkpoint="${PRETRAIN_SAVE_PATH}/${CONTINUATION_SOURCE_MODEL}/${CONTINUATION_SOURCE_MODEL}_${CONTINUATION_SOURCE_SELECTION}_ckpt.pth"
if [[ ! -s "$source_checkpoint" ]]; then
  echo "Missing or empty QAT source checkpoint: $source_checkpoint" >&2
  exit 2
fi
if [[ "$(realpath -m "$PRETRAIN_SAVE_PATH")" == "$(realpath -m "$FT_OUTPUT_SAVE_PATH")" ]]; then
  echo 'Use a separate FT_OUTPUT_SAVE_PATH to preserve the source experiment.' >&2
  exit 2
fi
if [[ -z "${COMB_LIST:-}" || "$COMB_LIST" == *$'\n'* ]]; then
  echo 'QAT continuation requires exactly one COMB_LIST record for the explicit source.' >&2
  exit 2
fi

export mode CONTINUATION_SOURCE_MODEL CONTINUATION_SOURCE_SELECTION
exec bash ./launch_scripts/run_kdcrd_then_ft.sbatch
