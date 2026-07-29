#!/bin/bash

REPO_ROOT="/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing"
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/run_mc45_toggle_ablation.sh"

###############################################################################################
# running with
# module swap slurm slurm/24.05.0.b1
# ( source ./launch_scripts/slurm_run_mc45_toggle_ablation.sh ) \
#  > ./logs/scheduler_slurm/slurm_mc45_toggle_ablation.log 2>&1 < /dev/null &
###############################################################################################

N_SERVERS="${N_SERVERS:-6}"
N_TRIALS="${N_TRIALS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/mc45_alternative_gaussian_2trials_59p50}"

export MC_45_CORNER_DIR="${MC_45_CORNER_DIR:-hardware_data/mc_45_corners}"
export ROOM_TEMPERATURE_C="${ROOM_TEMPERATURE_C:-25.0}"
export FULL_45_CORNER_C="${FULL_45_CORNER_C:-1.2e-13}"
# coupler non-linearity and variation
export MC_COUPLER_NONLINEAR_VARIATION_SOURCE="${MC_COUPLER_NONLINEAR_VARIATION_SOURCE:-CU_4500_r_vs_vin.csv}"
export MC_COUPLER_NONLINEAR_VARIATION_QUANTITY="${MC_COUPLER_NONLINEAR_VARIATION_QUANTITY:-resistance}"
export MC_COUPLER_NOMINAL_R="${MC_COUPLER_NOMINAL_R:-30e3}"
export ENABLE_NONLINEAR_R="${ENABLE_NONLINEAR_R:-true}"
export ENABLE_DIFF_MISMATCH="${ENABLE_DIFF_MISMATCH:-false}"
export NONLINEAR_R_CURVE_SHARING="${NONLINEAR_R_CURVE_SHARING:-per_coupler}"
export NONLINEAR_R_CURVE_SAMPLING="${NONLINEAR_R_CURVE_SAMPLING:-multivariate_gaussian}"
export NONLINEAR_R_CURVE_EDGE_CHUNK_SIZE="${NONLINEAR_R_CURVE_EDGE_CHUNK_SIZE:-65536}"
# Spin variation
export MC_SPIN_VARIATION_SOURCE="${MC_SPIN_VARIATION_SOURCE:-PVT_Monte_Carlo_Results_SPIN.csv}"
export ENABLE_SPIN_VARIATION="${ENABLE_SPIN_VARIATION:-true}"
# measured act function
export MC_RELU_MONTE_CARLO_SOURCE="${MC_RELU_MONTE_CARLO_SOURCE:-relu_monteCarlo}"
export ENABLE_MEASURED_ACTIVATION="${ENABLE_MEASURED_ACTIVATION:-true}"
export ACTIVATION_INTERPOLATION="${ACTIVATION_INTERPOLATION:-piecewise_linear}"
export ACTIVATION_FIT_CONSTRAINT="${ACTIVATION_FIT_CONSTRAINT:-auto}"
export ACTIVATION_NORMALIZE_POSITIVE_ENDPOINT="${ACTIVATION_NORMALIZE_POSITIVE_ENDPOINT:-false}"
export COMPILE_MEASURED_ACTIVATION="${COMPILE_MEASURED_ACTIVATION:-false}"
# noise
export ENABLE_SUMMING_CURRENT_NOISE="${ENABLE_SUMMING_CURRENT_NOISE:-true}"
export SUMMING_CURRENT_P="${SUMMING_CURRENT_P:-0.6e-12}"
export ENABLE_COUPLER_NOISE="${ENABLE_COUPLER_NOISE:-true}"
export COUPLER_NOISE_P="${COUPLER_NOISE_P:-0.6e-12}"
# DTC pulse
export ENABLE_DTC_NONIDEALITY="${ENABLE_DTC_NONIDEALITY:-true}"
export MC_DTC_PULSE_WIDTH_VARIATION_SOURCE="${MC_DTC_PULSE_WIDTH_VARIATION_SOURCE:-PVT_45corner_DTC_pulse_width.csv}"
export DTC_CHARACTERIZED_NOMINAL_FRACTION="${DTC_CHARACTERIZED_NOMINAL_FRACTION:-0.0625}"
export DTC_LEADING_EDGE_VARIATION_STD="${DTC_LEADING_EDGE_VARIATION_STD:-0.0}"
export DTC_LEADING_EDGE_JITTER_STD="${DTC_LEADING_EDGE_JITTER_STD:-0.005}"
export DTC_FALLING_EDGE_JITTER_STD="${DTC_FALLING_EDGE_JITTER_STD:-0.005}"

CORNER_GROUPS=()
corner_index=0
for process in TT FF SS FS SF; do
  for voltage in 0 1 2; do
    for temperature in 0 1 2; do
      shard=$((corner_index % N_SERVERS))
      CORNER_GROUPS[${shard}]="${CORNER_GROUPS[${shard}]} ${process}_V${voltage}_T${temperature}"
      corner_index=$((corner_index + 1))
    done
  done
done

for ((shard = 0; shard < N_SERVERS; shard++)); do
  corner_ids="${CORNER_GROUPS[${shard}]# }"
  output_dir="${OUTPUT_ROOT}/shard_${shard}"
  jid=$(
    sbatch --parsable \
      --export=ALL,IS_SLURM=1,REPO_ROOT="${REPO_ROOT}",N_TRIALS="${N_TRIALS}",CORNER_IDS="${corner_ids}",OUTPUT_DIR="${output_dir}" \
      "${SBATCH_SCRIPT}"
  )
  echo "submitted job ${jid}: shard=${shard}, corners=${corner_ids}"
  sleep "1"
done
