#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# ALL FOUR MODES: --exclude-avgpool-wrn, --output-dir, --mismatch-types,
# --wide-csv, --controls-source, --baseline-paired-root, --pcn-bn-root,
# --extra-data, --extra-registry, --wrn-ranking.
# --wrn-ranking dataset-severe-mean (default) matches the paper: per dataset/type,
# average four architectures over three severe levels, then hold the top three fixed.
# Alternatives: global-endpoint (legacy) and per-point (a separate top three per point).
# ALL FOUR MODES: --sizes 28_6,28_8 --datasets cifar10,cifar100 (defaults unchanged).
# PCN-PLOTS ONLY: --color-mode family|batchnorm|pooling and --highlight-top3
# affect WRN curves inside the PCN-versus-WRN plot, not the PCN curves.
# PCN-PLOTS ONLY: --pcn-models P0,P3 (default), any comma-separated registered
# PCN IDs (including loaded PB-F/PB-R), or all.
# PCN-PLOTS ONLY: --top3-plus-baseline selects those PCNs + recal-BN top3 + baseline.
# P0,P3 writes plots/pcn_vs_top3_baseline_wrns/; another selection is encoded in
# the folder name, for example plots/p0_p3_pb-r_vs_top3_baseline_wrns/.
# BOTH PLOT MODES ONLY: --uncertainty (mean +/- STD where loaded; OFF by default).
# wrn-plots has fixed frozen/recalibrated/no-BN colors in its two comparisons.
# Example: bash launch_scripts/generate_robustness_reports.sh \
#   --color-mode batchnorm --highlight-top3 --pcn-models P0,P3
# Each Python invocation below can also be run independently.
shared=()
pcn_plot=()
plot_options=()
while (($#)); do
  case "$1" in
    --exclude-avgpool-wrn) shared+=("$1"); shift ;;
    --output-dir|--mismatch-types|--wide-csv|--controls-source|--baseline-paired-root|--pcn-bn-root|--extra-data|--extra-registry|--sizes|--datasets|--wrn-ranking)
      [[ $# -ge 2 ]] || { echo "Missing value: $1" >&2; exit 2; }
      shared+=("$1" "$2"); shift 2 ;;
    --color-mode|--pcn-models)
      [[ $# -ge 2 ]] || { echo "Missing value: $1" >&2; exit 2; }
      pcn_plot+=("$1" "$2"); shift 2 ;;
    --highlight-top3|--top3-plus-baseline) pcn_plot+=("$1"); shift ;;
    --uncertainty) plot_options+=("$1"); shift ;;
    --help|-h) "$PYTHON" "$REPO_ROOT/baseline/generate_robustness_reports.py" --help; exit ;;
    *) echo "Unknown option: $1 (use --option value syntax)" >&2; exit 2 ;;
  esac
done

###############################################
# 1. PCN comparison Markdown
###############################################
# Shared data/filter options only. All available PCNs stay in Markdown.
"$PYTHON" "$REPO_ROOT/baseline/generate_robustness_reports.py" --mode pcn-md "${shared[@]}"

###############################################
# 2. PCN versus all WRNs plots
###############################################
# Shared options + PCN selection/WRN styling + optional uncertainty.
"$PYTHON" "$REPO_ROOT/baseline/generate_robustness_reports.py" --mode pcn-plots "${shared[@]}" "${pcn_plot[@]}" "${plot_options[@]}"

###############################################
# 3. WRN-only comparison Markdown
###############################################
# Shared data/filter options only.
"$PYTHON" "$REPO_ROOT/baseline/generate_robustness_reports.py" --mode wrn-md "${shared[@]}"

###############################################
# 4. WRN-only plots
###############################################
# BN/bias and baseline/top-three frozen/recal comparisons: fixed color policy.
"$PYTHON" "$REPO_ROOT/baseline/generate_robustness_reports.py" --mode wrn-plots "${shared[@]}" "${plot_options[@]}"
