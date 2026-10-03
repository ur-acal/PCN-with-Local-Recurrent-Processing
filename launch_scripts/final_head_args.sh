# Shared production classifier options. Omitted values inherit checkpoint defaults.
FINAL_HEAD_ARGS=()
for _head_key in FINAL_HEAD_TYPE FINAL_REPR_BITS FINAL_WEIGHT_BITS FINAL_BIAS_BITS \
                 FINAL_ACCUMULATOR_BITS FINAL_ADC_NOISE_LSB FINAL_HEAD_QUANTIZE FINAL_HEAD_CLAMP; do
  if [[ -n "${!_head_key:-}" ]]; then
    FINAL_HEAD_ARGS+=("--${_head_key,,}" "${!_head_key}")
  fi
done
unset _head_key
