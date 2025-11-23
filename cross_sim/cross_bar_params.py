base_params_args = {
    # 'ideal': ideal,  # When ideal is True, this should be expected to have the same acc as raw torch model
    ## Mapping style
    'core_style': "BALANCED",
    'Nslices': 1,
    ## Weight value representation and precision
    # 'weight_bits': weight_bits,
    'weight_percentile': 100,
    # 'digital_bias': digital_bias,
    ## Memory device
    'Rmin': 1e4,
    'Rmax': 1e6,
    'infinite_on_off_ratio': False,
    ###############################################
    'error_model': "generic",
    # 'alpha_error': noise_level,
    # 'proportional_error': proportional_error,
    ###############################################
    'noise_model': "none",
    'alpha_noise': 0.0,
    'proportional_noise': False,
    'drift_model': "none",
    't_drift': 0,
    ## Array properties
    'NrowsMax': 1152,
    'NcolsMax': None,
    'Rp_row': 0,  # ohms
    'Rp_col': 0,  # ohms
    'interleaved_posneg': False,
    'subtract_current_in_xbar': True,
    'current_from_input': True,
    ## Input quantization
    # 'input_bits': input_bits,
    'input_bitslicing': False,
    'input_slice_size': 1,
    ## ADC
    # 'adc_bits': adc_bits,
    'adc_range_option': "CALIBRATED",
    'adc_type': "generic",
    'adc_per_ibit': False,
    ## Simulation parameters
    # 'useGPU': useGPU
}