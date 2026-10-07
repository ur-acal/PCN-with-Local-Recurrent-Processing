"""Pooling-defined groups: a pool after layer N ends the stage at N."""


def stage_members(num_layers, pool_positions):
    positions = list(pool_positions)
    if num_layers < 1 or positions != sorted(set(positions)):
        raise ValueError('Pool positions must be unique and increasing; num_layers must be positive.')
    if any(p < 1 or p > num_layers for p in positions):
        raise ValueError('Pool positions must lie within the 1-based layer range.')
    ends = positions + ([] if positions and positions[-1] == num_layers else [num_layers])
    result, start = {}, 1
    for i, end in enumerate(ends, 1):
        result[f'stage_{i:02d}'] = [f'layer_{j:02d}' for j in range(start, end + 1)]
        start = end + 1
    result['final_linear'] = ['final_linear']
    return result
