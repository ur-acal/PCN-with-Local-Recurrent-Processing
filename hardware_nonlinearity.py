import torch


def interpolate_R_eff(v, v_grid, R_codes, R_left, R_slope, code_idx, proj_fn=None):
    """Piecewise-linear R(v) interpolation shared by ordinary and pulse MVMs."""
    if proj_fn is not None:
        v = proj_fn(v)

    interval_idx = torch.bucketize(v, v_grid) - 1
    interval_idx = interval_idx.clamp(min=0, max=v_grid.numel() - 2)
    n_codes = R_codes.numel()
    v_flat = v.reshape(-1)
    interval_flat = interval_idx.reshape(-1)

    if torch.is_tensor(code_idx):
        if code_idx.numel() == 1:
            column = min(max(int(code_idx.item()), 0), n_codes - 1)
            position = interval_flat * n_codes + column
        else:
            columns = code_idx.to(torch.long).reshape(-1).clamp(0, n_codes - 1)
            position = interval_flat * n_codes + columns
    else:
        column = min(max(int(code_idx), 0), n_codes - 1)
        position = interval_flat * n_codes + column

    left = R_left.reshape(-1)[position]
    slope = R_slope.reshape(-1)[position]
    v_left = v_grid[interval_flat]
    return (left + slope * (v_flat - v_left)).reshape_as(v)
