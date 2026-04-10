import torch
from TorchDiffEqPack.odesolver import odesolve as aca_ode_solve


def toy_rhs(t, y):
    out = torch.zeros_like(y)
    out[:, :, 8, 8] = 1.0
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    y0 = torch.zeros((2, 3, 16, 16), device=device, dtype=dtype)

    option_aca = {
        "method": "dopri5",
        "t0": torch.tensor(0.0, device=device, dtype=dtype),
        "t1": torch.tensor(6e-8, device=device, dtype=dtype),
        "t_eval": [
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(6e-6, device=device, dtype=dtype),
        ],
        "rtol": 1e-6,
        "atol": 1e-6,
        "h": None,
    }

    out = aca_ode_solve(toy_rhs, y0, option_aca)
    y1 = out[-1]

    dy = y1 - y0
    changed = (dy.abs() > 1e-12).any(dim=1)  # [B, H, W]

    print("changed positions sample0:")
    print(changed[0].nonzero(as_tuple=False))
    print("num changed positions sample0:", changed[0].sum().item())

    print("active pixel diff sample0:", dy[0, :, 8, 8].abs().max().item())

    mask = torch.ones((16, 16), device=device, dtype=torch.bool)
    mask[8, 8] = False
    print("non-active diff sample0:", dy[0, :, mask].abs().max().item())


if __name__ == "__main__":
    main()