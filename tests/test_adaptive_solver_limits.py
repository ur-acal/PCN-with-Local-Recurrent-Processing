import io
import unittest
from contextlib import redirect_stdout

import torch

from TorchDiffEqPack import odesolve


class _LinearODE(torch.nn.Module):
    def forward(self, _t, y):
        return y


class AdaptiveSolverLimitTests(unittest.TestCase):
    def test_retry_fallback_supports_automatic_initial_step(self):
        options = {
            "method": "dopri5",
            "t0": 0.0,
            "t1": 1.0,
            "t_eval": [0.0, 1.0],
            "h": None,
            "rtol": 1e-4,
            "atol": 1e-4,
            "neval_max": 1,
            "max_steps": 5000,
        }

        with redirect_stdout(io.StringIO()):
            result = odesolve(_LinearODE(), torch.ones(2), options)

        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.allclose(result[-1], torch.full((2,), torch.e), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
