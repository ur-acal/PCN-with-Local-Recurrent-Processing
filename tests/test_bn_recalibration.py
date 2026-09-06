import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bn_recalibration import recalibrate_batchnorm


class _PhysicalToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.enable_summing_current_noise = True
        self.enable_coupler_noise = True
        self.dtc_leading_edge_jitter_std = 0.005
        self.dtc_falling_edge_jitter_std = 0.005
        self._summing_noise_generators = {"old": object()}
        self._coupler_noise_generators = {"old": object()}
        self.flags_seen = []
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x):
        self.flags_seen.append((
            self.enable_summing_current_noise,
            self.enable_coupler_noise,
            self.dtc_leading_edge_jitter_std,
            self.dtc_falling_edge_jitter_std,
        ))
        return self.bn(x)


def test_recalibration_keeps_dynamic_nonidealities_enabled():
    model = _PhysicalToy()
    old_mean = model.bn.running_mean.clone()
    old_weight = model.bn.weight.detach().clone()
    old_bias = model.bn.bias.detach().clone()
    old_summing_generator = model._summing_noise_generators["old"]
    old_coupler_generator = model._coupler_noise_generators["old"]
    inputs = torch.randn(8, 2, 3, 3) + 3.0
    loader = DataLoader(TensorDataset(inputs, torch.zeros(8)), batch_size=4)

    stats = recalibrate_batchnorm(model, loader, torch.device("cpu"))

    assert stats == {"num_batchnorms": 1, "num_batches": 2, "num_samples": 8}
    assert all(flags == (True, True, 0.005, 0.005)
               for flags in model.flags_seen)
    assert model.enable_summing_current_noise
    assert model.enable_coupler_noise
    assert model.dtc_leading_edge_jitter_std == 0.005
    assert model.dtc_falling_edge_jitter_std == 0.005
    assert model._summing_noise_generators["old"] is old_summing_generator
    assert model._coupler_noise_generators["old"] is old_coupler_generator
    assert not model.bn.training
    assert model.bn.momentum == 0.1
    assert not torch.equal(model.bn.running_mean, old_mean)
    assert torch.equal(model.bn.weight, old_weight)
    assert torch.equal(model.bn.bias, old_bias)
