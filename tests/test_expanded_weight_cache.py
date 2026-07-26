from copy import deepcopy
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from validation import (
    MVMConv,
    Validator,
    conv2d_to_matrix_fixed_padding,
    expanded_weight_cache_fingerprint,
)


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.PcConvs = nn.ModuleList([_ToyBlock()])

    def forward(self, x):
        for layer in self.PcConvs:
            x = layer(x)
        return x


def _make_model():
    torch.manual_seed(17)
    return _ToyModel()


def _make_loader():
    generator = torch.Generator().manual_seed(23)
    inputs = torch.randn(4, 2, 5, 5, generator=generator)
    targets = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(inputs, targets), batch_size=2)


def _legacy_entry(model):
    conv = model.PcConvs[0].conv
    unrolled, _, _ = conv2d_to_matrix_fixed_padding(
        (2, 5, 5), conv.weight.detach(), stride=1, padding=1)
    return {
        "conv": {
            "weight": unrolled,
            "meta": {
                "padding": 1,
                "stride": 1,
                "ker_h": 3,
                "ker_w": 3,
                "inp_chan": 2,
                "out_chan": 3,
            },
        },
    }


def test_effective_weight_fingerprint_is_stable_and_weight_specific():
    first = _make_model()
    second = _make_model()

    first_key = expanded_weight_cache_fingerprint(first, (2, 5, 5))
    assert first_key == expanded_weight_cache_fingerprint(second, (2, 5, 5))
    assert first_key != expanded_weight_cache_fingerprint(second, (2, 6, 6))

    with torch.no_grad():
        second.PcConvs[0].conv.weight[0, 0, 0, 0].add_(1.0)
    assert first_key != expanded_weight_cache_fingerprint(second, (2, 5, 5))


def test_validator_reuses_content_addressed_expansion(tmp_path):
    original = _make_model()
    clean_state = deepcopy(original.state_dict())
    first = Validator(
        original, str(tmp_path), "cpu", _make_loader(), str(tmp_path))
    fingerprint = first.expanded_weight_cache_fingerprint
    assert isinstance(first.model.PcConvs[0].conv, MVMConv)

    reloaded = _make_model()
    reloaded.load_state_dict(clean_state)
    with patch(
            "validation.conv2d_to_matrix_fixed_padding",
            side_effect=AssertionError("cache should have been loaded")):
        second = Validator(
            reloaded, str(tmp_path), "cpu", _make_loader(), str(tmp_path))

    assert second.expanded_weight_cache_fingerprint == fingerprint
    assert isinstance(second.model.PcConvs[0].conv, MVMConv)
    assert list((tmp_path / ("cache_" + fingerprint)).glob(
        "expanded_weights_*.pth"))


def test_matching_legacy_cache_migrates_but_stale_cache_rebuilds(tmp_path):
    matching_dir = tmp_path / "matching"
    matching_dir.mkdir()
    matching_model = _make_model()
    torch.save(_legacy_entry(matching_model),
               matching_dir / "expanded_weights_0.pth")

    with patch(
            "validation.conv2d_to_matrix_fixed_padding",
            side_effect=AssertionError("matching legacy cache should migrate")):
        migrated = Validator(
            matching_model, str(matching_dir), "cpu", _make_loader(),
            str(matching_dir))
    migrated_dir = matching_dir / (
        "cache_" + migrated.expanded_weight_cache_fingerprint)
    assert (migrated_dir / "expanded_weights_0.pth").is_file()

    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    clean_model = _make_model()
    torch.save(_legacy_entry(clean_model), stale_dir / "expanded_weights_0.pth")
    changed_model = _make_model()
    with torch.no_grad():
        changed_model.PcConvs[0].conv.weight.add_(0.25)

    with patch(
            "validation.conv2d_to_matrix_fixed_padding",
            wraps=conv2d_to_matrix_fixed_padding) as expand:
        rebuilt = Validator(
            changed_model, str(stale_dir), "cpu", _make_loader(),
            str(stale_dir))
    assert expand.call_count == 1
    rebuilt_dir = stale_dir / (
        "cache_" + rebuilt.expanded_weight_cache_fingerprint)
    assert (rebuilt_dir / "expanded_weights_0.pth").is_file()
