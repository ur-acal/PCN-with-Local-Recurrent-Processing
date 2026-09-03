import io
import unittest

import torch
import torch.nn as nn

import baseline.cifar_resnet  # noqa: F401 - registers models with timm
from baseline.baseline_cifar_configs import build_model, get_baseline_config


MODEL_SPECS = {
    "wrn_28_2_cifar_avgpool": (True, True),
    "wrn_28_2_cifar_nobn_avgpool": (False, True),
    "wrn_28_2_cifar_avgpool_shortcut": (True, False),
    "wrn_28_2_cifar_nobn_avgpool_shortcut": (False, False),
}


def make_model(name: str):
    cfg = get_baseline_config(name, pretrained=False, case="custom_noresize")
    cfg.update({"dropout_rate": 0.0, "final_dropout_rate": 0.25})
    return build_model(name, cfg, num_classes=100)


class WRNAvgPoolVariantTests(unittest.TestCase):
    def test_structure_and_feature_sizes(self):
        for name, (uses_bn, pools_main_path) in MODEL_SPECS.items():
            with self.subTest(name=name):
                model = make_model(name)
                convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
                shortcut_convs = [(n, m) for n, m in convs if ".shortcut" in n]
                stride2_convs = [(n, m) for n, m in convs if m.stride == (2, 2)]
                avg_pools = [m for m in model.modules() if isinstance(m, nn.AvgPool2d)]
                batchnorms = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]

                self.assertEqual(len(convs), 25)
                self.assertEqual(shortcut_convs, [])
                self.assertEqual(len(stride2_convs), 0 if pools_main_path else 2)
                self.assertEqual(len(avg_pools), 4 if pools_main_path else 2)
                self.assertEqual(bool(batchnorms), uses_bn)
                self.assertTrue(all((m.bias is None) == uses_bn for _, m in convs))

                shapes = []
                hooks = [
                    layer.register_forward_hook(lambda _m, _i, out: shapes.append(tuple(out.shape)))
                    for layer in (model.layer1, model.layer2, model.layer3)
                ]
                model.eval()
                with torch.no_grad():
                    output = model(torch.randn(2, 3, 32, 32))
                for hook in hooks:
                    hook.remove()
                self.assertEqual(shapes, [(2, 32, 32, 32), (2, 64, 16, 16), (2, 128, 8, 8)])
                self.assertEqual(tuple(output.shape), (2, 100))

    def test_gradients_and_state_dict_reload(self):
        for name in MODEL_SPECS:
            with self.subTest(name=name):
                torch.manual_seed(7)
                model = make_model(name)
                model.train()
                loss = model(torch.randn(2, 3, 32, 32)).square().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                learned_modules = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
                self.assertTrue(all(m.weight.grad is not None for m in learned_modules))
                self.assertTrue(all(torch.isfinite(m.weight.grad).all() for m in learned_modules))

                model.eval()
                sample = torch.randn(2, 3, 32, 32)
                with torch.no_grad():
                    expected = model(sample)
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = make_model(name)
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                restored.eval()
                with torch.no_grad():
                    actual = restored(sample)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_existing_wrn_architectures_are_unchanged(self):
        for name in ("wrn_28_2_cifar", "wrn_28_2_cifar_nobn"):
            with self.subTest(name=name):
                model = make_model(name)
                shortcut_convs = [
                    module
                    for module_name, module in model.named_modules()
                    if ".shortcut" in module_name and isinstance(module, nn.Conv2d)
                ]
                stride2_convs = [
                    module for module in model.modules()
                    if isinstance(module, nn.Conv2d) and module.stride == (2, 2)
                ]
                self.assertEqual(len(shortcut_convs), 3)
                self.assertEqual(len(stride2_convs), 4)


if __name__ == "__main__":
    unittest.main()
