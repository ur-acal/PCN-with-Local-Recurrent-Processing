import unittest

import torch

from baseline.cifar_resnet import WideBasicBlock, WideResNetCIFAR
from baseline.cifar_wrn_pool_after_add import (
    WideBasicBlockPoolAfterAdd, WideResNetPoolAfterAddCIFAR,
)


class PoolAfterAddTests(unittest.TestCase):
    def test_blocks_outputs_gradients_and_recording(self):
        for bn, bias in ((True, False), (False, True), (False, False)):
            for stride in (1, 2):
                with self.subTest(bn=bn, bias=bias, stride=stride):
                    torch.manual_seed(12)
                    args = dict(in_planes=4, planes=8, dropout_rate=0.1,
                                stride=stride, use_batchnorm=bn, conv_bias=bias)
                    old = WideBasicBlock(**args, avgpool_main_downsample=True,
                                         avgpool_downsample_shortcut=True).double()
                    new = WideBasicBlockPoolAfterAdd(**args).double()
                    new.load_state_dict(old.state_dict(), strict=True)
                    recorded = []
                    hook = new.pre_pool.register_forward_hook(
                        lambda m, i, o: recorded.append(o.detach().clone()))
                    x = torch.randn(2, 4, 9, 9, dtype=torch.float64)
                    # Non-leaf clones allow the existing no-BN in-place ReLU.
                    a = x.clone().requires_grad_()
                    b = x.clone().requires_grad_()
                    torch.manual_seed(22)
                    y = old(a.clone())
                    torch.manual_seed(22)
                    z = new(b.clone())
                    torch.testing.assert_close(z, y, rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(new.main_downsample(recorded[0]), z)
                    self.assertEqual(recorded[0].shape[-2:], (9, 9))
                    y.square().sum().backward()
                    z.square().sum().backward()
                    torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-9)
                    for p, q in zip(old.parameters(), new.parameters()):
                        torch.testing.assert_close(p.grad, q.grad, rtol=1e-9, atol=1e-9)
                    hook.remove()

    def test_network_strict_load_all_sizes(self):
        for depth, width in ((16, 2), (16, 4), (28, 2), (28, 4)):
            for bn, bias in ((True, False), (False, True), (False, False)):
                with self.subTest(depth=depth, width=width, bn=bn, bias=bias):
                    args = dict(depth=depth, widen_factor=width, num_classes=100,
                                use_batchnorm=bn, conv_bias=bias,
                                final_dropout_rate=0.25)
                    old = WideResNetCIFAR(**args, avgpool_main_downsample=True,
                                         avgpool_downsample_shortcut=True).eval()
                    new = WideResNetPoolAfterAddCIFAR(**args).eval()
                    new.load_state_dict(old.state_dict(), strict=True)
                    old.load_state_dict(new.state_dict(), strict=True)
                    self.assertEqual(list(old.state_dict()), list(new.state_dict()))
                    with torch.no_grad():
                        x = torch.randn(2, 3, 32, 32)
                        torch.testing.assert_close(new(x.clone()), old(x.clone()),
                                                   rtol=2e-4, atol=2e-5)

    def test_reject_incompatible_structure(self):
        with self.assertRaises(ValueError):
            WideResNetPoolAfterAddCIFAR(avgpool_main_downsample=False)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
