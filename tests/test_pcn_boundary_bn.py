import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from pc_model import PCNet, PCNetBoundaryBN, PCNetNoBatchNorm, PCN_CLASSES
from ode_pc import ODEXInitFFFB, make_ode_block


class BoundaryBNTests(unittest.TestCase):
    def build(self, cls, depth=16, width=2, classes=100):
        outputs = [16] + [c * width for c in (16, 32, 64) for _ in range((depth - 4) // 6)]
        inputs = [3] + outputs[:-1]
        pools = [int(i > 1 and inputs[i] != outputs[i]) for i in range(len(outputs))]
        torch.manual_seed(123)
        return cls(inp_channels=inputs, out_channels=outputs, max_pool=pools,
                   num_classes=classes, dropout=0.25, first_bn=False, bias=False,
                   cls=0, bypass=False, tie_weights=False, tie_bp=False)

    def test_all_eight_architectures_differ_only_by_bn(self):
        for depth in (16, 28):
            for width in (2, 4):
                for classes in (10, 100):
                    old = self.build(PCNetNoBatchNorm, depth, width, classes)
                    new = self.build(PCNetBoundaryBN, depth, width, classes)
                    old_params = dict(old.named_parameters())
                    new_params = dict(new.named_parameters())
                    for name, value in old_params.items():
                        self.assertTrue(torch.equal(value, new_params[name]), name)
                    extra = set(new_params) - set(old_params)
                    self.assertTrue(extra)
                    self.assertTrue(all(n.startswith(('BNs.', 'BNend.')) for n in extra))
                    self.assertIsInstance(new.BNs[0], nn.Identity)
                    self.assertEqual(sum(isinstance(m, nn.BatchNorm2d) for m in new.modules()), len(new.ics))
                    for m in new.modules():
                        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                            self.assertIsNone(m.bias)
                        if isinstance(m, nn.BatchNorm2d):
                            self.assertEqual((m.eps, m.momentum), (1e-5, 0.1))
                    self.assertEqual(old.init_args, new.init_args)

    def test_ode_forward_and_gradients_with_bn_removed(self):
        def small(cls):
            torch.manual_seed(321)
            net = cls(inp_channels=[3, 4], out_channels=[4, 4], max_pool=[0, 1],
                      num_classes=10, dropout=0.25, first_bn=False, cls=0, bypass=False)
            return make_ode_block(net, ode_block=ODEXInitFFFB, method='dopri5', t_end=1.75, tol=1e-4)
        old, new = small(PCNetNoBatchNorm), small(PCNetBoundaryBN)
        new.BNs = nn.ModuleList([nn.Identity() for _ in new.BNs])
        new.BNend = nn.Identity()
        x = torch.randn(2, 3, 4, 4)
        for training in (False, True):
            old.train(training)
            new.train(training)
            old.zero_grad()
            new.zero_grad()
            torch.manual_seed(9)
            old_features, a = old(x, is_feat=True)
            torch.manual_seed(9)
            new_features, b = new(x, is_feat=True)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(old_features[0], new_features[0], rtol=0, atol=0)
            a.sum().backward()
            b.sum().backward()
            for name, p in old.named_parameters():
                other = dict(new.named_parameters())[name]
                if p.grad is None:
                    self.assertIsNone(other.grad)
                else:
                    torch.testing.assert_close(p.grad, other.grad, rtol=0, atol=0)

    def test_head_order_backward_and_registry_restore(self):
        net = self.build(PCNetBoundaryBN)
        events = []
        hook = net.BNend.register_forward_hook(lambda *args: events.append('bn'))
        dropout = F.dropout
        def observed_dropout(*args, **kwargs):
            events.append('dropout')
            return dropout(*args, **kwargs)
        with patch('pc_model.F.dropout', side_effect=observed_dropout):
            net(torch.randn(2, 3, 8, 8)).sum().backward()
        hook.remove()
        self.assertEqual(events, ['bn', 'dropout'])
        self.assertTrue(torch.isfinite(net.BNend.weight.grad).all())
        restored = PCN_CLASSES[type(net).__name__](**net.init_args['model_args'], **net.init_args['kwargs'])
        restored.load_state_dict(net.state_dict(), strict=True)
        self.assertIs(PCN_CLASSES['PCNet'], PCNet)
        self.assertIs(PCN_CLASSES['PCNetNoBatchNorm'], PCNetNoBatchNorm)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
