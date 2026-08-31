import unittest
from unittest.mock import patch

import torch

from baseline.cifar_resnet import WideResNetCIFAR


class WideResNetFinalDropoutTests(unittest.TestCase):
    def _model(self, final_dropout_rate):
        return WideResNetCIFAR(
            depth=16,
            widen_factor=2,
            num_classes=10,
            dropout_rate=0.0,
            final_dropout_rate=final_dropout_rate,
        )

    def test_final_dropout_is_between_final_bn_and_relu(self):
        model = self._model(0.25).train()
        x = torch.randn(2, 3, 32, 32)
        observed = {}

        def capture_bn(_module, _inputs, output):
            observed["bn"] = output.detach().clone()

        def identity_dropout(value, p, training):
            observed["dropout_input"] = value.detach().clone()
            observed["p"] = p
            observed["training"] = training
            return value

        hook = model.bn.register_forward_hook(capture_bn)
        try:
            with patch("baseline.cifar_resnet.F.dropout", side_effect=identity_dropout) as dropout:
                features = model.forward_features(x)
        finally:
            hook.remove()

        self.assertEqual(dropout.call_count, 1)
        self.assertEqual(observed["p"], 0.25)
        self.assertTrue(observed["training"])
        torch.testing.assert_close(observed["dropout_input"], observed["bn"])
        torch.testing.assert_close(features, torch.relu(observed["bn"]))

    def test_eval_logits_are_identical_for_existing_checkpoint_state(self):
        torch.manual_seed(7)
        source = self._model(0.0).eval()
        checkpoint = source.state_dict()
        aligned = self._model(0.25).eval()
        aligned.load_state_dict(checkpoint, strict=True)
        x = torch.randn(4, 3, 32, 32)

        with torch.no_grad():
            expected = source(x)
            actual = aligned(x)

        self.assertEqual(list(source.state_dict()), list(aligned.state_dict()))
        self.assertEqual(
            sum(parameter.numel() for parameter in source.parameters()),
            sum(parameter.numel() for parameter in aligned.parameters()),
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()

