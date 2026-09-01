import random
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline.train_resnet18_sigma_selection import evaluate_fixed_max_sqrt_trials


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2, 2, bias=True)

    def forward(self, x):
        x = self.bn(self.conv(x)).mean(dim=(2, 3))
        return self.fc(x)


class NoisyCheckpointSelectionTest(unittest.TestCase):
    def test_evaluation_is_deterministic_and_read_only(self):
        torch.manual_seed(7)
        model = TinyClassifier()
        inputs = torch.randn(12, 1, 4, 4)
        labels = torch.arange(12) % 2
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)

        def evaluate(dataloader):
            correct = 0
            total = 0
            with torch.no_grad():
                model.eval()
                for batch_inputs, batch_labels in dataloader:
                    predicted = model(batch_inputs).argmax(dim=1)
                    correct += int((predicted == batch_labels).sum())
                    total += batch_labels.numel()
            return correct / total, None, labels, labels

        state_before = {name: value.clone() for name, value in model.state_dict().items()}
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()

        first = evaluate_fixed_max_sqrt_trials(model, loader, evaluate, 0.07, 3, 123)
        second = evaluate_fixed_max_sqrt_trials(model, loader, evaluate, 0.07, 3, 123)

        self.assertEqual(first, second)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state_before[name]), name)
        self.assertEqual(random.getstate(), python_before)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_before[1]))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))


if __name__ == "__main__":
    unittest.main()
