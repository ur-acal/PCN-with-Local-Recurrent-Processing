import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from torch import nn

from pc_model import PCNet
from trainer_timm import TrainerCiFarTimmStyle, _NonfiniteTrainingLoss


class TrainingOutputPathTests(unittest.TestCase):
    def test_pcn_failure_files_are_separate_and_baseline_path_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            model = PCNet.__new__(PCNet)
            nn.Module.__init__(model)
            for name, net, relative in [
                ('PCNetNoBatchNorm_test', model, 'PCNetNoBatchNorm_test/training_collapse.json'),
                ('PCNetBoundaryBN_test', model, 'PCNetBoundaryBN_test/training_collapse.json'),
                ('baseline', nn.Linear(1, 1), 'training_collapse.json'),
            ]:
                event = {'reason': name}
                trainer = SimpleNamespace(
                    model=net, model_name=name, save_path=root, num_epochs=1,
                    collapse_monitor_enabled=True,
                    train_one_epoch=Mock(side_effect=_NonfiniteTrainingLoss(event)),
                    _record_training_collapse=TrainerCiFarTimmStyle._record_training_collapse,
                )
                TrainerCiFarTimmStyle.train(trainer)
                self.assertEqual(json.loads((Path(root) / relative).read_text()), event)
            self.assertEqual(len(list(Path(root).rglob('training_collapse.json'))), 3)


if __name__ == '__main__':
    unittest.main()
