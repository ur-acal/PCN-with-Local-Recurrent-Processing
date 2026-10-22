"""RGB ciFAIR replaces test data only, preserving the training task and split."""
import json
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from cifar_rgb import (CiFAIR10, CiFAIR100, cifar_task, require_rgb_alias,
                       rgb_dataset_class)
from trainer import TrainerCiFar, _CIFAR_STATS, _normalize_dataset_name


def test_resized_cifair_rejected_before_model_construction():
    from baseline import train_baseline_cifar as entry
    with patch("sys.argv", ["train", "--model_name", "resnet18",
                            "--dataset", "cifair10", "--img_type", "rgb",
                            "--case", "resize_scratch"]), \
         patch.object(entry, "build_model") as build:
        with pytest.raises(ValueError, match="native 3-channel 32x32"):
            entry.main()
        build.assert_not_called()


@pytest.mark.parametrize("classes", [10, 100])
def test_rgb_classes_and_normalization(classes):
    from torchvision import datasets
    canonical = f"cifar{classes}"
    alias = f"cifair{classes}"
    standard_cls = datasets.CIFAR100 if classes == 100 else datasets.CIFAR10
    assert rgb_dataset_class(alias, train=True) is standard_cls
    assert rgb_dataset_class(canonical, train=False) is standard_cls
    assert rgb_dataset_class(alias) is (CiFAIR100 if classes == 100 else CiFAIR10)
    assert _normalize_dataset_name(alias) == canonical == cifar_task(alias)
    assert _CIFAR_STATS[_normalize_dataset_name(alias)] == _CIFAR_STATS[canonical]
    with pytest.raises(ValueError, match="requires img_type=rgb"):
        require_rgb_alias(alias, "CiFAIR")
    require_rgb_alias(canonical, "CiFAIR")  # Existing sensor mode is unchanged.


@pytest.mark.parametrize("classes", [10, 100])
@pytest.mark.parametrize("validation_mode", [False, True])
def test_rgb_training_and_validation_split_are_identical(tmp_path, classes, validation_mode):
    class FakeCIFAR(Dataset):
        def __init__(self, root, train, download, transform):
            self.targets = (np.arange(10000) % classes).tolist()
            self.train = train
            self.transform = transform

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return torch.zeros(3, 32, 32), self.targets[index]

    class FakeCiFAIR(FakeCIFAR):
        pass

    def construct(name):
        trainer = object.__new__(TrainerCiFar)
        trainer.requested_dataset_name = name
        trainer.normalize_student_input = True
        trainer.validation_mode = validation_mode
        trainer.official_test_loader = None
        trainer.validation_data_root = str(tmp_path)
        trainer.validation_manifest = None
        trainer.validation_split_seed = 4096
        trainer.aug = False
        trainer._crd_enabled = False
        trainer.teacher_eval_loader = None
        trainer.batch_size = 8
        trainer.test_batch_size = 8
        trainer.num_workers = 0
        trainer._prepare_cifar("rgb", name)
        return trainer

    with patch(f"torchvision.datasets.CIFAR{classes}", FakeCIFAR), \
         patch(f"cifar_rgb.CiFAIR{classes}", FakeCiFAIR):
        a, b = construct(f"cifar{classes}"), construct(f"cifair{classes}")
    if not validation_mode:
        assert type(a.train_set) is type(b.train_set) is FakeCIFAR
        assert type(a.val_set) is FakeCIFAR
        assert type(b.val_set) is FakeCiFAIR
        assert len(a.train_set) == len(b.train_set) == 10000
        return
    assert a.train_set.indices == b.train_set.indices
    assert a.val_set.indices == b.val_set.indices
    assert a.validation_split_metadata == b.validation_split_metadata
    assert type(b.train_set.dataset) is FakeCIFAR
    assert type(b.val_set.dataset) is FakeCIFAR
    assert type(a.official_test_loader.dataset) is FakeCIFAR
    assert type(b.official_test_loader.dataset) is FakeCiFAIR
    assert len(a.train_set) == len(b.train_set) == 5000


@pytest.mark.parametrize("classes", [10, 100])
def test_shared_eval_uses_cifair_test_but_cifar_training(classes):
    import inference_utils
    with patch(f"cifar_rgb.CiFAIR{classes}") as fair, \
         patch(f"torchvision.datasets.CIFAR{classes}") as standard:
        fair.return_value = standard.return_value = []
        inference_utils.get_eval_data(img_type="rgb", task=f"cifair{classes}")
        assert fair.call_args.kwargs["train"] is False
        standard.assert_not_called()
        inference_utils.get_eval_data(img_type="rgb", task=f"cifair{classes}", train=True)
        assert standard.call_args.kwargs["train"] is True


def test_missing_cifair_never_falls_back_to_cifar():
    import inference_utils
    with patch("cifar_rgb.CiFAIR10", side_effect=RuntimeError("ciFAIR unavailable")), \
         patch("torchvision.datasets.CIFAR10") as standard:
        with pytest.raises(RuntimeError, match="ciFAIR unavailable"):
            inference_utils.get_eval_data(img_type="rgb", task="cifair10")
        standard.assert_not_called()


@pytest.mark.parametrize("classes", [10, 100])
def test_saved_config_rebuilds_narrow_model_and_strictly_loads(tmp_path, classes):
    from baseline.baseline_cifar_configs import build_model, get_baseline_config
    from baseline.evaluate_baseline_cifar import evaluate_checkpoint
    cfg = get_baseline_config("wrn_flexible_cifar_avgpool_shortcut", case="custom_noresize")
    cfg.update(wrn_depth=10, wrn_first_stage_channels=17, in_chans=3)
    model = build_model(cfg["model_name"], cfg, classes).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc.bias[0] = 5
    (tmp_path / "baseline_config.json").write_text(json.dumps({"cfg": cfg, "physical_feedforward": None}))
    checkpoint = tmp_path / "model" / "best_ckpt.pth"
    checkpoint.parent.mkdir()
    state = dict(net=model.state_dict(), dataset_name=f"cifar{classes}", img_type="rgb", epoch=3)
    torch.save(state, checkpoint)
    inputs = torch.randn(3, 3, 32, 32)
    dataset = torch.utils.data.TensorDataset(inputs, torch.zeros(3, dtype=torch.long))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    result = evaluate_checkpoint(checkpoint, f"cifair{classes}", test_loader=loader, device="cpu")
    assert result["top1_percent"] == 100 and result["samples"] == 3
    assert result["full_dataset"] is True
    assert ("top5_percent" in result) == (classes == 100)
    del state["net"]["conv1.weight"]
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        evaluate_checkpoint(checkpoint, f"cifair{classes}", test_loader=loader, device="cpu")


@pytest.mark.parametrize("supplied_loader", [False, True])
@pytest.mark.parametrize("source", ["checkpoint", "config", "legacy_path", "defaults"])
@pytest.mark.parametrize("normalize", [False, True])
def test_digital_evaluator_matches_trainer_input_preprocessing(tmp_path, supplied_loader, source, normalize):
    from PIL import Image
    from torchvision import transforms
    from baseline import evaluate_baseline_cifar as evaluation
    from baseline.baseline_cifar_configs import get_baseline_config

    bits, center = (None, False) if source == "defaults" else (2, True)
    settings = dict(input_quant_bits=bits, center_student_input=center)
    cfg = get_baseline_config("wrn_flexible_cifar_avgpool_shortcut", case="custom_noresize")
    cfg["in_chans"] = 3
    document = dict(cfg=cfg, normalize_student_input=normalize)
    state = dict(net={}, student_preprocessing=dict(normalize_student_input=normalize))
    if source == "checkpoint":
        state["student_preprocessing"].update(settings)
        # Checkpoint settings take precedence over the run config.
        document.update(input_quant_bits=None, center_student_input=False)
    elif source == "config":
        document.update(settings)
    (tmp_path / "baseline_config.json").write_text(json.dumps(document))
    directory = tmp_path / "model_iq2_ctr" if source == "legacy_path" else tmp_path
    directory.mkdir(exist_ok=True)
    checkpoint = directory / "model.pth"
    torch.save(state, checkpoint)

    class Images(Dataset):
        def __init__(self, *, transform=None, **kwargs):
            self.transform = transform or transforms.Compose([
                transforms.ToTensor(),
                *([transforms.Normalize(*_CIFAR_STATS["cifar10"])] if normalize else [])])

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return self.transform(Image.fromarray(np.full((32, 32, 3), 64, dtype=np.uint8))), 1

    received = []

    class Probe(torch.nn.Module):
        def forward(self, images):
            received.append(images.clone())
            mean = images.mean(dim=(1, 2, 3))
            logits = images.new_full((len(images), 10), -10)
            logits[:, 0], logits[:, 1] = mean, -mean
            return logits

    loader = torch.utils.data.DataLoader(Images(), batch_size=2) if supplied_loader else None
    with patch.object(evaluation, "build_model", return_value=Probe()), \
         patch.object(evaluation, "rgb_dataset_class", return_value=Images):
        result = evaluation.evaluate_checkpoint(
            checkpoint, "cifair10", batch_size=2, test_loader=loader, device="cpu")
    trainer = object.__new__(TrainerCiFar)
    trainer.input_quant_bits, trainer.center_student_input = bits, center
    raw = torch.stack([Images()[index][0] for index in range(3)])
    expected = trainer._prepare_student_inputs(raw)
    assert torch.equal(torch.cat(received), expected)
    assert result["top1_percent"] == (100 if expected.mean() < 0 else 0)
    assert result["input_quant_bits"] == bits
    assert result["center_student_input"] is center


def test_digital_checkpoint_explicit_disabled_settings_override_legacy_suffix(tmp_path):
    from baseline import evaluate_baseline_cifar as evaluation
    from baseline.baseline_cifar_configs import get_baseline_config
    cfg = get_baseline_config("wrn_flexible_cifar_avgpool_shortcut", case="custom_noresize")
    cfg["in_chans"] = 3
    (tmp_path / "baseline_config.json").write_text(json.dumps(dict(
        cfg=cfg, input_quant_bits=2, center_student_input=True)))
    directory = tmp_path / "model_iq2_ctr"
    directory.mkdir()
    checkpoint = directory / "model.pth"
    torch.save(dict(net={}, student_preprocessing=dict(
        input_quant_bits=None, center_student_input=False)), checkpoint)
    images = torch.full((1, 3, 32, 32), .251)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        images, torch.zeros(1, dtype=torch.long)))
    received = []

    class Probe(torch.nn.Module):
        def forward(self, batch):
            received.append(batch.clone())
            return batch.new_zeros((len(batch), 10))

    with patch.object(evaluation, "build_model", return_value=Probe()):
        result = evaluation.evaluate_checkpoint(checkpoint, "cifar10", test_loader=loader, device="cpu")
    assert torch.equal(received[0], images)
    assert result["input_quant_bits"] is None
    assert result["center_student_input"] is False


def test_digital_training_saves_input_preprocessing(tmp_path):
    from baseline import train_baseline_cifar as entry
    from input_preprocessing import student_preprocessing_metadata
    with patch("sys.argv", ["train", "--model_name", "wrn_flexible_cifar_avgpool_shortcut",
                            "--dataset", "cifair10", "--img_type", "rgb",
                            "--input_quant_bits", "2", "--center_student_input", "true",
                            "--output_dir", str(tmp_path), "--print_only", "true"]):
        entry.main()
    config = json.loads(next(tmp_path.rglob("baseline_config.json")).read_text())
    assert config["input_quant_bits"] == 2
    assert config["center_student_input"] is True
    metadata = student_preprocessing_metadata(SimpleNamespace(
        input_quant_bits=2, center_student_input=True))
    assert metadata["input_quant_bits"] == 2
    assert metadata["center_student_input"] is True
    # Older/synthetic trainers retain the previous no-op interpretation.
    defaults = student_preprocessing_metadata(SimpleNamespace())
    assert defaults["input_quant_bits"] is None
    assert defaults["center_student_input"] is False


@pytest.mark.parametrize("classes", [10, 100])
def test_training_cli_keeps_requested_dataset_in_paths(classes):
    from baseline.train_baseline_cifar import parse_args, infer_num_classes
    with patch("sys.argv", ["train", "--model_name", "wrn_flexible_cifar_avgpool_shortcut",
                            "--dataset", f"cifair{classes}", "--img_type", "rgb",
                            "--validation_mode", "true"]):
        args = parse_args()
    assert args.dataset == f"cifair{classes}"
    assert infer_num_classes(args.dataset) == classes


@pytest.mark.parametrize("classes", [10, 100])
def test_final_only_reports_last_without_a_second_test(tmp_path, classes):
    # Exercise the launcher's actual post-training branch without running
    # hundreds of epochs. Its trainer-side metrics are already computed once.
    import baseline.train_baseline_cifar as entry
    tree = ast.parse(Path(entry.__file__).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    branch = next(node for node in reversed(main.body) if isinstance(node, ast.If)
                  and "args.dataset.startswith('cifair')" == ast.unparse(node.test))
    wrapper = ast.parse("def finish():\n    pass\n")
    wrapper.body[0].body = [branch]
    trainer = SimpleNamespace(final_eval_only=True, save_path=str(tmp_path),
                              model_name="wrn", num_epochs=300,
                              final_test_results=dict(samples=10000, top1_percent=75.0))
    args = SimpleNamespace(dataset=f"cifair{classes}", normalize_student_input=True)
    namespace = dict(args=args, trainer=trainer, Path=Path, json=json)
    with patch("baseline.evaluate_baseline_cifar.evaluate_checkpoint") as evaluate:
        exec(compile(ast.fix_missing_locations(wrapper), entry.__file__, "exec"), namespace)
        namespace["finish"]()
        evaluate.assert_not_called()
    result = json.loads((tmp_path / "test_results.json").read_text())
    assert result["checkpoint"].endswith("wrn_last_ckpt.pth")
    assert result["samples"] == 10000 and result["top1_percent"] == 75.0
