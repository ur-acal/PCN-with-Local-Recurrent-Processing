#!/usr/bin/env python3
"""Evaluate saved native 32x32 digital RGB baselines on CIFAR or ciFAIR.

Architecture comes from the run's baseline_config.json, not current defaults.
No physical conversion, weight quantization, BN recalibration or test augmentation.
Saved student input quantization/centering is applied just as in training.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import baseline.cifar_resnet  # Register the native CIFAR/WRN architectures.
from baseline.baseline_cifar_configs import build_model
from cifar_rgb import RGB_DATASETS, cifar_task, rgb_dataset_class
from input_preprocessing import infer_preprocessing_from_path, prepare_student_input
from trainer import _CIFAR_STATS


def find_config(checkpoint, config=None):
    if config is not None:
        return Path(config)
    for parent in Path(checkpoint).resolve().parents:
        path = parent / "baseline_config.json"
        if path.is_file():
            return path
    raise FileNotFoundError("Cannot find baseline_config.json beside the saved run; supply --config")


def evaluate_checkpoint(checkpoint, dataset, *, data_dir="../data", batch_size=1024,
                        device=None, config=None, output=None, max_batches=0,
                        test_loader=None):
    if max_batches < 0 or batch_size < 1:
        raise ValueError("batch_size must be positive and max_batches must be nonnegative")
    task = cifar_task(dataset)
    checkpoint = Path(checkpoint).resolve()
    config_path = find_config(checkpoint, config)
    document = json.loads(config_path.read_text())
    if document.get("physical_feedforward"):
        raise ValueError("Use the physical evaluator for converted/QAT checkpoints")
    cfg = dict(document["cfg"])
    from cifar_rgb import require_native_rgb_config
    require_native_rgb_config(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_task = state.get("dataset_name") or document.get("training_dataset")
    if saved_task and cifar_task(saved_task) != task:
        raise ValueError(f"Checkpoint task {saved_task} does not match evaluation task {dataset}")
    if state.get("img_type", document.get("img_type", "rgb")).lower() != "rgb":
        raise ValueError("Cannot evaluate a sensor/RGGB checkpoint as a digital RGB model")
    if state.get("final_head", {}).get("type", "old_ideal") != "old_ideal":
        raise ValueError("This evaluator expects an ordinary digital classifier")
    cfg["pretrained"] = False  # Loading this checkpoint, not downloading initial weights.
    classes = 100 if task == "cifar100" else 10
    model = build_model(cfg["model_name"], cfg, classes)
    model.load_state_dict(state["net"], strict=True)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    preprocessing = state.get("student_preprocessing", {})
    normalize = preprocessing.get(
        "normalize_student_input", document.get("normalize_student_input", True))
    # New checkpoints are self-describing; older runs may only carry path
    # suffixes. Explicit None/False in metadata must override those suffixes.
    legacy_bits, legacy_center = infer_preprocessing_from_path(checkpoint)
    input_quant_bits = preprocessing.get(
        "input_quant_bits", document.get("input_quant_bits", legacy_bits))
    center_student_input = preprocessing.get(
        "center_student_input", document.get("center_student_input", legacy_center))
    default_mean, default_std = _CIFAR_STATS[task]
    mean = state.get("timm_mean") or default_mean
    std = state.get("timm_std") or default_std
    if test_loader is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            *([transforms.Normalize(mean, std)] if normalize else []),
        ])
        test_set = rgb_dataset_class(dataset)(
            root=data_dir, train=False, download=True, transform=transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    total = top1 = top5 = 0
    with torch.inference_mode():
        progress = tqdm(test_loader, desc=f"Test {dataset}", file=sys.stdout)
        for index, batch in enumerate(progress):
            if max_batches and index >= max_batches:
                break
            images, labels = batch[:2]
            if images.shape[1:] != (3, 32, 32):
                raise ValueError(f"Expected RGB 3x32x32, got {tuple(images.shape)}")
            labels = labels.to(device)
            images = prepare_student_input(
                images.to(device), input_quant_bits, center_student_input)
            predictions = model(images).topk(5, dim=1).indices
            top1 += predictions[:, 0].eq(labels).sum().item()
            top5 += predictions.eq(labels[:, None]).any(dim=1).sum().item()
            total += labels.numel()
            progress.set_postfix(acc=f"{100*top1/total:.2f}%")
    if not total:
        raise ValueError("Evaluation dataset is empty")
    result = dict(dataset=dataset, checkpoint=str(checkpoint), config=str(config_path),
                  checkpoint_epoch=state.get("epoch"), samples=total, top1_percent=100*top1/total,
                  normalize_student_input=normalize, mean=list(mean), std=list(std),
                  input_quant_bits=input_quant_bits, center_student_input=center_student_input,
                  max_batches=max_batches, full_dataset=(total == len(test_loader.dataset)))
    if classes == 100:
        result["top5_percent"] = 100*top5/total
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print("DIGITAL_TEST_RESULT " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=RGB_DATASETS, required=True)
    parser.add_argument("--config")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device")
    parser.add_argument("--output")
    parser.add_argument("--max_batches", type=int, default=0,
                        help="Optional diagnostic limit; zero evaluates the full test set")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.dataset, data_dir=args.data_dir,
                        batch_size=args.batch_size, device=args.device, config=args.config,
                        output=args.output, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
