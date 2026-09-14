"""CPU checkpoint equivalence check; no training, downloads or checkpoint writes.

Run from the repository root:
python -m baseline.verify_wrn_pool_after_add --samples 256
Use --samples 10000 for the full test datasets.
"""

import argparse
import json
from pathlib import Path

import torch
from torchvision import datasets, transforms
from trainer import _CIFAR_STATS

from .baseline_cifar_configs import build_model
from .cifar_wrn_pool_after_add import WideResNetPoolAfterAddCIFAR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("checkpoint/baselines_wrn28_2_avgpool_study"))
    parser.add_argument("--data-root", type=Path, default=Path("../data"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.samples <= 10000:
        parser.error("samples must be between 1 and 10000")
    torch.set_num_threads(2)
    results = []
    for dataset_name, dataset_class, classes in (
            ("cifar10", datasets.CIFAR10, 10),
            ("cifar100", datasets.CIFAR100, 100)):
        for name in ("wrn_28_2_cifar_avgpool", "wrn_28_2_cifar_nobn_avgpool"):
            root = args.checkpoint_root / dataset_name / "custom_noresize" / name
            paths = list(root.rglob("*best_ckpt.pth"))
            if len(paths) != 1:
                raise ValueError(f"Expected exactly one best checkpoint in {root}: {paths}")
            checkpoint = torch.load(paths[0], map_location="cpu", weights_only=False)
            cfg = json.loads((root / "baseline_config.json").read_text())["cfg"]
            old = build_model(name, cfg, num_classes=classes).eval()
            new = WideResNetPoolAfterAddCIFAR(
                depth=old.depth, widen_factor=old.widen_factor,
                num_classes=classes, in_chans=old.in_chans,
                base_width=old.conv1.out_channels, use_batchnorm=old.use_batchnorm,
                conv_bias=old.conv_bias, dropout_rate=old.dropout_rate,
                final_dropout_rate=old.final_dropout_rate, init_mode=old.init_mode,
            ).eval()
            state = checkpoint["net"]
            old.load_state_dict(state, strict=True)
            new.load_state_dict(state, strict=True)
            if checkpoint.get("use_model_data_config"):
                raise ValueError("This CIFAR verifier expects use_model_data_config=False")
            mean, std = _CIFAR_STATS[dataset_name]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(checkpoint.get("timm_mean") or mean,
                                     checkpoint.get("timm_std") or std),
            ])
            data = dataset_class(args.data_root, train=False, download=False,
                                 transform=transform)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(data, range(args.samples)), batch_size=16,
                num_workers=0, shuffle=False)
            correct_old = correct_new = changed = 0
            max_error = error_sq = reference_sq = 0.0
            with torch.no_grad():
                for x, labels in loader:
                    a, b = old(x.clone()), new(x.clone())
                    torch.testing.assert_close(b, a, rtol=2e-4, atol=2e-5)
                    pa, pb = a.argmax(1), b.argmax(1)
                    changed += int((pa != pb).sum())
                    correct_old += int((pa == labels).sum())
                    correct_new += int((pb == labels).sum())
                    max_error = max(max_error, float((a - b).abs().max()))
                    error_sq += float((a.double() - b.double()).square().sum())
                    reference_sq += float(a.double().square().sum())
            row = dict(dataset=dataset_name, model=name, checkpoint=str(paths[0]),
                       samples=args.samples, strict_load=True,
                       max_abs_logit_difference=max_error,
                       relative_l2_logit_difference=(error_sq / reference_sq)**0.5,
                       prediction_changes=changed, old_correct=correct_old,
                       new_correct=correct_new, rtol=2e-4, atol=2e-5)
            results.append(row)
            print(json.dumps(row), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
