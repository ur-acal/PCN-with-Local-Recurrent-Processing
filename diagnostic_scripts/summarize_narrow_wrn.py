#!/usr/bin/env python3
"""Export reproducible digital RGB and historical RGGB narrow-WRN summaries."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torchinfo import summary
from baseline.cifar_resnet import wrn_flexible_cifar_avgpool_shortcut


def export_summaries(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for classes in (10, 100):
        sections = [
            f"Narrow WRN-28 c27: CIFAR-{classes} / ciFAIR-{classes}\n",
            "Registered model: wrn_flexible_cifar_avgpool_shortcut\n"
            "Stem: 16; residual stages: 27/54/108; blocks per stage: 4/4/4.\n"
            "Main downsampling: stride-2 convolution; shortcuts: channel zero-padding, with AvgPool at downsampling.\n"
            "BatchNorm enabled; convolution biases disabled; final Linear bias enabled.\n"
            "Intermediate ReLU6; final ideal ReLU; final dropout 0.25 (inactive at eval).\n"
            "Unconverted digital model: FP32; no hardware scaling, QAT or distillation connector.\n"
            "Parameter storage below excludes BN running buffers, activations, optimizer and serialization overhead.\n"
            "torchinfo multiply-adds use one input image; they are not a measured latency.\n",
        ]
        for channels, size, label in ((3, 32, "Digital sRGB baseline"),
                                      (4, 16, "Historical packed-RGGB hardware-study architecture")):
            torch.manual_seed(4096)
            model = wrn_flexible_cifar_avgpool_shortcut(
                depth=28, first_stage_channels=27, num_classes=classes,
                in_chans=channels, final_dropout_rate=0.25).eval()
            params = sum(p.numel() for p in model.parameters())
            parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
            buffer_bytes = sum(b.numel()*b.element_size() for b in model.buffers())
            report = summary(model, input_size=(1, channels, size, size),
                             device="cpu", mode="eval", depth=5, verbose=0,
                             col_names=("input_size", "output_size", "num_params", "mult_adds"))
            assert report.total_params == params
            sections.append(
                f"\n{'='*80}\n{label}\n"
                f"Input: (1, {channels}, {size}, {size})\n"
                f"Total parameters: {params:,}\n"
                f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
                f"FP32 parameter bytes: {parameter_bytes:,} ({parameter_bytes/2**20:.6f} MiB)\n"
                f"Persistent buffer bytes: {buffer_bytes:,}\n"
                f"FP32 parameters + buffers: {(parameter_bytes+buffer_bytes)/2**20:.6f} MiB\n\n"
                f"{report}\n")
        path = output_dir / f"wrn28_c27_cifair{classes}_torchinfo.txt"
        path.write_text("\n".join(sections))
        outputs.append(path)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/narrow_wrn_model_summaries")
    args = parser.parse_args()
    torch.set_num_threads(2)
    for path in export_summaries(args.output_dir):
        print(path)
