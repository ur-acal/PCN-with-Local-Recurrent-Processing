#!/usr/bin/env python3
"""Generate fixed-MC18 zero-offset activation tables for diagnostics."""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measured_activation import CubicBSplineActivation


def _write_curve(path, vin, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Vin", "Vout_MC18"))
        writer.writerows(zip(vin.tolist(), values.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    vin, curves, _ = CubicBSplineActivation._load_csv(
        args.source, adapt_relu_offset=True)
    values = curves["MC18"]
    zero_positions = (vin == 0).nonzero(as_tuple=False).flatten()
    if zero_positions.numel() != 1:
        raise ValueError(
            "Expected exactly one Vin=0 sample, found {}."
            .format(zero_positions.numel()))
    zero_value = values[zero_positions.item()]
    shifted = values - zero_value

    output_dir = Path(args.output_dir)
    _write_curve(output_dir / "mc18_zero_offset.csv", vin, shifted)
    _write_curve(
        output_dir / "mc18_zero_offset_relu.csv", vin, shifted.clamp_min(0))
    print("MC18 centered value at Vin=0: {:.10g} V".format(zero_value.item()))


if __name__ == "__main__":
    main()
