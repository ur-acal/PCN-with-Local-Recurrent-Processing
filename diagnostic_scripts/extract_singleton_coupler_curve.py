#!/usr/bin/env python3
"""Extract one X/Y realization from a paired-column coupler CSV bank."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--curve-index", type=int, required=True,
                        help="Zero-based X/Y pair index.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--copies", type=int, default=1,
        help="Repeat the selected X/Y pair this many times. Use two when a "
             "training sampler requires distinct FF/FB indices; the curves "
             "remain numerically identical.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.curve_index < 0:
        parser.error("--curve-index must be nonnegative")
    if args.copies < 1:
        parser.error("--copies must be positive")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}")

    with args.source.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or len(rows[0]) % 2:
        parser.error("source must contain paired X/Y columns")

    pair_count = len(rows[0]) // 2
    if args.curve_index >= pair_count:
        parser.error(
            f"curve index {args.curve_index} is outside 0..{pair_count - 1}")

    columns = (2 * args.curve_index, 2 * args.curve_index + 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for row_number, row in enumerate(rows, start=1):
            if len(row) <= columns[1]:
                parser.error(
                    f"row {row_number} has {len(row)} columns; "
                    f"expected at least {columns[1] + 1}")
            pair = (row[columns[0]], row[columns[1]])
            writer.writerow(pair * args.copies)

    print(
        f"wrote {args.output}: source={args.source}, "
        f"curve_index={args.curve_index}, mcparamset={args.curve_index + 1}, "
        f"copies={args.copies}")


if __name__ == "__main__":
    main()
