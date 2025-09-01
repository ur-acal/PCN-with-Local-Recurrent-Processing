#!/usr/bin/env python3
"""
Usage:
  python -u shell_utils/merge_csvs.py <OUTPUT_CSV> <INPUT1.csv> [INPUT2.csv ...]
Simply concatenates input CSVs horizontally on the 'Noise' column.
Assumes every input has the same 'Noise' values in the same order.
"""

import sys, os
import pandas as pd

def main():
    if len(sys.argv) < 3:
        print("Usage: merge_csvs.py <OUTPUT_CSV> <INPUT1.csv> [INPUT2.csv ...]", file=sys.stderr)
        sys.exit(2)

    out_path = sys.argv[1]
    in_paths = sys.argv[2:]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Read each CSV and index by 'Noise'
    frames = []
    for p in in_paths:
        df = pd.read_csv(p)
        df = df.set_index("Noise")
        frames.append(df)

    # Horizontal concat; since 'Noise' is identical across inputs,
    # default outer join equals inner join here.
    merged = pd.concat(frames, axis=1)

    # Restore 'Noise' column and write
    merged.reset_index().to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
