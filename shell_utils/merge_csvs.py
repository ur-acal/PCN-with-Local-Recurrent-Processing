#!/usr/bin/env python3
import sys, os, csv

def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            return [], []
        rows = [row for row in r]
    return header, rows

def main():
    if len(sys.argv) < 3:
        print("Usage: merge_csvs.py <OUTPUT_CSV> <INPUT1.csv> [INPUT2.csv ...]", file=sys.stderr)
        sys.exit(2)

    out_path = sys.argv[1]
    in_paths = [p for p in sys.argv[2:] if os.path.isfile(p)]
    for p in sys.argv[2:]:
        if p not in in_paths:
            print(f"[WARN] Missing input CSV skipped: {p}", file=sys.stderr)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not in_paths:
        open(out_path, "w").close()
        print(f"Wrote {out_path}")
        return

    # Baseline from the first non-empty CSV
    base_hdr, base_rows, base_path = [], [], None
    for p in in_paths:
        h, r = read_csv(p)
        if h and r:
            base_hdr, base_rows, base_path = h, r, p
            break

    if not base_hdr:
        open(out_path, "w").close()
        print(f"Wrote {out_path}")
        return

    if base_hdr[0] != "Noise":
        print(f"[ERROR] First column in baseline '{base_path}' is {base_hdr[0]!r}, expected 'Noise'", file=sys.stderr)
        sys.exit(1)

    # Prepare output header: Noise + baseline non-Noise columns
    out_header = ["Noise"] + base_hdr[1:]
    base_noise = [row[0] for row in base_rows]
    nrows = len(base_rows)

    # Map Noise -> row (start with baseline non-Noise part)
    merged_rows = { base_rows[i][0]: base_rows[i][1:] for i in range(nrows) }

    # Append columns from the remaining files
    for p in in_paths:
        if p == base_path:
            continue
        hdr, rows = read_csv(p)
        if not hdr:
            continue
        if hdr[0] != "Noise":
            print(f"[ERROR] First column in '{p}' is {hdr[0]!r}, expected 'Noise'", file=sys.stderr)
            sys.exit(1)
        if len(rows) != nrows:
            print(f"[ERROR] Row count mismatch in '{p}': expected {nrows}, found {len(rows)}", file=sys.stderr)
            sys.exit(1)
        noise = [row[0] for row in rows]
        if noise != base_noise:
            print(f"[ERROR] Noise values differ between '{p}' and baseline '{base_path}'", file=sys.stderr)
            sys.exit(1)
        # extend header with this file's non-Noise columns
        out_header.extend(hdr[1:])
        # append this file's non-Noise cells to each row by Noise key
        for i in range(nrows):
            key = noise[i]
            merged_rows[key].extend(rows[i][1:])

    # Write output: keep baseline Noise order
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(out_header)
        for i in range(nrows):
            key = base_noise[i]
            w.writerow([key] + merged_rows[key])

    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
