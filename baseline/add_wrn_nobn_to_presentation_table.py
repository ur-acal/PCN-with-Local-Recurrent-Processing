#!/usr/bin/env python3
"""Add matched BN-free WRN results to an existing presentation table."""
import argparse
import csv
import re
from pathlib import Path

HEADING = re.compile(r"^## (cifar10|cifar100) (WRN_\d+_\d+) (additive|multiplicative)$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--markdown", required=True)
    p.add_argument("--presentation_csv", required=True)
    p.add_argument("--bn_free_aggregate_csv", required=True)
    p.add_argument("--output_markdown", default=None)
    p.add_argument("--output_csv", default=None)
    return p.parse_args()


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def key(row):
    return row["dataset"], row["architecture"], row["mismatch_type"], float(row["mismatch_level"])


def add_markdown_column(text, by_key):
    lines = text.splitlines()
    notation = [
        "",
        "WRN BN-free acc: Train the matched WRN topology without BatchNorm;",
        "apply mismatch to Conv/Linear parameters while excluding Conv biases.",
    ]
    if not any("WRN BN-free acc:" in line for line in lines):
        borders = [i for i, line in enumerate(lines) if line == "****************************************************************************"]
        if not borders:
            raise ValueError("Presentation notation border not found")
        border = borders[1] if len(borders) > 1 else borders[0]
        lines[border:border] = notation

    current = None
    used = set()
    for index, line in enumerate(lines):
        match = HEADING.match(line)
        if match:
            current = match.groups()
            continue
        if current is None or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if "WRN folded recal-BN acc" in cells:
            if "WRN BN-free acc" not in cells:
                insert_at = cells.index("folded BN recovery")
                cells.insert(insert_at, "WRN BN-free acc")
                lines[index] = "| " + " | ".join(cells) + " |"
            continue
        if cells and all(set(cell) <= {"-", ":"} for cell in cells):
            header = [cell.strip() for cell in lines[index - 1].strip("|").split("|")]
            if len(cells) < len(header):
                cells.insert(header.index("WRN BN-free acc"), "---:")
                lines[index] = "|" + "|".join(cells) + "|"
            continue
        try:
            level = float(cells[0])
        except (ValueError, IndexError):
            continue
        row_key = (*current, level)
        if row_key not in by_key:
            raise ValueError(f"Missing BN-free aggregate row for {row_key}")
        header = [cell.strip() for cell in lines[index - 2].strip("|").split("|")]
        insert_at = header.index("WRN BN-free acc")
        if len(cells) < len(header):
            cells.insert(insert_at, f"{float(by_key[row_key]['wrn_nobn_mean_accuracy']):.2f}")
            lines[index] = "| " + " | ".join(cells) + " |"
        used.add(row_key)

    expected = set(by_key)
    if used != expected:
        missing = sorted(expected - used)
        raise ValueError(f"BN-free rows not represented in markdown: {missing[:5]}")
    return "\n".join(lines) + "\n"


def add_csv_columns(rows, by_key):
    output = []
    for row in rows:
        bn_free = by_key[key(row)]
        merged = dict(row)
        merged["wrn_nobn_mean_accuracy"] = bn_free["wrn_nobn_mean_accuracy"]
        merged["wrn_nobn_std_accuracy"] = bn_free["wrn_nobn_std_accuracy"]
        merged["wrn_nobn_parameter_count"] = bn_free["parameter_count"]
        output.append(merged)
    return output


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    aggregate_rows = read_csv(args.bn_free_aggregate_csv)
    by_key = {key(row): row for row in aggregate_rows}
    markdown_path = Path(args.markdown)
    output_markdown = Path(args.output_markdown or args.markdown)
    output_csv = Path(args.output_csv or args.presentation_csv)
    output_markdown.write_text(add_markdown_column(markdown_path.read_text(), by_key))
    write_csv(output_csv, add_csv_columns(read_csv(args.presentation_csv), by_key))
    print(f"Wrote {output_markdown}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    run(parse_args())
