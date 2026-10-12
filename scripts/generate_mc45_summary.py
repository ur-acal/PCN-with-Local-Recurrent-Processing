#!/usr/bin/env python3
"""Generate one aggregate MC45 summary from sharded trial CSV files."""

import argparse
import csv
import statistics
from pathlib import Path


PROCESS_ORDER = ("TT", "FF", "SS", "FS", "SF")
VOLTAGE_ORDER = ("V0", "V1", "V2")
TEMPERATURE_ORDER = ("T0", "T1", "T2")
EXPECTED_CORNERS = tuple(
    "{}_{}_{}".format(process, voltage, temperature)
    for process in PROCESS_ORDER
    for voltage in VOLTAGE_ORDER
    for temperature in TEMPERATURE_ORDER
)


def load_sharded_trials(result_root):
    """Load and validate all ``shard_*/corner_trials.csv`` records."""
    result_root = Path(result_root)
    csv_paths = sorted(result_root.glob("shard_*/corner_trials.csv"))
    if not csv_paths:
        raise ValueError(
            "No shard_*/corner_trials.csv files under {}".format(result_root))

    trials = {}
    seen = set()
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"corner", "trial_index", "accuracy"}
            missing_fields = required - set(reader.fieldnames or ())
            if missing_fields:
                raise ValueError(
                    "{} is missing fields {}".format(
                        csv_path, sorted(missing_fields)))
            for row in reader:
                corner = row["corner"]
                trial_index = int(row["trial_index"])
                key = (corner, trial_index)
                if key in seen:
                    raise ValueError(
                        "Duplicate corner/trial record: {} trial {}".format(
                            corner, trial_index))
                seen.add(key)
                trials.setdefault(corner, []).append(
                    (trial_index, float(row["accuracy"])))

    expected = set(EXPECTED_CORNERS)
    actual = set(trials)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "Expected exactly 45 MC45 corners; missing={}, extra={}".format(
                missing, extra))

    trial_counts = {len(values) for values in trials.values()}
    if len(trial_counts) != 1 or not trial_counts or next(iter(trial_counts)) < 1:
        raise ValueError(
            "Every corner must have the same nonzero number of trials; "
            "found counts {}".format(sorted(trial_counts)))

    num_trials = next(iter(trial_counts))
    expected_indices = list(range(num_trials))
    for corner, values in trials.items():
        values.sort(key=lambda item: item[0])
        indices = [index for index, _ in values]
        if indices != expected_indices:
            raise ValueError(
                "{} has non-contiguous trial indices: {}".format(
                    corner, indices))
    return trials


def render_summary(trials, title=None, description=None):
    """Return an aggregate Markdown summary for validated MC45 trials."""
    num_trials = len(trials[EXPECTED_CORNERS[0]])
    means = {
        corner: statistics.mean(value for _, value in trials[corner])
        for corner in EXPECTED_CORNERS
    }
    stds = {
        corner: statistics.pstdev(value for _, value in trials[corner])
        for corner in EXPECTED_CORNERS
    }

    lowest_corner = min(EXPECTED_CORNERS, key=lambda corner: means[corner])
    highest_corner = max(EXPECTED_CORNERS, key=lambda corner: means[corner])
    individual_trials = [
        (value, corner, trial_index)
        for corner in EXPECTED_CORNERS
        for trial_index, value in trials[corner]
    ]
    lowest_trial = min(individual_trials)
    highest_trial = max(individual_trials)

    if title is None:
        title = "MC45 45-corner accuracy summary"
    if description is None:
        description = (
            "This file reports the aggregate 45-corner evaluation. Each "
            "corner reports the accuracy mean and population standard "
            "deviation across {} independently sampled hardware "
            "realizations.".format(num_trials))

    lines = [
        "# {}".format(title),
        "",
        description,
        "",
        "## Corner-level mapping",
        "",
        "| Voltage level | Voltage |",
        "|---|---:|",
        "| V0 | 0.9 VDD |",
        "| V1 | VDD |",
        "| V2 | 1.1 VDD |",
        "",
        "| Temperature level | Temperature |",
        "|---|---:|",
        "| T0 | -20 °C |",
        "| T1 | 25 °C |",
        "| T2 | 85 °C |",
        "",
        "## All 45 corners",
        "",
        "All standard deviations use the population definition.",
        "",
        "| Statistic over all 45 corners | Result |",
        "|---|---:|",
        "| Lowest corner mean | **{}: {:.4f}% ± {:.4f}%** |".format(
            lowest_corner, means[lowest_corner], stds[lowest_corner]),
        "| Highest corner mean | **{}: {:.4f}% ± {:.4f}%** |".format(
            highest_corner, means[highest_corner], stds[highest_corner]),
        "| Lowest single-trial accuracy | **{}, trial {}: {:.4f}%** |".format(
            lowest_trial[1], lowest_trial[2], lowest_trial[0]),
        "| Highest single-trial accuracy | **{}, trial {}: {:.4f}%** |".format(
            highest_trial[1], highest_trial[2], highest_trial[0]),
        "| Mean of corner means | **{:.4f}%** |".format(
            statistics.mean(means.values())),
        "| Std of corner means | **{:.4f}%** |".format(
            statistics.pstdev(means.values())),
        "",
        "| Corner | Mean (%) | Std (%) |",
        "|---|---:|---:|",
    ]
    lines.extend(
        "| {} | {:.4f} | {:.4f} |".format(
            corner, means[corner], stds[corner])
        for corner in EXPECTED_CORNERS
    )
    return "\n".join(lines) + "\n"


def generate_summary(result_root, output_path=None, title=None,
                     description=None):
    """Generate and write an aggregate summary; return its output path."""
    result_root = Path(result_root)
    if output_path is None:
        output_path = result_root / "summary.md"
    else:
        output_path = Path(output_path)
    trials = load_sharded_trials(result_root)
    contents = render_summary(trials, title=title, description=description)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(contents, encoding="utf-8")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_root", required=True,
        help="Directory containing shard_*/corner_trials.csv files")
    parser.add_argument(
        "--output",
        help="Output Markdown path (default: RESULT_ROOT/summary.md)")
    parser.add_argument("--title", help="Optional Markdown document title")
    parser.add_argument(
        "--description", help="Optional introductory description")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = generate_summary(
        args.result_root, output_path=args.output, title=args.title,
        description=args.description)
    print(output_path)


if __name__ == "__main__":
    main()
