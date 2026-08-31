import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baseline import generate_finaldrop_wrn_report as report


class FinalDropoutReportTests(unittest.TestCase):
    def _write_csv(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_report_joins_wrn_only_outputs_without_replacing_pcn_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.csv"
            mismatch_root = root / "mismatch"
            output = root / "report"
            base_rows = []

            for dataset, architecture in report.PAIRS:
                for mismatch_type in ("additive_max", "additive_rms", "multiplicative"):
                    base_rows.append(
                        {
                            "dataset": dataset,
                            "architecture": architecture,
                            "mismatch_type": mismatch_type,
                            "level": "0",
                            "PCN/WRN trials": "10/10",
                            "PCNetNoBatchNorm": "77.77",
                            "WRN folded frozen-BN": "70.00",
                        }
                    )
                for family, raw_types, scale in (
                    ("max_mul", ("additive", "multiplicative"), "max_abs"),
                    ("rms", ("additive",), "rms"),
                ):
                    for mode in ("folded", "unfolded"):
                        rows = []
                        for raw_type in raw_types:
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "architecture": architecture,
                                    "mismatch_type": raw_type,
                                    "mismatch_level": "0",
                                    "additive_scale_mode": scale,
                                    "num_trials": "10",
                                    "pcn_node_mean_accuracy": "",
                                    "paired_wrn_frozen_bn_mean_accuracy": "81.25",
                                    "wrn_recalibrated_bn_mean_accuracy": "82.50",
                                }
                            )
                        self._write_csv(
                            mismatch_root / family / mode / dataset / architecture / "full_aggregate.csv",
                            rows,
                        )
            self._write_csv(base_path, base_rows)

            argv = [
                "generate_finaldrop_wrn_report.py",
                "--base_csv",
                str(base_path),
                "--mismatch_root",
                str(mismatch_root),
                "--output_dir",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                report.main()

            generated = report.load_csv(output / "full_accuracy_comparison.csv")
            self.assertEqual(len(generated), len(base_rows))
            self.assertTrue(all(row["PCNetNoBatchNorm"] == "77.77" for row in generated))
            evaluated = [row for row in generated if row["mismatch_type"] != "additive_rms"]
            omitted = [row for row in generated if row["mismatch_type"] == "additive_rms"]
            self.assertTrue(all(row[report.NEW_COLUMNS[0]] == "81.25" for row in evaluated))
            self.assertTrue(all(row[report.NEW_COLUMNS[1]] == "82.50" for row in evaluated))
            self.assertTrue(all(row[report.NEW_COLUMNS[2]] == "82.50" for row in evaluated))
            self.assertTrue(all(row[report.NEW_COLUMNS[0]] == "N/A" for row in omitted))
            self.assertTrue(all(row[report.NEW_COLUMNS[1]] == "N/A" for row in omitted))
            self.assertTrue(all(row[report.NEW_COLUMNS[2]] == "N/A" for row in omitted))

    def test_rejects_wrn_output_containing_pcn_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset, architecture in report.PAIRS:
                for family, raw_types, scale in (
                    ("max_mul", ("additive", "multiplicative"), "max_abs"),
                    ("rms", ("additive",), "rms"),
                ):
                    for mode in ("folded", "unfolded"):
                        rows = [
                            {
                                "dataset": dataset,
                                "architecture": architecture,
                                "mismatch_type": raw_type,
                                "mismatch_level": "0",
                                "additive_scale_mode": scale,
                                "pcn_node_mean_accuracy": "99.0" if (
                                    dataset == "cifar10"
                                    and architecture == "WRN_16_2"
                                    and family == "max_mul"
                                    and mode == "folded"
                                    and raw_type == "additive"
                                ) else "",
                            }
                            for raw_type in raw_types
                        ]
                        self._write_csv(
                            root / family / mode / dataset / architecture / "full_aggregate.csv",
                            rows,
                        )
            with self.assertRaisesRegex(ValueError, "PCN reference"):
                report.load_result_index(root)


if __name__ == "__main__":
    unittest.main()

