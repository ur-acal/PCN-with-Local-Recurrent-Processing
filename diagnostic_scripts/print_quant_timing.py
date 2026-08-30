#!/usr/bin/env python3
"""Print per-layer quantization factors and direct-mode pulse timings."""

import argparse
import csv
from pathlib import Path

from diagnostic_config import (
    C_FARAD,
    N_CYCLES,
    R_OHM,
    T_END,
    WEIGHT_QUANT_FACTOR_BITS,
    add_model_arguments,
    layer_indices,
    load_model_state,
)


def floating_weight_max(full_state, layer, stage):
    prefix = "PcConvs.{}.{}conv".format(layer, stage)
    direct_key = prefix + ".weight"
    parametrized_key = prefix + ".parametrizations.weight.original"
    key = parametrized_key if parametrized_key in full_state else direct_key
    return float(full_state[key].abs().max())


def timing_rows(model_name=None, model_root=None):
    quantized, paths = load_model_state(
        model_name=model_name, model_root=model_root, full_param=False)
    full_state, _ = load_model_state(
        model_name=model_name, model_root=model_root, full_param=True)
    rows = []
    for layer in layer_indices(quantized):
        s_ff = float(quantized["PcConvs.{}.s_ff".format(layer)])
        s_fb = float(quantized["PcConvs.{}.s_fb".format(layer)])
        m_ff_q = 1.0 / s_ff
        m_fb_q = 1.0 / s_fb
        t_z_cycle = R_OHM * C_FARAD / s_fb
        t_y_cycle = (T_END / N_CYCLES) * R_OHM * C_FARAD / s_ff
        rows.append({
            "layer": layer + 1,
            "m_ff": floating_weight_max(full_state, layer, "FF"),
            "m_ff_q": m_ff_q,
            "s_ff": s_ff,
            "m_fb": floating_weight_max(full_state, layer, "FB"),
            "m_fb_q": m_fb_q,
            "s_fb": s_fb,
            "z_cycle_ns": t_z_cycle * 1e9,
            "y_cycle_ns": t_y_cycle * 1e9,
            "z_total_ns": N_CYCLES * t_z_cycle * 1e9,
            "y_total_ns": N_CYCLES * t_y_cycle * 1e9,
            "layer_total_ns": N_CYCLES * (t_z_cycle + t_y_cycle) * 1e9,
        })
    return rows, paths


def print_table(rows, paths):
    print("Task: {}".format(paths["task"]))
    print("Model: {}".format(paths["model_name"]))
    print("Quantized checkpoint: {}".format(paths["quantized"]))
    print("R = {:.6g} ohm, C = {:.6g} F, RC = {:.6f} ns".format(
        R_OHM, C_FARAD, R_OHM * C_FARAD * 1e9))
    print("Direct scaling, t_end = {}, cycles = {}, weight_quant_factor_bits = {}".format(
        T_END, N_CYCLES, WEIGHT_QUANT_FACTOR_BITS))
    print("Timing uses the s_ff/s_fb buffers stored in the quantized inference checkpoint; m^Q = 1/s.")
    print()
    header = (
        "Layer", "m_ff", "m_ff^Q", "s_ff", "m_fb", "m_fb^Q", "s_fb",
        "z/cycle ns", "y/cycle ns", "z total ns", "y total ns", "total ns")
    widths = (5, 9, 9, 9, 9, 9, 9, 11, 11, 11, 11, 11)
    print(" ".join("{:>{}}".format(value, width)
                   for value, width in zip(header, widths)))
    for row in rows:
        values = (
            row["layer"], row["m_ff"], row["m_ff_q"], row["s_ff"],
            row["m_fb"], row["m_fb_q"], row["s_fb"], row["z_cycle_ns"],
            row["y_cycle_ns"], row["z_total_ns"], row["y_total_ns"],
            row["layer_total_ns"])
        formatted = ("{}",) + ("{:.6f}",) * 6 + ("{:.3f}",) * 5
        print(" ".join("{:>{}}".format(fmt.format(value), width)
                       for value, fmt, width in zip(values, formatted, widths)))
    print()
    print("All layers: z = {:.3f} ns, y = {:.3f} ns, total = {:.3f} ns".format(
        sum(row["z_total_ns"] for row in rows),
        sum(row["y_total_ns"] for row in rows),
        sum(row["layer_total_ns"] for row in rows)))


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("CSV: {}".format(path))


def main():
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--csv", default=None, help="Optional output CSV path.")
    args = parser.parse_args()
    rows, paths = timing_rows(args.model_name, args.model_dir)
    print_table(rows, paths)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()

