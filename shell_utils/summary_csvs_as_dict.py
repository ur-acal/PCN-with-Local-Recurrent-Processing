#!/usr/bin/env python3
import argparse
import csv
import os
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Tuple


def _parse_channels(model_name: str) -> int:
    # "..._3K1S112C_..." -> 112 (avoid confusion with "_C100_" classes)
    m = re.search(r"S(\d+)C", model_name)
    if not m:
        raise ValueError(f"Cannot parse channels from model name: {model_name}")
    return int(m.group(1))


def _parse_layers_and_alblc(model_name: str) -> Tuple[int, str]:
    # "..._10Layers2l1l4_..." -> (10, "2l1l4")
    m = re.search(r"(\d+)Layers([^_|\s]+)", model_name)
    if not m:
        raise ValueError(f"Cannot parse layers/alblc from model name: {model_name}")
    layers = int(m.group(1))
    alblc = m.group(2)
    return layers, alblc


def _parse_acc_float(acc_str: str) -> float:
    # "12.79±0.04%" or "12.79%" -> 12.79
    s = acc_str.strip()
    m = re.match(r"([-+]?\d+(?:\.\d+)?)", s)
    if not m:
        raise ValueError(f"Cannot parse accuracy from: {acc_str}")
    return float(m.group(1))


def _prefix_before_pcn(model_name: str) -> str:
    """
    Assumption (per your instruction):
      - Split by "_"
      - The FIRST token that starts with "PCNet" is the PCN anchor
      - Prefix = everything before that token (joined by "_")
      - If prefix contains "QAT" or "NT", return prefix; else return "".

    Examples:
      "PCNetNoBatchNorm_..." -> prefix "" (no QAT/NT)
      "QAT5bNT0p15mulPCNetNoBatchNorm_..." -> prefix "QAT5bNT0p15mul"
      "QATxxx_PCNetNoBatchNorm_..." -> prefix "QATxxx"
      "NTxxx_PCNetNoBatchNorm_..." -> prefix "NTxxx"
    """
    toks = model_name.split("_")
    # Find the first token that begins with "PCNet"; don't loop over all toks.
    # If toks[0] starts with PCNet -> no prefix.
    if toks and toks[0].startswith("PCNet"):
        return ""

    # Otherwise, assume toks[0] is the prefix and the PCN token is embedded after it
    # OR appears as toks[1]. For safety, we define prefix as all tokens before the first
    # token that startswith "PCNet", but we avoid looping by handling common patterns:
    #
    # - "QAT...PCNet..." (no underscore before PCNet): toks[0] contains PCNet => split there
    # - "QAT..._PCNet..." : toks[1] startswith PCNet => prefix is toks[0]
    #
    # If neither matches, fall back to toks[0] as prefix.
    if toks:
        if "PCNet" in toks[0] and not toks[0].startswith("PCNet"):
            # e.g. "QAT5bNT0p15mulPCNetNoBatchNorm"
            prefix = toks[0].split("PCNet", 1)[0]
        elif len(toks) >= 2 and toks[1].startswith("PCNet"):
            # e.g. "QAT5bNT0p15mul_PCNetNoBatchNorm"
            prefix = toks[0]
        else:
            prefix = toks[0]

        prefix = prefix.strip("_")
        if ("QAT" in prefix) or ("NT" in prefix):
            return prefix
    return ""


def summary_csvs_to_dict(csv_paths: List[str]) -> Dict[float, Dict[int, Dict[int, List[Tuple[str, str]]]]]:
    """
    Returns:
      d[noise_level][channels][layers] -> list of (label, acc_str), sorted by acc desc

    label is:
      - "alblc" if no QAT/NT prefix
      - "{prefix}_{alblc}" if prefix (before PCNet...) contains QAT/NT
    """
    d = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for path in csv_paths:
        if not os.path.isfile(path):
            continue

        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            continue

        header = rows[0]
        if len(header) < 2 or header[0].strip() != "Noise":
            raise ValueError(f"Unexpected header in {path}: first column must be 'Noise'")

        model_names = header[1:]

        meta = []
        for name in model_names:
            ch = _parse_channels(name)
            layers, alblc = _parse_layers_and_alblc(name)
            prefix = _prefix_before_pcn(name)
            label = f"{prefix}_{alblc}" if prefix else alblc
            meta.append((ch, layers, label))

        for r in rows[1:]:
            if not r or len(r) < 2:
                continue
            nl_str = r[0].strip()
            if nl_str == "":
                continue
            nl = float(nl_str)

            for col_idx, (ch, layers, label) in enumerate(meta, start=1):
                if col_idx >= len(r):
                    continue
                cell = r[col_idx].strip()
                if cell == "":
                    continue
                # keep original string with ± and %
                d[nl][ch][layers].append((label, cell))

    # convert to plain dicts + sort lists by parsed float accuracy DESC
    out: Dict[float, Dict[int, Dict[int, List[Tuple[str, str]]]]] = {}
    for nl, dd1 in d.items():
        out[nl] = {}
        for ch, dd2 in dd1.items():
            out[nl][ch] = {}
            for layers, lst in dd2.items():
                lst_sorted = sorted(lst, key=lambda t: _parse_acc_float(t[1]), reverse=True)
                out[nl][ch][layers] = lst_sorted

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="output pickle path")
    ap.add_argument("csvs", nargs="+", help="input merged CSV paths")
    args = ap.parse_args()

    if args.out:
        out_pkl = args.out
    else:
        first = os.path.abspath(args.csvs[0])
        out_pkl = os.path.join(os.path.dirname(first), "summary_dict.pkl")

    d = summary_csvs_to_dict(args.csvs)

    os.makedirs(os.path.dirname(os.path.abspath(out_pkl)), exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Wrote pickle: {out_pkl}")
    print(f"[INFO] noise levels: {len(d)}")


if __name__ == "__main__":
    main()
