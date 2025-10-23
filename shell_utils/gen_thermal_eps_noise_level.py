#!/usr/bin/env python3
import os, re, sys, csv
from collections import OrderedDict

EPS_RE = re.compile(r'^\s*Thermal noise eps:\s*(\S+)\s*$')
ACC_RE = re.compile(r'Noise level:\s*([0-9.]+)\s*,\s*Acc:\s*([0-9.]+)(?:±([0-9.]+))?%')

def parse_log(path: str):
    table = OrderedDict()   # eps -> { noise_level_float -> "mean%"/"mean±std%" }
    eps_order = []
    noise_levels = set()
    cur_eps = None

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m_eps = EPS_RE.search(line)
            if m_eps:
                cur_eps = m_eps.group(1).strip()
                if cur_eps not in table:
                    table[cur_eps] = {}
                    eps_order.append(cur_eps)
                continue

            m_acc = ACC_RE.search(line)
            if m_acc and cur_eps is not None:
                nl = float(m_acc.group(1))
                mean = m_acc.group(2)
                std  = m_acc.group(3)
                acc_str = f"{mean}±{std}%" if std is not None else f"{mean}%"
                table[cur_eps][nl] = acc_str
                noise_levels.add(nl)

    return sorted(noise_levels), eps_order, table

def write_csv(log_path: str, noise_levels, eps_order, table):
    here   = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "parse_res", "multi_thermal_eps")
    os.makedirs(outdir, exist_ok=True)

    # basename with .csv (e.g., xxx.log -> xxx.csv)
    base   = os.path.splitext(os.path.basename(log_path))[0] + ".csv"
    out_fp = os.path.join(outdir, base)

    with open(out_fp, 'w', newline='', encoding='utf-8') as fp:
        w = csv.writer(fp)
        w.writerow(["Noise level"] + eps_order)
        for nl in noise_levels:
            row = [nl] + [table.get(eps, {}).get(nl, "") for eps in eps_order]
            w.writerow(row)

    print(f"Wrote {out_fp}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_by_eps.py /path/to/xxx.log", file=sys.stderr)
        sys.exit(1)
    log_path = sys.argv[1]
    noise_levels, eps_order, table = parse_log(log_path)
    write_csv(log_path, noise_levels, eps_order, table)

if __name__ == "__main__":
    main()
