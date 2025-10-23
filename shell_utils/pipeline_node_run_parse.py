#!/usr/bin/env python3
import os, re, sys, csv
from collections import OrderedDict

# Old patterns (kept)
MODEL_LINE_RE = re.compile(r'Model name:\s*(.+?)\s*-{2,}\s*$')

# New: accept optional "±std" after accuracy
# e.g., "Acc:80.94±0.08%" or "Acc:79.82%"
ACC_LINE_RE   = re.compile(r'Noise level:\s*([0-9.]+)\s*,\s*Acc:\s*([0-9.]+)(?:±([0-9.]+))?%')

# New: optional section header
THERMAL_RE    = re.compile(r'^\s*Thermal noise eps:\s*(\S+)\s*$')

def parse_log(path: str):
    """
    Returns:
      noise_levels: sorted list of floats
      data: OrderedDict[str, dict[float, str]]  # column_key -> noise_level -> "Acc%" (std ignored for CSV)
    """
    data = OrderedDict()
    current_model = None
    current_eps   = None

    def current_key():
        # Prefer most specific label first
        if current_model and current_eps is not None:
            return f"{current_model} | eps={current_eps}"
        if current_model:
            return current_model
        if current_eps is not None:
            return f"eps={current_eps}"
        return "default"

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Section headers
            m_model = MODEL_LINE_RE.search(line)
            if m_model:
                current_model = m_model.group(1).strip()
                # data.setdefault(current_key(), {})
                continue

            m_eps = THERMAL_RE.search(line)
            if m_eps:
                # handle "None" literally too
                eps_txt = m_eps.group(1).strip()
                current_eps = eps_txt
                # data.setdefault(current_key(), {})
                continue

            # Acc lines
            a = ACC_LINE_RE.search(line)
            if a:
                nl  = float(a.group(1))
                mean = a.group(2)
                std = a.group(3)
                acc = f"{mean}±{std}%" if std is not None else f"{mean}%"
                key = current_key()
                data.setdefault(key, {})
                data[current_key()][nl] = acc

    noise_levels = sorted({nl for d in data.values() for nl in d.keys()})
    return noise_levels, data

def write_csv(log_path: str, noise_levels, data):
    here   = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "parse_res", "neural_ode_res")
    os.makedirs(outdir, exist_ok=True)

    exp_name = os.path.basename(os.path.dirname(os.path.abspath(log_path)))
    out_csv  = os.path.join(outdir, f"{exp_name}.csv")

    col_names = list(data.keys())  # preserve first-seen order

    with open(out_csv, 'w', newline='', encoding='utf-8') as fp:
        w = csv.writer(fp)
        w.writerow(["Noise"] + col_names)
        for nl in noise_levels:
            row = [nl]
            for c in col_names:
                row.append(data.get(c, {}).get(nl, ""))
            w.writerow(row)

    print(f"Wrote {out_csv}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_eval_log.py /path/to/merged.log", file=sys.stderr)
        sys.exit(1)
    log_path = sys.argv[1]
    noise_levels, data = parse_log(log_path)
    write_csv(log_path, noise_levels, data)

if __name__ == "__main__":
    main()
