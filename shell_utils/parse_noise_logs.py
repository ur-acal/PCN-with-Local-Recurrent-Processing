"""
parse_noise_logs.py  < master_log  > acc_vs_noise.csv
log_dir ../logs/noisy_test/master...log

In local terminal, run
scp rongzeng@100.97.246.53:/home/rongzeng/_workspce_old/repos/pcn/PCN-with-Local-Recurrent-Processing/shell_utils/parse_res/....csv .
"""

import re, sys, csv
from collections import OrderedDict, defaultdict

model_order          = []                  # preserve appearance order
acc_for_noise_model  = defaultdict(dict)   # {noise : {model : acc}}

re_model = re.compile(r"^=+ (.+?) =+$")
re_data = re.compile(r"Noise level:\s*([0-9.]+),\s*Acc:([0-9.]+%)")

current_model = None

for line in sys.stdin:
    m = re_model.match(line)
    if m:                                  # ===== Model name =====
        current_model = m.group(1).strip()
        if current_model not in model_order:
            model_order.append(current_model)
        continue

    d = re_data.match(line)                # Noise level: x, Acc:y%
    if d and current_model:
        noise = float(d.group(1))
        acc = d.group(2)
        acc_for_noise_model[noise][current_model] = acc

# ---------- write CSV ----------
writer = csv.writer(sys.stdout)
writer.writerow(["noise_level"] + model_order)

for noise in sorted(acc_for_noise_model):
    row = [noise] + [acc_for_noise_model[noise].get(m, "") for m in model_order]
    writer.writerow(row)
