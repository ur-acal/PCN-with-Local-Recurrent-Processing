import json
import logging
import os
import sys
import time

import torch

ROOT = "/home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN"
sys.path[:0] = ["/tmp", ROOT]
os.chdir(ROOT)
logging.disable(logging.WARNING)

import switch
from debug_strang_real_block import build
from diagnose_switch_yoshida4 import measured_forward, rel_l2, set_tolerance
from inference_utils import get_test_data
from ode_pc import ODEXInitFFFB
from switch import ODEXInitFFFBPixelSwitchYoshida4


@torch.no_grad()
def main():
    torch.manual_seed(4096)
    x, _ = next(iter(get_test_data(
        test_bs=1, img_type="scanGFI", task="cifar100")))
    x = x.cuda()

    ref_net, _ = build(ODEXInitFFFB, 1)
    set_tolerance(ref_net.PcConvs[0], 1e-6)
    reference, _, _, _ = measured_forward(ref_net, x, __import__("ode_pc"))

    net, _ = build(ODEXInitFFFBPixelSwitchYoshida4, 1)
    block = net.PcConvs[0]
    result = {
        "dtype": str(x.dtype),
        "reference_tolerance": 1e-6,
        "float32_epsilon": torch.finfo(x.dtype).eps,
        "cases": [],
    }
    for n_iters in (5, 10):
        block.n_iters = n_iters
        for tol in (1e-9, 1e-10, 1e-11):
            set_tolerance(block, tol)
            value, counts, runtime, negative_max = measured_forward(
                net, x, switch, track_negative=True)
            row = {
                "n_iters": n_iters,
                "requested_tol": tol,
                "rel_l2": rel_l2(value, reference),
                "solves": counts["solves"],
                "nfe": counts["nfe"],
                "runtime_s": runtime,
                "negative_stage_max_abs": negative_max,
                "finite": bool(torch.isfinite(value).all()),
            }
            result["cases"].append(row)
            print(json.dumps(row), flush=True)

    path = os.path.join(ROOT, "results/switch_yoshida4_tight_tolerance.json")
    with open(path, "w") as fp:
        json.dump(result, fp, indent=2)
    print("RESULT_JSON=" + path, flush=True)


if __name__ == "__main__":
    main()

