import json
import logging
import os
import sys
import time

import torch

ROOT = "/home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN"
sys.path.insert(0, "/tmp")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

logging.disable(logging.WARNING)

import ode_pc
import switch
from debug_strang_real_block import build
from inference_utils import get_test_data
from ode_pc import ODEXInitFFFB
from switch import (
    ODEXInitFFFBPixelSwitchEfficient,
    ODEXInitFFFBPixelSwitchExplicit,
    ODEXInitFFFBPixelSwitchStrang,
)

TOLS = (1e-5, 1e-6, 1e-7, 1e-8)
N_ITERS = (5, 10, 20, 40)


def set_tolerance(block, tol):
    block.tol = float(tol)
    for name in ("option_aca", "option_init", "option_patch"):
        option = getattr(block, name, None)
        if option is not None:
            option["rtol"] = float(tol)
            option["atol"] = float(tol)


def measured_forward(net, x, module):
    counters = {"solves": 0, "nfe": 0}
    original = module.aca_ode_solve

    def counted_solve(fn, y0, options, *args, **kwargs):
        counters["solves"] += 1

        def counted_fn(t, y):
            counters["nfe"] += 1
            return fn(t, y)

        return original(counted_fn, y0, options, *args, **kwargs)

    module.aca_ode_solve = counted_solve
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        value = net.PcConvs[0](x)
        torch.cuda.synchronize()
    finally:
        module.aca_ode_solve = original
    return value, counters, time.perf_counter() - started


def rel_l2(value, reference):
    return float((value - reference).norm() / reference.norm())


@torch.no_grad()
def main():
    torch.manual_seed(4096)
    x, _ = next(iter(get_test_data(
        test_bs=1, img_type="scanGFI", task="cifar100")))
    x = x.cuda()

    result = {
        "reference_tolerance": 1e-6,
        "dtype": "float32",
        "cases": [],
        "reference_controls": [],
    }

    ref_net, _ = build(ODEXInitFFFB, 1)
    ref_block = ref_net.PcConvs[0]
    reference_outputs = {}
    for tol in (1e-6, 1e-7, 1e-8):
        set_tolerance(ref_block, tol)
        value, counts, runtime = measured_forward(ref_net, x, ode_pc)
        reference_outputs[tol] = value.detach().clone()
        result["reference_controls"].append({
            "tol": tol,
            "relative_to_1e-6": None,
            "solves": counts["solves"],
            "nfe": counts["nfe"],
            "runtime_s": runtime,
        })

    reference = reference_outputs[1e-6]
    for row in result["reference_controls"]:
        row["relative_to_1e-6"] = rel_l2(
            reference_outputs[row["tol"]], reference)

    methods = (
        ("jacobi", ODEXInitFFFBPixelSwitchEfficient),
        ("lie", ODEXInitFFFBPixelSwitchExplicit),
        ("strang", ODEXInitFFFBPixelSwitchStrang),
    )
    for method_name, block_cls in methods:
        net, _ = build(block_cls, 1)
        block = net.PcConvs[0]
        for n_iters in N_ITERS:
            block.n_iters = n_iters
            for tol in TOLS:
                set_tolerance(block, tol)
                value, counts, runtime = measured_forward(net, x, switch)
                row = {
                    "method": method_name,
                    "n_iters": n_iters,
                    "tol": tol,
                    "rel_l2": rel_l2(value, reference),
                    "solves": counts["solves"],
                    "nfe": counts["nfe"],
                    "runtime_s": runtime,
                }
                result["cases"].append(row)
                print(json.dumps(row), flush=True)
        del net
        torch.cuda.empty_cache()

    out_path = os.path.join(
        ROOT, "results/switch_solver_tolerance_floor_diagnostic.json")
    with open(out_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print("RESULT_JSON=" + out_path, flush=True)


if __name__ == "__main__":
    main()

