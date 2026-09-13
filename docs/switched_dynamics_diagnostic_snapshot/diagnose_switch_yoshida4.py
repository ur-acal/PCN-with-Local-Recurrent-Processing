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

import ode_pc
import switch
from debug_strang_real_block import build
from inference_utils import get_test_data
from ode_pc import ODEXInitFFFB
from switch import (
    ODEXInitFFFBPixelSwitchExplicit,
    ODEXInitFFFBPixelSwitchStrang,
    ODEXInitFFFBPixelSwitchYoshida4,
)

TOLS = (1e-5, 1e-6, 1e-7, 1e-8)
ALL_ITERS = (1, 2, 5, 10, 20, 40)
SWEEP_ITERS = (5, 10, 20, 40)


def set_tolerance(block, tol):
    block.tol = float(tol)
    for name in ("option_aca", "option_init", "option_patch"):
        option = getattr(block, name, None)
        if option is not None:
            option["rtol"] = float(tol)
            option["atol"] = float(tol)


def measured_forward(net, x, module, track_negative=False):
    counters = {"solves": 0, "nfe": 0}
    original = module.aca_ode_solve
    block = net.PcConvs[0]
    block._track_yoshida_diagnostics = bool(track_negative)
    block._yoshida_negative_max_abs = 0.0

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
        value = block(x)
        torch.cuda.synchronize()
    finally:
        module.aca_ode_solve = original
        block._track_yoshida_diagnostics = False
    return (value, counters, time.perf_counter() - started,
            float(block._yoshida_negative_max_abs))


def rel_l2(value, reference):
    return float((value - reference).norm() / reference.norm())


def negative_stage_control(block, wrapper, x):
    """Compare Yoshida's signed wrapper-aware RHS with explicit -F_p."""
    xin = wrapper.proj_fn(wrapper.inp_scale * x)
    y0 = block.init_y(xin)
    block._strang_active_pixel = (0, 0)
    installed = block._make_ode_fn(xin)
    del block._strang_active_pixel
    signed = block._make_signed_fixed_pixel_ode_fn(xin, 0, 0, -1.0)
    t = torch.zeros((), device=y0.device, dtype=y0.dtype)
    got = signed(t, y0)
    expected = -installed(t, y0)
    if hasattr(block, "_strang_active_pixel"):
        del block._strang_active_pixel
    return {
        "max_abs_difference": float((got - expected).abs().max()),
        "relative_difference": float((got - expected).norm() / expected.norm()),
    }


@torch.no_grad()
def main():
    torch.manual_seed(4096)
    x, _ = next(iter(get_test_data(
        test_bs=1, img_type="scanGFI", task="cifar100")))
    x = x.cuda()

    result = {
        "reference_tolerance": 1e-6,
        "dtype": "float32",
        "input_seed": 4096,
        "cases": [],
        "reference_controls": [],
        "negative_stage_control": None,
        "activation_saturation": "not instrumented",
    }

    ref_net, _ = build(ODEXInitFFFB, 1)
    ref_block = ref_net.PcConvs[0]
    reference_outputs = {}
    for tol in (1e-6, 1e-7, 1e-8):
        set_tolerance(ref_block, tol)
        value, counts, runtime, _ = measured_forward(
            ref_net, x, ode_pc)
        reference_outputs[tol] = value.detach().clone()
        result["reference_controls"].append({
            "tol": tol, "relative_to_1e-6": None,
            "solves": counts["solves"], "nfe": counts["nfe"],
            "runtime_s": runtime,
        })
    reference = reference_outputs[1e-6]
    for row in result["reference_controls"]:
        row["relative_to_1e-6"] = rel_l2(
            reference_outputs[row["tol"]], reference)

    methods = (("yoshida4", ODEXInitFFFBPixelSwitchYoshida4),)
    for method_name, block_cls in methods:
        net, wrappers = build(block_cls, 1)
        block = net.PcConvs[0]
        if method_name == "yoshida4":
            result["negative_stage_control"] = negative_stage_control(
                block, wrappers[0], x)
        for n_iters in ALL_ITERS:
            block.n_iters = n_iters
            tolerances = TOLS if n_iters in SWEEP_ITERS else (1e-6,)
            for tol in tolerances:
                set_tolerance(block, tol)
                value, counts, runtime, negative_max = measured_forward(
                    net, x, switch, track_negative=(method_name == "yoshida4"))
                row = {
                    "method": method_name,
                    "n_iters": n_iters,
                    "tol": tol,
                    "rel_l2": rel_l2(value, reference),
                    "solves": counts["solves"],
                    "nfe": counts["nfe"],
                    "runtime_s": runtime,
                    "negative_stage_max_abs": (
                        negative_max if method_name == "yoshida4" else None),
                }
                result["cases"].append(row)
                print(json.dumps(row), flush=True)
        del net
        torch.cuda.empty_cache()

    out_path = os.path.join(
        ROOT, "results/switch_yoshida4_only_diagnostic.json")
    with open(out_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print("RESULT_JSON=" + out_path, flush=True)


if __name__ == "__main__":
    main()

