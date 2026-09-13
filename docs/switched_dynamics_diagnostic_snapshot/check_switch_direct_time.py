import json
import os
import sys
import time

import torch

ROOT = "/home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN"
sys.path[:0] = ["/tmp", ROOT]
os.chdir(ROOT)

from debug_strang_real_block import build
from inference_utils import get_test_data
from ode_pc import ODEXInitFFFB
from switch import ODEXInitFFFBPixelSwitchExplicit, ODEXInitFFFBPixelSwitchStrang


def run(block, x, direct):
    block.scale_RHS = not direct
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = block(x)
    torch.cuda.synchronize()
    return y, time.perf_counter() - t0


@torch.no_grad()
def main():
    torch.manual_seed(4096)
    x, _ = next(iter(get_test_data(test_bs=1, img_type="scanGFI", task="cifar100")))
    x = x.cuda()
    ref_net, _ = build(ODEXInitFFFB, 1)
    ref = ref_net.PcConvs[0](x)
    result = []
    for name, cls in (("lie", ODEXInitFFFBPixelSwitchExplicit),
                      ("strang", ODEXInitFFFBPixelSwitchStrang)):
        net, _ = build(cls, 1)
        block = net.PcConvs[0]
        for n in (1, 5, 20):
            block.n_iters = n
            direct, direct_s = run(block, x, True)
            scaled, scaled_s = run(block, x, False)
            result.append({
                "method": name,
                "n_iters": n,
                "direct_vs_reference": float((direct-ref).norm()/ref.norm()),
                "scaled_vs_reference": float((scaled-ref).norm()/ref.norm()),
                "direct_vs_scaled": float((direct-scaled).norm()/scaled.norm()),
                "direct_runtime_s": direct_s,
                "scaled_runtime_s": scaled_s,
            })
            print(json.dumps(result[-1]), flush=True)
    path = os.path.join(ROOT, "results/switch_direct_time_sanity.json")
    with open(path, "w") as fp:
        json.dump(result, fp, indent=2)
    print("RESULT_JSON=" + path)


if __name__ == "__main__":
    main()

