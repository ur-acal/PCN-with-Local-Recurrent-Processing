"""Bounded real Level-3 inference; compare the original head on identical states.

The production inference path/curves/noises are unchanged. After each complete
forward, replay the old /q -> ReLU -> scaled measured GAP -> Linear head on the
same last-PCN tensor. This isolates the coordinate change without resampling
nonidealities or allocating a second expanded model. No checkpoints are saved.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--gpu-memory-fraction", type=float, default=.06)
    args = p.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["DATALOADER_NUM_WORKERS"] = "0"
    os.environ["NUM_WORKERS"] = "0"
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    import torch
    from torch.nn import functional as F
    import ode_inference as entry
    from validation import MVMConv
    from pc_model import logits_for_loss

    torch.set_num_threads(2)
    if not 0 < args.gpu_memory_fraction <= .1:
        p.error("Memory fraction must be in (0, .1] to protect concurrent training")
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    command = shlex.split(next(line for line in Path(args.log).read_text().splitlines()
                              if line.startswith("COMMAND:")).removeprefix("COMMAND:"))
    argv = command[command.index("ode_inference.py")+1:]
    argv = [str(ROOT / v.split("PCN-with-Local-Recurrent-Processing/")[1])
            if "PCN-with-Local-Recurrent-Processing/" in v else v for v in argv]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    argv += ["--test_bs", str(args.batch_size), "--noisy_trials", "1",
             "--mem_frac", str(args.gpu_memory_fraction),
             "--expanded_w_dir", str(out.parent / "expanded_weights"),
             "--hw_val_path", str(out.parent / "hardware_validation")]
    report = dict(source_log=args.log, argv=argv, batches=[], device=args.device,
                  reference="original head replayed on identical final PCN states",
                  gpu_memory_fraction=args.gpu_memory_fraction)
    original_validator = entry.Validator

    class Done(Exception):
        pass

    def validator(*vargs, **kwargs):
        result = original_validator(*vargs, **kwargs)
        model = result.model
        assert model.states_are_physical
        assert model.global_avg_pool2d.input_scale == 1
        assert all(isinstance(b.FFconv, MVMConv) and isinstance(b.FBconv, MVMConv)
                   for b in model.PcConvs)
        report["both_ff_fb_expanded"] = True
        last_state = {}
        def capture(module, inputs, output):
            last_state["x"] = output.detach()
        model.PcConvs[-1].register_forward_hook(capture)
        def compare(module, inputs, output):
            assert not module.training, "This audit replays the eval head (dropout disabled)"
            # The pinned model has no spatial pool after its last PCN layer.
            assert not module.max_pool[-1]
            q = module.state_q
            old_feature = F.relu(last_state.pop("x") / q)
            pool = module.global_avg_pool2d
            previous_scale = pool.input_scale
            pool.input_scale = q
            try:
                old_pooled = pool(old_feature).flatten(1)
            finally:
                pool.input_scale = previous_scale
            old_logits = F.linear(old_pooled, module.linear.weight, module.linear.bias)
            restored = logits_for_loss(output, module)
            torch.testing.assert_close(restored, old_logits, rtol=2e-5, atol=2e-5)
            assert torch.equal(output.argmax(1), old_logits.argmax(1))
            row = dict(max_abs_diff=float((restored-old_logits).abs().max()),
                       new_raw=output.detach().cpu().tolist(),
                       old_logits=old_logits.detach().cpu().tolist(),
                       predictions=output.argmax(1).cpu().tolist(),
                       peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else 0)
            report["batches"].append(row)
            out.write_text(json.dumps(report, indent=2)+"\n")
            print("HEAD_AUDIT", len(report["batches"]), row["max_abs_diff"], flush=True)
            if len(report["batches"]) >= args.batches:
                raise Done()
        model.register_forward_hook(compare)
        return result

    entry.Validator = validator
    sys.argv = ["ode_inference.py", *argv]
    try:
        entry.run_ode_inference()
    except Done:
        print("Bounded Level-3 comparison completed; all predictions match.", flush=True)


if __name__ == "__main__":
    main()
