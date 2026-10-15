"""Opt-in bounded execution of unmodified queued Python command lines.

Only loaded through the launcher's explicit REHEARSAL_ONLY switch. Real
parsers, dataset/model/teacher/optimizer initialization and checkpoint saving
run normally. Instrumentation redirects generated artifacts, limits the train
loop to one batch, and runs one MC45 corner/one inference batch. Production
processes never add this directory to PYTHONPATH.
"""
import argparse
import os
from pathlib import Path
import sys

if os.environ.get("REHEARSAL_ONLY") == "true":
    root = Path(os.environ["REHEARSAL_ARTIFACT_ROOT"]).resolve()
    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(repo), str(repo / "baseline")]
    original_parse = argparse.ArgumentParser.parse_args
    hooked = False

    def redirect(value):
        if not value:
            return value
        path = Path(value).resolve()
        if path.is_relative_to(root):
            return str(path)
        return str(root / path.relative_to(repo))

    class OneBatch:
        def __init__(self, loader):
            self.loader = loader

        def __len__(self):
            return 1

        def __iter__(self):
            yield next(iter(self.loader))

        def __getattr__(self, name):
            return getattr(self.loader, name)

    def training_hooks():
        from trainer_timm import TrainerCiFarTimmStyle
        original_init = TrainerCiFarTimmStyle.__init__

        def initialize(self, *a, **kw):
            original_init(self, *a, **kw)
            for name in ("train_dataloader", "val_dataloader", "teacher_eval_loader"):
                loader = getattr(self, name, None)
                if loader is not None:
                    setattr(self, name, OneBatch(loader))
            print("REHEARSAL_INITIALIZATION_PASS", self.model_name, flush=True)

        def train(self):
            import math
            loss = self.train_one_epoch(0)
            assert math.isfinite(float(loss)), loss
            self.scheduler.step(1)
            accuracy, _, _, _ = self.evaluate(self.val_dataloader)
            best = self._save_model_ckpt(accuracy, 1, "_best_ckpt.pth")
            self._save_model_ckpt(accuracy, 1, "_last_ckpt.pth")
            assert Path(best).is_file(), best
            print("REHEARSAL_TRAIN_SAVE_PASS", self.model_name, "loss=", loss,
                  "checkpoint=", best, flush=True)

        TrainerCiFarTimmStyle.__init__ = initialize
        TrainerCiFarTimmStyle.train = train

    def corner_hook():
        import subprocess
        from data_utils import MC45CornerData
        original_init = MC45CornerData.__init__
        original_popen = subprocess.Popen

        class StartupPopen(original_popen):
            def wait(self, *a, **kw):
                status = super().wait(*a, **kw)
                if status == 85:
                    print("REHEARSAL_EVAL_INITIALIZATION_PASS (stopped before cache generation)", flush=True)
                    raise SystemExit(0)
                return status

        def initialize(self, *a, **kw):
            original_init(self, *a, **kw)
            assert len(self.corners) == 45
            self.corners = self.corners[:1]
            print("REHEARSAL_CORNER_BANK_PASS", self.corners[0]["id"], flush=True)

        MC45CornerData.__init__ = initialize
        subprocess.Popen = StartupPopen

    def stop_before_expansion():
        from validation import Validator
        from feedforward_validation import FeedForwardCNNValidator

        def initialized(*a, **kw):
            print("REHEARSAL_CACHE_BOUNDARY_PASS: checkpoint, hardware and dataset initialized; no accuracy measured", flush=True)
            raise SystemExit(85)

        Validator.__init__ = initialized
        FeedForwardCNNValidator.__init__ = initialized

    def parse(self, *a, **kw):
        global hooked
        args = original_parse(self, *a, **kw)
        script = Path(sys.argv[0]).name
        if script in ("train_ode_cifar.py", "train_baseline_cifar.py"):
            field = "output_save_path" if script == "train_ode_cifar.py" else "output_dir"
            setattr(args, field, redirect(getattr(args, field)))
            if not hooked:
                training_hooks()
                hooked = True
        elif script in ("run_toggle_nonideality_ablation.py", "run_feedforward_mc45_ablation.py"):
            for field in ("output_dir", "model_dir", "checkpoint", "expanded_w_dir", "expanded_weight_dir"):
                if hasattr(args, field):
                    setattr(args, field, redirect(getattr(args, field)))
            args.max_eval_batches = 1
            if not hooked:
                corner_hook()
                hooked = True
        elif script in ("ode_inference.py", "evaluate_physical_feedforward_cifar.py"):
            args.max_eval_batches = 1
            if not hooked:
                stop_before_expansion()
                hooked = True
        return args

    argparse.ArgumentParser.parse_args = parse
