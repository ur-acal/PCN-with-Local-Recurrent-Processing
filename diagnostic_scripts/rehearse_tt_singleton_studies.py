#!/usr/bin/env python3
"""Exercise real Slurm job bodies locally; mock only submission/activation.

All generated artifacts live under a new, explicitly supplied /tmp directory.
The production stage scripts/arguments are used, with bounded workloads added
only by the opt-in rehearsal hooks. No scheduler or real training job is queued.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
JOBS = (("pcn", "cifar10"), ("wrn_c27", "cifar10"),
        ("resnet110", "cifar10"), ("resnet110", "cifar100"))


def rehearse(root, original, improved, jobs=JOBS):
    root = Path(root).resolve()
    if not root.is_relative_to(Path("/tmp")) or root == Path("/tmp"):
        raise ValueError("Choose a new directory under /tmp")
    root.mkdir(parents=True, exist_ok=False)
    binary = root / "infrastructure"
    binary.mkdir()
    fixtures = {
        "sbatch": '#!/bin/bash\nset -euo pipefail\nprintf "MOCK_SBATCH %s\\n" "$*"\nexec bash "${@: -1}"\n',
        "activate": '# Only environment activation is mocked.\nreturn 0\n',
        "conda": '#!/bin/bash\n[[ "$1" == activate ]]\n',
    }
    for name, content in fixtures.items():
        path = binary / name
        path.write_text(content)
        path.chmod(0o755)
    results = []
    for model, dataset in jobs:
        tag = model + "_" + dataset
        output = root / tag
        env = dict(os.environ,
                   PATH=str(binary) + os.pathsep + str(Path(sys.executable).parent)
                        + os.pathsep + os.environ["PATH"],
                   REPO_ROOT=str(ROOT), REHEARSAL_ONLY="true",
                   REHEARSAL_ARTIFACT_ROOT=str(output),
                   REHEARSAL_BATCHES="2", MODEL_KIND=model, TASK=dataset,
                   SELECTION_SAMPLES="128", SELECTION_BATCHES="1",
                   ONE_OVER_Q_CANDIDATES="4 5",
                   COUPLER_MONTE_TT_SINGLETON=str(Path(original).resolve()),
                   COUPLER_MONTE_V2_TT_SINGLETON=str(Path(improved).resolve()),
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   MPLCONFIGDIR=str(root / "matplotlib"))
        for key in ("STUDY_TAG", "STUDY_ARTIFACT_ROOT", "CNN_PRETRAIN_CKPT",
                    "PCN_PRETRAIN_ROOT", "PYTHONPATH", "PREFLIGHT_ONLY"):
            env.pop(key, None)
        log = root / (tag + ".log")
        print("REHEARSING", tag, log, flush=True)
        with log.open("w") as stream:
            result = subprocess.run(
                ["bash", str(ROOT / "launch_scripts/slurm_search_tt_singleton_config.sh")],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        text = log.read_text()
        proofs = list((output / "expansion_proofs").glob("*.pth"))
        expected_saves = 2 if model == "pcn" else 3
        passed = (result.returncode == 0 and len(proofs) == 2
                  and text.count("REHEARSAL_TRAIN_SAVE_PASS") == expected_saves
                  and text.count("REHEARSAL_EVAL_INITIALIZATION_PASS") == 2
                  and "STUDY_COMPLETE rehearsal=true" in text)
        if model != "pcn":
            passed &= text.count("SELECTION_COMPLETE") == 2
        results.append(dict(model=model, dataset=dataset, passed=bool(passed),
                            returncode=result.returncode, log=str(log),
                            expansion_proofs=[str(p) for p in proofs]))
        (root / "report.json").write_text(json.dumps(results, indent=2) + "\n")
        if not passed:
            raise RuntimeError(f"Rehearsal failed: {log}")
        print("REHEARSAL_PIPELINE_PASS", tag, flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_root", required=True)
    parser.add_argument("--original_singleton", required=True)
    parser.add_argument("--improved_singleton", required=True)
    parser.add_argument("--job", choices=[m + ":" + d for m, d in JOBS], action="append")
    args = parser.parse_args()
    jobs = tuple(tuple(job.split(":")) for job in args.job) if args.job else JOBS
    rehearse(args.artifact_root, args.original_singleton, args.improved_singleton, jobs)
