"""Claim 1/2 diagnostic entry point. Default: inventory only; --execute runs it."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from torch import nn

from .activation_diagnostics import run_diagnostics, summarize, units_for, selected_parameters
from .run_baseline import (BNRecalibrationConfig, build_model, get_baseline_config,
                           build_test_loader, build_bn_calibration_loader)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (ROOT.parent / "papers/hardware-native-neural-ode/shared/evidence"
                  / "generic-mismatch/activation_claims")
SIZES = ("16_2", "16_4", "28_2", "28_4")
FAMILIES = {
    "wrn": dict(directory="baselines", label="std_BN_wd5e-4_lr1e-1_fd0"),
    "wd1e3": dict(directory="baselines_wd1e3_pilot", label="std_BN_wd1e-3_lr1e-1_fd0"),
    "finaldrop": dict(directory="baselines_wd1e3_finaldrop025",
                      label="std_BN_wd1e-3_lr1e-1_fd0p25"),
    "r4": dict(row=4, source="legacy", label="avgSC_BN_wd1e-3_lr1e-1_fd0p25"),
    "r7": dict(row=7, source="legacy", label="avgMain_BN_wd1e-3_lr1e-1_fd0p25"),
    "r10": dict(row=10, source="legacy", label="std_BN_wd1e-3_lr1e-2_fd0p25_ptinit"),
    "r14": dict(row=14, source="pool", label="maxMain_BN_wd1e-3_lr1e-1_fd0p25"),
    "r17": dict(row=17, source="pool", label="avgMain_BN_wd1e-3_lr1e-2_fd0p25_ptinit"),
    "r18": dict(row=18, source="pool", label="maxMain_BN_wd1e-3_lr1e-2_fd0p25_ptinit"),
}


def wrn_paths(args, family, dataset, size):
    family_spec = FAMILIES[family]
    if "directory" in family_spec:
        name = f"wrn_{size}_cifar"
        parent = args.wrn_root / family_spec["directory"] / dataset / "custom_noresize" / name
    else:
        row = family_spec["row"]
        name = f"wrn_{size}_cifar_control_r{row}"
        root = args.legacy_control_root if family_spec["source"] == "legacy" else args.pool_control_root
        parent = root / f"row{row}" / dataset / size / "checkpoints" / dataset / "custom_noresize" / name
    run = f"custom_noresize_{dataset}_{name}"
    return name, parent / run / f"{run}_best_ckpt.pth", parent / "baseline_config.json"


def inventory(args):
    rows = []
    for dataset in args.datasets.split(","):
        if dataset not in ("cifar10", "cifar100"):
            raise ValueError(dataset)
        for size in args.sizes.split(","):
            if size not in SIZES:
                raise ValueError(size)
            index = SIZES.index(size) + (4 if dataset == "cifar100" else 0)
            for family in args.families.split(","):
                if family == "pcn":
                    depth, width = map(int, size.split("_"))
                    tail = "7Layers0l1l2" if depth == 16 else "13Layers0l3l6"
                    name = ("TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_"
                            "dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_"
                            + ("C100_" if dataset == "cifar100" else "")
                            + f"3K1S{64 * width}C_0.25Dropout_{tail}_2Pool_1REP")
                    checkpoint = args.pcn_root / name / f"{name}_best_ckpt.pth"
                    config = None
                else:
                    if family not in FAMILIES:
                        raise ValueError(family)
                    name, checkpoint, config = wrn_paths(args, family, dataset, size)
                rows.append(dict(id=f"{dataset}_{size}_{family}", dataset=dataset, size=size,
                                 family=family, name=name, model_index=index,
                                 family_label=FAMILIES.get(family, {}).get("label", "PCN_noX"),
                                 checkpoint=str(checkpoint.resolve()),
                                 config=str(config.resolve()) if config else None))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate model selection")
    return rows


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def load_model(spec, device):
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    state = {k.removeprefix("module."): v for k, v in checkpoint["net"].items()}
    if spec["family"] == "pcn":
        from inference_utils import load_and_prepare_model
        from pc_model import PCNetNoBatchNorm
        from pc_conv import PCConvNoisy
        from ode_pc import ODEXInitFFFB
        if checkpoint.get("net_type") != "PCNetNoBatchNorm":
            raise ValueError("Expected a PCNetNoBatchNorm checkpoint")
        model = load_and_prepare_model(
            spec["checkpoint"], device, model_struct=PCNetNoBatchNorm,
            pc_conv_layer=PCConvNoisy, data_parallel=False, fuse_bn=False,
            noise_to_bn=False, noise_to_linear=False, noise_to_conv_bias=False,
            noise_level=0.0, ode_params=dict(ode_block=ODEXInitFFFB, method="dopri5",
                t_end=1.75, tol=1e-4, return_init=False, mismatch_type="mul",
                additive_scale_mode="max_abs"))
        # The historical loader is permissive; audit all actual learned parameters.
        for name, p in model.named_parameters():
            if name not in state or not torch.equal(p.detach().cpu(), state[name].cpu()):
                raise ValueError(f"PCN checkpoint mismatch: {name}")
        if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
            raise ValueError("Unexpected BN in no-BN PCN")
        depth, width = map(int, spec["size"].split("_"))
        blocks = (depth - 4) // 6
        expected_channels = [16] + [c for c in (16 * width, 32 * width, 64 * width)
                                    for _ in range(blocks)]
        if list(model.ocs) != expected_channels:
            raise ValueError("PCN channel schedule does not match the selected WRN pair")
        for name, p in model.named_parameters():
            if ".b0." in name and torch.count_nonzero(p):
                raise ValueError("Nonzero b0 outside the agreed checkpoint policy")
        cfg = dict(timm_input_size=(3, 32, 32))
        kind = "pcn"
    else:
        cfg = get_baseline_config(spec["name"], pretrained=False, case="custom_noresize")
        cfg.update(json.loads(Path(spec["config"]).read_text())["cfg"])
        cfg["timm_input_size"] = tuple(cfg["timm_input_size"])
        num_classes = 100 if spec["dataset"] == "cifar100" else 10
        if spec["family"] == "r7":
            # AvgPool commutes with addition. Load the original row-7 weights into
            # the equivalent pool-after-add block so the pre-pooling sum is explicit.
            from .cifar_wrn_pool_after_add import WideResNetPoolAfterAddCIFAR
            from .wrn_control_specs import model_options
            depth, width = map(int, spec["size"].split("_"))
            options = model_options(7)
            model = WideResNetPoolAfterAddCIFAR(
                depth=depth, widen_factor=width, num_classes=num_classes,
                in_chans=cfg.get("in_chans", 3), pool_type="avg", **options)
        else:
            model = build_model(spec["name"], cfg, num_classes=num_classes)
        model.load_state_dict(state, strict=True)
        model.to(device)
        kind = "wrn"
    model.eval()
    expected_units = 7 if spec["size"].startswith("16") else 13
    if len(units_for(model, kind)) != expected_units:
        raise ValueError("Architecture/unit-count mismatch")
    if any(isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.bias is not None
           for m in model.modules()):
        raise ValueError("Unexpected convolution bias in selected top-three/no-BN PCN cohort")
    return model, kind, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--datasets", default="cifar10,cifar100")
    parser.add_argument("--sizes", default=",".join(SIZES))
    parser.add_argument("--families", default="pcn,wrn,wd1e3,finaldrop")
    parser.add_argument("--claims", default="1,2")
    parser.add_argument("--condition", choices=["multiplicative", "max_additive"])
    parser.add_argument("--level", type=float)
    parser.add_argument("--cache-root", type=Path,
                        help="Shared clean caches, verified by checkpoint/config/source signature")
    parser.add_argument("--pcn-root", type=Path, default=ROOT / "saved_ckpt")
    parser.add_argument("--wrn-root", type=Path, default=ROOT / "checkpoint")
    parser.add_argument("--legacy-control-root", type=Path,
                        default=ROOT / "checkpoint/baselines_wrn_controls")
    parser.add_argument("--pool-control-root", type=Path,
                        default=ROOT / "logs/wrn_pool_controls_slurm")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Result/cache directory (default: paper generic-mismatch evidence)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--smoke", action="store_true",
                        help="Two test/calibration images, one trial; not scientific results")
    args = parser.parse_args()
    grids = {"multiplicative": [0.30, 0.35, 0.40], "max_additive": [0.08, 0.09, 0.10]}
    if args.level is not None and (not args.condition or args.level not in grids[args.condition]):
        parser.error("level requires condition and must belong to its agreed grid")
    claims = tuple(int(c) for c in args.claims.split(","))
    if not claims or set(claims) - {1, 2} or len(set(claims)) != len(claims):
        parser.error("claims must be 1, 2, or 1,2")
    if args.trials < 1 or args.batch_size < 1:
        parser.error("trials and batch-size must be positive")
    specs = inventory(args)
    for spec in specs:
        spec["checkpoint_exists"] = Path(spec["checkpoint"]).is_file()
    if not args.execute:
        print(json.dumps(specs, indent=2))
        return
    if any(not r["checkpoint_exists"] or (r["config"] and not Path(r["config"]).is_file())
           for r in specs):
        raise FileNotFoundError("Missing checkpoint/config; inspect inventory before executing")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly for a smoke test")
    if device.type == "cpu":
        torch.set_num_threads(2)
    # Refuse overwrites, including prior smoke or partial runs.
    args.output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    revision=revision, models=specs, smoke=args.smoke,
                    implementation_hashes={p: sha256(ROOT / p) for p in (
                        "baseline/activation_diagnostics.py", "baseline/run_activation_diagnostics.py",
                        "baseline/run_baseline.py", "pc_model.py", "ode_pc.py", "mismatch_utils.py")})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for spec in specs:
        output = args.output / spec["id"]
        output.mkdir()
        state_path = output / "state.json"
        state_path.write_text(json.dumps(dict(status="running")))
        try:
            model, kind, cfg = load_model(spec, device)
            loader_args = argparse.Namespace(dataset=spec["dataset"], data_dir=str(args.data_root),
                batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
            loader = build_test_loader(model, loader_args, cfg)
            recal_cfg = BNRecalibrationConfig(enabled=True, num_samples="5120", batch_size=128,
                subset_seed=20240618, num_workers=args.workers,
                diagnostics_dir=str(output / "bn_diagnostics"))
            if args.smoke:
                loader = torch.utils.data.DataLoader(torch.utils.data.Subset(loader.dataset, [0, 1]),
                                                     batch_size=2, num_workers=0)
                recal_cfg.num_samples = "2"
            elif len(loader.dataset) != 10000:
                raise ValueError("Production run requires the full 10,000-image test dataset")
            calibration = (build_bn_calibration_loader(model, loader_args, cfg, recal_cfg)
                           if kind == "wrn" else None)
            checkpoint_hash = sha256(spec["checkpoint"])
            (output / "model.json").write_text(json.dumps(dict(spec=spec,
                checkpoint_sha256=checkpoint_hash, cfg=cfg,
                units=[n for n, _ in units_for(model, kind)],
                selected_parameters=sorted(selected_parameters(model))), indent=2, default=str))
            cache_signature = dict(checkpoint_sha256=checkpoint_hash, cfg=cfg,
                dataset=spec["dataset"], data_root=str(args.data_root.resolve()),
                batch_size=loader.batch_size, samples=len(loader.dataset),
                torch_version=str(torch.__version__), device=str(device),
                implementation_hashes=manifest["implementation_hashes"])
            # Normalize tuples/JSON scalars before comparing with persisted metadata.
            cache_signature = json.loads(json.dumps(cache_signature, default=str))
            selected_levels = ({args.condition: [args.level] if args.level is not None
                                else grids[args.condition]} if args.condition else grids)
            if args.smoke:
                selected_levels = {k: v[:1] for k, v in selected_levels.items()}
            rows = run_diagnostics(model, kind, loader, calibration, recal_cfg, output, device,
                claims=claims, trials=1 if args.smoke else args.trials,
                levels=selected_levels, base_seed=args.base_seed, model_index=spec["model_index"],
                cache_directory=args.cache_root / spec["id"] if args.cache_root else None,
                cache_signature=cache_signature)
            summarize(rows, output)
            state_path.write_text(json.dumps(dict(status="complete", smoke=args.smoke)))
            del model
        except Exception as exc:
            state_path.write_text(json.dumps(dict(status="failed", error=repr(exc))))
            raise


if __name__ == "__main__":
    main()
