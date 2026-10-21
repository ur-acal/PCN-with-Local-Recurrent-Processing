"""Shared engine for local-weight and full-network activation diagnostics."""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from .run_baseline import FixedMismatchHelper, recalibrate_batchnorm_statistics


class RelativeError:
    """Both agreed full-dataset metrics, with explicit undefined-image records."""

    def __init__(self):
        self.samples = self.valid = 0
        self.error_energy = self.reference_energy = self.ratio_sum = 0.0
        self.zeros = []

    def update(self, noisy, clean, ids):
        if noisy.shape != clean.shape or noisy.shape[0] != len(ids):
            raise ValueError("Activation shape/sample alignment mismatch")
        a, b = noisy.detach().double(), clean.detach().double()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise FloatingPointError("Nonfinite activation; refusing a misleading metric")
        e = (a - b).flatten(1).square().sum(1).cpu()
        r = b.flatten(1).square().sum(1).cpu()
        valid = r > 0
        self.samples += len(ids)
        self.valid += int(valid.sum())
        self.error_energy += float(e.sum())
        self.reference_energy += float(r.sum())
        self.ratio_sum += float((e[valid] / r[valid]).sqrt().sum())
        for i in (~valid).nonzero().flatten().tolist():
            self.zeros.append(dict(sample_id=int(ids[i]), absolute_l2=float(e[i].sqrt())))

    def result(self):
        return dict(
            samples=self.samples, valid_images=self.valid,
            zero_reference_images=len(self.zeros),
            dataset_relative_l2=(math.sqrt(self.error_energy / self.reference_energy)
                                 if self.reference_energy > 0 else None),
            mean_image_relative_l2=(self.ratio_sum / self.valid if self.valid else None),
            error_energy=self.error_energy, reference_energy=self.reference_energy,
        )


def units_for(model, kind):
    if kind == "pcn":
        return [(f"PcConvs.{i}", layer) for i, layer in enumerate(model.PcConvs)]
    return [("conv1", model.conv1)] + [
        (f"layer{s}.{i}", layer)
        for s in (1, 2, 3) for i, layer in enumerate(getattr(model, f"layer{s}"))]


def measurement_module(layer):
    """Return the agreed feature boundary for a diagnostic unit."""
    return layer.pre_pool if hasattr(layer, "pre_pool") else layer


class Capture:
    """Clone immediately: WRN uses in-place ReLU at subsequent boundaries."""

    def __init__(self, model, kind, save_inputs=False):
        self.inputs, self.outputs, self.handles = {}, {}, []
        for name, layer in units_for(model, kind):
            if save_inputs:
                def before(module, args, key=name):
                    self.inputs[key] = args[0].detach().cpu().clone()
                self.handles.append(layer.register_forward_pre_hook(before))
            def after(module, args, output, key=name):
                self.outputs[key] = output.detach().cpu().clone()
            self.handles.append(measurement_module(layer).register_forward_hook(after))
        if kind == "wrn":
            def head_before(module, args):
                self.outputs["final_feature"] = args[0].detach().cpu().clone()
            self.handles.append(model.bn.register_forward_pre_hook(head_before))
        else:
            def pcn_final(module, args, output):
                feature = output.detach()
                if model.max_pool[-1]:
                    feature = model.max_pool2d(feature)
                self.outputs["final_feature"] = feature.cpu().clone()
            self.handles.append(model.PcConvs[-1].register_forward_hook(pcn_final))

    def close(self):
        for handle in self.handles:
            handle.remove()


def cache_clean(model, kind, loader, directory, device):
    """One clean, original-BN pass; batch files avoid an all-dataset RAM cache."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    capture = Capture(model, kind, save_inputs=True)
    count = correct = 0
    paths = []
    model.eval()
    try:
        with torch.no_grad():
            for batch, (images, labels) in enumerate(loader):
                capture.inputs.clear()
                capture.outputs.clear()
                logits = model(images.to(device)).detach().cpu()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite clean logits")
                ids = torch.arange(count, count + len(labels))
                batch_dir = directory / f"batch_{batch:05d}"
                batch_dir.mkdir()
                path = batch_dir / "samples.pt"
                torch.save(dict(images=images, labels=labels, sample_ids=ids,
                                logits=logits), path)
                # Claim 1 reads only one unit, not every layer's large activations.
                for name, output in capture.outputs.items():
                    torch.save(dict(input=capture.inputs.get(name), output=output),
                               batch_dir / f"{name}.pt")
                paths.append(path)
                correct += int((logits.argmax(1) == labels).sum())
                count += len(labels)
    finally:
        capture.close()
    if count == 0:
        raise ValueError("Empty test dataset")
    return paths, dict(samples=count, correct=correct, accuracy=100 * correct / count)


def selected_parameters(model, unit=None):
    """Conv/ConvTranspose weights and classifier weight/bias, never BN/b0."""
    selected = set()
    for name, module in model.named_modules():
        if unit is not None and not (name == unit or name.startswith(unit + ".")):
            continue
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            selected.add(f"{name}.weight")
        elif isinstance(module, nn.Linear):
            selected.add(f"{name}.weight")
            if module.bias is not None:
                selected.add(f"{name}.bias")
    if not selected:
        raise ValueError(f"No eligible mismatch tensors for {unit}")
    return selected


def make_helper(model, selected, condition):
    helper = FixedMismatchHelper(
        model, noise_sigma=0, noise_type=("additive" if condition == "max_additive"
                                        else "multiplicative"),
        noise_to_norm=False, include_buffers=False, additive_scale_mode="max_abs",
        exclude_param_names=set(dict(model.named_parameters())) - selected,
    )
    helper.snapshot_clean_state()
    return helper


class LogitDiagnostics:
    """Classification-aware C2 metrics, accumulated exactly across images."""

    def __init__(self):
        self.centered = RelativeError()
        self.samples = 0
        self.clean_margin_sum = self.noisy_margin_sum = 0.0

    @staticmethod
    def margin(logits, labels):
        values = logits.detach().double()
        true = values.gather(1, labels[:, None]).squeeze(1)
        rivals = values.clone()
        rivals.scatter_(1, labels[:, None], -torch.inf)
        return true - rivals.max(1).values

    def update(self, noisy, clean, labels, ids):
        a, b = noisy.detach().double(), clean.detach().double()
        self.centered.update(a - a.mean(1, keepdim=True),
                             b - b.mean(1, keepdim=True), ids)
        self.clean_margin_sum += float(self.margin(b, labels).sum())
        self.noisy_margin_sum += float(self.margin(a, labels).sum())
        self.samples += len(labels)

    def result(self):
        return dict(centered_logits=self.centered.result(),
                    centered_zero_references=self.centered.zeros,
                    clean_margin_mean=self.clean_margin_sum / self.samples,
                    noisy_margin_mean=self.noisy_margin_sum / self.samples,
                    margin_degradation=(self.clean_margin_sum - self.noisy_margin_sum) / self.samples)


def save_logits(path, batches):
    """One ordered full-dataset file, not one file per inference batch."""
    values = {key: torch.cat([b[key].detach().cpu() for b in batches])
              for key in ('logits', 'labels', 'sample_ids')}
    n = len(values['sample_ids'])
    if (values['logits'].ndim != 2 or len(values['logits']) != n or len(values['labels']) != n
            or values['sample_ids'].unique().numel() != n or not torch.isfinite(values['logits']).all()):
        raise ValueError('Invalid or duplicate logit samples')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = torch.load(path, map_location='cpu', weights_only=True)
        if any(not torch.equal(previous[k], v) for k, v in values.items()):
            raise ValueError(f'Existing logits differ; refusing overwrite: {path}')
        return
    temporary = path.with_suffix('.pt.tmp')
    torch.save(values, temporary)
    temporary.replace(path)


def consolidate_clean_logits(paths, destination):
    batches = []
    for path in paths:
        batch = torch.load(path, map_location='cpu', weights_only=True)
        batches.append({k: batch[k] for k in ('logits', 'labels', 'sample_ids')})
    save_logits(destination, batches)


def evaluate_cached(model, kind, paths, device, unit=None, *, logits_only=False, logits_path=None):
    if unit is not None and logits_only:
        raise ValueError('Logits-only diagnostics require Claim 2')
    capture = Capture(model, kind) if unit is None and not logits_only else None
    module = dict(units_for(model, kind)).get(unit) if unit is not None else None
    metrics = {}
    transitions = dict(clean_correct=0, noisy_correct=0, wrong_to_correct=0,
                       correct_to_wrong=0, samples=0)
    predictions = []
    logit_metrics = LogitDiagnostics()
    logit_batches = []
    model.eval()
    try:
        with torch.no_grad():
            for path in paths:
                batch = torch.load(path, map_location="cpu", weights_only=True)
                if unit is not None:
                    # Use pristine cached input; never a preceding noisy layer.
                    reference_unit = torch.load(path.parent / f"{unit}.pt",
                                                map_location="cpu", weights_only=True)
                    measured = {}
                    boundary = measurement_module(module)
                    handle = boundary.register_forward_hook(
                        lambda _module, _args, output: measured.setdefault(
                            "output", output.detach().cpu().clone()))
                    try:
                        returned = module(reference_unit["input"].to(device).clone())
                    finally:
                        handle.remove()
                    output = measured.get("output", returned.detach().cpu())
                    outputs = {unit: output}
                else:
                    if capture:
                        capture.outputs.clear()
                    logits = model(batch["images"].to(device)).cpu()
                    outputs = dict(capture.outputs if capture else {}, logits=logits)
                    clean_pred = batch["logits"].argmax(1)
                    pred = logits.argmax(1)
                    labels = batch["labels"]
                    if logits_path is not None:
                        logit_batches.append(dict(logits=logits, labels=labels, sample_ids=batch['sample_ids']))
                    logit_metrics.update(logits, batch['logits'], labels, batch['sample_ids'])
                    c, n = clean_pred == labels, pred == labels
                    transitions["samples"] += len(labels)
                    transitions["clean_correct"] += int(c.sum())
                    transitions["noisy_correct"] += int(n.sum())
                    transitions["wrong_to_correct"] += int((~c & n).sum())
                    transitions["correct_to_wrong"] += int((c & ~n).sum())
                    predictions.extend(zip(batch["sample_ids"].tolist(), labels.tolist(),
                                           clean_pred.tolist(), pred.tolist()))
                for name, output in outputs.items():
                    if name == "logits":
                        reference = batch["logits"]
                    elif unit is not None:
                        reference = reference_unit["output"]
                    else:
                        reference = torch.load(path.parent / f"{name}.pt",
                                               map_location="cpu", weights_only=True)["output"]
                    metrics.setdefault(name, RelativeError()).update(
                        output, reference, batch["sample_ids"])
    finally:
        if capture:
            capture.close()
    if unit is None:
        if logits_path is not None:
            save_logits(logits_path, logit_batches)
        transitions.update(logit_metrics.result())
        for key in ('correct_to_wrong', 'wrong_to_correct'):
            transitions[key + '_percent'] = 100 * transitions[key] / transitions['samples']
    return metrics, transitions, predictions


def save_csv(path, rows, fields=None):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostics(model, kind, test_loader, calibration_loader, recal_cfg,
                    output, device, *, claims=(1, 2), levels=None, trials=10,
                    base_seed=123, model_index=0, cache_directory=None,
                    cache_signature=None, clean_logits_path=None):
    """Run one checkpoint. A fresh output directory prevents accidental mixing."""
    levels = levels or {"multiplicative": [0.30, 0.35, 0.40],
                        "max_additive": [0.08, 0.09, 0.10]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Save every parameter and buffer, including unused PCN b0 and BN counters.
    clean = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    cache = Path(cache_directory) if cache_directory else output / "clean"
    if cache.exists():
        if cache_directory is None:
            raise FileExistsError(cache)
        metadata = json.loads((cache / "complete.json").read_text())
        if cache_signature is None or metadata["signature"] != cache_signature:
            raise ValueError("Clean-cache identity mismatch; refusing reuse")
        paths = sorted(cache.glob("batch_*/samples.pt"))
        if len(paths) != metadata["batches"]:
            raise ValueError("Incomplete clean cache")
        clean_result = metadata["clean_result"]
    else:
        paths, clean_result = cache_clean(model, kind, test_loader, cache, device)
        (cache / "complete.json").write_text(json.dumps(dict(signature=cache_signature,
            batches=len(paths), clean_result=clean_result), indent=2))
    all_units = [name for name, _ in units_for(model, kind)]
    clean_logits_path = Path(clean_logits_path) if clean_logits_path else output / 'clean_logits.pt'
    consolidate_clean_logits(paths, clean_logits_path)
    (output / 'clean_reference.json').write_text(json.dumps(dict(
        clean_cache=str(cache.resolve()), clean_logits=str(clean_logits_path.resolve()),
        cache_signature=cache_signature), indent=2))
    (output / "clean_summary.json").write_text(json.dumps(clean_result, indent=2))
    bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    policies = ["frozen", "recalibrated"] if bn else ["no_bn"]
    rows = []
    try:
        for claim in claims:
            for target in (all_units if claim == 1 else [None]):
                selected = selected_parameters(model, target)
                for condition, values in levels.items():
                    model.load_state_dict(clean, strict=True)
                    helper = make_helper(model, selected, condition)
                    for level in values:
                        for trial in range(trials):
                            helper.restore_clean_state()
                            model.eval()
                            helper.noise_sigma = level
                            helper.seed = (base_seed + 100000 * model_index
                                           + 1000 * int(round(level * 1e6)) + trial)
                            helper.add_noise()
                            trial_dir = (output / f"claim{claim}" / (target or "network")
                                         / condition / f"level_{level:g}" / f"trial_{trial}")
                            trial_dir.mkdir(parents=True, exist_ok=False)
                            (trial_dir / "perturbation.json").write_text(json.dumps(dict(
                                seed=helper.seed, selected_parameters=sorted(selected),
                                condition=condition, level=level, trial=trial,
                                scope=target or "network"), indent=2))
                            for policy in policies:
                                if policy == "recalibrated":
                                    if calibration_loader is None or recal_cfg is None:
                                        raise ValueError("Recalibration requires a training-data loader")
                                    recalibrate_batchnorm_statistics(
                                        model, calibration_loader, device, recal_cfg,
                                        f"claim{claim}_{target or 'network'}_{condition}", level, trial)
                                out = trial_dir / policy
                                out.mkdir()
                                metrics, transition, predictions = evaluate_cached(
                                    model, kind, paths, device, target,
                                    logits_path=out / 'noisy_logits.pt' if claim == 2 else None)
                                trial_rows, zeros = [], []
                                for name, metric in metrics.items():
                                    row = dict(claim=claim, target=target or "network",
                                               condition=condition, level=level, trial=trial,
                                               bn_policy=policy, layer=name, **metric.result())
                                    trial_rows.append(row)
                                    zeros.extend(dict(layer=name, **z) for z in metric.zeros)
                                # Unit average includes stem/first PCN layer, not duplicate head metrics.
                                for metric_name in ("dataset_relative_l2", "mean_image_relative_l2"):
                                    vals = [r[metric_name] for r in trial_rows if r["layer"] in all_units]
                                    transition[f"mean_layer_{metric_name}"] = (
                                        sum(vals) / len(vals) if vals and all(v is not None for v in vals)
                                        else None)
                                if claim == 2:
                                    transition["noisy_accuracy"] = (
                                        100 * transition["noisy_correct"] / transition["samples"])
                                    save_csv(out / "predictions.csv", [dict(zip(
                                        ("sample_id", "label", "clean_prediction", "noisy_prediction"), p))
                                        for p in predictions])
                                save_csv(out / "metrics.csv", trial_rows)
                                save_csv(out / "zero_references.csv", zeros,
                                         ["layer", "sample_id", "absolute_l2"])
                                (out / "summary.json").write_text(json.dumps(transition, indent=2))
                                rows.extend(trial_rows)
                                save_csv(output / "per_trial.csv", rows)
                                print(f"claim={claim} unit={target or 'network'} {condition} "
                                      f"level={level} trial={trial} BN={policy} complete", flush=True)
    finally:
        model.load_state_dict(clean, strict=True)
        model.eval()
    return rows


def summarize(rows, output):
    groups = defaultdict(list)
    keys = ("claim", "target", "condition", "level", "bn_policy", "layer")
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    aggregate = []
    for key, group in groups.items():
        row = dict(zip(keys, key))
        row["trials"] = len(group)
        for metric in ("dataset_relative_l2", "mean_image_relative_l2"):
            values = [r[metric] for r in group if r[metric] is not None]
            row[metric] = statistics.mean(values) if len(values) == len(group) else None
            row[metric + "_std"] = statistics.pstdev(values) if len(values) == len(group) else None
        row["zero_reference_images_per_trial"] = group[0]["zero_reference_images"]
        aggregate.append(row)
    save_csv(Path(output) / "aggregate.csv", aggregate)
    layer_groups = defaultdict(list)
    for row in rows:
        if row["layer"] not in ("logits", "final_feature"):
            layer_groups[(row["claim"], row["condition"], row["level"],
                          row["bn_policy"], row["trial"])].append(row)
    layer_means = []
    for key, group in layer_groups.items():
        row = dict(zip(("claim", "condition", "level", "bn_policy", "trial"), key))
        row["units_including_stem"] = len(group)
        for metric in ("dataset_relative_l2", "mean_image_relative_l2"):
            values = [r[metric] for r in group]
            row[metric] = statistics.mean(values) if all(v is not None for v in values) else None
        layer_means.append(row)
    save_csv(Path(output) / "layer_means.csv", layer_means)
    lines = ["# Activation Diagnostics", "",
             "Each checkpoint uses its own original-BN clean reference. "
             "Layer 0 includes the stem/first PCN layer. No epsilon is used.", "",
             "| Claim | Target | Mismatch | Level | BN | Layer | Trials | Dataset relative L2 | Mean image relative L2 |",
             "|---|---|---|---:|---|---|---:|---:|---:|"]
    for r in aggregate:
        values = [r[k] for k in keys] + [r["trials"], r["dataset_relative_l2"],
                                         r["mean_image_relative_l2"]]
        lines.append("| " + " | ".join("N/A" if v is None else str(v) for v in values) + " |")
    lines += ["", "Per-trial prediction transitions and across-layer means are in each "
              "trial's summary.json. layer_means.csv contains equal-unit means for both claims "
              "including the stem. Per-image predictions and zero-reference flags are saved alongside trials."]
    (Path(output) / "summary.md").write_text("\n".join(lines) + "\n")
