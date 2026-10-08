"""Coordinate one sampled PVT corner across Level-2 fine-tuning draws."""

import re

import torch

from utils import mc45_corner_ids, mc_training_curve_corner_indices


_ACTIVATION_CORNER_RE = re.compile(
    r"^(TT|FF|SS|FS|SF)_(-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))_([0-2])_",
    re.IGNORECASE)


def _activation_corner_curve_indices(corner_names):
    parsed = []
    for index, name in enumerate(corner_names):
        match = _ACTIVATION_CORNER_RE.match(str(name))
        if match is None:
            raise ValueError(
                "FT corner-coupled measured activation requires the full "
                "MC45 directory bank; unrecognized curve {!r}.".format(name))
        parsed.append((index, match.group(1).upper(),
                       float(match.group(2)), int(match.group(3))))
    temperatures = sorted({entry[2] for entry in parsed})
    if len(temperatures) != 3:
        raise ValueError(
            "FT corner-coupled measured activation requires three "
            "characterized temperatures.")
    mapping = {corner: [] for corner in mc45_corner_ids()}
    for index, process, temperature, voltage in parsed:
        corner = "{}_V{}_T{}".format(
            process, voltage, temperatures.index(temperature))
        mapping.setdefault(corner, []).append(index)
    return {corner: tuple(indices) for corner, indices in mapping.items()
            if indices}


def _validated_mc45_mapping(mapping, component):
    expected = mc45_corner_ids()
    actual = tuple(corner for corner in expected if corner in mapping)
    missing = sorted(set(expected) - set(mapping))
    extra = sorted(set(mapping) - set(expected))
    if missing or extra or actual != expected:
        raise ValueError(
            "FT corner-coupled {} bank does not match MC45: missing={}, "
            "extra={}.".format(component, missing, extra))
    counts = {corner: len(mapping[corner]) for corner in expected}
    invalid = {corner: count for corner, count in counts.items()
               if count != 100}
    if invalid:
        raise ValueError(
            "FT corner-coupled {} requires 100 empirical curves per "
            "corner; got {}.".format(component, invalid))
    return {corner: tuple(mapping[corner]) for corner in expected}


class FTCornerCoupledSampler:
    """Transient per-forward corner selection shared by registered consumers."""

    def __init__(self, model):
        self.model = model
        self.corners = mc45_corner_ids()
        self.active_corner = None

    def sample(self):
        index = int(torch.randint(len(self.corners), (1,)).item())
        self.active_corner = self.corners[index]
        self.model._last_ft_corner_coupled_corner = self.active_corner
        return self.active_corner

    def clear(self):
        self.active_corner = None
        self.model._last_ft_corner_coupled_corner = None

    def allowed_indices(self, mapping):
        if self.active_corner is None:
            return None
        try:
            return mapping[self.active_corner]
        except KeyError as error:
            raise RuntimeError(
                "Active FT corner {} is missing from a registered curve bank."
                .format(self.active_corner)) from error


def configure_ft_corner_coupled_sampling(
        model, *, activation_corner_mode="fixed",
        include_measured_activation=False):
    """Couple empirical FT curve draws to one MC45 corner per train forward.

    Fixed measured activations intentionally do not participate.  Evaluation
    clears the active corner and therefore retains its original sampling logic.
    """
    previous_hook = getattr(model, "_ft_corner_coupled_sampling_hook", None)
    if previous_hook is not None:
        previous_hook.remove()

    sampler = FTCornerCoupledSampler(model)
    participants = []

    packages = []
    seen_packages = set()
    for module in model.modules():
        package = getattr(module, "_nonlinear_R_training_pkg", None)
        if package is not None and id(package) not in seen_packages:
            seen_packages.add(id(package))
            packages.append(package)
    for package in packages:
        mode = package.get("nonlinear_R_train_mode")
        if mode != "exact_curve":
            raise ValueError(
                "FT corner-coupled nonlinear-R sampling requires "
                "nonlinear_R_train_mode=exact_curve, got {!r}.".format(mode))
        mapping = mc_training_curve_corner_indices(
            package.get("nonlinear_R_table"), mode,
            package.get("nonlinear_R_corner_range", "all"))
        package["ft_corner_curve_indices"] = _validated_mc45_mapping(
            mapping, "nonlinear-R")
        package["ft_corner_coupled_sampler"] = sampler
        participants.append("nonlinear-R")

    from measured_pooling import MeasuredAvgPool2d
    for pool in (module for module in model.modules()
                 if isinstance(module, MeasuredAvgPool2d) and
                 module.enable_nonideality):
        if pool.curve_gaussian:
            raise ValueError(
                "FT corner-coupled sampling does not support Gaussian "
                "measured-pooling banks.")
        if pool.training_curve_mode != "exact_curve":
            raise ValueError(
                "FT corner-coupled measured pooling requires "
                "training_curve_mode=exact_curve.")
        mapping = mc_training_curve_corner_indices(
            pool.training_curve_source, pool.training_curve_mode,
            pool.training_corner_range)
        pool._ft_corner_curve_indices = _validated_mc45_mapping(
            mapping, "measured-pooling")
        pool._ft_corner_coupled_sampler = sampler
        participants.append("measured-pooling")

    if (include_measured_activation and
            str(activation_corner_mode).lower() == "random_per_forward"):
        from measured_activation import (
            MEASURED_ACTIVATION_TYPES)
        activations = tuple(module for module in model.modules()
                            if isinstance(module, MEASURED_ACTIVATION_TYPES))
        if not activations:
            raise ValueError(
                "FT corner-coupled random measured activation requires at "
                "least one measured activation module.")
        for activation in activations:
            mapping = _activation_corner_curve_indices(
                activation.corner_names)
            activation._ft_corner_curve_indices = _validated_mc45_mapping(
                mapping, "measured-activation")
            activation._ft_corner_coupled_sampler = sampler
        participants.append("measured-activation")

    participants = tuple(dict.fromkeys(participants))
    if not participants:
        raise ValueError(
            "FT corner-coupled sampling was enabled, but no supported "
            "corner-dependent empirical training bank is active.")

    # Future FT nonidealities with corner-dependent sampling must register
    # with this coordinator so all enabled components use the same corner.
    def _select_corner_before_forward(module, inputs):
        if module.training:
            sampler.sample()
        else:
            sampler.clear()

    model._ft_corner_coupled_sampler = sampler
    model._last_ft_corner_coupled_corner = None
    model._ft_corner_coupled_sampling_hook = model.register_forward_pre_hook(
        _select_corner_before_forward, prepend=True)
    return participants
