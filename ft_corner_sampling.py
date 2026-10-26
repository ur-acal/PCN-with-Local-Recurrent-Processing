"""Coordinate one sampled PVT corner across Level-2 fine-tuning draws."""

import re

import torch

from utils import mc45_corner_ids, mc_training_curve_corner_indices


_ACTIVATION_CORNER_RE = re.compile(
    r"^(TT|FF|SS|FS|SF)_(-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))_([0-2])_",
    re.IGNORECASE)
_CURRENT_ACTIVATION_CORNER_RE = re.compile(
    r"^RELU_(TT|FF|SS|FS|SF)G([0-8])_MC[0-9]+$", re.IGNORECASE)


def _activation_corner_curve_indices(corner_names):
    parsed = []
    for index, name in enumerate(corner_names):
        current = _CURRENT_ACTIVATION_CORNER_RE.match(str(name))
        if current is not None:
            # Same voltage-major, temperature-minor numbering as MC45CornerData.
            voltage, temperature = divmod(int(current.group(2)), 3)
            parsed.append((index, current.group(1).upper(),
                           (-20.0, 25.0, 85.0)[temperature], voltage))
            continue
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
    """Couple empirical curves or TC Gaussian factors per training forward.

    Fixed measured activations intentionally do not participate.  Evaluation
    clears the active corner and therefore retains its original sampling logic.
    """
    previous_hook = getattr(model, "_ft_corner_coupled_sampling_hook", None)
    if previous_hook is not None:
        previous_hook.remove()

    sampler = FTCornerCoupledSampler(model)
    participants = []

    # TC uses corner-specific covariance or code-conditioned empirical curves.
    # Share each bank across the backbone, pooling and any existing analog head.
    tc_banks = {}

    def register_tc(module, path, grid, factor):
        from tc_nonidealities import prepare_tc_corner_factors, TCCornerFactorBank
        key = (path, tuple(grid.detach().cpu().double().tolist()),
               factor.device, factor.dtype)
        if key not in tc_banks:
            tc_banks[key] = TCCornerFactorBank(prepare_tc_corner_factors(
                path, grid, dtype=factor.dtype, device=factor.device))
        module._tc_corner_factors = tc_banks[key]
        module._ft_corner_coupled_sampler = sampler

    for module in model.modules():
        package = getattr(module, "_tc_resistance_package", None)
        if package is None:
            package = getattr(module, "_tc_curve_package", None)
        if package is not None:
            if package.empirical_bank is not None:
                if getattr(module, 'mat', None) is not None:
                    raise ValueError('Corner-coupled FT requires dense TC convolutions.')
                if not package.empirical_bank.get('corner_indices'):
                    raise ValueError('Corner-coupled TC empirical FT requires an MC45 directory bank.')
                module._tc_empirical_coupling = True
                module._ft_corner_coupled_sampler = sampler
                participants.append('nonlinear-R')
                continue
            if getattr(module, "mat", None) is not None or package.empirical_bank is not None:
                raise ValueError("TC corner-coupled FT requires dense Gaussian curves.")
            register_tc(module, package.covariance_path, package.v_grid, package.factor)
            participants.append("nonlinear-R")

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
            path = getattr(pool, "tc_covariance_path", None)
            if path is None or pool.gaussian_quantity != "resistance" or pool.gaussian_value_scale != 1.:
                raise ValueError("FT corner coupling requires TC Gaussian measured-pooling banks.")
            register_tc(pool, path, pool.gaussian_v_grid, pool.gaussian_factor)
            participants.append("measured-pooling")
            continue
        if getattr(pool, '_tc_empirical_mapping', None) is not None:
            pool._ft_corner_curve_indices = pool._tc_empirical_mapping
            pool._ft_corner_coupled_sampler = sampler
            participants.append('measured-pooling')
            continue
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
