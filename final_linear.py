"""Physical analog and integer-digital classifier heads.

Stored weight/bias remain in model coordinates. Runtime circuitry is rebuilt
from production configuration; it never changes checkpoint parameter names.

Production controls (environment names are uppercase):
  final_head_type: old_ideal (unchanged default), analog, digital.
  Analog inherits w_bits, weight_quant_factor_bits, R/C, toggle timing and
  the last physical block's supported noise/spin/coupler/DTC configuration.
  Bias is an additional coupler column driven by an ideal constant q voltage.
  Derived duration = RC/s_w; fixed duration = toggle_y_time/s_w. No extra ReLU.
  Pretraining is ideal, with gain gamma=T_base/RC (1 in derived mode);
  scale_train_recipe rescales initialization and optimizer for both W and b.
  Digital defaults: signed 8-bit ADC, 8-bit weights, 32-bit bias/accumulator,
  zero ADC noise. s_v=VDD/2**(B-1), s_w=(2**(Bw-1)-1)/max(abs(W)).
  Raw A=round(s_w W) @ ADC(g_v) + round(q*s_w*b/s_v).
  Round-to-even STE; ADC/bias code ranges and accumulator saturate, never wrap.
  final_adc_noise_lsb specifies Gaussian standard deviation in ADC LSB units.
  logits_for_loss restores analog raw/q or digital raw*s_v/(q*s_w).
  Classification uses raw outputs; SRRL feature restoration remains independent.
  measured_activation_scope controls ReLU replacement, independently of head type.
  final_head_quantize=False is a coordinate diagnostic bypass; clamp=False
  disables analog rails/digital accumulator clipping, not finite ADC code limits.

Use distinct experiment/output directories for different heads. Checkpoints
persist head kind/config, with explicit CLI options overriding saved settings.
"""
import inspect
import math
import os

import torch
from torch import nn
from torch.nn import functional as F


DIGITAL_DEFAULTS = dict(final_repr_bits=8, final_weight_bits=8,
                        final_bias_bits=32, final_accumulator_bits=32,
                        final_adc_noise_lsb=0.0)
HEAD_DEFAULTS = dict(**DIGITAL_DEFAULTS, final_head_quantize=True,
                     final_head_clamp=True)


def add_final_head_args(parser, include_type=True):
    if include_type:
        parser.add_argument('--final_head_type', choices=('old_ideal', 'analog', 'digital'),
                            default=os.environ.get('FINAL_HEAD_TYPE') or None)
    for name in HEAD_DEFAULTS:
        if name in ('final_head_quantize', 'final_head_clamp'):
            def boolean(value):
                if str(value).lower() not in ('true', 'false', '1', '0'):
                    raise ValueError('Expected true or false')
                return str(value).lower() in ('true', '1')
            dtype = boolean
        else:
            dtype = float if name == 'final_adc_noise_lsb' else int
        parser.add_argument('--' + name, type=dtype,
                            default=os.environ.get(name.upper()) or None)


def config_from_args(args, inherited=None):
    cfg = dict(inherited or {})
    cfg.update({key: getattr(args, key) for key in HEAD_DEFAULTS
                if getattr(args, key, None) is not None})
    return validate_config(cfg)


def head_load_overrides(args):
    result = {}
    if getattr(args, 'final_head_type', None) is not None:
        result['final_head_type'] = args.final_head_type
    explicit = {key: getattr(args, key) for key in HEAD_DEFAULTS
                if getattr(args, key, None) is not None}
    if explicit:
        result['final_head_config'] = explicit
    return result


def analog_recipe_scale(model):
    head = getattr(model, 'linear', getattr(model, 'fc', None))
    return getattr(model, 'linear_train_scale', head.recipe_gain) if isinstance(head, AnalogLinear) else 1.


def _feedforward_features_hook(model, inputs, output):
    if isinstance(output, tuple) and len(output) == 2 and getattr(model, 'states_are_physical', False):
        features, logits = output
        # Only the final feature changed coordinates when its early /q was
        # removed. Preserve the existing intermediate-feature contract.
        return [*features[:-1], features[-1] / model.state_q], logits
    return output


def configure_feedforward_head(model, args, physical=True):
    if getattr(model, 'final_head_type', 'old_ideal') == 'old_ideal':
        return
    from physical_feedforward import iter_physical_blocks
    blocks = list(iter_physical_blocks(model))
    if not blocks:
        raise ValueError('Nonideal feedforward classifier needs converted physical blocks')
    q = args.v_dd / args.one_over_q
    configure_model_head(model, physical=physical, q=q, v_dd=args.v_dd,
                         template=blocks[-1], family='tc' if args.tc_feedforward else 'toggle',
                         timing=args.toggle_timing_mode, base_time=args.toggle_y_time,
                         R=args.R, C=args.C, w_bits=args.w_bits,
                         weight_quant_factor_bits=args.weight_quant_factor_bits)
    model.linear_train_scale = model.fc.recipe_gain * (getattr(args, 'scale_train_recipe', 0.) or 1.)
    if not getattr(model, '_head_features_hook_installed', False):
        model.register_forward_hook(_feedforward_features_hook)
        model._head_features_hook_installed = True


def configure_model_head(model, *, physical, q, v_dd, template=None,
                         family='toggle', timing='derived', base_time=5e-9,
                         R=50e3, C=500e-15, **hardware):
    head = getattr(model, 'linear', getattr(model, 'fc', None))
    if not isinstance(head, NonidealLinear):
        return
    head.configure(physical=physical, q=q, v_dd=v_dd, template=template,
                   family=family, timing=timing, base_time=base_time, R=R, C=C)
    if isinstance(head, AnalogLinear):
        head._hardware = hardware


def select_model_head(model, kind=None, config=None):
    """Explicit conversion preserving original parameter objects and names."""
    name = 'linear' if hasattr(model, 'linear') else 'fc'
    previous = getattr(model, name, None)
    kind = kind or getattr(model, 'final_head_type', 'old_ideal')
    cfg = validate_config(config or getattr(model, 'final_head_config', None))
    if kind != 'old_ideal':
        if not isinstance(previous, nn.Linear):
            raise TypeError('Nonideal heads require a supported PCN/CIFAR CNN linear or fc classifier')
        with torch.random.fork_rng(devices=[]):
            head = make_final_linear(kind, previous.in_features, previous.out_features,
                                     previous.bias is not None, cfg)
        head.weight, head.bias = previous.weight, previous.bias
        setattr(model, name, head)
    elif isinstance(previous, NonidealLinear):
        raise ValueError('Converting a nonideal checkpoint to old_ideal requires an explicit migration')
    model.final_head_type, model.final_head_config = kind, cfg
    if hasattr(model, 'init_args'):
        model.init_args['model_args'].update(final_head_type=kind, final_head_config=cfg)


def round_clip_ste(x, lo, hi):
    return (x + (x.round() - x).detach()).clamp(lo, hi)


def validate_config(config):
    cfg = dict(HEAD_DEFAULTS, **(config or {}))
    for key in ("final_repr_bits", "final_weight_bits", "final_bias_bits",
                "final_accumulator_bits"):
        value = cfg[key]
        if isinstance(value, bool) or int(value) != value or not 2 <= value <= 32:
            raise ValueError(f"{key} must be an integer in [2, 32]")
        cfg[key] = int(value)
    if not math.isfinite(cfg["final_adc_noise_lsb"]) or cfg["final_adc_noise_lsb"] < 0:
        raise ValueError("final_adc_noise_lsb must be finite and nonnegative")
    return cfg


class NonidealLinear(nn.Linear):
    """A common interface consumed by logits_for_loss and checkpoint loaders."""
    def __init__(self, in_features, out_features, bias=True, config=None):
        super().__init__(in_features, out_features, bias=bias)
        self.head_config = validate_config(config)
        self.physical = False
        self.q = 1.0
        self.v_dd = .5
        self.recipe_gain = 1.0

    def configure(self, *, physical, q, v_dd, **kwargs):
        if not math.isfinite(q) or q <= 0 or not math.isfinite(v_dd) or v_dd <= 0:
            raise ValueError("Classifier q and v_dd must be finite and positive")
        self.physical, self.q, self.v_dd = bool(physical), float(q), float(v_dd)

    def logits_for_loss(self, output):
        return (output / self.q if self.physical else output).to(self.weight.dtype)


class DigitalLinear(NonidealLinear):
    """Signed ADC + integer MVM. No analog timing or coupler model."""
    def _weight_normalizer(self):
        maximum = self.weight.detach().double().abs().max()
        return torch.where(maximum > 0,
                           maximum.clamp_min(torch.finfo(torch.float64).tiny).reciprocal(),
                           maximum.new_tensor(1.))

    def logits_for_loss(self, output):
        if not self.physical or not self.head_config['final_head_quantize']:
            return super().logits_for_loss(output)
        # Derive from master parameters, not mutable per-replica forward state.
        sw = ((1 << (self.head_config['final_weight_bits'] - 1)) - 1) * self._weight_normalizer()
        sv = self.v_dd / (1 << (self.head_config['final_repr_bits'] - 1))
        return (output * (sv / (self.q * sw))).to(self.weight.dtype)

    def forward(self, inputs):
        if not self.physical:
            return F.linear(inputs, self.weight, self.bias)
        cfg = self.head_config
        if not cfg["final_head_quantize"]:
            x = inputs.clamp(-self.v_dd, self.v_dd) if cfg["final_head_clamp"] else inputs
            return F.linear(x, self.weight, None if self.bias is None else self.q * self.bias)
        from ode_pc import PulseQuantizationImpl
        Q = (1 << (cfg["final_weight_bits"] - 1)) - 1
        # Same arithmetic/STE as PulseSymQuantizeWeight; no pulse simulation.
        norm = self._weight_normalizer()
        sw = Q * norm
        sv = self.v_dd / (1 << (cfg["final_repr_bits"] - 1))
        # Float64 exactly represents integer accumulations through 53 bits.
        # Reject configurations whose worst-case unbounded sum exceeds that.
        adc_limit = 1 << (cfg["final_repr_bits"] - 1)
        bias_limit = 1 << (cfg["final_bias_bits"] - 1)
        if self.in_features * adc_limit * Q + bias_limit > (1 << 53):
            raise ValueError("Digital accumulator emulation exceeds exact float64 integer range")
        x = inputs
        if cfg["final_adc_noise_lsb"]:
            x = x + torch.randn_like(x) * (cfg["final_adc_noise_lsb"] * sv)
        a = round_clip_ste(x.double() / sv, -adc_limit, adc_limit - 1)
        Q_tensor = norm.new_tensor(Q)
        K = PulseQuantizationImpl.apply(self.weight.double(), norm, -Q_tensor, Q_tensor)
        b = None if self.bias is None else round_clip_ste(
            self.bias.double() * (self.q * sw.double() / sv), -bias_limit, bias_limit - 1)
        acc = F.linear(a, K, b)
        limit = 1 << (cfg["final_accumulator_bits"] - 1)
        if cfg["final_head_clamp"]:
            acc = acc.clamp(-limit, limit - 1)
        return acc


class AnalogLinear(NonidealLinear):
    """Bias-augmented one-shot MVM using the existing physical FF kernels."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Runtime-only circuitry: no duplicate trainable parameters/checkpoint keys.
        object.__setattr__(self, "_circuit", None)
        object.__setattr__(self, "_expanded_module", None)
        object.__setattr__(self, "_template", None)
        self.expanded = False

    def _apply(self, fn, recurse=True):
        # Runtime circuitry is deliberately not registered in state_dict.
        object.__setattr__(self, '_circuit', None)
        object.__setattr__(self, '_expanded_module', None)
        return super()._apply(fn, recurse=recurse)

    def train(self, mode=True):
        if mode:
            object.__setattr__(self, '_expanded_module', None)
        return super().train(mode)

    def _pulse_mode(self):
        return self.head_config['final_head_quantize'] and self._runtime['family'] == 'toggle' and (
            self.expanded or getattr(self._template, 'physical_level', 2) == 3 or
            (not hasattr(self._template, 'physical_level') and
             getattr(self._template, 'supports_pulse_training', False)))

    def begin_evaluation_trial(self):
        self.expanded = True
        object.__setattr__(self, '_circuit', None)
        object.__setattr__(self, '_expanded_module', None)

    def configure(self, *, template=None, family="toggle", physical=False,
                  q=1., v_dd=.5, timing="derived", base_time=5e-9,
                  R=50e3, C=500e-15, **kwargs):
        super().configure(physical=physical, q=q, v_dd=v_dd)
        if timing not in ("fixed", "derived") or min(R, C, base_time) <= 0:
            raise ValueError("Invalid analog classifier timing")
        self.recipe_gain = base_time / (R * C) if timing == "fixed" else 1.
        self._runtime = dict(family=family, timing=timing, base_time=base_time, R=R, C=C)
        object.__setattr__(self, "_template", template)
        object.__setattr__(self, "_circuit", None)
        object.__setattr__(self, "_expanded_module", None)

    def _build_circuit(self, inputs):
        from physical_feedforward import AveragedPhysicalBasicBlock, PulsePhysicalBasicBlock
        from physical_feedforward_tc import TCPhysicalBasicBlock
        template = self._template
        if template is None:
            raise RuntimeError("Physical analog head requires the production FF configuration")
        options = {}
        for key in inspect.signature(AveragedPhysicalBasicBlock.__init__).parameters:
            if hasattr(template, key) and key not in (
                    "conv1", "conv2", "physical", "layer_idx", "norm1", "norm2",
                    "act1", "act2", "between", "main_downsample", "post_add_pool", "shortcut"):
                options[key] = getattr(template, key)
        # TC PCN stores its settings on the ODE block as a configuration dict,
        # unlike TC feedforward blocks' public scalar attributes.
        tc_config = getattr(template, '_tc_noise_cfg', {})
        options.update({key: value for key, value in tc_config.items()
                        if key in inspect.signature(AveragedPhysicalBasicBlock.__init__).parameters})
        options.update({key: value for key, value in getattr(self, '_hardware', {}).items()
                        if key in inspect.signature(AveragedPhysicalBasicBlock.__init__).parameters})
        # Level-3 evaluators install corner-specific mismatch after wrapping.
        options['noise_level'] = getattr(template, 'noise_level', options.get('noise_level', 0.))
        runtime = self._runtime
        options.update(R=runtime['R'], C=runtime['C'], v_dd=self.v_dd,
                       one_over_q=self.v_dd/self.q, physical=True,
                       layer_idx=int(getattr(template, "layer_idx", 0)) + 104729,
                       toggle_timing_mode=runtime['timing'], toggle_y_time=runtime['base_time'],
                       z_over_y_time=1.)
        family = runtime['family']
        mapped_tc = family == 'tc' and (isinstance(template, TCPhysicalBasicBlock) or
                                      getattr(template, '_tc_current_mode', False) or
                                      getattr(template, 'tc_nonidealities', False))
        cls = TCPhysicalBasicBlock if mapped_tc else (
            PulsePhysicalBasicBlock if self._pulse_mode() else AveragedPhysicalBasicBlock)
        if mapped_tc:
            options.update(tc_covariance_table=getattr(template, 'tc_covariance_table', None),
                           tc_curve_sampling=getattr(template, 'tc_curve_sampling',
                                                     getattr(template, '_tc_curve_sampling', 'histogram')),
                           tc_noise_reference_R=getattr(template, 'tc_noise_reference_R',
                                                       tc_config.get('tc_noise_reference_R', 50e3)),
                           one_shot_conv=True)
        with torch.random.fork_rng(devices=[]):
            conv = nn.Conv2d(self.in_features + int(self.bias is not None), self.out_features, 1, bias=False)
        block = cls(conv, **options).to(inputs)
        block.weight_scale = getattr(template, 'weight_scale', 1.)
        # The matrix is supplied differentiably from this head on every forward.
        del conv._parameters['weight']
        if not self.head_config['final_head_clamp']:
            block.project_state = lambda x: x
        for name in ('_nonlinear_R_training_pkg', '_nonlinear_R_pkg'):
            if hasattr(template, name):
                setattr(block, name, dict(getattr(template, name)))
        package = getattr(template, '_tc_curve_package',
                          getattr(template, '_tc_resistance_package', None))
        if package is not None:
            block._tc_resistance_package = package
            block._tc_curve_seed = getattr(template, 'nonlinear_R_curve_seed', 4096) or 4096
        object.__setattr__(self, '_circuit', block)
        self._built_pulse_mode = self._pulse_mode()
        return block

    def forward(self, inputs):
        if not self.physical:
            return self.recipe_gain * F.linear(inputs, self.weight, self.bias)
        from ode_pc import QuantizationImpl, PulseQuantizationImpl, _symmetric_qat_weight_scale
        if self._circuit is not None and self._built_pulse_mode != self._pulse_mode():
            object.__setattr__(self, '_circuit', None)
            object.__setattr__(self, '_expanded_module', None)
        block = self._circuit or self._build_circuit(inputs)
        block.train(self.training)
        augmented = self.weight if self.bias is None else torch.cat((self.weight, self.bias[:, None]), 1)
        if self.head_config['final_head_quantize']:
            scale = _symmetric_qat_weight_scale(augmented, block.weight_quant_factor_bits)
            if not torch.isfinite(scale):
                scale = augmented.new_tensor(1.)
            impl = PulseQuantizationImpl if self._pulse_mode() else QuantizationImpl
            limit = augmented.new_tensor(block.q_hi)
            weights = (impl.apply(augmented, scale, -limit, limit)
                       if impl is PulseQuantizationImpl else
                       impl.apply(augmented, scale, -limit, limit, block.weight_scale))
            block._uses_quantized_weight_scale = True
        else:
            scale, weights = augmented.new_tensor(1.), augmented
            block._uses_quantized_weight_scale = False
        block.scale1.copy_(scale)
        if hasattr(block.conv1, 'mat'):
            raise RuntimeError("Analog inference matrix must be rebuilt after changing parameters")
        block.conv1.weight = weights[:, :, None, None]
        block.clean_params['conv1'] = block.conv1.weight
        if not self.expanded and block.noise_level:
            # Same per-forward training/static evaluation mismatch lifecycle.
            if self.training or block._training_pulse_mismatch is None:
                block.begin_training_pulse_mismatch(block.noise_level, block.mismatch_type)
        if self.expanded:
            if self.training:
                raise ValueError("Expanded analog head is inference-only; use dense QAT for training")
            from validation import MVMConv
            dense_conv = block.conv1
            # Hold physical assignments throughout the trial, not per batch.
            cached = getattr(self, '_expanded_module', None)
            version = (self.weight._version, None if self.bias is None else self.bias._version)
            if getattr(self, '_expanded_version', None) != version:
                cached = None
            if cached is None:
                mat = weights.detach().to_sparse_csr()
                mvm = MVMConv(mat, dict(padding=0, stride=1, ker_h=1, ker_w=1,
                                       inp_chan=weights.shape[1], out_chan=self.out_features))
                mvm.clean_mat_values = mat.values().clone()
                from validation import build_unrolled_dtc_metadata
                mvm.set_dtc_metadata(*build_unrolled_dtc_metadata(
                    mat, (weights.shape[1], 1, 1), mvm.meta))
                if hasattr(block, '_nonlinear_R_pkg'):
                    pkg = dict(block._nonlinear_R_pkg)
                    if pkg.get('nonlinear_R_curve_seed') is not None:
                        pkg['nonlinear_R_curve_seed'] += 104729
                    mvm.enable_csv(**pkg)
                mvm.add_noise(block.noise_level, block.mismatch_type,
                              q_hi=block.q_hi, weight_scale=block.weight_scale)
                object.__setattr__(self, '_expanded_module', mvm)
                self._expanded_version = version
                cached = mvm
            block.conv1 = cached
        x = inputs if self.bias is None else torch.cat((inputs, inputs.new_full((inputs.shape[0], 1), self.q)), 1)
        try:
            output = block(x[:, :, None, None]).flatten(1)
        finally:
            if self.expanded:
                block.conv1 = dense_conv
        return output


def make_final_linear(kind, in_features, out_features, bias=True, config=None):
    cls = {'analog': AnalogLinear, 'digital': DigitalLinear}.get(kind)
    if cls is None:
        raise ValueError(f"Unknown nonideal head: {kind}")
    return cls(in_features, out_features, bias=bias, config=config)
