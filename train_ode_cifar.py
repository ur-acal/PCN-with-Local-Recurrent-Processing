from typing import List
from collections import OrderedDict
import torch
import os
import argparse
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ode_pc import ODEBLOCK_CLASSES, make_ode_block, ODEWrapper_CLASSES, wrap_ode_block, QUANTIZER_CLASSES


class RGGBToRGBTransform(nn.Module):
    """Transform RGGB (B, 4, 16, 16) to RGB (B, 3, 32, 32).

    RGGB format assumes 4 channels packed from a 2x2 Bayer pattern:
    - Channel 0: Red
    - Channel 1: Green (from position 0,1)
    - Channel 2: Green (from position 1,0)
    - Channel 3: Blue

    The transformation upsamples each channel to 32x32 using bilinear interpolation
    and averages the two green channels to produce standard RGB format.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (B, 4, 16, 16) - RGGB format
        # Split channels: R, G1, G2, B
        r = x[:, 0:1, :, :]   # (B, 1, 16, 16)
        g1 = x[:, 1:2, :, :]  # (B, 1, 16, 16)
        g2 = x[:, 2:3, :, :]  # (B, 1, 16, 16)
        b = x[:, 3:4, :, :]   # (B, 1, 16, 16)

        # Upsample each channel to 32x32 using bilinear interpolation
        r_up = F.interpolate(r, scale_factor=2, mode='bilinear', align_corners=False)
        g1_up = F.interpolate(g1, scale_factor=2, mode='bilinear', align_corners=False)
        g2_up = F.interpolate(g2, scale_factor=2, mode='bilinear', align_corners=False)
        b_up = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)

        # Average the two G channels
        g = (g1_up + g2_up) / 2.0

        # Combine to RGB: (B, 3, 32, 32)
        rgb = torch.cat([r_up, g, b_up], dim=1)

        return rgb


class ModelWithRGGBTransform(nn.Module):
    """Wrapper that applies RGGB to RGB transform before the model.

    Supports both direct construction and reconstruction from saved init_args.
    Stores ODE conversion params to ensure reconstructed model matches original.
    Delegates attribute access to the inner model for compatibility with
    training/saving code that expects attributes like PcConvs, etc.
    """
    # Registry of inner model classes for reconstruction
    _model_registry = {}
    # Registry of pc_conv classes for reconstruction
    _pc_conv_registry = {}
    # ODE params for reconstruction (set by set_ode_params)
    _ode_params = None
    # Original pc_conv class (before ODE conversion)
    _original_pc_conv_class = None

    @classmethod
    def register_model_class(cls, model_class):
        """Register a model class for reconstruction."""
        cls._model_registry[model_class.__name__] = model_class

    @classmethod
    def register_pc_conv_class(cls, pc_conv_class):
        """Register a pc_conv class for reconstruction."""
        cls._pc_conv_registry[pc_conv_class.__name__] = pc_conv_class

    @classmethod
    def set_original_pc_conv_class(cls, pc_conv_class):
        """Store the original pc_conv class (before ODE conversion) for reconstruction."""
        cls._original_pc_conv_class = pc_conv_class
        cls.register_pc_conv_class(pc_conv_class)

    @classmethod
    def set_ode_params(cls, ode_block, method, t_end, tol, n_steps, **ode_kwargs):
        """Store ODE conversion params for reconstruction."""
        cls._ode_params = {
            'ode_block': ode_block,
            'method': method,
            't_end': t_end,
            'tol': tol,
            'n_steps': n_steps,
            'ode_kwargs': ode_kwargs,
        }

    def __init__(self, model=None, transform=None, _rggb_wrapper_inner_class=None,
                 _rggb_ode_block=None, _rggb_ode_method=None, _rggb_ode_t_end=None,
                 _rggb_ode_tol=None, _rggb_ode_n_steps=None, _rggb_pc_conv_class=None,
                 **kwargs):
        super().__init__()

        # Check if this is reconstruction from saved init_args
        # The trainer merges model_args and kwargs, so _rggb_wrapper_inner_class comes as a kwarg
        if _rggb_wrapper_inner_class is not None:
            # Reconstruction mode
            inner_class_name = _rggb_wrapper_inner_class

            # Get inner model class from registry
            inner_class = self._model_registry.get(inner_class_name)
            if inner_class is None:
                raise ValueError(f"Unknown inner model class: {inner_class_name}. "
                               f"Available: {list(self._model_registry.keys())}")

            # Get pc_conv class from registry if specified
            if _rggb_pc_conv_class and _rggb_pc_conv_class in self._pc_conv_registry:
                kwargs['pc_conv_layer'] = self._pc_conv_registry[_rggb_pc_conv_class]

            self.transform = RGGBToRGBTransform()
            # Create inner model (use _pcnet to avoid ".linear.weight" pattern collision with data_utils.py)
            self._pcnet = inner_class(**kwargs)

            # Apply ODE conversion if params are provided
            if _rggb_ode_block is not None and self._ode_params is not None:
                ode_block = self._ode_params['ode_block']
                self._pcnet = make_ode_block(
                    pc_net=self._pcnet,
                    ode_block=ode_block,
                    noise_level=0.0,
                    method=_rggb_ode_method or self._ode_params['method'],
                    t_end=_rggb_ode_t_end or self._ode_params['t_end'],
                    tol=_rggb_ode_tol or self._ode_params['tol'],
                    n_steps=_rggb_ode_n_steps or self._ode_params['n_steps'],
                    **self._ode_params.get('ode_kwargs', {})
                )

            self._inner_class_name = inner_class_name
            self._pc_conv_class_name = _rggb_pc_conv_class
        else:
            # Direct construction mode
            if model is None:
                raise ValueError("model must be provided for direct construction")
            self.transform = transform if transform is not None else RGGBToRGBTransform()
            # Use _pcnet to avoid ".linear.weight" pattern collision with data_utils.py
            self._pcnet = model
            self._inner_class_name = model.__class__.__name__
            # Use the original pc_conv class (set before ODE conversion)
            self._pc_conv_class_name = None
            if self._original_pc_conv_class is not None:
                self._pc_conv_class_name = self._original_pc_conv_class.__name__
            # Register the inner model class
            self.register_model_class(model.__class__)

    @property
    def init_args(self):
        """Return init_args that can reconstruct this wrapped model.

        The trainer extracts model_args and kwargs from init_args and merges them,
        so we put the wrapper flags inside model_args.
        """
        inner_init_args = self._pcnet.init_args if hasattr(self._pcnet, 'init_args') else {}
        model_args = inner_init_args.get('model_args', {}).copy()
        model_args['_rggb_wrapper_inner_class'] = self._inner_class_name

        # Include ODE params for reconstruction
        if self._ode_params is not None:
            model_args['_rggb_ode_block'] = self._ode_params['ode_block'].__name__
            model_args['_rggb_ode_method'] = self._ode_params['method']
            model_args['_rggb_ode_t_end'] = self._ode_params['t_end']
            model_args['_rggb_ode_tol'] = self._ode_params['tol']
            model_args['_rggb_ode_n_steps'] = self._ode_params['n_steps']

        # Include pc_conv class for reconstruction
        if self._pc_conv_class_name:
            model_args['_rggb_pc_conv_class'] = self._pc_conv_class_name

        return {
            'model_args': model_args,
            'kwargs': inner_init_args.get('kwargs', {}),
        }

    def forward(self, x):
        x = self.transform(x)
        return self._pcnet(x)

    def state_dict(self, *args, **kwargs):
        """Override to rename .linear. keys to .fc. to avoid data_utils.py pattern collision."""
        sd = super().state_dict(*args, **kwargs)
        # Rename .linear. to .fc. to avoid triggering quantization mapping in data_utils.py
        result = {}
        for k, v in sd.items():
            if '.linear.' in k:
                new_key = k.replace('.linear.', '.fc.')
                result[new_key] = v
            else:
                result[k] = v
        return result

    def load_state_dict(self, state_dict, strict=True):
        """Override to handle both .linear. and .fc. key formats."""
        # Map .fc. keys back to .linear. for loading
        mapped_sd = {}
        for k, v in state_dict.items():
            if '.fc.' in k:
                new_key = k.replace('.fc.', '.linear.')
                mapped_sd[new_key] = v
            else:
                mapped_sd[k] = v
        return super().load_state_dict(mapped_sd, strict=strict)

    def __getattr__(self, name):
        # Don't delegate init_args - we have our own property for that
        if name == 'init_args':
            raise AttributeError(name)
        # First try to get from this module's __dict__ or nn.Module
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Delegate to the inner model
            return getattr(self._pcnet, name)


from pc_conv import PCConv, PartialTiedPCConv
from pc_model import PCNet, PCNetWithMiddleConv, PCN_CLASSES, PC_CONV_CLASS
from trainer import TrainerCiFar
from inference_utils import load_and_prepare_model, test_once


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_args():
    p = argparse.ArgumentParser(description="Train PCNet on CIFAR with neural ode")
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_ckpt")
    # TrainerCiFar args
    p.add_argument("--save_path",     type=str,   default=model_save_path)
    p.add_argument("--img_type", type=str, default="rgb")
    p.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--task", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--optim",         type=str,   choices=["SGD", "Adam"], default="SGD",
                   help="optimizer")
    p.add_argument("--weight_decay",  type=float, default=1e-3)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--lr_reduce_on", type=str, default="80,122,150,225,262")
    p.add_argument("--num_epochs",    type=int,   default=300)
    p.add_argument("--max_g_norm", type=float, default=None)
    p.add_argument("--warmup_epoch",  type=int,   default=0)
    p.add_argument("--cosine_t0", type=int, default=None,
                   help="T0 of cosine annealing schedule; if None, using default reduce on epoch scheduler")
    p.add_argument("--aug", type=str2bool, default=False)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--model_name", type=str, default=None,
                   help="Resume from a checkpoint. None means training from scratch")
    p.add_argument("--ckpt", type=str, default="best")
    # PCNet / PCConv args
    p.add_argument("--inp_channels",  type=int, nargs="+", default=[3,  64, 64, 128, 128, 256, 256, 512],
                   help="list of input-channel sizes, e.g. 3 16 32")
    p.add_argument("--out_channels",  type=int, nargs="+", default=[64, 64, 128, 128, 256, 256, 512, 512],
                   help="list of output-channel sizes")
    p.add_argument("--stride", type=int, nargs="+",
                   default=1)
    p.add_argument("--kernel_size", type=int, nargs="+",
                   default=3)
    p.add_argument("--first_ksz", type=int, default=3)
    p.add_argument("--first_stride", type=int, default=1)
    p.add_argument("--first_pad", type=str, choices=["same", "valid"], default="valid")
    p.add_argument("--max_pool",      type=int, nargs="+",
                   default=[False, False, True, False, True, False, False, False])
    p.add_argument("--avg_pooling", type=str2bool, default=False)
    p.add_argument("--separable", type=str, nargs="+", default=None)
    p.add_argument("--patch_dim", type=int, default=None)
    p.add_argument("--num_classes",   type=int, default=10)
    # ODE hyper-params
    p.add_argument("--ode_block", type=str, choices=list(ODEBLOCK_CLASSES.keys()),
                        default="ODEBlockPC")
    p.add_argument("--method", type=str, default="dopri5")
    p.add_argument("--tol", type=float, default=1e-3, help="ODE solver tolerance")
    p.add_argument("--n_steps", type=float, default=10, help="ODE solver number of steps")
    p.add_argument("--t_end", type=float, default=1.0, help="Stop time of the solver")
    p.add_argument("--offset_eps", type=float, default=None,
                   help="Noise level of the offset or noise level of noise in sde (used in wrapper)")
    p.add_argument("--patch_node", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    p.add_argument("--patch_stride", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    p.add_argument("--patch_cycle", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    p.add_argument("--patch_pad", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    p.add_argument("--fold_scalar", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    # Quantization-aware training related args
    p.add_argument("--ode_wrapper", type=str, choices=list(ODEWrapper_CLASSES.keys()) + [None],
                        default=None)
    p.add_argument("--qat_cls", type=str, choices=list(QUANTIZER_CLASSES.keys()) + [None],
                   default=None)
    p.add_argument("--R", type=float, default=1e5, help="Resistance")
    p.add_argument("--R_max", type=lambda s: None if s.lower() in {"none", ""} else float(s),
                   default=None, help="Maximum meaningful Resistance")
    p.add_argument("--C", type=float, default=49e-15, help="Capacitance")
    p.add_argument("--v_dd", type=float, default=1.0, help="V_DD")
    p.add_argument("--enob", type=lambda s: None if s.lower() in {"none", ""} else int(s),
                   default=None, help="The effective number of bits applied to the output spins.")
    p.add_argument("--w_bits", type=int, default=8, help="weight quantized bits")
    p.add_argument("--tie_cap", type=str2bool, default=False)
    p.add_argument("--one_over_q", type=float, default=10, help="1/q")
    # Noise-inject training related args
    p.add_argument('--noise_level', default=None, type=float,
                        help='noise level in noise inject training. None means normal training without noise injection')
    p.add_argument('--noise_type', default='mul', type=str, choices=['mul', 'add'],
                        help='Multiplicative or additive noise')
    p.add_argument("--sde_noise_type", type=str, default="mul", choices=["mul", "add"],
                   help="Only useful when self.eps is set in the ODESolver class")
    p.add_argument("--teacher_ckpt", type=str, default=None,
                   help="Path to a teacher checkpoint for knowledge distillation.")
    p.add_argument("--teacher_arch", type=str, default=None,
                   help="Identifier for the teacher architecture (e.g., efficientnet-b4 or efficientnet_v2_s).")
    p.add_argument("--teacher_arch_source", type=str, default="auto",
                   choices=["auto", "torchvision", "hankyul2"],
                   help="EfficientNetV2 implementation to use for the teacher (auto=detect from checkpoint).")
    p.add_argument("--teacher_input_size", type=int, default=224,
                   help="Spatial size for teacher inputs (resize + optional center crop).")
    p.add_argument("--teacher_center_crop", type=str2bool, default=True,
                   help="Whether to center crop teacher inputs after resizing.")
    p.add_argument(
        "--distill_method",
        type=str,
        default="kd",
        choices=["none", "kd", "crd", "kd_crd", "kd+crd"],
        help="Distillation strategy to apply.",
    )
    p.add_argument(
        "--contrast_method",
        type=str,
        default="memory",
        choices=["memory", "moco"],
        help="Contrast method in CRDLoss.",
    )
    p.add_argument(
        "--neg_sample",
        type=str,
        default="index",
        choices=["index", "label"],
        help="Ways of sampling negative samples in CRD.",
    )
    p.add_argument("--distill_alpha", type=float, default=0.0,
                   help="Weight assigned to the teacher KL loss term.")
    p.add_argument("--distill_temperature", type=float, default=1.0,
                   help="Temperature used in distillation soft targets.")
    p.add_argument("--crd_feat_dim", type=int, default=128, help="Projection dimension for CRD.")
    p.add_argument("--crd_k", type=int, default=16384, help="Number of negatives in the CRD memory bank.")
    p.add_argument("--crd_temperature", type=float, default=0.07, help="Temperature for CRD logits.")
    p.add_argument("--crd_momentum", type=float, default=0.5, help="Momentum for CRD memory updates.")
    p.add_argument("--crd_beta", type=float, default=0.8, help="Weight for the CRD objective.")
    # PCConv hyper-params
    # p.add_argument("--kernel_size",   type=int, default=3)
    # p.add_argument("--stride",        type=int, default=1)
    p.add_argument("--padding",       type=int, default=1)
    p.add_argument("--dropout",       type=float, default=0.0)
    p.add_argument("--bias",          action="store_true")
    p.add_argument("--tie_weights",   type=str2bool, default=False)
    p.add_argument("--tie_bp",        type=str2bool, default=False)
    p.add_argument("--bypass",        type=str2bool, default=False)
    p.add_argument("--relu_bp", type=str2bool, default=False)
    p.add_argument("--use_pc", type=str2bool, default=True)
    p.add_argument("--pcn", type=str, choices=list(PCN_CLASSES.keys()) + [None],
                   default=None)
    p.add_argument("--pc_conv", type=str, choices=list(PC_CONV_CLASS.keys()) + [None],
                        default=None)
    p.add_argument("--tie_method", type=str, choices=["kernel_random", "random", None], default=None,
                   help="method used to select positions in the kernel to tie between FF/FB")
    p.add_argument("--tie_frac", type=float, default=1.0,
                   help="fraction to tie weights of FF/FB")
    p.add_argument("--mem_frac", type=float, default=1.0)
    p.add_argument("--test_only", type=str2bool, default=False)
    p.add_argument("--rggb_to_rgb", type=str2bool, default=False,
                   help="Convert RGGB (16x16x4) input to RGB (32x32x3) before model. "
                        "This will automatically adjust inp_channels[0] from 4 to 3.")
    return p.parse_args()


def evaluate_teacher(model: torch.nn.Module, trainer: TrainerCiFar) -> float:
    device = trainer.device
    dataloader = getattr(trainer, "teacher_eval_loader", trainer.val_dataloader)
    model = model.to(device)
    model.eval()
    correct = 0
    correct_top5 = 0
    total = 0
    compute_top5 = getattr(trainer, "dataset_name", "cifar10") == "cifar100"
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[:2]
            else:
                inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = trainer._prepare_teacher_inputs(inputs)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if compute_top5:
                max_k = min(5, outputs.size(1))
                topk = outputs.topk(max_k, dim=1).indices
                correct_top5 += topk.eq(labels.view(-1, 1)).any(dim=1).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    if compute_top5:
        top5_acc = correct_top5 / total if total > 0 else 0.0
        logging.warning(
            "Teacher evaluation top1: %.2f%%, top5: %.2f%%",
            accuracy * 100.0,
            top5_acc * 100.0,
        )
    else:
        logging.warning("Teacher evaluation accuracy: %.2f%%", accuracy * 100.0)
    return accuracy

def parse_n0n1n2(inp, out):
    # ignore idx 0 (3/4 -> c0), since it's always an "expansion" but not a stage boundary
    exps = [i for i, (ic, oc) in enumerate(zip(inp, out)) if (i != 0 and oc > ic)]

    def count_same(l, r):
        return sum(1 for i in range(l, r) if inp[i] == out[i])

    if len(exps) < 2:
        n0 = count_same(1, len(inp))  # all same-channel layers are c0->c0
        n1 = 0
        n2 = 0
    else:
        e1, e2 = exps[0], exps[1]     # now these are c0->c1 and c1->c2
        n0 = count_same(1, e1)
        n1 = count_same(e1 + 1, e2)
        n2 = count_same(e2 + 1, len(inp))
    return "l".join([str(n0), str(n1), str(n2)])

def _constr_model_name(args, rep=1):
    name_dict = {str(True): "with", str(False): "no"}
    model_name = 'NODE_PPCN'
    if args.pcn is not None and args.pcn != "PCNet":
        model_name = args.pcn
    if args.pc_conv is not None:
        model_name += "_" + args.pc_conv
    if args.offset_eps is not None:
        model_name += "_{}eps".format(args.offset_eps)
    model_name += '_{}'.format(args.ode_block) \
                  + '_{}Solver'.format(args.method) + '_{}TEnd'.format(str(args.t_end)) + '_{}Tol_'.format(str(args.tol)) \
                  + str(args.weight_decay) + 'WD_' \
                  + str(args.batch_size) + 'BS_'
    if args.cosine_t0 is not None:
        model_name += "{}CosLR{}T0_".format(str(args.learning_rate), args.cosine_t0)
    else:
        model_name += str(args.learning_rate) + 'LR_'
    if args.dataset == "cifar100":
        model_name += "C100_"
    _ksz = args.kernel_size if not isinstance(args.kernel_size, List) else args.kernel_size[0]
    _stride = args.stride if not isinstance(args.stride, List) else args.stride[0]
    model_name += "{}K{}S{}C_".format(_ksz, _stride, max(args.inp_channels)) \
                  + str(args.dropout) + 'Dropout_'\
                  + str(len(args.inp_channels)) + "Layers{}_".format(parse_n0n1n2(args.inp_channels, args.out_channels)) \
                  + str(len([_ for _ in args.max_pool if _])) + "Pool"

    if args.tie_method is not None:
        model_name += "_" + args.tie_method + "TieMethod_" + str(args.tie_frac) + "TieFrac"
    distill_method = getattr(args, "distill_method", "none")
    if distill_method and distill_method.lower() != "none":
        method_tag = distill_method.replace(" ", "").replace(",", "-")
        model_name += "_" + method_tag + "Distill"
        if getattr(args, "distill_alpha", 0.0) > 0.0:
            model_name += "_a" + str(args.distill_alpha).replace('.', 'p')
        if getattr(args, "distill_temperature", None) not in (None, 0.0, 1.0):
            model_name += "_t" + str(args.distill_temperature).replace('.', 'p')
    if args.img_type != "rgb":
        model_name += "_" + args.img_type
    if getattr(args, "rggb_to_rgb", False):
        model_name += "_rggb2rgb"
    model_name = model_name + "_" + str(rep) + 'REP'
    if args.model_name is not None or args.ode_wrapper is not None or args.noise_level is not None:
        if args.ode_wrapper is None:
            ft_prefix = "ft"
        elif args.qat_cls is None or args.qat_cls == "SymQuantizeWeight":
            ft_prefix = "QAT{}b".format(args.w_bits)
        else:
            ft_prefix = "QAT{}b{}".format(args.w_bits, args.qat_cls)
        if args.noise_level is not None:
            ft_prefix += "NT{}{}".format(str(args.noise_level).replace('.', 'p'), args.noise_type)
        if args.model_name is not None:
            eps_val = args.model_name.split("_")[2]
            ode_blk = args.model_name.split("_")[3]
            orig_rep = args.model_name.split("_")[-1]
            model_name = ft_prefix + args.model_name.split(orig_rep)[0].replace(
                eps_val, "{}eps".format(args.offset_eps)).replace(
                ode_blk, "{}".format(args.ode_block)) + str(rep) + 'REP'
        else:
            # Add DT short for direct training to distinguish from finetuning.
            model_name = "DT" + ft_prefix + model_name
    return model_name

def get_model_name(args):
    rep = 1
    model_name = _constr_model_name(args, rep)
    model_dir = os.path.join(args.save_path, model_name)
    while os.path.exists(model_dir):
        rep += 1
        model_name = _constr_model_name(args, rep)
        model_dir = os.path.join(args.save_path, model_name)
    return model_name


def _replace_first_conv(model: nn.Module, in_channels: int) -> None:
    first_conv = None
    first_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            first_conv_name = name
            break
    if first_conv is None or first_conv_name is None:
        raise RuntimeError("No Conv2d layer found to update input channels.")
    if first_conv.in_channels == in_channels:
        return
    new_conv = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode,
    )
    with torch.no_grad():
        if in_channels >= first_conv.in_channels:
            new_conv.weight[:, :first_conv.in_channels].copy_(first_conv.weight)
            extra = first_conv.weight.mean(dim=1, keepdim=True)
            repeat = in_channels - first_conv.in_channels
            if repeat > 0:
                new_conv.weight[:, first_conv.in_channels:].copy_(extra.repeat(1, repeat, 1, 1))
        else:
            new_conv.weight.copy_(first_conv.weight[:, :in_channels])
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)
    parent = model
    path, attr = first_conv_name.rsplit(".", 1) if "." in first_conv_name else ("", first_conv_name)
    if path:
        for part in path.split("."):
            parent = getattr(parent, part)
    setattr(parent, attr, new_conv)


def _strip_state_dict_prefix(state_dict, prefix):
    if any(key.startswith(prefix) for key in state_dict):
        return OrderedDict(
            (key.replace(prefix, "", 1), value) for key, value in state_dict.items()
        )
    return state_dict


def _load_teacher_state_dict(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("net", "model", "state_dict"):
            if key in checkpoint:
                nested = checkpoint[key]
                if isinstance(nested, nn.Module):
                    nested = nested.state_dict()
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Teacher checkpoint at {ckpt_path} does not contain a state dict.")
    state_dict = OrderedDict(checkpoint)
    state_dict = _strip_state_dict_prefix(state_dict, "module.")
    state_dict = _strip_state_dict_prefix(state_dict, "model.")
    return state_dict


def _infer_teacher_source(state_dict):
    if any(key.startswith("stem.") for key in state_dict) and any(key.startswith("blocks.") for key in state_dict):
        return "hankyul2"
    if any(key.startswith("features.") for key in state_dict):
        return "torchvision"
    return "unknown"


def build_teacher_model(args, student_in_channels=None):
    method = getattr(args, "distill_method", "none")
    method_lower = method.lower()
    needs_kd = "kd" in method_lower and args.distill_alpha > 0.0
    needs_crd = "crd" in method_lower
    if not args.teacher_ckpt or not (needs_kd or needs_crd):
        return None
    if args.teacher_arch is None:
        raise ValueError("teacher_arch must be specified when using a teacher checkpoint.")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state_dict = _load_teacher_state_dict(args.teacher_ckpt, device)
    inferred_in_channels = None
    if student_in_channels is not None:
        inferred_in_channels = int(student_in_channels)
    elif isinstance(args.inp_channels, list) and args.inp_channels:
        inferred_in_channels = int(args.inp_channels[0])
    elif isinstance(args.inp_channels, int):
        inferred_in_channels = int(args.inp_channels)

    if args.teacher_arch.startswith("efficientnet_v2"):
        arch_source = getattr(args, "teacher_arch_source", "auto")
        if arch_source == "auto":
            inferred_source = _infer_teacher_source(state_dict)
            arch_source = inferred_source if inferred_source != "unknown" else "torchvision"
        logging.warning("Teacher arch source: %s", arch_source)
        if arch_source == "hankyul2":
            teacher_model = torch.hub.load(
                "hankyul2/EfficientNetV2-pytorch",
                args.teacher_arch,
                nclass=args.num_classes,
                skip_validation=True,
            )
        elif arch_source == "torchvision":
            try:
                import torchvision.models as models
            except ImportError as exc:
                raise ImportError(
                    "torchvision is required to load EfficientNetV2 teacher checkpoints."
                ) from exc
            builder = getattr(models, args.teacher_arch, None)
            if builder is None:
                raise ValueError(f"Unsupported EfficientNetV2 architecture: {args.teacher_arch}")
            teacher_model = builder(weights=None)
            classifier = getattr(teacher_model, "classifier", None)
            if isinstance(classifier, nn.Sequential):
                classifier[-1] = nn.Linear(classifier[-1].in_features, args.num_classes)
            elif isinstance(classifier, nn.Linear):
                teacher_model.classifier = nn.Linear(classifier.in_features, args.num_classes)
            else:
                raise RuntimeError("Unexpected classifier head for EfficientNetV2.")
        else:
            raise ValueError(f"Unsupported teacher_arch_source: {arch_source}")
        if inferred_in_channels is not None:
            _replace_first_conv(teacher_model, inferred_in_channels)
    elif args.teacher_arch.startswith("efficientnet"):
        logging.warning("Teacher arch source: {}".format("efficientnet_pytorch"))
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError as exc:
            raise ImportError(
                "efficientnet_pytorch is required to load EfficientNet teacher checkpoints."
            ) from exc
        teacher_model = EfficientNet.from_name(args.teacher_arch)
        if teacher_model._fc.out_features != args.num_classes:
            teacher_model._fc = torch.nn.Linear(teacher_model._fc.in_features, args.num_classes)
        if inferred_in_channels is not None and hasattr(teacher_model, "_change_in_channels"):
            teacher_model._change_in_channels(inferred_in_channels)
    else:
        raise ValueError(f"Unsupported teacher architecture: {args.teacher_arch}")

    missing, unexpected = teacher_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logging.warning("Teacher state dict load: missing=%s unexpected=%s", missing, unexpected)
    teacher_model.to(device)
    teacher_model.eval()
    return teacher_model

def main():
    args = get_args()
    if args.test_only:
        logging.basicConfig(level=logging.INFO)
    if args.dataset == "cifar100" and args.num_classes == 10:
        logging.warning("Overriding num_classes to 100 for CIFAR-100.")
        args.num_classes = 100

    # Handle RGGB to RGB conversion: adjust input channels from 4 to 3
    # Transform upsamples 16x16 to 32x32, model architecture stays the same (just different input channels)
    if args.rggb_to_rgb:
        if isinstance(args.inp_channels, list) and len(args.inp_channels) > 0:
            if args.inp_channels[0] == 4:
                args.inp_channels[0] = 3
                logging.warning("RGGB to RGB mode enabled: adjusted inp_channels[0] from 4 to 3")
            elif args.inp_channels[0] != 3:
                logging.warning("RGGB to RGB mode enabled but inp_channels[0]=%d (expected 4). "
                              "Setting to 3 for RGB input.", args.inp_channels[0])
                args.inp_channels[0] = 3
        logging.warning("RGGB to RGB transformation will be applied: (B, 4, 16, 16) -> (B, 3, 32, 32)")

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, device=0)

    # map optimizer name -> class
    # optim_type = getattr(optim, args.optim)
    loss_fn = nn.CrossEntropyLoss()

    model_args = {
        "inp_channels": args.inp_channels,
        "out_channels": args.out_channels,
        "max_pool": args.max_pool,
        "num_classes": args.num_classes,
        "kernel_size": args.kernel_size if not (isinstance(args.kernel_size, List) and len(args.kernel_size) == 1) else args.kernel_size[0],
        "stride": args.stride if not (isinstance(args.stride, List) and len(args.stride) == 1) else args.stride[0],
        "padding": args.padding if args.patch_dim is None else "same",
        "cls": 0,
        "bias": args.bias,
        "lr": 0.0,
        "tie_weights": args.tie_weights,
        "tie_bp": args.tie_bp,
        "relu_between": True,
        "bypass": args.bypass,
        "relu_bp": args.relu_bp,
        "use_pc": args.use_pc,
        "first_bn": False,
        "dropout": args.dropout,
        "patch_dim": args.patch_dim,
        "separable": args.separable,
        "avg_pooling": args.avg_pooling,
        "first_ksz": args.first_ksz,
        "first_stride": args.first_stride,
        "first_pad": args.first_pad,
    }
    if args.pc_conv and "Noisy" in args.pc_conv:
        default_noise = 0.0 if args.noise_level is None else args.noise_level
        model_args["noise_level"] = float(default_noise)

    # Select PCConv Module to use
    pc_conv_mod = PCConv
    if args.pc_conv is not None:
        pc_conv_mod = PC_CONV_CLASS.get(args.pc_conv, PCConv)
    elif args.tie_method is not None:
        pc_conv_mod = PartialTiedPCConv
        model_args.update({"tie_method": args.tie_method, "tie_frac": args.tie_frac})
    model_args.update({"pc_conv_layer": pc_conv_mod})
    logging.warning("----- Using PC Conv layer: {} -----".format(pc_conv_mod.__name__))

    # Store original pc_conv class for RGGB wrapper reconstruction (before ODE conversion)
    if args.rggb_to_rgb:
        ModelWithRGGBTransform.set_original_pc_conv_class(pc_conv_mod)

    # Select PCNet model to use
    pcn_model = PCN_CLASSES.get(args.pcn, PCNet)
    logging.warning("----- Using PCN model: {} -----".format(pcn_model.__name__))

    # build model
    if args.model_name is None:
        model = pcn_model(**model_args)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        ckpt_path = os.path.join(args.save_path, args.model_name, args.model_name + "_{}_ckpt.pth".format(args.ckpt))
        noisy_params = {"noise_level": 0.0, "weight": None}
        model = load_and_prepare_model(model_path=ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu",
                                       model_struct=pcn_model,
                                       pc_conv_layer=pc_conv_mod, data_parallel=False,
                                       noise_to_bn=False, noise_to_linear=False,
                                       fuse_bn=False, conv_only=False, ode_params=None,
                                       **noisy_params)
        model.dropout = args.dropout

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    model_name = get_model_name(args)
    logging.warning("input channels: {}".format(model.ics))
    logging.warning("output channels: {}".format(model.ocs))
    logging.warning("max pooling: {}".format(model.max_pool))
    logging.warning("pooling layer: {}".format(model.max_pool2d))
    logging.warning("dropout rate: {}".format(model.dropout))
    logging.warning("Total number of parameters: {}".format(total_params / 1e6))
    logging.warning("Model name: {}".format(model_name))
    logging.info("----- Printing out model parameter names: -----")
    for name, param in model.named_parameters():
        logging.info("name: {}, shape: {}, param count: {}".format(name, param.shape, param.numel()))

    # convert block to Neural ode
    # Todo: The offset eps in ode_block is currently useless. Need to pass that to the wrapper.
    ode_kw, ode_kwargs = ["offset_eps", "sde_noise_type", "patch_node", "patch_stride", "patch_cycle", "patch_pad", "fold_scalar"], {}
    for _name, _val in vars(args).items():
        if _name in ode_kw and _val is not None:
            ode_kwargs[_name] = _val
    ode_block = ODEBLOCK_CLASSES[args.ode_block]
    student_in_channels = None
    if hasattr(model, "ics") and getattr(model, "ics"):
        try:
            student_in_channels = int(model.ics[0])
        except (TypeError, ValueError, IndexError):
            student_in_channels = None
    model = make_ode_block(
        pc_net=model, ode_block=ode_block, noise_level=0.0, method=args.method, t_end=args.t_end,
        tol=args.tol, n_steps=args.n_steps, **ode_kwargs)
    logging.warning("PcConv converted to ODEBlock: {}".format(ode_block.__name__))
    logging.warning("t_end: {}".format(args.t_end))
    logging.warning("method: {}".format(args.method))
    logging.warning("tol: {}".format(args.tol))

    # Apply RGGB to RGB transformation wrapper if enabled
    if args.rggb_to_rgb:
        rggb_transform = RGGBToRGBTransform()
        model = ModelWithRGGBTransform(model, rggb_transform)
        # Store ODE params for reconstruction during save/load
        ModelWithRGGBTransform.set_ode_params(
            ode_block=ode_block,
            method=args.method,
            t_end=args.t_end,
            tol=args.tol,
            n_steps=args.n_steps,
            **ode_kwargs
        )
        logging.warning("Model wrapped with RGGB to RGB transformation")

    # sanity check if resume training
    if args.model_name is not None:
        logging.warning("Before training, evaluate the accuracy of the loaded model")
        # test_once(model, device='cuda' if torch.cuda.is_available() else 'cpu', model_name=args.model_name)
        model.train()

    # wrap blocks for QAT
    if args.ode_wrapper is not None:
        wrapper_params = {"ode_wrapper": ODEWrapper_CLASSES[args.ode_wrapper], "calib_path": None,
                          "R": args.R, "R_max": args.R_max, "C": args.C, "v_dd": args.v_dd, "w_bits": args.w_bits,
                          "enob": args.enob, "qat_cls": QUANTIZER_CLASSES[args.qat_cls],
                          "tie_cap": args.tie_cap, "one_over_q": args.one_over_q}
        model, _ = wrap_ode_block(model, **wrapper_params)
        logging.warning("ODEBlock in network wrapped, ode_wrapper_params={}".format(wrapper_params))

    # Get trainer
    logging.warning("Training task: {}".format(args.task))
    logging.warning("lr reduce on: {}, max grad norm: {}, cosine annealing T0: {}".format(
        args.lr_reduce_on, args.max_g_norm, args.cosine_t0))
    teacher_model = None
    method_lower = args.distill_method.lower()
    needs_teacher = ("crd" in method_lower) or ("kd" in method_lower and args.distill_alpha > 0.0)
    if needs_teacher:
        if not args.teacher_ckpt:
            raise ValueError(f"distill_method={args.distill_method} requires --teacher_ckpt.")
        teacher_model = build_teacher_model(args, student_in_channels=student_in_channels)
        logging.warning("Training with {}, contrast method: {}, negative sampling method: {}".format(
            args.distill_method, args.contrast_method, args.neg_sample))
    elif args.teacher_ckpt:
        logging.warning("teacher_ckpt provided but distillation disabled by configuration; ignoring teacher.")
    trainer = TrainerCiFar(
        model         = model,
        model_name    = model_name,
        save_path     = args.save_path,
        batch_size    = args.batch_size,
        optim_type    = args.optim,
        weight_decay  = args.weight_decay,
        loss_fn       = loss_fn,
        learning_rate = args.learning_rate,
        num_epochs    = args.num_epochs,
        warmup_epoch  = args.warmup_epoch,
        lr_reduce_on  = args.lr_reduce_on,
        test_bs       = args.batch_size,
        max_norm      = args.max_g_norm,
        aug           = args.aug,
        T0            = args.cosine_t0,
        eval_every    = args.eval_every,
        img_type      = args.img_type,
        dataset_name  = args.dataset,
        noise_level   = args.noise_level,
        noise_type    = args.noise_type,
        contrast_method = args.contrast_method,
        neg_sample    = args.neg_sample,
        distill_alpha = args.distill_alpha,
        distill_temperature = args.distill_temperature,
        distill_method = args.distill_method,
        crd_feat_dim = args.crd_feat_dim,
        crd_k = args.crd_k,
        crd_temperature = args.crd_temperature,
        crd_momentum = args.crd_momentum,
        crd_beta = args.crd_beta,
        teacher_model = teacher_model,
        teacher_input_size = args.teacher_input_size,
        teacher_center_crop = args.teacher_center_crop,
    )

    if teacher_model is not None:
        evaluate_teacher(teacher_model, trainer)

    if args.test_only:
        _ = model(next(iter(trainer.train_dataloader))[0].to(trainer.device))
        logging.info("Test model forward only. Exit without training the model.")
        logging.info("Result: {}".format(_))
        exit(0)

    trainer.train()

if __name__ == "__main__":
    main()
