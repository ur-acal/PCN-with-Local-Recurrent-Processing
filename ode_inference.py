import torch
import torch.nn as nn
import torchvision
import os
import pickle
import argparse
import logging

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from copy import deepcopy
from thop import profile, clever_format

from pc_conv import PCConvNoisy, PCConv, PartialTiedPCConv
from pc_model import PCNet, PCNetWithMiddleConv, PCN_CLASSES, PC_CONV_CLASS
from inference_utils import load_and_prepare_model, replace_transpose_conv, get_test_data, test_once
from ode_pc import make_ode_block, is_adaptive, ODEBLOCK_CLASSES, ODEWrapper_CLASSES, wrap_ode_block
from cross_sim_inference import calibrate_input

import logging
log = logging.getLogger(__name__)
log.propagate = False

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PCConvNoisy tests with custom noise parameters"
    )
    parser.add_argument("--model_dir",  type=str, required=True,
                        help="Directory containing the saved model checkpoint")
    parser.add_argument("--ckpt", type=str, default="best")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Identifier or filename of the model to load")
    parser.add_argument("--img_type", type=str, default="rgb")
    parser.add_argument("--pc_conv", type=str, choices=list(PC_CONV_CLASS.keys())+[None],
                   default=None)
    parser.add_argument("--ode_block", type=str, choices=list(ODEBLOCK_CLASSES.keys()) + [None],
                        default=None)
    parser.add_argument("--ode_wrapper", type=str, choices=list(ODEWrapper_CLASSES.keys()) + [None],
                        default=None)
    parser.add_argument("--state_calib", type=str, required=False,
                        help="The calibration result of ode intermediate states for each layer")
    parser.add_argument("--R", type=float, default=1e5, help="Resistance")
    parser.add_argument("--C", type=float, default=49e-15, help="Capacitance")
    parser.add_argument("--v_dd", type=float, default=1.0, help="V_DD")
    parser.add_argument("--w_bits", type=int, default=8, help="weight quantized bits")
    parser.add_argument("--w_quant_mode", type=str, default="min_max", help="min_max or perc")
    parser.add_argument("--w_perc", type=float, default=0.99999, help="percentile for quantization")
    parser.add_argument("--method", type=str, default="dopri5")
    parser.add_argument("--t_end", type=float, default=None, help="Stop time of the solver")
    parser.add_argument("--n_steps", type=float, default=10, help="ODE solver number of steps")
    parser.add_argument("--tol", type=float, default=1e-3, help="ODE solver tolerance")
    parser.add_argument("--ts_scale", type=int, default=10,
                        help="Scale factor of time step; Applies to fixed grid methods")
    parser.add_argument("--d_start", type=float, default=0.1,
                        help="Difference between real t_end and position to start sweep")
    parser.add_argument("--d_end", type=float, default=0.1,
                        help="Difference between position to end sweep and real t_end")
    parser.add_argument("--n_sweep_left", type=int, default=5, help="Number of swept t_end")
    parser.add_argument("--n_sweep_right", type=int, default=5, help="Number of swept t_end")
    parser.add_argument("--conv_only", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False)
    parser.add_argument("--mem_frac", type=float, default=1.0)
    parser.add_argument("--count_mac", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False)
    parser.add_argument("--test_only", type=lambda v: v.lower() in ('yes','true','t','1'),
                        default=False)
    return parser.parse_args()


def get_t_end(args):
    if "TEnd" in args.model_name:
        return float(args.model_name.split("TEnd")[0].split("_")[-1])
    elif "CLS" in args.model_name and "LRPC" in args.model_name:
        cycles = float(args.model_name.split("CLS")[0].split("_")[-1])
        lr_pc = float(args.model_name.split("LRPC")[0].split("_")[-1])
        return cycles * lr_pc
    else:
        assert args.t_end is not None
        return args.t_end


def run_test_only(args, test_dataloader, ckpt_path, pc_conv, device):
    logging.info("----- Running one forward pass for model: {} -----".format(args.model_name))
    t_end = get_t_end(args)
    noisy_params = {"noise_level": 0.2, "weight": None}
    ode_params = {"ode_block": ODEBLOCK_CLASSES[args.ode_block], "t_end": t_end, "method": args.method,
                  "tol": args.tol, "ts_scale": args.ts_scale, "n_steps": args.n_steps}
    wrapper_params = {"ode_wrapper": ODEWrapper_CLASSES[args.ode_wrapper], "calib_path": args.state_calib,
                      "R": args.R, "C": args.C, "v_dd": args.v_dd, "w_bits": args.w_bits,
                      "w_quant_mode": args.w_quant_mode,
                      "w_perc": args.w_perc} if args.ode_wrapper is not None else None
    net_ = load_and_prepare_model(model_path=ckpt_path, device=device, model_struct=PCNet,
                                  pc_conv_layer=pc_conv, data_parallel=False,
                                  noise_to_bn=True, noise_to_linear=True,
                                  fuse_bn=False, conv_only=args.conv_only, ode_params=ode_params,
                                  ode_wrapper_params=wrapper_params,
                                  **noisy_params)
    logging.warning("Model input channels: {}".format(net_.ics))
    logging.warning("Model output channels: {}".format(net_.ocs))
    logging.warning("Model pooling layers: {}".format(net_.max_pool))
    net_.eval()
    # test_batch = next(iter(test_dataloader))[0].to(device)[:512]
    # _ = net_(test_batch)
    # _, predicted_raw = torch.max(_, 1)
    if not args.count_mac:
        test_once(net_, test_dataloader, device, args.model_name)
    # net_.recover_params()
    # test_once(net_, test_dataloader, device)

    logging.info("===== Inspecting the range of the weights =====")
    for _name, _p in net_.named_parameters():
        if "ff" in _name.lower():
            print("Name: {}, max: {}, min: {}, median: {}, mean: {}, sum's mean: {}".format(
                _name, _p.max(), _p.min(), _p.median(), _p.mean(), _p.view(_p.shape[0], -1).sum(-1).mean()))
        elif "fb" in _name.lower():
            print("Name: {}, max: {}, min: {}, median: {}, mean: {}, sum's mean: {}".format(
                _name, _p.max(), _p.min(), _p.median(), _p.mean(), _p.view(_p.shape[1], -1).sum(-1).mean()))
        else:
            print("Name: {}, max: {}, min: {}, median: {}, mean: {}".format(
                _name, _p.max(), _p.min(), _p.median(), _p.mean()))
    input_ranges = calibrate_input(model=net_, device=device, model_name=args.model_name,
                                   calib_bs=256,
                                   calib_samples=256, percentile=0.995,
                                   symmetric=False, save_to=None, img_type=args.img_type)
    for _t, _range in input_ranges.items():
        print("Type: {}".format(_t))
        print(_range)
        print("================================")

    if args.count_mac:
        with torch.no_grad():
            macs, params = profile(net_, inputs=(next(iter(test_dataloader))[0].to(device)[:1],), verbose=False)
            logging.warning("Model Params: {}, Solver: {}, N steps: {}, MAC Count: {}".format(
                clever_format([params], "%.1f"), args.method, args.n_steps, clever_format([macs], "%.1f")
            ))


def run_ode_inference():
    args = parse_args()
    if args.test_only:
        # set level in the very beginning before calling logging.warning, otherwise the line below will not work
        logging.basicConfig(level=logging.INFO)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, device=0)

    # Get pc_conv_layer to use
    pc_conv = PC_CONV_CLASS.get(args.pc_conv, PCConvNoisy)
    logging.warning("----- Using PC Conv layer: {} -----".format(pc_conv.__name__))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataloader = get_test_data(test_bs=128, img_type=args.img_type)
    ckpt_path = os.path.join(args.model_dir, args.model_name, args.model_name + "_{}_ckpt.pth".format(args.ckpt))

    with torch.no_grad():
        if args.test_only:
            run_test_only(args, test_dataloader, ckpt_path, pc_conv, device)
            exit(0)

    # noise_level_list_ = [0, 0.05, 0.1, 0.15, .20, .25, .30, .35, .40]
    noise_level_list_ = [0, 0.1, .20, .30, .40]
    noisy_trials = 20
    gt_t_end = get_t_end(args)

    # Get t_end_list for experiments
    # For single t_end, set d_start=0, d_end=1, n_sweep=1
    sweep_start, sweep_end = gt_t_end - args.d_start, gt_t_end + args.d_end
    t_end_before, t_end_after = torch.tensor([]), torch.tensor([])
    if args.d_start > 0:
        t_end_before = torch.arange(sweep_start, gt_t_end, args.d_start / args.n_sweep_left, dtype=torch.float32)
        t_end_before = t_end_before[t_end_before < gt_t_end]
    if args.d_end > 0:
        t_end_after = torch.arange(gt_t_end, sweep_end, args.d_end / args.n_sweep_right, dtype=torch.float32)
    t_end_list = torch.cat([t_end_before, t_end_after]).tolist()

    logging.warning("Running ODE pcn inference, method: {}, tol: {}".format(args.method, args.tol))
    acc_dict = {}
    for t_end in t_end_list:
        logging.warning("Current t_end: {}, ground truth t_end: {}".format(t_end, gt_t_end))
        ode_params = {"ode_block": ODEBLOCK_CLASSES[args.ode_block], "t_end": t_end, "method": args.method,
                      "tol": args.tol, "ts_scale": args.ts_scale, "n_steps": args.n_steps}
        wrapper_params = {"ode_wrapper": ODEWrapper_CLASSES[args.ode_wrapper], "calib_path": args.state_calib,
                          "R": args.R, "C": args.C, "v_dd": args.v_dd, "w_bits": args.w_bits,
                          "w_quant_mode": args.w_quant_mode,
                          "w_perc": args.w_perc} if args.ode_wrapper is not None else None
        noise_acc_spec = {}
        max_real_t, min_real_t, real_t_end = t_end, t_end, t_end
        for noise_level in noise_level_list_:
            trials = noisy_trials if noise_level > 0 else 1
            acc_list = []
            for t in range(trials):
                noisy_params = {"noise_level": noise_level, "weight": None}
                with torch.no_grad():
                    net_ = load_and_prepare_model(model_path=ckpt_path, device=device, model_struct=PCNet,
                                                  pc_conv_layer=pc_conv, data_parallel=False,
                                                  noise_to_bn=True, noise_to_linear=True,
                                                  fuse_bn=False, conv_only=args.conv_only, ode_params=ode_params,
                                                  ode_wrapper_params=wrapper_params,
                                                  **noisy_params)
                real_t_list = torch.tensor([_.integration_time[-1].cpu() for _ in net_.PcConvs])
                max_real_t, min_real_t, avg_real_t = real_t_list.max(), real_t_list.min(), real_t_list.mean()
                max_real_t, min_real_t, avg_real_t = f"{max_real_t.item():.2g}", f"{min_real_t.item():.2g}", f"{avg_real_t.item():.2g}"
                real_t_end = avg_real_t
                net_.eval()
                total = 0
                correct = 0

                for batch_idx, (inputs, targets) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), disable=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    with torch.no_grad():
                        output_tensor = net_(inputs)
                        if torch.isnan(output_tensor).any():
                            logging.warning("=====> Output tensor contains nan values. <=====")

                    # Get the predicted class
                    _, predicted = torch.max(output_tensor, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                # Calculate the accuracy
                accuracy = 100 * correct / total
                acc_list.append(accuracy)
                log.warning(f'Test Accuracy at noise level {noise_level}: {accuracy:.2f}%')
            avg_acc = sum(acc_list) / len(acc_list)
            noise_acc_spec[noise_level] = acc_list
            log.warning("Average test acc over {} trials is {}".format(trials, avg_acc))
        ###################################################################################################
        # Noisy Experiment finished for one t_end
        ###################################################################################################
        log.warning("-------- Final Result ODEBlock with t_end: {}, real_t_end: {} --------".format(t_end, real_t_end))
        log.warning("-------- Model name: {} --------".format(args.model_name))
        if args.ode_wrapper is not None:
            log.warning("wrapper params: {}".format(wrapper_params))
        for _nl, _acc in noise_acc_spec.items():
            log.warning("t_end: {}, real_t_end: {}, min_real_t: {}, max_real_t: {}, Noise level: {}, Acc:{:.2f}%".format(
                t_end, real_t_end, min_real_t, max_real_t, _nl, sum(_acc) / len(_acc)))

        acc_dict[real_t_end] = {"noise_acc_spec": noise_acc_spec, "t": (t_end, real_t_end, min_real_t, max_real_t)}

    # save noise acc spec to a pkl
    if args.ode_wrapper is None:
        spec_path = os.path.join(
            "logs/ode_noisy_acc", "TEnd{}_{}_{}_{}_{}NoiseLevel.pkl".format(
                str(round(t_end_list[0], 2)).replace(".", "p"),
                str(round(t_end_list[-1], 2)).replace(".", "p"),
                args.method, args.model_name, len(noise_level_list_)))
    else:
        spec_path = os.path.join(
            "logs/ode_noisy_acc", "TEnd{}_{}_{}Cap_{}_{}_{}{}b_{}NL.pkl".format(
                str(round(t_end_list[0], 2)).replace(".", "p"),
                str(round(t_end_list[-1], 2)).replace(".", "p"),
                args.C, args.method, args.model_name, args.ode_wrapper, args.w_bits,
                len(noise_level_list_)))
    with open(spec_path, "wb") as fp:
        pickle.dump(acc_dict, fp)
    log.warning("-------- ODEBlock Noisy experiment finished, spec saved to {} --------".format(spec_path))


if __name__ == "__main__":
    run_ode_inference()
