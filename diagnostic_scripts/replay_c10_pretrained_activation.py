"""Read-only accuracy replay of C10 PCN pretrained activation hypotheses.

Reuses production model loading, toggle dynamics, measured activation and data
helpers. Does not quantize weights, add hardware defects, train, or write caches.
"""
import argparse
import functools
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import inference_utils
from inference_utils import get_test_data, load_and_prepare_model
from ode_pc import ToggleODEXInitFFFB, make_ode_block
from pc_conv import PCConvReLU6
from final_linear import configure_model_head
from train_ode_cifar import configure_unitless_measured_activation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['old_ideal', 'old_0906', 'old_1022',
                                          'new_1022_control'], required=True)
    parser.add_argument('--batches', type=int, default=0)
    options = parser.parse_args()
    assert torch.cuda.is_available(), 'GPU unavailable; refusing unintended CPU run'
    random.seed(4096)
    np.random.seed(4096)
    torch.manual_seed(4096)
    torch.cuda.manual_seed_all(4096)
    if options.case.startswith('old_'):
        run = 'C10_C96_454_16L_allReLU_analog_biasfalse_iq12_toggle_odexinit'
        selector = 'best'
    else:
        run = ('C10_C96L16_cc_randSpin_fullRange_1022ReLU_vdd0p5_analog_'
               'finalOnly_warmPT5_iq12_toggle_odexinit')
        selector = 'last'
    path = next((ROOT/'saved_ckpt_runs'/run).glob(
        f'TIMMPCNet*1REP/*_{selector}_ckpt.pth'))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = torch.load(path, map_location='cpu', weights_only=False)
    model = load_and_prepare_model(
        str(path), 'cuda', pc_conv_layer=PCConvReLU6,
        fuse_bn=False, conv_only=False, noise_level=0.,
        noise_to_bn=False, noise_to_linear=False)
    configure_model_head(model, physical=False, q=.1, v_dd=.5,
                         timing='fixed', base_time=10e-9, R=50e3, C=500e-15)
    measured = options.case != 'old_ideal'
    bank = '0906_RELU_Voltage' if options.case == 'old_0906' else '1022_RELU_Voltage'
    model = make_ode_block(
        model, ode_block=ToggleODEXInitFFFB, noise_level=0.,
        method='dopri5', t_end=1.75, tol=1e-4, n_steps=5,
        toggle_n_cycles=5, toggle_fast_path=True,
        toggle_timing_mode='fixed', toggle_y_time=10e-9, z_over_y_time=1.,
        toggle_timing_R=50e3, toggle_timing_C=500e-15,
        odexinit_scaling_mode='direct',
        unitless_measured_pullback_mode='direct' if measured else 'none',
        unitless_pullback_q=.1)
    if measured:
        configure_unitless_measured_activation(
            model, str(ROOT/'hardware_data/mc_45_corners'/bank/'tt_25_1.csv'),
            corner='MC18', num_parameters=10,
            normalize_positive_endpoint=False, fit_constraint='auto',
            physical_v_dd=.5, pullback_scale=.1,
            interpolation='piecewise_linear', fuse_measured_activation=True)
    model.eval()
    # TrainerCiFar's official validation dataset explicitly uses seed=0;
    # the generic inference helper otherwise leaves the dataset seed unset.
    inference_utils.MyNoiseCIFARDataset = functools.partial(
        inference_utils.MyNoiseCIFARDataset, seed=0)
    loader = get_test_data(test_bs=128, img_type='CiFAIR', task='cifar10',
                           input_quant_bits=12, center_student_input=False)
    correct = total = 0
    with torch.no_grad():
        for index, (inputs, targets) in enumerate(loader):
            outputs = model(inputs.cuda())
            correct += int(outputs.argmax(1).eq(targets.cuda()).sum())
            total += targets.numel()
            if (index+1) % 10 == 0:
                print(options.case, index+1, 'accuracy', 100*correct/total, flush=True)
            if options.batches and index+1 >= options.batches:
                break
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    report = dict(case=options.case, checkpoint=str(path), sha256=before,
                  recorded_accuracy_percent=100*metadata['acc'],
                  samples=total, accuracy_percent=100*correct/total,
                  measured=measured, measured_bank=bank if measured else None,
                  pullback_q=.1, timing='fixed', Ty_seconds=10e-9,
                  Tz_seconds=10e-9, cycles=5, VDD=.5,
                  weights_quantized=False, input_bits=12, sensor_dataset_seed=0,
                  analog_head_physical=False)
    output = ROOT/'results/C10_pretrained_activation_replay'
    output.mkdir(exist_ok=True)
    (output/f'{options.case}_{total}samples.json').write_text(
        json.dumps(report, indent=2)+'\n')
    print('PRETRAIN_REPLAY', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
