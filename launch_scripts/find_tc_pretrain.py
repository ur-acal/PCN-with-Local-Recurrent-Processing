"""Find one completed PCN pretrain in an explicitly selected experiment root.

Legacy checkpoints store architecture/epoch but encode ODE state and image type
in the model name. Check both, plus root preprocessing metadata; never use logs.
Only load trusted, locally supplied checkpoints (their pickle contains classes).
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch


def find_pretrain(root, *, inp, out, pool, task, img_type, block,
                  pcn='PCNetNoBatchNorm', bits=None, center=False,
                  stride=(1,), kernel=(3,), selection='last', normalize_student_input=None):
    root = Path(root).resolve(strict=True)
    config = json.loads((root / 'run_config.json').read_text())
    if (config.get('input_quant_bits'), config.get('center_student_input')) != (bits, center):
        raise ValueError('Experiment preprocessing metadata does not match request')
    image = img_type.lower()
    if image not in ('rgb', 'cifair', 'scangfi'):
        raise ValueError('Unsupported image type for verified TC discovery')
    inp = list(inp)
    inp[0] = 3 if image == 'rgb' else 4
    classes = {'cifar10': 10, 'cifar100': 100}[task]
    expand = lambda values: list(values) * len(out) if len(values) == 1 else list(values)
    matches, rejected = [], []
    for directory in sorted(root.iterdir()):
        name = directory.name
        path = directory / f'{name}_{selection}_ckpt.pth'
        if not directory.is_dir() or not path.is_file() or 'QAT' in name:
            continue
        if directory.resolve().parent != root or path.resolve().parent != directory.resolve():
            raise ValueError(f'Checkpoint escapes requested experiment: {path}')
        if f'_{block}_' not in name:
            continue
        try:
            recorded_image = ('cifair' if '_CiFAIR_' in name else
                              'scangfi' if '_scanGFI_' in name else 'rgb')
            if recorded_image != image or ('_C100_' in name) != (classes == 100):
                raise ValueError('dataset/image name mismatch')
            d = torch.load(path, map_location='cpu', weights_only=False)
            if (image == 'rgb' and normalize_student_input is not None and
                    d.get('student_preprocessing', {}).get('normalize_student_input', True)
                    != normalize_student_input):
                raise ValueError('RGB input normalization mismatch')
            m = d['init_args']['model_args']
            expected = dict(inp_channels=inp, out_channels=list(out), max_pool=list(pool),
                            num_classes=classes, stride=expand(stride), kernel_size=expand(kernel),
                            avg_pooling=True, first_bn=False)
            for key, value in expected.items():
                if m.get(key) != value:
                    raise ValueError(f'{key} mismatch')
            if d['net_type'] != pcn or m['pc_conv_layer'].__name__ != 'PCConvReLU6':
                raise ValueError('model/activation class mismatch')
            kw = d['init_args']['kwargs']
            if any(kw.get(k) is not False for k in ('bias', 'tie_weights', 'tie_bp', 'bypass')):
                raise ValueError('unsupported architecture flags')
            # Best checkpoints alone do not establish completion of pretraining.
            completed = d if selection == 'last' else torch.load(
                directory / f'{name}_last_ckpt.pth', map_location='cpu', weights_only=False)
            if completed.get('epoch') != 300 or completed.get('acc') is None:
                raise ValueError('no completed 300-epoch pretraining checkpoint')
            net = d['net']
            if any('parametrizations.' in k for k in net):
                raise ValueError('QAT checkpoint is not pretraining')
            if net['linear.weight'].shape[0] != classes:
                raise ValueError('classifier shape mismatch')
            for i, channels in enumerate(out):
                if net[f'PcConvs.{i}.FFconv.weight'].shape[:2] != (channels, inp[i]):
                    raise ValueError(f'layer {i} weight shape mismatch')
            matches.append(path)
        except (KeyError, ValueError, OSError, RuntimeError, AttributeError) as exc:
            rejected.append(f'{name}: {exc}')
    if len(matches) != 1:
        raise ValueError(f'Expected one matching pretrain in {root}; found {len(matches)}. '
                         + '; '.join(str(p) for p in matches) + '; '.join(rejected))
    return matches[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    for key in ('inp', 'out', 'pool', 'stride', 'kernel'):
        p.add_argument('--' + key, required=True, help='Space-separated integers')
    for key in ('task', 'img_type', 'block', 'pcn'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--bits', default='none')
    p.add_argument('--center', choices=('true', 'false'), default='false')
    p.add_argument('--selection', choices=('last', 'best'), default='last')
    from rgb_teacher_preprocessing import normalization_bool
    p.add_argument('--normalize_student_input', type=normalization_bool,
                   default=os.environ.get('NORMALIZE_STUDENT_INPUT') or None)
    a = vars(p.parse_args())
    for key in ('inp', 'out', 'pool', 'stride', 'kernel'):
        a[key] = [int(v) for v in a[key].split()]
    a['bits'] = None if a['bits'].lower() == 'none' else int(a['bits'])
    a['center'] = a['center'] == 'true'
    try:
        path = find_pretrain(**a)
    except (ValueError, OSError) as exc:
        p.exit(2, f'Pretrain discovery failed: {exc}\n')
    print(f'Verified pretrained checkpoint: {path}', file=sys.stderr)
    print(path.parent.name)


if __name__ == '__main__':
    main()
