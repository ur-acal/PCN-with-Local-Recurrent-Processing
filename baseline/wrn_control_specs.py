"""Pinned WRN controls; row 1 is an existing reference, not a training job."""

SIZES = ('16_2', '16_4', '28_2', '28_4')
DATASETS = ('cifar10', 'cifar100')
SLURM_ROWS = (2, 4, 5, 6, 7, 8, 9, 10)
# Row 10 is the additional standard-WRN LR/default-initialization control.
ROWS = {
    2: (False, True, False, False),
    3: (False, False, False, False),
    4: (True, False, True, False),
    5: (False, True, True, False),
    6: (False, False, True, False),
    7: (True, False, True, True),
    8: (False, True, True, True),
    9: (False, False, True, True),
    10: (True, False, False, False),
}


def model_name(row, size):
    if row not in ROWS or size not in SIZES:
        raise ValueError(f'Unknown WRN control: row={row}, size={size}')
    return f'wrn_{size}_cifar_control_r{row}'


def model_options(row):
    bn, bias, shortcut, main = ROWS[row]
    return dict(use_batchnorm=bn, conv_bias=bias,
                avgpool_downsample_shortcut=shortcut, avgpool_main_downsample=main,
                init_mode='pytorch' if row == 10 else 'wrn',
                dropout_rate=0.0, final_dropout_rate=0.25)


def training_override(row):
    return ('num_epochs=300,eval_every=5,weight_decay=0.001,'
            'dropout_rate=0.0,final_dropout_rate=0.25,'
            f'lr={0.01 if row == 10 else 0.1}')
