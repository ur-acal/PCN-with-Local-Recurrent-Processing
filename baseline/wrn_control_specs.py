"""Pinned WRN controls; row 1 is an existing reference, not a training job."""

SIZES = ('16_2', '16_4', '28_2', '28_4')
DATASETS = ('cifar10', 'cifar100')
SLURM_ROWS = (2, 4, 5, 6, 7, 8, 9, 10)
NEW_POOL_ROWS = (11, 12, 13, 14, 15, 16, 17, 18)
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
    # MaxPool counterparts of rows 4-9; rows 14-16 add before shared pooling.
    11: (True, False, True, False),
    12: (False, True, True, False),
    13: (False, False, True, False),
    14: (True, False, True, True),
    15: (False, True, True, True),
    16: (False, False, True, True),
    # Combined LR/default-init controls: AvgPool and MaxPool, respectively.
    17: (True, False, True, True),
    18: (True, False, True, True),
}


def model_name(row, size):
    if row not in ROWS or size not in SIZES:
        raise ValueError(f'Unknown WRN control: row={row}, size={size}')
    return f'wrn_{size}_cifar_control_r{row}'


def model_options(row):
    bn, bias, shortcut, main = ROWS[row]
    options = dict(use_batchnorm=bn, conv_bias=bias,
                avgpool_downsample_shortcut=shortcut, avgpool_main_downsample=main,
                init_mode='pytorch' if row in (10, 17, 18) else 'wrn',
                dropout_rate=0.0, final_dropout_rate=0.25)
    if row in NEW_POOL_ROWS:
        options.update(pool_type='avg' if row == 17 else 'max', pool_after_add=main)
    return options


def training_override(row):
    return ('num_epochs=300,eval_every=5,weight_decay=0.001,'
            'dropout_rate=0.0,final_dropout_rate=0.25,'
            f'lr={0.01 if row in (10, 17, 18) else 0.1}')
