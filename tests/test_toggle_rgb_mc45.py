"""RGB MC45 forwarding without changing historical standalone inference."""
import sys
from unittest.mock import patch

from test_pcn_mc45_pipeline_options import resolve_eval
from scripts.run_toggle_nonideality_ablation import parse_args, build_command


def test_rgb_pipeline_forwards_task_format_and_batch_size(tmp_path):
    for task in ('cifar10', 'cifar100'):
        argv = resolve_eval(tmp_path, TASK=task, IMG_TYPE='rgb',
                            POST_FT_EVAL_BATCH_SIZE='32')
        with patch.object(sys, 'argv', argv[1:]):
            args = parse_args()
        command = build_command(args, 'all_known')
        assert command[command.index('--task') + 1] == task
        assert command[command.index('--img_type') + 1] == 'rgb'
        assert command[command.index('--test_bs') + 1] == '32'


def test_omitted_eval_batch_size_retains_128(tmp_path):
    argv = resolve_eval(tmp_path)
    assert argv[argv.index('--test_bs') + 1] == '128'


def test_explicit_task_and_format_override_model_name_inference():
    with patch.object(sys, 'argv', ['runner', '--model_name', 'TIMMQAT_C100_CiFAIR',
                                   '--task', 'cifar10', '--img_type', 'rgb']):
        command = build_command(parse_args(), 'all_known')
    assert command[command.index('--task') + 1] == 'cifar10'
    assert command[command.index('--img_type') + 1] == 'rgb'


def test_legacy_standalone_model_name_inference_is_unchanged():
    for name, task, img in (('TIMMQAT_C100_CiFAIR', 'cifar100', 'CiFAIR'),
                            ('TIMMQAT_plain', 'cifar10', 'scanGFI')):
        with patch.object(sys, 'argv', ['runner', '--model_name', name]):
            command = build_command(parse_args(), 'all_known')
        assert command[command.index('--task') + 1] == task
        assert command[command.index('--img_type') + 1] == img
