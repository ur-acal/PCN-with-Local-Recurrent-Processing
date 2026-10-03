"""Fresh defaults and checkpoint compatibility at the training entry point."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('explicit,checkpoint,expected', [
    (None, None, 'all'),
    ('pc_only', None, 'pc_only'),
    ('all', None, 'all'),
    (None, {}, 'pc_only'),
    (None, {'measured_activation_scope': 'all'}, 'all'),
    (None, {'measured_activation_scope': 'pc_only'}, 'pc_only'),
    ('all', {}, 'all'),
    ('pc_only', {'measured_activation_scope': 'all'}, 'pc_only'),
    ('all', {'measured_activation_scope': 'pc_only'}, 'all'),
])
def test_training_scope_resolution(explicit, checkpoint, expected):
    # Execute the actual entry-point assignment without launching training or
    # duplicating its expression in the test.
    path = Path(__file__).resolve().parents[1] / 'baseline/train_baseline_cifar.py'
    nodes = [node for node in ast.walk(ast.parse(path.read_text()))
             if isinstance(node, ast.Assign)
             and any(ast.unparse(target) == 'args.measured_activation_scope'
                     for target in node.targets)]
    assert len(nodes) == 1
    args = SimpleNamespace(measured_activation_scope=explicit)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'),
         {'args': args, 'checkpoint': checkpoint})
    assert args.measured_activation_scope == expected
