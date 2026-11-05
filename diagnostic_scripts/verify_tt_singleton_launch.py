"""Validate real queued launcher commands without running training/evaluation.

Uses production argument parsers, deserializes FT input checkpoints on CPU,
and resolves/validates every corner's actual child inference command. Future
FT output checkpoints are checked for consistent paths, not fabricated.
"""
import io
import math
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "baseline")]


def parse(script, arguments, function="parse_args"):
    namespace = runpy.run_path(str(ROOT / script), run_name="launch_preflight")
    with patch.object(sys, "argv", [script, *arguments]):
        return namespace, namespace[function]()


def readable_checkpoint(path):
    import torch
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state.get("net", state.get("state_dict", state.get("model_state_dict", state)))
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Empty checkpoint: {path}")
    tensors = [v for v in weights.values() if isinstance(v, torch.Tensor)]
    if not tensors or any(not torch.isfinite(t).all() for t in tensors):
        raise ValueError(f"Invalid checkpoint tensors: {path}")
    return str(path)


def check_training(script, namespace, args):
    import numpy as np
    # Confirm the singleton is an actual paired-column conductance bank.
    bank = np.loadtxt(args.nonlinear_R_table, delimiter=",", skiprows=1)
    assert bank.shape[1] in (2, 4), bank.shape
    assert np.isfinite(bank).all() and (bank[:, 1::2] > 0).all()
    if bank.shape[1] == 4:
        assert np.array_equal(bank[:, :2], bank[:, 2:])
    assert args.fuse_measured_activation and args.enable_measured_activation
    epochs = (args.num_epochs if script == "train_ode_cifar.py" else
              namespace["parse_kv_overrides"](args.override)["num_epochs"])
    assert args.enable_measured_pooling and epochs == 50
    assert Path(args.activation_curve_path).is_file()
    assert Path(args.teacher_ckpt).is_file()
    assert shutil.which("scangen"), "scangen missing from queued environment PATH"
    from measured_activation import PiecewiseLinearActivation
    activation = PiecewiseLinearActivation(args.activation_curve_path, args.v_dd,
                                          corner=args.activation_corner)
    import torch
    assert torch.isfinite(activation(torch.zeros(1))).all()
    if script == "train_ode_cifar.py":
        raw = os.environ["MODEL_NAME_OVERRIDE"]
        source = Path(args.save_path) / raw / (raw + "_best_ckpt.pth")
        assert args.toggle_timing_mode == "derived" and args.one_over_q == 1
    else:
        source = args.resume_checkpoint
        assert args.toggle_timing_mode == "fixed"
        assert abs(args.toggle_y_time - 15e-9) < 1e-18
        expected_q = float(os.environ["WRN_ONE_OVER_Q"])
        assert math.isfinite(expected_q) and expected_q > 0
        assert args.one_over_q == expected_q
    return readable_checkpoint(source)


def check_evaluation(script, namespace, args):
    from data_utils import MC45CornerData
    catalog = MC45CornerData(
        args.mc_45_corner_dir,
        spin_variation_source=args.mc_spin_variation_source,
        dtc_pulse_width_variation_source=args.mc_dtc_pulse_width_variation_source,
        relu_monte_carlo_source=args.mc_relu_monte_carlo_source,
        coupler_nonlinear_variation_source=args.mc_coupler_nonlinear_variation_source,
        coupler_nonlinear_variation_quantity=args.mc_coupler_nonlinear_variation_quantity,
        coupler_nominal_R=args.mc_coupler_nominal_R)
    assert len(catalog.corners) == 45 and args.n_trials == 1
    assert not args.corner_ids and args.fuse_measured_activation
    captured = []
    if script == "scripts/run_toggle_nonideality_ablation.py":
        for i, corner in enumerate(catalog.corners):
            command, _ = namespace["build_corner_command"](args, corner, i)
            captured.append(command)
        child = "ode_inference.py"
    else:
        class Captured(Exception):
            pass

        def capture(command, **kwargs):
            captured.append(command)
            raise Captured

        def log_open(path, mode="r", *a, **kw):
            if mode == "w":
                return io.StringIO()
            return original_open(path, mode, *a, **kw)

        original_open = Path.open
        for corner in catalog.corners:
            with patch("subprocess.Popen", capture), patch.object(Path, "open", log_open):
                try:
                    namespace["run_corner"](args, corner, [None])
                except Captured:
                    pass
        child = "baseline/evaluate_physical_feedforward_cifar.py"
    child_namespace = runpy.run_path(str(ROOT / child), run_name="launch_preflight")
    for corner, command in zip(catalog.corners, captured):
        start = command.index(child)
        with patch.object(sys, "argv", command[start:]):
            resolved = child_namespace["parse_args"]()
        assert resolved.fuse_measured_activation
        assert Path(resolved.nonlinear_R_table).is_file()
        assert Path(resolved.activation_curve_path).is_file()
        assert len(corner["coupler"]["curve_indices"]) == 100
    assert len(captured) == 45
    return len(captured)


def s2_rehearsal_assertions(script, arguments):
    """Assert the real command, executed blocks, wrappers and saved metadata."""
    import torch
    import ode_pc

    def flag(key, default=None):
        key = '--' + key
        assert arguments.count(key) <= 1, ('duplicate argument', key)
        return arguments[arguments.index(key) + 1] if key in arguments else default

    evaluation = script == 'ode_inference.py'
    ft = not evaluation and flag('model_name') is not None
    stage = 'eval' if evaluation else 'ft' if ft else 'pretrain'
    pulse = evaluation or (ft and os.environ['TOGGLE_MODE'] == 's2_dtc_pulse')
    expected_block = 'S2DTCPulseFFFB' if pulse else 'S2DTCFFFB'
    expected_steps = os.environ.get('S2_DTC_SUBSTEPS', '2' if pulse else '1')
    if evaluation:
        expected_steps = os.environ.get('S2_DTC_EVAL_SUBSTEPS', '2')
    elif ft:
        expected_steps = os.environ.get('S2_DTC_FT_SUBSTEPS', expected_steps)
    expected_method = os.environ.get('S2_DTC_METHOD', 'rk2')
    if evaluation:
        expected_method = os.environ.get('S2_DTC_EVAL_METHOD', expected_method)
    expected_y = os.environ.get('S2_Y_INIT_WITH', 'x')
    expected_z = os.environ.get('S2_Z_INIT_WITH', '0')
    expected_pairs = int(os.environ.get('S2_DTC_ORDERING_PAIRS', '0')) if ft else 0
    assert int(flag('s2_dtc_ordering_pairs', '0')) == expected_pairs
    assert flag('ode_block') == expected_block
    assert flag('method') == expected_method
    assert flag('s2_dtc_substeps') == expected_steps, (stage, flag('s2_dtc_substeps'), expected_steps)
    assert flag('y_init_with', 'x') == expected_y
    assert flag('z_init_with', '0') == expected_z
    assert flag('toggle_timing_mode') == 'fixed'
    assert float(flag('toggle_y_time')) == 1e-8
    assert float(flag('z_over_y_time')) == 1
    assert flag('measured_activation_scope') == 'all'
    assert flag('enable_measured_activation') == 'true'
    if ft or evaluation:
        expected_wrapper = 'S2DTCQATTester2State' if evaluation else 'S2DTCQATWrapper2State'
        assert flag('ode_wrapper') == expected_wrapper
        assert flag('ckpt') == 'last', (stage, flag('ckpt'))
        assert flag('enable_measured_pooling') == 'true'

    observed = set()
    wrappers = set()
    original_wrap = ode_pc.wrap_ode_block
    correction_calls = []
    original_correction = ode_pc._S2DTCMixin._s2_ordering_correction
    def checked_correction(block, *args, **kwargs):
        correction_calls.append(block.layer_idx)
        result = original_correction(block, *args, **kwargs)
        assert all(torch.isfinite(v).all() for v in result)
        return result
    ode_pc._S2DTCMixin._s2_ordering_correction = checked_correction
    gradient_layers = set()

    def checked_wrap(*args, **kwargs):
        result = original_wrap(*args, **kwargs)
        for wrapper in result[1]:
            wrappers.add(type(wrapper).__name__)
            assert type(wrapper).__name__ == expected_wrapper
        return result
    if ft or evaluation:
        ode_pc.wrap_ode_block = checked_wrap

    def block_check(module, inputs):
        if not getattr(module, 's2_dtc', False) or id(module) in observed:
            return
        assert type(module).__name__ == expected_block
        assert module.method == expected_method
        assert module.s2_dtc_substeps == int(expected_steps)
        assert module.s2_dtc_ordering_pairs == expected_pairs
        assert module.y_init_with == expected_y and module.z_init_with == expected_z
        assert module.toggle_n_cycles == 5 and module.reset_z is False
        assert bool(getattr(module, 'physical', False)) == (ft or evaluation)
        if ft or evaluation:
            assert module.enable_spin_variation
            assert module.enable_summing_current_noise and module.enable_coupler_noise
        assert 'measured' in type(module.act_fn).__name__.lower() or 'piecewise' in type(module.act_fn).__name__.lower()
        observed.add(id(module))
        if ft and expected_pairs:
            for conv in (module.FFconv, module.FBconv):
                def check_gradient(gradient, layer=module.layer_idx):
                    assert torch.isfinite(gradient).all()
                    if gradient.abs().max() > 0:
                        gradient_layers.add(layer)
                conv.parametrizations.weight.original.register_hook(check_gradient)
        print('REHEARSAL_BLOCK_PASS ' + json.dumps(dict(stage=stage, layer=module.layer_idx,
              block=expected_block, method=module.method, substeps=module.s2_dtc_substeps,
              y_init=module.y_init_with, z_init=module.z_init_with)), flush=True)

    hook = torch.nn.modules.module.register_module_forward_pre_hook(block_check)
    original_save = torch.save
    saves = []

    def checked_save(obj, path, *a, **kw):
        if isinstance(obj, dict) and 's2_dtc' in obj.get('init_args', {}):
            meta = obj['init_args']['s2_dtc']
            assert meta['ode_block'] == expected_block and meta['method'] == expected_method
            assert meta['s2_dtc_substeps'] == int(expected_steps)
            assert meta.get('s2_dtc_ordering_pairs', 0) == expected_pairs
            assert meta['y_init_with'] == expected_y and meta['z_init_with'] == expected_z
            saves.append(str(path))
        return original_save(obj, path, *a, **kw)
    torch.save = checked_save

    def finish():
        hook.remove()
        ode_pc._S2DTCMixin._s2_ordering_correction = original_correction
        if expected_pairs:
            assert correction_calls and len(gradient_layers) == len(observed), (correction_calls, gradient_layers)
        else:
            assert not correction_calls
        assert observed, 'No S2 computation observed'
        if ft or evaluation:
            assert wrappers, 'No physical wrapper observed'
        if not evaluation:
            assert any(p.endswith('_last_ckpt.pth') for p in saves), saves
        print('REHEARSAL_STAGE_PASS ' + json.dumps(dict(stage=stage, blocks=len(observed),
              ordering_pairs=expected_pairs, correction_calls=len(correction_calls), gradient_layers=sorted(gradient_layers),
              wrappers=sorted(wrappers), checkpoints=saves)), flush=True)
    return finish


def s2_dtc_rehearsal_child(arguments):
    """Opt-in real pipeline rehearsal, never imported by production launchers.

    Only shorten epochs/loaders/evaluation. Run the actual training entrypoint,
    teacher, optimizer, checkpoint writer and inference subprocesses.
    """
    import torch
    if arguments[0] == "-u":
        arguments = arguments[1:]
    script, arguments = arguments[0], arguments[1:]
    torch.set_num_threads(2)
    finish = None
    if script in ('train_ode_cifar.py', 'ode_inference.py'):
        finish = s2_rehearsal_assertions(script, arguments)
    if script == "train_ode_cifar.py":
        from trainer_timm import TrainerCiFarTimmStyle
        from torch.utils.data import DataLoader, Subset
        original = TrainerCiFarTimmStyle._prepare_cifar

        def limited_data(self, *args, **kwargs):
            original(self, *args, **kwargs)
            batch_path = os.environ.get('S2_DIAGNOSTIC_BATCH')
            if batch_path and not Path(batch_path).exists():
                batch = next(iter(DataLoader(Subset(self.val_set, range(128)), batch_size=128, num_workers=0)))
                torch.save(batch, batch_path)
            self.train_set = Subset(self.train_set, range(4))
            self.val_set = Subset(self.val_set, range(4))
            self.train_dataloader = DataLoader(self.train_set, batch_size=2, num_workers=0)
            self.val_dataloader = DataLoader(self.val_set, batch_size=2, num_workers=0)
            self.teacher_eval_loader = self.val_dataloader
        TrainerCiFarTimmStyle._prepare_cifar = limited_data
        arguments += ["--num_epochs", "1", "--batch_size", "2", "--num_workers", "0",
                      "--warmup_epoch", "0", "--skip_eval_epochs", "0", "--eval_every", "1",
                      "--health_check_epochs", "", "--final_eval_only", "true"]
    elif script == "scripts/run_toggle_nonideality_ablation.py":
        arguments += ["--max_eval_batches", "1", "--test_bs", "2", "--n_trials", "1"]
        # Children normally invoke sys.executable directly, bypassing the shell
        # shim. Instrument them too, without changing production evaluation.
        import subprocess
        original_popen = subprocess.Popen

        def checked_popen(command, *a, **kw):
            if isinstance(command, list) and 'ode_inference.py' in command:
                child = command[command.index('ode_inference.py'):]
                command = [sys.executable, str(Path(__file__)), '--s2-dtc-rehearsal-child', *child]
            return original_popen(command, *a, **kw)
        subprocess.Popen = checked_popen
    elif script == "shell_utils/pipeline_node_run_parse.py":
        # Its output directory is global. Do not mix rehearsal rows with results.
        print("REHEARSAL: skipped global summary parser")
        return
    print("REHEARSAL_COMMAND " + json.dumps([script, *arguments]), flush=True)
    sys.argv = [script, *arguments]
    runpy.run_path(str(ROOT / script), run_name="__main__")
    if finish is not None:
        finish()


def main():
    arguments = sys.argv[1:]
    if arguments[0] == "--s2-dtc-rehearsal-child":
        return s2_dtc_rehearsal_child(arguments[1:])
    if arguments[0] == "--check-handoff":
        records = [json.loads(line.split(" ", 1)[1])
                   for line in Path(arguments[1]).read_text().splitlines()
                   if line.startswith("PREFLIGHT_PASS ")]
        assert len(records) == 2
        assert records[0]["expected_checkpoint"] == records[1]["checkpoint"]
        print("PREFLIGHT_HANDOFF_PASS " + records[1]["checkpoint"])
        return
    if arguments[0] == "-u":
        arguments.pop(0)
    script = arguments.pop(0)
    function = "get_args" if script == "train_ode_cifar.py" else "parse_args"
    namespace, args = parse(script, arguments, function)
    if script in ("train_ode_cifar.py", "baseline/train_baseline_cifar.py"):
        source = check_training(script, namespace, args)
        if script == "train_ode_cifar.py":
            name = namespace["_constr_model_name"](args, rep=1)
            expected = Path(args.output_save_path) / name / (name + "_best_ckpt.pth")
        else:
            from input_preprocessing import append_preprocessing_suffix, resolve_preprocessing
            bits, center = resolve_preprocessing(args.resume_checkpoint,
                                                 args.input_quant_bits, args.center_student_input)
            output = append_preprocessing_suffix(args.output_dir, bits, center)
            name = f"{args.case}_{args.dataset}_{args.model_name}"
            expected = Path(output) / args.dataset / args.case / args.model_name / name / (name + "_full_param_best_ckpt.pth")
        print("PREFLIGHT_PASS " + json.dumps({"script": script, "source": source,
              "singleton": args.nonlinear_R_table, "epochs": 50,
              "expected_checkpoint": str(expected)}))
    else:
        corners = check_evaluation(script, namespace, args)
        checkpoint = (str(Path(args.model_dir) / args.model_name / (args.model_name + "_best_ckpt.pth"))
                      if script == "scripts/run_toggle_nonideality_ablation.py" else args.checkpoint)
        print("PREFLIGHT_PASS " + json.dumps({"script": script, "corners": corners,
              "trials": args.n_trials, "family": args.mc_coupler_nonlinear_variation_source,
              "checkpoint": checkpoint}))


if __name__ == "__main__":
    main()
