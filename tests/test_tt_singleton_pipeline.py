"""Opt-in pipeline, q-selection and direct-15ns reparameterization regressions."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class SelectionTests(unittest.TestCase):
    def test_selection_data_root_matches_training_and_explicit_override_wins(self):
        from inference_utils import _get_scangfi_location
        from trainer import _resolve_cifar_data_root
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scanGFI").mkdir()
            for task, name in (("cifar10", "ciFAIR10"), ("cifar100", "ciFAIR100")):
                (root / "scanGFI" / (name + ".h5")).touch()
                with patch.dict(os.environ, {"SCANGEN_DATA_ROOT": directory}, clear=True):
                    expected = str(_resolve_cifar_data_root("CiFAIR", task))
                    self.assertEqual(_get_scangfi_location(task, "CiFAIR"), (expected, name))
                    with patch.dict(os.environ, {"SCAN_TEST_DATA": str(root / "override.h5")}):
                        self.assertEqual(_get_scangfi_location(task, "CiFAIR"), (directory, "override"))

    def test_sensor_noise_is_opt_in_for_calibration_loader(self):
        import inference_utils as data
        import torch
        from torch.utils.data import TensorDataset
        with tempfile.TemporaryDirectory() as directory:
            def configure(command, **kwargs):
                Path(command.split()[-1]).write_text(json.dumps({"noise": {}}))
            with patch.object(data, "_get_scangfi_location", return_value=(directory, "ciFAIR10")), \
                 patch.object(data.subprocess, "run", side_effect=configure), \
                 patch.object(data, "MyNoiseCIFARDataset", return_value=TensorDataset(torch.zeros(2, 1))) as dataset:
                for noisy in (False, True):
                    data.get_bn_calibration_data(img_type="CiFAIR", **({"noisy_inp": True} if noisy else {}))
                    self.assertEqual(dataset.call_args.kwargs["noisy_inp"], noisy)
                    self.assertFalse(dataset.call_args.kwargs["augment"])

    def test_preflight_accepts_selected_q_other_than_five(self):
        from types import SimpleNamespace
        from diagnostic_scripts import verify_tt_singleton_launch as preflight
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "curve.csv"
            bank.write_text("V,G\n-0.1,0.00002\n0.1,0.00002\n")
            args = SimpleNamespace(nonlinear_R_table=str(bank), fuse_measured_activation=True,
                                   enable_measured_activation=True, enable_measured_pooling=True,
                                   override="", activation_curve_path=str(bank), v_dd=0.1,
                                   activation_corner="TT", teacher_ckpt=str(bank),
                                   resume_checkpoint="source", toggle_timing_mode="fixed",
                                   toggle_y_time=15e-9, one_over_q=4)
            with patch.dict(os.environ, {"WRN_ONE_OVER_Q": "4"}), \
                 patch.object(preflight.shutil, "which", return_value="scangen"), \
                 patch("measured_activation.PiecewiseLinearActivation", return_value=lambda x: x), \
                 patch.object(preflight, "readable_checkpoint", return_value="source"):
                self.assertEqual(preflight.check_training("train_baseline_cifar.py",
                                 {"parse_kv_overrides": lambda _: {"num_epochs": 50}}, args), "source")

    def test_every_candidate_uses_same_source_and_deterministic_tie_break(self):
        from baseline.evaluate_physical_feedforward_cifar import parse_args
        from diagnostic_scripts.select_feedforward_one_over_q import select
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "source.pth"
            checkpoint.write_bytes(b"source identity")
            with patch("sys.argv", ["select", "--model_name", "resnet110_cifar",
                                    "--checkpoint", str(checkpoint),
                                    "--evaluation_split", "train"]):
                args = parse_args()
            seen = []
            def evaluate(current):
                seen.append(copy.copy(current))
                return 60 if len(seen) < 3 else {4: 45, 5: 45, 8: 40}[current.one_over_q]
            with patch("baseline.evaluate_physical_feedforward_cifar.evaluate_once", side_effect=evaluate):
                winner = select(args, [8, 5, 4], Path(directory) / "result")
            self.assertEqual(winner, 4)
            self.assertEqual(len(seen), 5)
            self.assertTrue(all(item.checkpoint == str(checkpoint) for item in seen))
            self.assertTrue(all(item.evaluation_split == "train" for item in seen))
            data = json.loads((Path(directory) / "result/selection.json").read_text())
            self.assertEqual(len(data["candidates"]), 3)
            self.assertEqual(data["selected_one_over_q"], 4)

    def test_accuracy_drop_warns_and_completes_selection(self):
        from baseline.evaluate_physical_feedforward_cifar import parse_args
        from diagnostic_scripts.select_feedforward_one_over_q import select
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "source.pth"
            checkpoint.write_bytes(b"source identity")
            output = Path(directory) / "result"
            with patch("sys.argv", ["select", "--model_name", "resnet110_cifar",
                                    "--checkpoint", str(checkpoint)]):
                args = parse_args()
            with patch("baseline.evaluate_physical_feedforward_cifar.evaluate_once",
                       side_effect=[95.3125, 88.203125, 10.0, 12.0]), \
                 patch("builtins.print") as printed:
                winner = select(args, [4, 8], output)
            self.assertEqual(winner, 8)
            data = json.loads((output / "selection.json").read_text())
            self.assertFalse(data["sanity"]["passed"])
            self.assertEqual(data["sanity"]["absolute_difference_pp"], 7.109375)
            self.assertEqual(len(data["candidates"]), 2)
            self.assertEqual((output / "selected_one_over_q.txt").read_text(), "8\n")
            self.assertTrue(any("SELECTION_SANITY_WARNING" in str(call)
                                for call in printed.call_args_list))

    def test_nonfinite_accuracy_still_blocks_selection(self):
        from baseline.evaluate_physical_feedforward_cifar import parse_args
        from diagnostic_scripts.select_feedforward_one_over_q import select
        for values in ([float('nan')], [95., 88., float('inf')]):
            with self.subTest(values=values), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "source.pth"
                checkpoint.touch()
                output = Path(directory) / "result"
                with patch("sys.argv", ["select", "--model_name", "resnet110_cifar",
                                        "--checkpoint", str(checkpoint)]):
                    args = parse_args()
                with patch("baseline.evaluate_physical_feedforward_cifar.evaluate_once",
                           side_effect=values):
                    with self.assertRaisesRegex(RuntimeError, "Non-finite selection accuracy"):
                        select(args, [4], output)
                self.assertFalse((output / "selected_one_over_q.txt").exists())

    def test_failed_candidate_cannot_leave_success_selection(self):
        from baseline.evaluate_physical_feedforward_cifar import parse_args
        from diagnostic_scripts.select_feedforward_one_over_q import select
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "source.pth"
            checkpoint.touch()
            output = Path(directory) / "result"
            output.mkdir()
            (output / "selected_one_over_q.txt").write_text("128\n")
            with patch("sys.argv", ["select", "--model_name", "resnet110_cifar", "--checkpoint", str(checkpoint)]):
                args = parse_args()
            with patch("baseline.evaluate_physical_feedforward_cifar.evaluate_once",
                       side_effect=[60, 60, 40, RuntimeError("candidate failed")]):
                with self.assertRaisesRegex(RuntimeError, "candidate failed"):
                    select(args, [4, 5], Path(directory) / "result")
            self.assertFalse((Path(directory) / "result/selected_one_over_q.txt").exists())

    def test_evaluation_activation_override_is_opt_in(self):
        from baseline import evaluate_physical_feedforward_cifar as evaluator
        for explicit in (False, True):
            argv = ["eval", "--model_name", "resnet110_cifar", "--checkpoint", "unused"]
            if explicit:
                argv += ["--intermediate_activation", "relu6", "--final_dropout_rate", "0.25"]
            with patch("sys.argv", argv):
                args = evaluator.parse_args()
            captured = {}
            def construct(name, cfg, count):
                captured.update(cfg)
                raise RuntimeError("construction reached")
            with patch.object(evaluator, "build_model", side_effect=construct):
                with self.assertRaisesRegex(RuntimeError, "construction reached"):
                    evaluator.evaluate_once(args)
            self.assertEqual(captured.get("intermediate_activation", "relu"),
                             "relu6" if explicit else "relu")
            self.assertEqual(captured.get("final_dropout_rate", 0), 0.25 if explicit else 0)


class DirectTimingTests(unittest.TestCase):
    def test_15ns_matches_5ns_initialization_and_sgd_in_effective_coordinates(self):
        import torch
        from baseline.cifar_resnet import CIFARResNet, WideResNetCIFAR
        from baseline.train_baseline_cifar import scale_feedforward_initial_weights
        from physical_feedforward import convert_wide_resnet_to_physical
        from trainer_timm import TrainerCiFarTimmStyle
        torch.set_num_threads(2)
        constructors = [
            lambda: CIFARResNet(depth=8, base_width=2, in_chans=4, intermediate_activation="relu6"),
            lambda: WideResNetCIFAR(depth=10, base_width=2, widen_factor=2,
                                   in_chans=4, intermediate_activation="relu6",
                                   avgpool_main_downsample=True, avgpool_downsample_shortcut=True),
        ]
        for constructor in constructors:
            with self.subTest(model=constructor):
                torch.manual_seed(4096)
                source = constructor()
                models = []
                for time in (5e-9, 15e-9):
                    model = convert_wide_resnet_to_physical(
                        copy.deepcopy(source), physical=False, physical_level=2,
                        toggle_timing_mode="fixed", toggle_y_time=time,
                        R=50e3, C=500e-15, z_over_y_time=1,
                        qat=False, enable_spin_variation=False,
                        enable_summing_current_noise=False, enable_coupler_noise=False)
                    scale = time / (50e3 * 500e-15)
                    with torch.no_grad():
                        scale_feedforward_initial_weights(model, scale, scale)
                    trainer = object.__new__(TrainerCiFarTimmStyle)
                    trainer.model = model
                    trainer.scale_train_recipe = True
                    trainer.ff_train_scale = trainer.fb_train_scale = scale
                    trainer.bias_lr_multiplier = 1
                    trainer.bias_weight_decay = None
                    groups = trainer._composed_optimizer_parameters(0.1, 0.001)
                    optimizer = torch.optim.SGD(groups, lr=0.1, momentum=0.9, nesterov=True)
                    models.append((model, optimizer, scale))
                x = torch.randn(2, 4, 16, 16)
                for step in range(2):
                    outputs = []
                    for model, optimizer, _ in models:
                        optimizer.zero_grad()
                        prediction = model(x)
                        outputs.append(prediction.detach())
                        prediction.square().mean().backward()
                        optimizer.step()
                    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-5, rtol=1e-4)
                parameters = [dict(model.named_parameters()) for model, _, _ in models]
                for name, parameter in parameters[0].items():
                    factor = 3 if ".ode_block.conv" in name and name.endswith(("weight", "bias")) else 1
                    torch.testing.assert_close(parameter, parameters[1][name] * factor, atol=1e-5, rtol=1e-4)


class StageFailureTests(unittest.TestCase):
    def test_pcn_python_pipe_failure_cannot_reuse_stale_best(self):
        import re
        root = Path(__file__).resolve().parents[1]
        script = root / "launch_scripts/run_cifair100_tt_singleton_ft_eval.sh"
        text = script.read_text()
        raw = re.search(r'^PCN_RAW_NAME="(.*)"$', text, re.M)[1]
        source = re.search(r'^PCN_SOURCE_ROOT="(.*)"$', text, re.M)[1]
        qat = "TIMMQAT5bNoneaNT0p0mul" + raw
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            teacher = directory / "teacher.pth"
            singleton = directory / "singleton.csv"
            source_ckpt = directory / source / raw / (raw + "_best_ckpt.pth")
            stale = directory / "saved_ckpt_runs/stale_ft" / qat / (qat + "_best_ckpt.pth")
            for path in (teacher, singleton, source_ckpt, stale):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            launch = directory / "launch_scripts/run_toggle_ode_mixed_ft.sh"
            launch.parent.mkdir()
            launch.write_text("#!/bin/bash\nset -e\nfalse | tee /dev/null\n")
            env = dict(os.environ, REPO_ROOT=str(directory), MODEL_KIND="pcn",
                       TASK="cifar100", COUPLER_FAMILY="coupler_monte",
                       TT_SINGLETON=str(singleton), TEACHER_CKPT=str(teacher),
                       RUN_TAG="stale", STAGE="ft", REHEARSAL_ONLY="false",
                       PREFLIGHT_ONLY="false")
            for key in ("STUDY_ARTIFACT_ROOT", "PCN_PRETRAIN_ROOT"):
                env.pop(key, None)
            result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)


class StandaloneStageTests(unittest.TestCase):
    def test_existing_production_study_tag_is_rejected_without_overwriting(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            logs = directory / "logs/existing"
            logs.mkdir(parents=True)
            status = logs / "status.log"
            status.write_text("original result\n")
            env = dict(os.environ, REPO_ROOT=str(directory), MODEL_KIND="pcn", TASK="cifar10",
                       COUPLER_MONTE_TT_SINGLETON="unused", COUPLER_MONTE_V2_TT_SINGLETON="unused",
                       STUDY_TAG="existing", REHEARSAL_ONLY="false")
            env.pop("STUDY_ARTIFACT_ROOT", None)
            result = subprocess.run(["bash", str(root / "launch_scripts/run_tt_singleton_study.sh")],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("use a new STUDY_TAG", result.stderr)
            self.assertEqual(status.read_text(), "original result\n")

    def test_resnet_defaults_and_rehearsal_handoff_are_self_contained(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            launch = directory / "launch_scripts"
            launch.mkdir()
            child = '''#!/bin/bash
set -euo pipefail
[[ "$INTERMEDIATE_ACTIVATION" == relu6 && "$FINAL_DROPOUT_RATE" == 0.25 ]]
[[ "$EXTRA_OVERRIDE" == *intermediate_activation=relu6,final_dropout_rate=0.25* ]]
name="custom_noresize_${TASK}_${MODEL_NAME}"
dest="${OUTPUT_DIR}_iq12/${TASK}/custom_noresize/${MODEL_NAME}/${name}"
mkdir -p "$dest"
touch "$dest/${name}_full_param_best_ckpt.pth"
'''
            (launch / "run_feedforward_physical_ft.sh").write_text(child)
            (launch / "run_feedforward_mc45_ablation.sh").write_text('''#!/bin/bash
set -euo pipefail
[[ -f "$MODEL_CKPT" ]]
[[ "$INTERMEDIATE_ACTIVATION" == relu6 && "$FINAL_DROPOUT_RATE" == 0.25 ]]
[[ "$MODEL_CKPT" == "$REHEARSAL_ARTIFACT_ROOT"/* ]]
''')
            source = directory / "source.pth"
            source.touch()
            csv = directory / "singleton.csv"
            csv.touch()
            env = dict(os.environ, REPO_ROOT=str(directory), MODEL_KIND="resnet110",
                       TASK="cifar100", COUPLER_FAMILY="coupler_monte", STAGE="all",
                       TT_SINGLETON=str(csv), TEACHER_CKPT=str(source), CNN_PRETRAIN_CKPT=str(source),
                       WRN_ONE_OVER_Q="4", REHEARSAL_ONLY="true", PREFLIGHT_ONLY="false",
                       REHEARSAL_ARTIFACT_ROOT=str(directory / "rehearsal"))
            for key in ("STUDY_ARTIFACT_ROOT", "CNN_ARCH_OVERRIDE", "INTERMEDIATE_ACTIVATION", "FINAL_DROPOUT_RATE"):
                env.pop(key, None)
            result = subprocess.run(["bash", str(root / "launch_scripts/run_cifair100_tt_singleton_ft_eval.sh")],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((directory / "saved_ckpt_runs").exists())
            self.assertFalse((directory / "logs").exists())


class ToggleFullFTModeTests(unittest.TestCase):
    def test_existing_pipeline_ft_modes_and_mc45_handoff(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            binary = directory / "bin"
            binary.mkdir()
            fake = binary / "python"
            fake.write_text("#!" + sys.executable + '''
import json, os, sys
from pathlib import Path
if sys.argv[1] == '-u': del sys.argv[1]
if sys.argv[1] == '-':
    os.execv(os.environ['REAL_PYTHON'], [os.environ['REAL_PYTHON'], *sys.argv[1:]])
args = sys.argv[2:]
def value(key, default=None):
    return args[args.index(key)+1] if key in args else default
with open(os.environ['TRACE'], 'a') as stream:
    stream.write(json.dumps(dict(script=sys.argv[1], args=args))+'\\n')
if sys.argv[1] == 'baseline/train_baseline_cifar.py':
    if os.environ.get('FAIL_FT') == 'true': sys.exit(7)
    model, task = value('--model_name'), value('--dataset')
    name = f'custom_noresize_{task}_{model}'
    output = Path(value('--output_dir')+'_iq12')/task/'custom_noresize'/model/name
    output.mkdir(parents=True, exist_ok=True)
    for kind in ('best', 'last', 'full_param_best', 'full_param_last'):
        (output/f'{name}_{kind}_ckpt.pth').touch()
''')
            fake.chmod(0o755)
            source = directory / "independent_15ns_best_ckpt.pth"
            source.touch()
            trace = directory / "trace.jsonl"
            env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ["PATH"],
                       REAL_PYTHON=sys.executable, TRACE=str(trace), REPO_ROOT=str(root),
                       MODEL_NAME="wrn_flexible_cifar_avgpool_shortcut", TASK="cifar100",
                       IMG_TYPE="CiFAIR", TC_FEEDFORWARD="false", VALIDATION_MODE="false",
                       FINAL_EVAL_ONLY="false", MODEL_CKPT=str(source),
                       FT_OUTPUT_DIR=str(directory/"ft"), RESULT_PATH=str(directory/"eval"),
                       INPUT_QUANT_BITS="12", CENTER_STUDENT_INPUT="false", WRN_DEPTH="28",
                       WRN_FIRST_STAGE_CHANNELS="27", TOGGLE_TIMING_MODE="fixed",
                       TOGGLE_Y_TIME="15e-9", WEIGHT_QUANT_FACTOR_BITS="1", ENOB="none",
                       R_VAL="50e3", C_VAL="500e-15", FT_NUM_EPOCHS="140",
                       ENABLE_MEASURED_POOLING="true", ENABLE_DTC_NONIDEALITY="false")
            for key in ("EXTRA_OVERRIDE", "N_TRIALS", "MC_COUPLER_NONLINEAR_VARIATION_SOURCE",
                        "ENABLE_BN_RECALIBRATION", "EVAL_ENABLE_DTC_NONIDEALITY"):
                env.pop(key, None)
            for mode in ("ft_only", "ft_and_eval"):
                for family, voltage, q, quantity_variable, quantity in (
                        ("coupler_full_range", "0.5", "20", "NONLINEAR_R_MC_QUANTITY", "conductance"),
                        ("coupler_monte_v2", "0.1", "5", "NONLINEAR_R_MC_QUANTITY", "resistance"),
                        ("coupler_monte_v2", "0.1", "5", "MC_COUPLER_NONLINEAR_VARIATION_QUANTITY", "resistance")):
                    trace.write_text("")
                    current_env = dict(env)
                    for key in ("NONLINEAR_R_MC_QUANTITY", "MC_COUPLER_NONLINEAR_VARIATION_QUANTITY"):
                        current_env.pop(key, None)
                    current_env[quantity_variable] = quantity
                    result = subprocess.run(
                        ["bash", str(root/"launch_scripts/run_feedforward_pretrain_then_ft.sh")],
                        cwd=root, env=dict(current_env, mode=mode, NONLINEAR_R_TABLE=family,
                                           V_DD=voltage, TOGGLE_ONE_OVER_Q=q),
                        text=True, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
                    rows = [json.loads(line) for line in trace.read_text().splitlines()]
                    self.assertEqual(len(rows), 1 if mode == "ft_only" else 2)
                    train = rows[0]["args"]
                    self.assertEqual(train[train.index("--resume_checkpoint")+1], str(source))
                    self.assertIn("num_epochs=140", train[train.index("--override")+1])
                    self.assertEqual(train[train.index("--nonlinear_R_mc_quantity")+1], quantity)
                    if mode == "ft_and_eval":
                        self.assertEqual(rows[1]["script"], "scripts/run_feedforward_mc45_ablation.py")
                        args = rows[1]["args"]
                        for flag, expected in {
                            "--physical_level": "3", "--v_dd": voltage, "--one_over_q": q,
                            "--toggle_y_time": "15e-9", "--toggle_timing_mode": "fixed",
                            "--mc_coupler_nonlinear_variation_source": family,
                            "--mc_coupler_nonlinear_variation_quantity": quantity,
                            "--nonlinear_R_curve_sharing": "per_coupler",
                            "--activation_curve_sharing": "per_spin", "--n_trials": "1",
                            "--bn_recalibrate": "false", "--enable_dtc_nonideality": "true",
                        }.items():
                            self.assertEqual(args[args.index(flag)+1], expected)
                        self.assertTrue(args[args.index("--checkpoint")+1].endswith("_full_param_best_ckpt.pth"))
            trace.write_text("")
            result = subprocess.run(
                ["bash", str(root/"launch_scripts/run_feedforward_pretrain_then_ft.sh")],
                cwd=root, env=dict(env, mode="ft_and_eval", FAIL_FT="true"),
                text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(len(trace.read_text().splitlines()), 1)
            trace.write_text("")
            result = subprocess.run(
                ["bash", str(root/"launch_scripts/run_feedforward_pretrain_then_ft.sh")], cwd=root,
                env=dict(env, mode="ft_and_eval", NONLINEAR_R_MC_QUANTITY="resistance",
                         MC_COUPLER_NONLINEAR_VARIATION_QUANTITY="conductance"), capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(trace.read_text(), "")


if __name__ == "__main__":
    unittest.main()
