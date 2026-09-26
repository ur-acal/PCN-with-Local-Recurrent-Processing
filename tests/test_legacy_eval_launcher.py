"""Resolve the real legacy shell launcher without starting inference."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = ('TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_'
        'dopri5Solver_1.75TEnd_C100_16Layers_scanGFI_6REP')


class LegacyEvalLauncherTests(unittest.TestCase):
    def resolve(self, name, mode):
        with tempfile.TemporaryDirectory() as logs:
            env = os.environ.copy()
            for key in ('TC_NONIDEALITIES', 'DIFF_MISMATCH', 'THERMAL_NOISE'):
                env.pop(key, None)
            env.update(MODEL_NAMES_STR=name, EVAL_MODE=mode, BASE_LOGDIR=logs)
            script = '''
python() { printf '%s\\n' "$@"; }
parallel() { run_model "${MODEL_NAMES[0]}" dopri5 5 49e-15; }
source launch_scripts/run_ode_wrapped_inference.sh
'''
            return subprocess.check_output(['bash', '-c', script], cwd=ROOT,
                                           env=env, text=True, stderr=subprocess.STDOUT)

    def test_fp_is_unwrapped_without_code_mismatch(self):
        out = self.resolve(BASE, 'fp')
        self.assertIn('--ode_wrapper\nnone\n', out)
        self.assertIn('--diff_mismatch\nfalse\n', out)
        self.assertIn('--thermal_noise\nfalse\n', out)

    def test_scaled_fp_uses_rc_without_code_mismatch(self):
        out = self.resolve(BASE, 'scaled_fp')
        self.assertIn('--ode_wrapper\nODEWrapperRC\n', out)
        self.assertIn('--diff_mismatch\nfalse\n', out)

    def test_qat_defaults_are_preserved(self):
        out = self.resolve('TIMMQAT5b8aNT0p25mul' + BASE, 'auto')
        self.assertIn('--ode_wrapper\nQATTester1State\n', out)
        self.assertIn('--enob\n8\n', out)
        self.assertIn('--diff_mismatch\ntrue\n', out)
        self.assertIn('--thermal_noise\ntrue\n', out)


if __name__ == '__main__':
    unittest.main()
