"""Pinned pre-TC implementation versus current disabled paths, in fresh processes.

Baseline modules are loaded read-only from Git into memory. No checkout,
worktree mutation, training or dataset evaluation occurs.
"""
import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '3ea543408fc958f44eab1e0bd92bc97ab017034a'
PROBE = r'''
import importlib.abc, importlib.util, json, pathlib, subprocess, sys
root=pathlib.Path.cwd()
revision=sys.argv[1]
files=['ode_pc.py','validation.py','train_ode_cifar.py','ode_inference.py',
       'TorchDiffEqPack/odesolver/base.py','TorchDiffEqPack/odesolver/adaptive_grid_solver.py',
       'TorchDiffEqPack/odesolver/fixed_grid_solver.py','TorchDiffEqPack/odesolver/ode_solver.py']
mapping={p[:-3].replace('/','.'):p for p in files}
if revision != 'current':
    class Loader(importlib.abc.MetaPathFinder,importlib.abc.Loader):
        def find_spec(self,fullname,path=None,target=None):
            if fullname in mapping:
                return importlib.util.spec_from_file_location(fullname,root/mapping[fullname],loader=self)
        def create_module(self,spec): return None
        def exec_module(self,module):
            path=mapping[module.__name__]
            source=subprocess.check_output(['git','show',revision+':'+path],text=True)
            exec(compile(source,str(root/path),'exec'),module.__dict__)
    sys.meta_path.insert(0,Loader())
sys.path.insert(0,str(root/'tests'))
import torch
from test_tc_legacy_regression import CASES,capture
result={'models':{},'solvers':{}}
for case,noisy in [(c,False) for c in CASES]+[('toggle',True),('toggle_qat',True),('pulse',True)]:
    key=case+('_noisy' if noisy else '')
    result['models'][key]=capture(case,noisy)
    if revision=='current':
        assert result['models'][key]==capture(case,noisy,tc_flag=False),key
from TorchDiffEqPack.odesolver import odesolve
for two in (False,True):
    for method in ('euler','dopri5'):
        for regenerate in (False,True):
            torch.manual_seed(711)
            class RHS(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.2))
                def forward(self,t,y):
                    if two:return tuple(self.scale*v for v in y)
                    return self.scale*y
            rhs=RHS()
            x=torch.tensor([.1,.2],requires_grad=True)
            initial=(x,x*.5) if two else x
            options=dict(method=method,t0=0.,t1=1.,h=.2,t_eval=[1.],
                eps=(.01,.02) if two else .01,noise_type='addi',regenerate_graph=regenerate)
            out=odesolve(rhs,initial,options)
            tensors=out if isinstance(out,tuple) else (out,)
            grads=torch.autograd.grad(sum(v.square().sum() for v in tensors),(x,rhs.scale))
            result['solvers'][str((two,method,regenerate))]=[v.detach().flatten().tolist() for v in (*tensors,*grads)]
from train_ode_cifar import get_args
from ode_inference import parse_args
sys.argv=['inspection']
result['train_defaults']=vars(get_args())
sys.argv=['inspection','--model_dir','./inspection','--model_name','inspection']
result['eval_defaults']=vars(parse_args())
import test_toggle_unitless_measured_pullback as pullback
result['pullback']={}
for name in sorted(n for n in vars(pullback) if n.startswith('test_')):
    try:
        getattr(pullback,name)();result['pullback'][name]='pass'
    except Exception as error:
        result['pullback'][name]=type(error).__name__
print('AUDIT_JSON '+json.dumps(result,default=str))
'''


class TCDisabledAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = []
        for revision in (BASELINE,'current'):
            run = subprocess.run([sys.executable,'-c',PROBE,revision],cwd=ROOT,
                env=dict(os.environ,OMP_NUM_THREADS='1'),text=True,capture_output=True,timeout=90)
            if run.returncode:
                raise AssertionError(run.stderr[-6000:])
            cls.results.append(json.loads(next(line[len('AUDIT_JSON '):]
                for line in run.stdout.splitlines() if line.startswith('AUDIT_JSON '))))

    def test_seeded_outputs_match_and_qat_weight_gradients_are_close(self):
        old,new = self.results
        self.assertEqual(set(old['models']), set(new['models']))
        for case, old_fields in old['models'].items():
            self.assertEqual(
                set(old_fields), set(new['models'][case]), case)
            for field, expected in old_fields.items():
                actual = new['models'][case][field]
                if field not in ('ff_grad', 'fb_grad'):
                    self.assertEqual(expected, actual, case+':'+field)
                    continue
                self.assertEqual(len(expected), len(actual), case+':'+field)
                for old_value, new_value in zip(expected, actual):
                    self.assertTrue(
                        math.isclose(old_value, new_value,
                                     rel_tol=5e-7, abs_tol=5e-8),
                        f'{case}:{field}: {old_value} != {new_value}')
        self.assertEqual(old['solvers'],new['solvers'])

    def test_existing_cli_defaults_and_pullback_regressions(self):
        old,new = self.results
        for stage in ('train_defaults','eval_defaults'):
            for key,value in old[stage].items():
                self.assertEqual(value,new[stage][key],stage+':'+key)
            self.assertFalse(new[stage]['tc_nonidealities'])
        self.assertEqual(old['pullback'],new['pullback'])
        self.assertEqual([k for k,v in new['pullback'].items() if v!='pass'],[])

    def test_toggle_class_bodies_only_have_reviewed_fixes(self):
        previous = subprocess.check_output(['git','show',BASELINE+':ode_pc.py'],cwd=ROOT,text=True)
        # Keep the original baseline and allow only these exact, reviewed fixes.
        # Comparing the full class ASTs below still catches any other change,
        # including removal of the guard or broadening the expansion condition.
        fixes = [
            ("level_sum = coupler_level_sum.to(device=state.device, dtype=state.dtype)",
             "level_sum = coupler_level_sum.to(device=state.device, dtype=state.dtype).clamp_min(0.0)"),
            ("    def run_z_stage(self, y_hold, z, T_z):\n"
             "        # get the number of quantization levels",
             "    def run_z_stage(self, y_hold, z, T_z):\n"
             "        if getattr(self, \"_pulse_unrolling\", False) and isinstance(self.FBconv, nn.Conv2d):\n"
             "            self.FBconv(y_hold)\n"
             "        # get the number of quantization levels"),
            ("    def run_y_stage(self, y, h_hold, T_y):\n"
             "        num_slices = self._num_slices(\"y\")",
             "    def run_y_stage(self, y, h_hold, T_y):\n"
             "        if getattr(self, \"_pulse_unrolling\", False) and isinstance(self.FFconv, nn.Conv2d):\n"
             "            self.FFconv(h_hold)\n"
             "        num_slices = self._num_slices(\"y\")"),
            ("        self.beta_c = self.beta * self.k / self.R\n"
             "        self.inp_scale = self.q if self.is_first else 1\n"
             "        self.out_scale = self.q if self.is_last else 1\n\n"
             "        self.time_scaler = self.R * self.C / self.alpha",
             "        self.beta_c = self.beta * self.k / self.R\n"
             "        self.inp_scale = self.q if self.is_first else 1\n"
             "        self.out_scale = (self.q if self.is_last and\n"
             "                          not getattr(self, \"physical_head_output\", False) else 1)\n\n"
             "        self.time_scaler = self.R * self.C / self.alpha"),
        ]
        for old, new in fixes:
            self.assertEqual(previous.count(old), 1, "Pinned baseline changed: " + old)
            previous = previous.replace(old, new, 1)
        def classes(source):
            return {node.name:ast.dump(node,include_attributes=False)
                    for node in ast.parse(source).body
                    if isinstance(node,ast.ClassDef) and node.name.startswith('Toggle')}
        self.assertEqual(classes(previous),classes((ROOT/'ode_pc.py').read_text()))


if __name__=='__main__': unittest.main()
