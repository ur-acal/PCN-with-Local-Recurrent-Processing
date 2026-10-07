import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'diagnostic_scripts'))
from make_current_limits import generate
from current_stages import stage_members
from rhs_current_clamp import read_summary


class HelperTests(unittest.TestCase):
    def args(self, directory, **kwargs):
        values = dict(num_layers=3, pool_positions=[2], granularity='layer',
                      branches='separate', no_final_linear=False, summary=None,
                      percentile=95., bound=[], name=None, output_dir=directory, overwrite=False)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_four_manual_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            for granularity in ('layer', 'stage'):
                for branches in ('separate', 'pooled'):
                    members = stage_members(3, [2])
                    groups = members if granularity == 'stage' else {
                        layer: [layer] for layers in members.values() for layer in layers}
                    bound, expected = [], {}
                    for i, (group, layers) in enumerate(groups.items(), 1):
                        for j, branch in enumerate(('COMBINED',) if branches == 'pooled' else
                                                   (('FF',) if group == 'final_linear' else ('FF', 'FB'))):
                            low, high = -i-j-.25, i+j+.75
                            bound.append((group, branch, str(low), str(high)))
                            for layer in layers:
                                expected[layer, branch] = (low*1e-6, high*1e-6)
                    args = self.args(directory, granularity=granularity, branches=branches, bound=bound)
                    path = generate(args)
                    _, rows, _ = read_summary(path, 99)  # explicit bounds ignore percentile
                    self.assertEqual(expected, {key: (r['lower_A'], r['upper_A']) for key,r in rows.items()})
                    with self.assertRaises(FileExistsError):
                        generate(args)
                    args.overwrite = True
                    generate(args)

    def test_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            for bound in ([], [('layer_01','FF','nan','1')],
                          [('layer_01','FF','2','1')], [('layer_01','FF','-1','1')]*2):
                with self.assertRaises(ValueError):
                    generate(self.args(directory, bound=bound))
            valid = [('layer_01','COMBINED','-1','1')]
            for name in ('é'*128+'.md', '../bad.md'):
                with self.assertRaises(ValueError):
                    generate(self.args(directory, num_layers=1, pool_positions=[], no_final_linear=True,
                                       branches='pooled', bound=valid, name=name))
            for pools in ([2,1],[2,2],[0],[4]):
                with self.assertRaises(ValueError):
                    stage_members(3,pools)

    def test_stage_summary_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.md'
            source.write_text('# Total combined summing-current distributions\n\n'
                              '| Layer | Branch | Lower 95% (µA) | Upper 95% (µA) |\n'
                              '|---|---|---|---|\n| stage_01 | combined | -2 | 3 |\n'
                              '| stage_02 | combined | -4 | 5 |\n| final_linear | combined | -6 | 7 |\n')
            path = generate(self.args(directory, summary=str(source), granularity='stage', branches='pooled'))
            generated = path.read_text()
            self.assertIn('Source summary:', generated)
            self.assertIn('Source summary SHA-256:', generated)
            _, rows, _ = read_summary(path, 95)
            self.assertEqual(rows['layer_02','COMBINED']['lower_A'], -2e-6)
            self.assertEqual(rows['layer_03','COMBINED']['upper_A'], 5*1e-6)

    def test_stage_samples_not_layer_means(self):
        import torch
        from plot_toggle_current_distributions import MomentSink, HistogramSink, make_specifications
        members = stage_members(6, [2,5])
        self.assertEqual(members['stage_02'], ['layer_03','layer_04','layer_05'])
        samples = [torch.tensor([-2., 0., 1.]), torch.tensor([7.])]
        sink = MomentSink()
        for x in samples:
            sink.update('stage_01','FF',x,x)
        stats = sink.results()
        self.assertEqual(stats['total']['separate','stage_01','FF']['mean_A'], 1.5)
        hist = HistogramSink(make_specifications(stats, 10, 10))
        for x in samples:
            hist.update('stage_01','FF',x,x)
        self.assertEqual(hist.cpu_counts()['total']['combined','stage_01','combined'].sum(),4)

    def test_recorder_stage_routing(self):
        import torch
        from plot_toggle_current_distributions import CurrentRecorder, MomentSink
        def block():
            return SimpleNamespace(_brownian_increment=lambda *a: None,
                _coupler_brownian_increment=lambda *a: None,
                integrate_pulse_slice=lambda state,*a,**kw: state,
                _stage_capacitance=lambda stage: 1.)
        model = SimpleNamespace(PcConvs=[block() for _ in range(3)], linear=None)
        members = stage_members(3,[2])
        mapping = {layer: group for group,layers in members.items() for layer in layers}
        sink = MomentSink()
        recorder = CurrentRecorder(model,sink,mapping)
        originals = [b.integrate_pulse_slice for b in model.PcConvs]
        recorder.attach()
        try:
            for i,b in enumerate(model.PcConvs):
                for stage in ('z','y'):
                    b.integrate_pulse_slice(torch.zeros(2),1.,None,stage,0,
                                            constant_rhs=torch.full((2,),float(i)))
        finally:
            recorder.detach()
        rows = sink.results()['total']
        self.assertEqual(rows['separate','stage_01','FF']['count'],4)
        self.assertEqual(rows['combined','stage_01','combined']['count'],8)
        self.assertEqual(rows['separate','stage_02','FB']['mean_A'],2.)
        self.assertEqual(originals,[b.integrate_pulse_slice for b in model.PcConvs])

    def test_stage_plot_tables_expand(self):
        import torch
        from plot_toggle_current_distributions import MomentSink, HistogramSink, make_specifications, write_outputs
        with tempfile.TemporaryDirectory() as directory:
            sink = MomentSink()
            observations = []
            for group in ('stage_01','stage_02','final_linear'):
                for branch in (('FF',) if group == 'final_linear' else ('FF','FB')):
                    x = torch.tensor([-1e-6,0.,2e-6])
                    observations.append((group,branch,x,x))
                    sink.update(*observations[-1])
            first = sink.results()
            specs = make_specifications(first,5,5)
            hist = HistogramSink(specs)
            for row in observations:
                hist.update(*row)
            plot_args = SimpleNamespace(summary_bound_percentiles='95,99',plot_bound_percentile=99.,
                                        plot_lower_sigma=4.,plot_upper_sigma=4.)
            root = Path(directory)/'plots'
            write_outputs(root,plot_args,[],first,specs,hist.cpu_counts(),[],{},preserve_run_files=True)
            for mode,branches in (('separate','separate'),('combined','pooled')):
                source = root/'total'/mode/'summary.md'
                path = generate(self.args(directory,summary=source,granularity='stage',branches=branches))
                _,rows,_ = read_summary(path,95)
                self.assertEqual(len(rows),7 if branches=='separate' else 4)
                self.assertTrue((root/'total'/mode/'stage_01_ff.png' if mode=='separate'
                                 else root/'total'/mode/'stage_01_combined.png').exists())
            import json
            (root/'run_config.json').write_text(json.dumps({'stage_members':stage_members(3,[2])}))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                generate(self.args(directory,summary=root/'total/combined/summary.md',
                                   granularity='stage',branches='pooled',pool_positions=[1]))


if __name__ == '__main__':
    unittest.main()
