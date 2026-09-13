import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('report_r10k', Path(__file__).resolve().parents[1]/'scripts/report_switch_r10k.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def test_complete_coverage_flags_and_preserved_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'results'
            out.mkdir()
            paper = Path(tmp)/'paper.md'
            history = 'Historical measurements stay here.\n'
            paper.write_text('# Study\n\n'+history)
            (out/'status.json').write_text(json.dumps({'status':'running', 'failed_cases':{}}))
            (out/'reference_summary.json').write_text(json.dumps([{'tol':1e-6,'accuracy':.65}]))
            for index, error, accuracy in [(0,.1,.5), (1,.2,.6)]:
                config = report.CONFIGS[index]
                folder = out/'cases'/report.case_id(config)
                folder.mkdir(parents=True)
                for batch in range(7):
                    (folder/f'batch_{batch}.pt').touch()
                value = dict(case=folder.name, method=config[0], m=config[1], n_iters=config[2],tol=config[3],
                             n_samples=896, accuracy=accuracy, reference_accuracy=.65,
                             logit_relative_error=error, layers=[dict(index=j,input_relative_error=0.,
                             pre_quant_relative_error=error,output_relative_error=error,
                             rail_fraction=.01) for j in range(16)])
                (folder/'summary.json').write_text(json.dumps(value))
            self.assertEqual(report.render(out,paper), (2,5184,3))
            text = paper.read_text()
            self.assertIn(history,text)
            self.assertEqual(sum(line.startswith('| C') for line in text.splitlines()),324)
            self.assertIn('L: C002; A: C002; F: C002',text)
            self.assertIn('pending (0/7)',text)
            with open(out/'layer_metrics.csv',newline='') as handle:
                layers = list(csv.DictReader(handle))
            self.assertEqual(len(layers),5184)
            self.assertEqual(layers[-1]['output_relative_error'],'')
            with open(out/'error_accuracy_inversions.csv',newline='') as handle:
                pairs = list(csv.DictReader(handle))
            self.assertTrue(all(p['lower_actual_accuracy']=='True' for p in pairs))
            self.assertTrue(all(p['only_block_size_differs']=='False' for p in pairs))
            report.render(out,paper)
            self.assertEqual(text,paper.read_text())

    def test_smaller_error_without_larger_gap_is_not_flagged(self):
        a = dict(id='A',case='a',accuracy=.7,accuracy_gap_pp=5,
                 logit_relative_error=.1,mean_layer_relative_error=.1,first_layer_relative_error=.1,
                 method='Jacobi',m=1,n_iters=5,tol=1e-6)
        b = dict(a,id='B',case='b',accuracy=.6,logit_relative_error=.2,
                 mean_layer_relative_error=.2,first_layer_relative_error=.2)
        self.assertEqual(report.inversions([a,b]),[])


if __name__ == '__main__':
    unittest.main()
