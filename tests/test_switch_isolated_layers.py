import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from torch import nn
import study_switch_isolated_layers as diag


class Wrapper:
    v_dd=10.
    def _quantize_output(self,x):return x


class Block(nn.Module):
    def __init__(self,wrapper,factor):
        super().__init__();self.wrapper=wrapper;self.factor=factor;self.calls=[]
    def forward(self,x,layer):
        self.calls.append((layer,x.clone()))
        return self.wrapper._quantize_output(x*self.factor)


class Tests(unittest.TestCase):
    def test_each_layer_gets_its_own_reference_input(self):
        for layer,value,factor in [(0,1.,2.),(1,7.,3.)]:
            w=Wrapper();b=Block(w,factor)
            x=torch.full((128,1,2,2),value)
            trace={'input':x,'pre_quant':x*factor,'output':x*factor}
            record=diag.evaluate_layer(b,w,trace,layer,device='cpu')
            self.assertTrue(torch.equal(b.calls[0][1],x))
            self.assertEqual(b.calls[0][0],layer)
            self.assertEqual(len(b.calls),1)
            self.assertFalse(record['errors']['output']['difference_sq'].any())
            self.assertEqual(w._quantize_output.__func__,Wrapper._quantize_output)

    def test_pooled_error_and_csv_coverage(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(diag,'OUT',Path(tmp)):
            config=diag.CONFIGS[0];folder=Path(tmp)/'cases'/diag.s.case_id(config)/'layers';folder.mkdir(parents=True)
            for batch,diff,ref in [(0,4.,4.),(1,9.,16.)]:
                value=dict(batch=batch,layer=1,n_samples=128,rail_sum=0.,seconds=1.,peak_allocated_bytes=1)
                for kind in diag.KINDS:
                    value.update({kind+'_difference_sq':diff,kind+'_reference_sq':ref,kind+'_relative_error':(diff/ref)**.5})
                (folder/f'batch_{batch}_layer_00.json').write_text(json.dumps(value))
            _,layers=diag.collect(config)
            self.assertAlmostEqual(layers[0]['output_relative_error'],(13/20)**.5)
            diag.report('running',{})
            with open(Path(tmp)/'layer_metrics.csv') as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),832)
            self.assertEqual(rows[0]['n_samples'],'256')
            self.assertEqual(rows[1]['output_relative_error'],'')
            with open(Path(tmp)/'layer_batch_metrics.csv') as f:self.assertEqual(len(list(csv.DictReader(f))),1664)
            self.assertEqual(len(diag.CONFIGS),52)
            self.assertEqual(json.loads((Path(tmp)/'status.json').read_text())['completed'],0)


if __name__=='__main__':unittest.main()
