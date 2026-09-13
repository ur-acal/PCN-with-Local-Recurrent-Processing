import unittest
from unittest.mock import patch
import torch
from torch import nn
from test_switch_coupled_blocks import make,euler
from switch_coloring import ColorLie,ColorStrang,plan_for,validate_groups,validate_numerical_independence,color_flow,read_sites,output_shape
class ColoringTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);torch.manual_seed(42)
 def test_geometry_and_irregular(self):
  for shape in ((4,4),(7,9),(16,16)):
   for m in (1,2,3,4):
    op=make(ColorStrang,m);bs,ds,gs=plan_for(op,shape);validate_groups(bs,ds,gs)
    if shape==(16,16) and m==1:self.assertEqual(len(gs),9)
    if shape==(4,4) and m in (2,3):self.assertEqual(len(gs),len(bs))
  with self.assertRaises(ValueError):validate_groups(bs,ds,[[0,0]])
  bs,ds,gs=plan_for(make(ColorStrang,1),(4,4))
  with self.assertRaises(ValueError):validate_groups(bs,ds,[list(range(len(bs)))])
 def test_operator_geometry(self):
  for conv in (nn.Conv2d(1,1,3,stride=2,padding=2,dilation=2,bias=False),nn.ConvTranspose2d(1,1,3,stride=2,padding=1,output_padding=1,bias=False)):
   conv=conv.double();conv.weight.data.fill_(1);y=torch.ones(1,1,4,5,dtype=torch.float64,requires_grad=True);out=conv(y)
   self.assertEqual(tuple(out.shape[-2:]),output_shape(conv,(4,5)))
   for i in range(out.shape[-2]):
    for j in range(out.shape[-1]):
     grad=torch.autograd.grad(out[0,0,i,j],y,retain_graph=True)[0];actual={tuple(v) for v in (grad[0,0]!=0).nonzero().tolist()}
     self.assertEqual(actual,read_sites(conv,{(i,j)},(4,5)))
 def test_numeric_pair_independence(self):
  for shape in ((4,4),(5,7)):
   for m in (1,2,3,4):
    op=make(ColorStrang,m);y=torch.rand(1,1,*shape,dtype=torch.float64)
    validate_numerical_independence(op,y,y,atol=1e-13,rtol=1e-13)
 def test_order_and_simultaneous_commit(self):
  op=make(ColorStrang,1);y=torch.rand(1,1,4,4,dtype=torch.float64);bs,ds,gs=plan_for(op,(4,4));g=max(gs,key=len)
  a=color_flow(op,y,y,bs,g,.001);b=color_flow(op,y,y,bs,list(reversed(g)),.001);torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
  seen=[]
  def flow(x,state,block,duration):seen.append(state.clone());return state+1
  with patch.object(op,'local_block_flow',side_effect=flow):color_flow(op,y,y,bs,g,.001)
  self.assertTrue(all(torch.equal(t,y) for t in seen))
 def test_schedule_and_installed_wrapper(self):
  op=make(ColorStrang,2);y=torch.ones(1,1,4,4,dtype=torch.float64);bs,ds,gs=plan_for(op,(4,4));calls=[]
  def flow(x,state,block,duration):calls.append((block,duration));return state
  with patch.object(op,'local_block_flow',side_effect=flow):op._run_explicit_pixel_switch(y)
  self.assertEqual([c[0] for c in calls],[bs[i] for group in list(gs)+list(reversed(gs)) for i in group]);self.assertTrue(all(c[1]==.005 for c in calls))
  op=make(ColorStrang,4);raw=op._make_ode_fn;op._make_ode_fn=lambda x:lambda t,z:3*raw(x)(t,z)
  with patch('switch.aca_ode_solve',side_effect=euler):a=op._run_explicit_pixel_switch(y)
  expected=y.clone();fn=op._make_coupled_block_ode_fn((0,0,4,4))
  for _ in range(2):expected=expected+.005*3*fn(0.,expected)
  torch.testing.assert_close(a,expected,rtol=1e-12,atol=1e-12);self.assertFalse(hasattr(op,'_active_spatial_block'))
 def test_full_feature_map_matches_raster(self):
  from switch import ODEXInitFFFBPixelSwitchExplicit,ODEXInitFFFBPixelSwitchStrang
  y=torch.rand(1,1,4,4,dtype=torch.float64)
  for colored,raster in ((ColorLie,ODEXInitFFFBPixelSwitchExplicit),(ColorStrang,ODEXInitFFFBPixelSwitchStrang)):
   a=make(colored,4);b=make(raster,4)
   torch.testing.assert_close(a._run_explicit_pixel_switch(y),b._run_explicit_pixel_switch(y),atol=0,rtol=0)
if __name__=='__main__':unittest.main()
