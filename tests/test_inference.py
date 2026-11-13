import unittest
import numpy as np
import torch
from inference import fuse,compose_color
class InferenceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def test_same_band_identity(self):
  rng=np.random.default_rng(7);v=rng.uniform(.05,.9,(64,96)).astype('float32');rgb=np.repeat(v[...,None],3,-1)
  for mode in ['analytic-contrast','analytic-intensity']:
   np.testing.assert_allclose(fuse(rgb,v,mode)['value'],v,atol=1e-6,rtol=0)
 def test_flat_geometric_mean(self):
  rgb=np.full((64,96,3),.2,np.float32);nir=np.full((64,96),.9,np.float32)
  np.testing.assert_allclose(fuse(rgb,nir)['value'],np.sqrt((.2+.03)*(.9+.03))-.03,atol=1e-6,rtol=0)
 def test_invalid_registration_falls_back(self):
  rgb=np.full((64,96,3),.2,np.float32);nir=np.full((64,96),.9,np.float32)
  np.testing.assert_array_equal(fuse(rgb,nir,valid=np.zeros_like(nir,bool))['color'],rgb)
 def test_black_rgb_color(self):
  np.testing.assert_array_equal(compose_color(np.zeros((2,2,3),np.float32),np.full((2,2),.5,np.float32)),np.full((2,2,3),.5,np.float32))
 def test_invalid_inputs(self):
  rgb=np.zeros((64,96,3),np.float32);nir=np.zeros((64,96),np.float32)
  for bad in [nir[:32],nir+np.nan,nir+2]:
   with self.assertRaises(ValueError):fuse(rgb,bad)
if __name__=='__main__':unittest.main()
