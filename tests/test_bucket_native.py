"""Numerical tests of native sample visibility, independent of Blender/GPU."""
import importlib.util,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work/RetroMan-1.0.24'
def load(name):
 spec=importlib.util.spec_from_file_location(name,SRC/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
Bucket=load('bucket_native').Bucket
pyramid=load('grid_filter').pyramid

def quad(z,color,left=-1,right=1,end_shift=0):
 p=np.array([[left,-1,z,1],[right,-1,z,1],[right,1,z,1],[left,1,z,1]],dtype=np.float32)
 q=p.copy();q[:,0]+=end_shift
 return np.array([np.r_[p[ids].reshape(-1),q[ids].reshape(-1),color] for ids in ([0,1,2],[0,2,3])],dtype=np.float32)

def render(records,size=8,tile=8,axis=2,lens=(0,0,1),background=(0,0,0,0)):
 out=np.empty((size,size,4),np.float32)
 for y in range(0,size,tile):
  for x in range(0,size,tile):
   w=min(tile,size-x);h=min(tile,size-y)
   with Bucket(x,y,w,h,size,size,axis,lens,'BOX') as b:
    b.push(records);out[y:y+h,x:x+w]=b.finish(np.tile(background,(w*h,1)))
 return out

class BucketTests(unittest.TestCase):
 def test_transparency_order_and_analytic_color(self):
  a=quad(-.5,(1,0,0,.5));b=quad(.5,(0,0,1,.5));records=np.concatenate((a,b))
  forward=render(records);reverse=render(records[::-1])
  np.testing.assert_array_equal(forward,reverse)
  np.testing.assert_allclose(forward,np.broadcast_to((2/3,0,1/3,.75),forward.shape),atol=1e-6)
 def test_tiny_opacity_survives_compositing(self):
  image=render(quad(0,(1,0,0,1e-9)),axis=1)
  np.testing.assert_allclose(image[:,:,3],1e-9,rtol=1e-6,atol=0)
  np.testing.assert_allclose(image[:,:,:3],np.broadcast_to((1,0,0),image[:,:,:3].shape),atol=1e-6)
 def test_shared_edge_has_single_owner(self):
  image=render(quad(0,(1,0,0,.5)),axis=1)
  np.testing.assert_allclose(image[:,:,3],.5)
 def test_opaque_occludes_layers_behind(self):
  image=render(np.concatenate((quad(.5,(1,0,0,1)),quad(-.5,(0,1,0,1)))))
  np.testing.assert_array_equal(image,np.broadcast_to((0,1,0,1),image.shape))
 def test_bucket_boundaries_are_bit_exact(self):
  records=quad(0,(.2,.6,1,.5),-.9,.3,.6)
  np.testing.assert_array_equal(render(records,tile=8),render(records,tile=2))
 def test_motion_coverage_between_endpoints(self):
  image=render(quad(0,(1,1,1,1),-.9,-.7,1.6),size=16,axis=8)
  self.assertGreater(float(image[:,7:9,3].mean()),.05)
  self.assertLess(float(image[:,:,3].max()),.5)
 def test_crossing_transparency_changes_order_per_sample(self):
  a=quad(0,(1,0,0,.5));b=quad(0,(0,0,1,.5))
  for rows,sign in ((a,1),(b,-1)):
   rows[:,2:12:4]=rows[:,0:12:4]*sign*.5;rows[:,14:24:4]=rows[:,12:24:4]*sign*.5
  image=render(np.concatenate((a,b)),axis=4)
  self.assertGreater(image[4,1,0],image[4,1,2]);self.assertLess(image[4,6,0],image[4,6,2])
 def test_lens_focus_and_defocus(self):
  q=quad(0,(1,1,1,1),-.3,.3)
  pinhole=render(q,size=16,axis=8)
  focused=render(q,size=16,axis=8,lens=(8,0,1))
  defocused=render(q,size=16,axis=8,lens=(8,0,2))
  np.testing.assert_array_equal(pinhole,focused)
  self.assertGreater(float(defocused[:,3,3].mean()),float(focused[:,3,3].mean()))
 def test_minified_checker_pyramid_mean(self):
  data=np.ones((64,64,4),np.float32);data[:,:,:3]=(np.indices((64,64)).sum(axis=0)%2)[:,:,None]
  packed,size=pyramid(data,64,64)
  np.testing.assert_array_equal(packed[-1,0],(.5,.5,.5,1));self.assertEqual(size,(64,64,7))
 def test_near_plane_clipping_and_winding(self):
  p=np.array([[-1,-1,-2,1],[1,-1,0,1],[0,1,0,1]],np.float32)
  a=np.r_[p.reshape(-1),p.reshape(-1),(1,1,1,.5)].reshape(1,28)
  b=np.r_[p[::-1].reshape(-1),p[::-1].reshape(-1),(1,1,1,.5)].reshape(1,28)
  image=render(a,axis=4)
  np.testing.assert_array_equal(image,render(b,axis=4))
  self.assertGreater(float(image[:,:,3].sum()),0)
  self.assertLessEqual(float(image[:,:,3].max()),.5)
 def test_alpha_filtering_does_not_bleed_transparent_color(self):
  packed,size=pyramid(np.array([1,0,0,1,0,0,1,0],np.float32),2,1,True)
  np.testing.assert_array_equal(packed[-1,0],(.5,0,0,.5))
 def test_non_power_of_two_constant_pyramid(self):
  data=np.full((7,11,4),.37,np.float32);packed,size=pyramid(data,11,7)
  np.testing.assert_allclose(packed[-1,0],.37)
