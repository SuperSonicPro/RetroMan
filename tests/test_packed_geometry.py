"""Dense geometry access must stay bounded and agree with owned source arrays."""
import ast,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work/RetroMan-1.0.24'
tree=ast.parse((SRC/'mesh_extract.py').read_text(encoding='utf-8'))
classes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
space={'np':np,'Vector':tuple,'VertexData':lambda p,n,uv:NS(p=p,n=n,uv=uv),'TriangleData':lambda a,b,c,m:NS(a=a,b=b,c=c,material=m),'checkpoint':lambda c:None}
exec(compile(ast.Module(classes,type_ignores=[]),'mesh_extract','exec'),space)
class Tests(unittest.TestCase):
 def geometry(self):
  source=space['PackedTriangles']();materials=[object(),object()]
  for j in range(2):
   block=space['MeshArrays'](np.arange(18,dtype=np.float32).reshape(6,3)+100*j,np.arange(6),np.ones((6,3)),np.zeros((6,2)),np.array([[0,1,2],[3,4,5]]),np.array([0,1]),materials)
   source.append(block)
  return source,materials
 def test_random_access_does_not_expand_mesh(self):
  source,mats=self.geometry()
  for block in source.blocks:block.triangles=lambda *a:(_ for _ in ()).throw(AssertionError('Expanded mesh'))
  tri=source[3];self.assertEqual(tri.a.p,(109.,110.,111.));self.assertIs(tri.material,mats[1]);self.assertIs(source[-1],tri)
  self.assertEqual(len(source[::-1]),4)
  with self.assertRaises(IndexError):source[4]
 def test_gpu_bounds_upload_crosses_blocks_with_correct_materials(self):
  source,mats=self.geometry();a=source.bound_points(1,4,{id(mats[0]):.2,id(mats[1]):.7})
  expected=np.array([[source[i].a.p,source[i].b.p,source[i].c.p] for i in range(1,4)])
  np.testing.assert_array_equal(a[:,:,:3],expected)
  np.testing.assert_allclose(a[:,:,3],np.broadcast_to(np.array([.7,.2,.7])[:,None],(3,3)))
 def test_index_updates_after_append(self):
  source,mats=self.geometry();source[0];source.append(source.blocks[0]);self.assertEqual(source[4].a.p,source[0].a.p)
 def test_shutter_reuse_requires_exact_geometry_and_materials(self):
  import copy
  source,_=self.geometry()
  for block in source.blocks:block.materials=[NS(alpha=1.,texture=NS(pixels=np.ones(4),width=1,height=1)),NS(alpha=.5,texture=None)]
  other=copy.deepcopy(source);self.assertTrue(source.matches(other))
  other.blocks[0].materials[0].texture.pixels[0]=.5
  self.assertFalse(source.matches(other))
  other=copy.deepcopy(source);other.blocks[0].materials[0].alpha=.2
  self.assertFalse(source.matches(other))
  other=copy.deepcopy(source);other.blocks[1].positions[0,0]+=.001
  self.assertFalse(source.matches(other))
