"""Model Blender 5.2 OpenGL blend factors against actual renderer mode selections."""
import ast
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work/RetroMan-1.0.24'
def modes(cls_name,method):
    tree=ast.parse((SRC/'gpu_renderer.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls_name)
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method)
    return [n.args[0].value for n in sorted(ast.walk(fn),key=lambda n:getattr(n,"lineno",0)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='blend_set' and n.args and isinstance(n.args[0],ast.Constant)]
def blend(mode,src,dst):
    if mode=='ADDITIVE':return [src[i]*src[3]+dst[i] for i in range(3)]+[dst[3]]
    if mode=='ADDITIVE_PREMULT':return [a+b for a,b in zip(src,dst)]
    raise AssertionError(mode)
class BlendTests(unittest.TestCase):
    def test_light_rgb_contributes_without_changing_alpha(self):
        selected=modes('GPURenderer','_render_scene_pass')[-1]
        self.assertEqual(blend(selected,[.4,.2,.1,0],[.1,.1,.1,.5]),[.5,.30000000000000004,.2,.5])
        self.assertEqual(blend('ADDITIVE',[.4,.2,.1,0],[.1,.1,.1,.5]),[.1,.1,.1,.5])
    def test_weighted_samples_preserve_rgb_and_alpha(self):
        selected=modes('GPUAccumulator','add')[-1]
        for pixel in ([.8,.4,.2,1.],[.4,.2,.1,.5],[0,0,0,0]):
            dst=[0,0,0,0]
            for weight in (.25,.75):dst=blend(selected,[x*weight for x in pixel],dst)
            for actual,expected in zip(dst,pixel):self.assertAlmostEqual(actual,expected)
if __name__=='__main__':unittest.main()
