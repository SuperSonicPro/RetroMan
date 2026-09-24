"""Material state reuse must preserve draw order and per-chunk REYES bindings."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work/RetroMan-1.0.24'
tree=ast.parse((SRC/'gpu_renderer.py').read_text(encoding='utf-8'))
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPURenderer')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_draw_material_pass')
ns={};exec(compile(ast.Module([fn],type_ignores=[]),'draw-pass','exec'),ns)
class Tests(unittest.TestCase):
    def exercise(self,reyes):
        a,b=object(),object();items=[NS(material=m) for m in (a,a,b,a)]
        geometry=[];materials=[];draws=[]
        obj=NS(use_reyes=reyes,
               _bind_geometry=lambda shader,d:geometry.append(d),
               _bind_material=lambda shader,d,**kw:materials.append(d),
               _draw_one=lambda d,shader:draws.append(d))
        ns['_draw_material_pass'](obj,items,object(),base_pass=True)
        self.assertEqual(draws,items)
        self.assertEqual(materials,[items[i] for i in (0,2,3)])
        self.assertEqual(geometry,items if reyes else materials)
        ns['_draw_material_pass'](obj,[items[-1]],object(),base_pass=False)
        self.assertEqual(len(materials),4) # New pass always rebinds.
    def test_triangle_material_runs(self):self.exercise(False)
    def test_reyes_rebinds_each_chunk(self):self.exercise(True)

class ShadowTests(unittest.TestCase):
    def test_zero_opacity_cannot_occlude(self):
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_draw_depth_geometry')
        space={};exec(compile(ast.Module([fn],type_ignores=[]),'depth-pass','exec'),space)
        clear=NS(material=NS(alpha=0,base_color=(1,1,1,1)))
        solid=NS(material=NS(alpha=1,base_color=(1,1,1,1)))
        drawn=[]
        obj=NS(use_reyes=False,_drawables=lambda:[clear,solid],_bind_geometry=lambda *a,**k:None,_draw_one=lambda d,s:drawn.append(d))
        space['_draw_depth_geometry'](obj,object())
        self.assertEqual(drawn,[solid])
