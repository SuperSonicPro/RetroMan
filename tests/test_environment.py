"""World translation conserves constant radiance and produces classic ambient fill."""
import ast,math,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work/RetroMan-1.0.24'
tree=ast.parse((SRC/'illumination.py').read_text(encoding='utf-8'))
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'sh_basis','environment_diffuse','evaluate_environment'}]
ns={'np':np,'math':math,'checkpoint':lambda cancel:cancel() if cancel else None}
exec(compile(ast.Module(functions,type_ignores=[]),'illumination','exec'),ns)
class EnvironmentTests(unittest.TestCase):
    def test_constant_world_is_independent_of_normal(self):
        c=(.4,.8,1.2);coeff=ns['environment_diffuse'](None,1,c)
        for normal in [(1,0,0),(-1,0,0),(0,1,0),(0,0,1),(0,0,-1)]:
            for x,y in zip(ns['evaluate_environment'](coeff,normal),c):self.assertAlmostEqual(x,y,places=7)
    def test_constant_hdr_conserves_strength(self):
        tex=SimpleNamespace(width=8,height=4,pixels=[.2,.4,.6,1]*32)
        coeff=ns['environment_diffuse'](tex,2,(0,0,0))
        for normal in [(1,0,0),(0,1,0),(0,0,1)]:
            for x,y in zip(ns['evaluate_environment'](coeff,normal),(.4,.8,1.2)):self.assertLess(abs(x-y),.002)
    def test_directional_image_becomes_constant_ambient(self):
        row=[v for i in range(64) for v in ((1,1,1,1) if 16<=i<48 else (0,0,0,1))]
        tex=SimpleNamespace(width=64,height=32,pixels=row*32)
        coeff=ns['environment_diffuse'](tex,1,(0,0,0))
        self.assertAlmostEqual(ns['evaluate_environment'](coeff,(1,0,0))[0],.5)
        self.assertAlmostEqual(ns['evaluate_environment'](coeff,(-1,0,0))[0],.5)
    def test_environment_integration_can_cancel(self):
        tex=SimpleNamespace(width=1,height=1,pixels=[1,1,1,1])
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):ns['environment_diffuse'](tex,1,(0,0,0),cancel)
