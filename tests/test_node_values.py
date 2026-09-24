import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work/RetroMan-1.0.24'
spec=importlib.util.spec_from_file_location('node_values',SRC/'node_values.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Socket:
    def __init__(self,value=0,output=False,node=None):
        self.default_value=value;self.is_output=output;self.node=node
        self.links=[];self.type='VALUE'
    @property
    def is_linked(self):return bool(self.links)
    def as_pointer(self):return id(self)
    def link(self,s):self.links=[NS(from_socket=s)];return self

def math_node(op,*values):
    node=NS(bl_idname='ShaderNodeMath',mute=False,operation=op,use_clamp=False,inputs=[Socket(v) for v in values])
    return Socket(output=True,node=node)

class ConstantTests(unittest.TestCase):
    def test_linked_default_not_used(self):
        result=Socket(99).link(math_node('MULTIPLY',.125,2))
        self.assertEqual(m.constant(result),.25)
    def test_zero_division_and_clamp(self):
        self.assertEqual(m.constant(math_node('DIVIDE',1,0)),0)
        s=math_node('ADD',2,3);s.node.use_clamp=True
        self.assertEqual(m.constant(s),1)
    def test_cycle_is_rejected(self):
        s=math_node('ADD',1,2);s.node.inputs[0].link(s)
        with self.assertRaises(m.UnsupportedValue):m.constant(s)
    def test_depth_is_bounded(self):
        s=Socket(1)
        for _ in range(80):s=Socket(0).link(s)
        with self.assertRaises(m.UnsupportedValue):m.constant(s)
    def test_unsupported_does_not_return_linked_default(self):
        s=Socket(99).link(Socket(output=True,node=NS(bl_idname='ShaderNodeTexNoise',mute=False)))
        with self.assertRaises(m.UnsupportedValue):m.constant(s)
    def test_nonfinite_rejected(self):
        with self.assertRaises(m.UnsupportedValue):m.constant(Socket(float('nan')))
