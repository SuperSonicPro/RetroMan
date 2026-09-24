import ast,unittest
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work/RetroMan-1.0.24'
tree=ast.parse((SRC/'materials.py').read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='missing_shader_libraries');space={};exec(compile(ast.Module([fn],type_ignores=[]),'assets','exec'),space)
class Tests(unittest.TestCase):
 def test_nested_missing_library_deduplicated_and_cycles_terminate(self):
  root=NS(as_pointer=lambda:1,nodes=[]);missing=NS(as_pointer=lambda:2,is_missing=True,library=NS(filepath='//SAIOTemplates.blend'),name='SAIO Shader')
  root.nodes=[NS(bl_idname='ShaderNodeGroup',node_tree=t) for t in (root,missing,missing,None)]
  self.assertEqual(space['missing_shader_libraries'](root),['//SAIOTemplates.blend'])
