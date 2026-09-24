import importlib.util
from pathlib import Path
import sys, types, unittest, ast
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work/RetroMan-1.0.24'
spec=importlib.util.spec_from_file_location('params',SRC/'shader_parameters.py')
params=importlib.util.module_from_spec(spec);spec.loader.exec_module(params)
NS=types.SimpleNamespace
class UBO:
    def __init__(self,data):self.data=list(data);self.updates=0
    def update(self,data):self.data=list(data);self.updates+=1
class Raw:
    def uniform_block(self,name,ubo):self.bound=(name,ubo)
class Info:
    def __init__(self):self.sources={}
    def typedef_source(self,s):self.typedef=s
    def uniform_buf(self,*args):self.block=args
    def vertex_source(self,s):self.sources['vertex']=s
    def fragment_source(self,s):self.sources['fragment']=s
class Tests(unittest.TestCase):
    def setUp(self):
        self.raw=Raw();self.gpu=NS(types=NS(GPUUniformBuf=UBO,GPUShaderCreateInfo=Info),shader=NS(create_from_info=lambda info:self.raw))
        self.mock=patch.dict(sys.modules,{'gpu':self.gpu});self.mock.start()
    def tearDown(self):self.mock.stop()
    def test_std140_alignment_and_matrix_transpose(self):
        s=params.ParameterShader(self.raw,{'scalar':'FLOAT','vector':'VEC3','matrix':'MAT4','flag':'BOOL'})
        s.uniform_float('scalar',2);s.uniform_float('vector',(3,4,5))
        matrix=[[r*4+c for c in range(4)] for r in range(4)]
        s.uniform_float('matrix',matrix);s.uniform_bool('flag',True)
        self.assertEqual(list(s.data[:8]),[2,0,0,0,3,4,5,0])
        self.assertEqual(list(s.data[8:24]),[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15])
        self.assertEqual(s.data[24],1)
        self.assertIs(s.prepare_draw(),self.raw)
        s.prepare_draw();self.assertEqual(s.ubo.updates,1)
        s.uniform_float('scalar',6);s.prepare_draw();self.assertEqual(s.ubo.updates,2)
    def test_identical_parameters_skip_upload_but_mutations_upload(self):
        s=params.ParameterShader(self.raw,{'vector':'VEC3','matrix':'MAT4','scalar':'FLOAT'})
        v=[1,2,3];matrix=[[r*4+c for c in range(4)] for r in range(4)]
        s.uniform_float('vector',v);s.uniform_float('matrix',matrix)
        s.uniform_float('scalar',1.0);s.prepare_draw()
        for _ in range(3):
            s.uniform_float('vector',v);s.uniform_float('matrix',matrix)
            s.uniform_float('scalar',1.0+1e-10);s.prepare_draw()
        self.assertEqual(s.ubo.updates,1)
        v[1]=7;matrix[0][1]=99
        s.uniform_float('vector',v);s.uniform_float('matrix',matrix);s.prepare_draw()
        self.assertEqual(s.ubo.updates,2)
        self.assertEqual(s.ubo.data[1],7)
        self.assertEqual(s.ubo.data[8],99)

    def test_source_rewrite_and_buffer_binding(self):
        info=params.ParameterInfo()
        for kind,name in [('BOOL','enabled'),('INT','count'),('VEC3','tint'),('MAT4','view')]:info.push_constant(kind,name)
        info.vertex_source('void main(){ gl_Position = view * vec4(1.0); }')
        info.fragment_source('void main(){ if(enabled && count > 0) color = tint; }')
        info.build()
        self.assertEqual(info.info.block,(0,'RetroManParameters','rmParams'))
        self.assertIn('rmParams.enabled.x != 0.0',info.info.sources['fragment'])
        self.assertIn('int(rmParams.count.x)',info.info.sources['fragment'])
        self.assertIn('rmParams.view',info.info.sources['vertex'])
        self.assertIn('vec4 tint;',info.info.typedef)
    def test_clear_uses_keywords_on_actual_shadow_binding_method(self):
        tree=ast.parse((SRC/'gpu_renderer.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPURenderer')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_bind_shadow')
        ns={'Matrix':NS(Identity=lambda n:None), '_texture_option':lambda tex,name,value: getattr(tex,name)(value)};exec(compile(ast.Module([fn],type_ignores=[]),'shadow','exec'),ns)
        calls=[]
        class Texture:
            def clear(self,*,format,value):calls.append((format,value))
            def filter_mode(self,v):pass
        shader=NS(uniform_int=lambda *a:None,uniform_float=lambda *a:None,uniform_sampler=lambda *a:None)
        obj=NS(shadow_cache={},rscene=NS(settings=NS(shadow_bias=.01,shadow_softness=1)),_white_depth=None,gpu=NS(types=NS(GPUTexture=lambda *a,**kw:Texture())))
        ns['_bind_shadow'](obj,shader,object())
        self.assertEqual(calls,[('FLOAT',(1.0,))])
if __name__=='__main__':unittest.main()
