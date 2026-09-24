"""No Blender required. Behavioral tests use narrow doubles, not GPU execution."""
import ast
import contextlib
import importlib.util
import io
import pathlib
import re
import sys
import types
import unittest
import zipfile
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work/RetroMan-1.0.24'
NS = types.SimpleNamespace

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

pkg = types.ModuleType('rm'); pkg.__path__ = [str(SRC)]; sys.modules['rm'] = pkg
runtime = load('rm.runtime', SRC/'runtime.py')

class Buffer:
    def __init__(self, width, rows, start):
        self.width, self.rows, self.start = width, rows, start
        self.dimensions = (rows, width, 4)
    def __getitem__(self, index):
        # Mimic a shaped GPU buffer: top-level indexing addresses rows.
        if index >= self.rows:
            raise IndexError('buffer index out of range')
        return [[float(self.start)] * 4] * self.width
    def to_list(self):
        if tuple(self.dimensions) != (self.width*self.rows, 4):
            raise AssertionError('readback was not reshaped')
        return [[float(self.start + i//self.width), 0., 0., 1.] for i in range(self.width*self.rows)]

class Framebuffer:
    def __init__(self): self.calls=[]
    def bind(self): return contextlib.nullcontext()
    def read_color(self, x,y,w,h,*args):
        self.calls.append((x,y,w,h)); return Buffer(w,h,y)

class ReadbackTests(unittest.TestCase):
    def test_shaped_readback_non_square_and_order(self):
        fb=Framebuffer(); progress=[]
        pixels=runtime.read_framebuffer(fb,3,65,progress=progress.append)
        self.assertEqual(len(pixels),195)
        self.assertEqual([c[3] for c in fb.calls],[32,32,1])
        self.assertEqual([pixels[i*3][0] for i in range(65)],list(range(65)))
        self.assertEqual(progress[-1],1.)
    def test_cancel_before_read(self):
        fb=Framebuffer()
        with self.assertRaises(runtime.RenderCancelled): runtime.read_framebuffer(fb,2,2,cancel=lambda:True)
        self.assertEqual(fb.calls,[])
    def test_cancel_between_strips(self):
        fb=Framebuffer()
        with self.assertRaises(runtime.RenderCancelled):
            runtime.read_framebuffer(fb,3,65,cancel=lambda: bool(fb.calls))
        self.assertEqual(len(fb.calls),1)
    def test_original_flat_indexing_fails(self):
        buf=Buffer(3,2,0)
        with self.assertRaises(IndexError):
            [[buf[i],buf[i+1],buf[i+2],buf[i+3]] for i in range(0,24,4)]

class EngineBase:
    def __init__(self,*a,**kw):
        self.events=[];self.cancelled=False;self.error=None
        self.result=NS(layers=[NS(passes={'Combined':NS(rect=None)})])
    def update_stats(self,*a): self.events.append(('stats',a))
    def update_progress(self,p): self.events.append(('progress',p))
    def test_break(self): return self.cancelled
    def report(self,*a): self.events.append(('report',a))
    def error_set(self,e): self.error=e
    def begin_result(self,*a): self.events.append(('begin',a));return self.result
    def update_result(self,r): self.events.append(('update',r))
    def end_result(self,r,**kw): self.events.append(('end',kw))
    def frame_set(self,*a): self.events.append(('frame',a))

bpy=types.ModuleType('bpy');bpy.types=NS(RenderEngine=EngineBase,Operator=object,Panel=object)
sys.modules['bpy']=bpy
sys.modules['mathutils']=NS(Vector=lambda x:x)
scene_adapter=types.ModuleType('rm.scene_adapter')
scene_adapter.extract_scene=lambda *a,**kw:None
scene_adapter.extract_lights=lambda *a,**kw:[]
scene_adapter.viewport_camera=lambda *a:None
sys.modules['rm.scene_adapter']=scene_adapter
raster=types.ModuleType('rm.raster');raster.build_shadow_maps=lambda *a,**kw:None;raster.render_scene=lambda *a,**kw:None
sys.modules['rm.raster']=raster
engine=load('rm.engine',SRC/'engine.py')
gpu_stub=types.ModuleType('rm.gpu_renderer');gpu_stub.gpu_available=lambda:True;gpu_stub.gpu_info=lambda:{'backend':'TEST'}
sys.modules['rm.gpu_renderer']=gpu_stub

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.settings=NS(render_device='AUTO',geometry_mode='TRIANGLES',quality='DRAFT',show_translation_warnings=False,pixel_samples=1)
        self.scene=NS(settings=self.settings,width=2,height=2,triangles=[],lights=[],camera=NS(dof_enabled=False))
        self.dg=NS(scene=NS(retroman=self.settings,render=NS(use_motion_blur=False)))
        self.engine=engine.RETROMAN_RenderEngine()
        self.patcher=patch.object(engine,'extract_scene',return_value=(self.scene,[]));self.patcher.start()
        self.silent=contextlib.redirect_stdout(io.StringIO());self.silent.__enter__()
    def tearDown(self):self.patcher.stop();self.silent.__exit__(None,None,None)
    def ends(self):return [e[1] for e in self.engine.events if e[0]=='end']
    def test_success_publishes_and_closes_once(self):
        pixels=[[.1,.2,.3,1]]*4
        self.engine._render_gpu_effects=lambda *a:(pixels,NS(),self.scene,[],1)
        self.engine.render(self.dg)
        self.assertEqual(self.engine.result.layers[0].passes['Combined'].rect,pixels)
        self.assertEqual(self.ends(),[{'cancel':False}])
        self.assertEqual([e for e in self.engine.events if e[0]=='progress'][-1],('progress',1.0))
    def test_gpu_failure_does_not_launch_cpu(self):
        self.engine._render_gpu_effects=lambda *a: (_ for _ in ()).throw(ValueError('uniform missing'))
        self.engine._render_cpu=lambda *a:self.fail('CPU fallback was invoked')
        self.engine.render(self.dg)
        self.assertIn('uniform missing',self.engine.error)
        self.assertEqual(self.ends(),[])
    def test_gpu_unavailable_does_not_launch_cpu(self):
        with patch.object(gpu_stub,'gpu_available',return_value=False):self.engine.render(self.dg)
        self.assertIn('no hardware GPU',self.engine.error)
        self.assertEqual(self.ends(),[])
    def test_explicit_cpu(self):
        self.settings.render_device='CPU'
        self.engine._render_cpu=lambda *a:[[0,0,0,1]]*4
        self.engine.render(self.dg)
        self.assertEqual(self.ends(),[{'cancel':False}])
    def test_cancel_is_not_gpu_failure(self):
        self.engine._render_gpu_effects=lambda *a:(_ for _ in ()).throw(runtime.RenderCancelled())
        self.engine.render(self.dg)
        self.assertIsNone(self.engine.error)
        self.assertEqual(self.ends(),[])
    def test_cancel_during_translation_no_result(self):
        with patch.object(engine,'extract_scene',side_effect=runtime.RenderCancelled()):self.engine.render(self.dg)
        self.assertEqual(self.ends(),[])
        self.assertIsNone(self.engine.error)
    def test_rect_write_error_closes_result(self):
        class BadPass:
            @property
            def rect(self):return None
            @rect.setter
            def rect(self,v):raise ValueError('bad rect')
        self.engine.result.layers[0].passes['Combined']=BadPass()
        self.engine._render_gpu_effects=lambda *a:([[0]*4]*4,NS(),self.scene,[],1)
        self.engine.render(self.dg)
        self.assertIn('bad rect',self.engine.error)
        self.assertEqual(self.ends(),[{'cancel':True}])

class SamplingTests(unittest.TestCase):
    setUp = EngineTests.setUp
    tearDown = EngineTests.tearDown
    # Reuse fixtures only; sampling checks exercise the real sampling orchestration.
    def test_motion_cancel_restores_frame(self):
        self.dg.scene.render.use_motion_blur=True
        self.dg.scene.frame_current=12
        self.dg.scene.frame_subframe=0.25
        self.settings.temporal_samples=2
        sampling=NS(camera_sample=lambda *a,**kw:(None,None,None),pixel_jitter=lambda *a:(0,0),
                    reconstruction_weight=lambda *a:1.,shutter_offsets=lambda *a:[-0.1,0.1])
        accum=lambda *a:NS()
        def cancelled(*a,**kw):raise runtime.RenderCancelled()
        with patch.dict(sys.modules,{'rm.sampling':sampling}), patch.object(gpu_stub,'GPUAccumulator',accum,create=True), patch.object(gpu_stub,'GPURenderer',cancelled,create=True):
            with self.assertRaises(runtime.RenderCancelled):
                self.engine._render_gpu_effects_legacy(self.dg,self.scene,[])
        self.assertEqual([e for e in self.engine.events if e[0]=='frame'][-1],('frame',(12,0.25)))

    def test_first_sample_is_published(self):
        self.settings.pixel_samples=2
        sampling=NS(camera_sample=lambda *a,**kw:(None,None,None),pixel_jitter=lambda *a:(0,0),
                    reconstruction_weight=lambda *a:1.,shutter_offsets=lambda *a:[0]*4)
        self.dg.scene.frame_current=1;self.dg.scene.frame_subframe=0
        pixels=[[.1,.2,.3,1]]*4
        calls=[]
        renderer=NS(reflection_map=None,color_texture=object(),render_to_texture=lambda **kw:calls.append('draw'),read_pixels=lambda **kw:pixels)
        accum=NS(add=lambda *a:calls.append('accum'),read_pixels=lambda **kw:pixels)
        self.engine._active_result=self.engine.result
        with patch.dict(sys.modules,{'rm.sampling':sampling}), patch.object(gpu_stub,'GPUAccumulator',lambda *a:accum,create=True), patch.object(gpu_stub,'GPURenderer',lambda *a,**kw:renderer,create=True):
            result=self.engine._render_gpu_effects_legacy(self.dg,self.scene,[])
        self.assertEqual(result[-1],4)
        self.assertEqual(calls.count('draw'),4)
        self.assertEqual(self.engine.result.layers[0].passes['Combined'].rect,pixels)


class StructureTests(unittest.TestCase):
    def test_all_python_compiles(self):
        for file in SRC.glob('*.py'):
            with self.subTest(file=file.name):compile(file.read_text(encoding='utf-8'),str(file),'exec')
    def test_version_matches(self):
        import tomllib
        self.assertEqual(tomllib.loads((SRC/'blender_manifest.toml').read_text(encoding='utf-8'))['version'],'1.0.24')
        self.assertIn('"version": (1, 0, 24)',(SRC/'__init__.py').read_text(encoding='utf-8'))
        self.assertEqual((SRC/'VERSION').read_text(encoding='utf-8').strip(),'1.0.24')
    def test_shader_uniforms_are_referenced(self):
        # Execute only shader factories against a declarative GPU API double.
        tree=ast.parse((SRC/'gpu_renderer.py').read_text(encoding='utf-8'))
        keep=[n for n in tree.body if isinstance(n,(ast.Assign,ast.AnnAssign)) or isinstance(n,ast.FunctionDef) and n.name in {'_grid_shading_sources','_add_common_vertex_resources','_make_interface','_create_shader_set'}]
        infos=[]
        class Info:
            def __init__(self):self.constants=[];self.sources=[];infos.append(self)
            def build(self):return self
            def push_constant(self,t,n):self.constants.append(n)
            def vertex_source(self,s):self.sources.append(s)
            fragment_source=vertex_source
            def __getattr__(self,n):return lambda *a,**kw:None
        fake=NS(types=NS(GPUShaderCreateInfo=Info,GPUStageInterfaceInfo=lambda *a:Info()),shader=NS(create_from_info=lambda i:i))
        namespace={'ParameterInfo': Info}
        with patch.dict(sys.modules,{'gpu':fake}):
            exec(compile(ast.Module(keep,type_ignores=[]),'factories','exec'),namespace)
            namespace['_create_shader_set'](reyes=False)
            namespace['_create_shader_set'](reyes=True)
        for info in infos:
            for name in info.constants:
                with self.subTest(uniform=name):self.assertRegex('\n'.join(info.sources),r'\b'+name+r'\b')
    def test_viewport_identity_and_activation(self):
        ui=load('rm.ui',SRC/'ui.py')
        shading=NS(type='MATERIAL');area=NS(type='VIEW_3D',spaces=NS(active=NS(shading=shading)),tag_redraw=lambda:None)
        context=NS(screen=NS(areas=[area]),scene=NS(render=NS(engine='BLENDER_EEVEE')))
        op=ui.RETROMAN_OT_activate_viewport();op.report=lambda *a:None
        self.assertEqual(op.execute(context),{'FINISHED'})
        self.assertEqual(shading.type,'RENDERED');self.assertEqual(context.scene.render.engine,'RETROMAN')
        self.assertFalse(engine.RETROMAN_RenderEngine.bl_use_eevee_viewport)
        self.assertIn('RETROMAN 1.0.24 |',(SRC/'engine.py').read_text(encoding='utf-8'))
    def test_optional_gpu_capability_does_not_disable_gpu(self):
        tree=ast.parse((SRC/'gpu_renderer.py').read_text(encoding='utf-8'))
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'gpu_info','gpu_available'}]
        fake=NS(platform=NS(backend_type_get=lambda:'OPENGL',device_type_get=lambda:'NVIDIA'),capabilities=NS())
        ns={}
        with patch.dict(sys.modules,{'gpu':fake}):
            exec(compile(ast.Module(funcs,type_ignores=[]),'gpu_info','exec'),ns)
            self.assertTrue(ns['gpu_available']())
            self.assertGreater(len(ns['gpu_info']()['warnings']),0)
    def test_cpu_scanlines_have_cancel_checks(self):
        tree=ast.parse((SRC/'raster.py').read_text(encoding='utf-8'))
        for func in ('render_scene','_raster_depth_triangle'):
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==func)
            loops=[n for n in ast.walk(node) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='y']
            self.assertTrue(loops)
            for loop in loops:self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='checkpoint' for n in ast.walk(loop)))

if __name__=='__main__':unittest.main(verbosity=2)
