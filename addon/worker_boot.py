# SPDX-License-Identifier: GPL-3.0-or-later
"""Private worker entry point. Inputs are created in a private temporary directory."""
import sys,pathlib,importlib.util,pickle,json,os,traceback
import bpy
folder,addon=map(pathlib.Path,sys.argv[sys.argv.index('--')+1:])
name='retroman_worker_addon'
spec=importlib.util.spec_from_file_location(name,addon/'__init__.py',submodule_search_locations=[str(addon)])
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
class SceneReader(pickle.Unpickler):
    def find_class(self,module,name):
        # The owned snapshot's addon package may be bl_ext.<repository>.retroman.
        if module.endswith('.render_worker'):module='retroman_worker_addon.render_worker'
        if module.endswith('.model'):module='retroman_worker_addon.model'
        if module.endswith('.mesh_extract'):module='retroman_worker_addon.mesh_extract'
        return super().find_class(module,name)
class Engine:
    count=0
    def emit(self,**event):
        with (folder/'events.jsonl').open('a', encoding='utf-8') as file:file.write(json.dumps(event)+'\n')
    def test_break(self):return False
    def _status(self,text):self.emit(type='status',text=text)
    def update_progress(self,value):self.emit(type='progress',value=value)
    def _publish_bucket(self,x,y,pixels):
        import numpy as np
        self.count+=1;filename=f'tile-{self.count}.npy';np.save(folder/filename,pixels,allow_pickle=False)
        self.emit(type='tile',x=x,y=y,file=filename)
def run():
    try:
        import gpu,numpy as np
        if bpy.app.background:gpu.init()
        with (folder/'input.pickle').open('rb') as file:start,end,warnings,motion,_=SceneReader(file).load()
        from retroman_worker_addon.bucket_render import render_scenes
        from retroman_worker_addon.engine import RETROMAN_RenderEngine
        from types import SimpleNamespace
        engine=Engine()
        fast=start.settings.geometry_mode=='TRIANGLES' or (start.settings.geometry_mode=='AUTO' and start.settings.quality=='DRAFT')
        if fast and not motion:
            Engine._render_gpu=RETROMAN_RenderEngine._render_gpu
            Engine.update_stats=lambda self,category,text:self._status(text)
            Engine._publish_pixels=lambda self,p:self._publish_bucket(0,0,np.asarray(p,dtype=np.float32).reshape(start.height,start.width,4))
            dg=SimpleNamespace(scene=SimpleNamespace(retroman=start.settings,render=SimpleNamespace(use_motion_blur=False),frame_current=0,frame_subframe=0),view_layer=None)
            pixels,renderer,scene,warnings,samples=RETROMAN_RenderEngine._render_gpu_effects_legacy(engine,dg,start,warnings)
            engine._publish_pixels(pixels)
        else:
            if fast:warnings=list(warnings)+['Fast geometry with motion uses the GPU bucket sampler for moving coverage']
            pixels,renderer,scene,warnings,samples=render_scenes(engine,start,end,warnings,motion)
        np.save(folder/'pixels.npy',pixels,allow_pickle=False)
        result={'ok':True,'warnings':warnings,'samples':samples,'renderer':{'bucket_statistics':getattr(renderer,'bucket_statistics',{}),'micropolygon_count':renderer.micropolygon_count,'geometry_label':renderer.geometry_label,'shadow_warnings':getattr(renderer,'shadow_warnings',[]),'reflection_warnings':getattr(renderer,'reflection_warnings',[]),'resource_warnings':getattr(renderer,'resource_warnings',[])}}
    except Exception:
        result={'ok':False,'error':traceback.format_exc()};print(result['error'],flush=True)
    temp=folder/'complete.tmp';temp.write_text(json.dumps(result), encoding='utf-8');temp.replace(folder/'complete.json')
    # The parent owns lifetime. Avoid driver shutdown hangs after a finished frame.
    if not bpy.app.background:bpy.ops.wm.quit_blender()
if bpy.app.background:run()
else:bpy.app.timers.register(run,first_interval=.1)
