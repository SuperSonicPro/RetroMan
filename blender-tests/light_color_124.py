import bpy,pathlib,sys,importlib.util,json,traceback,os
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];out={'blender':bpy.app.version_string,'tests':[]}
def run():
 try:
  p=ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work'/os.environ.get('RM_VERSION','RetroMan-1.0.24')
  spec=importlib.util.spec_from_file_location('rm_lightqa',p/'__init__.py',submodule_search_locations=[str(p)]);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);m.register()
  from rm_lightqa.scene_adapter import extract_scene
  from rm_lightqa.bucket_render import render_scenes
  class Engine:
   def test_break(self):return False
   def _status(self,*a):pass
   def update_progress(self,*a):pass
  engine=Engine();bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
  sc=bpy.context.scene;sc.render.engine='RETROMAN';sc.render.resolution_x=32;sc.render.resolution_y=32;sc.render.resolution_percentage=100
  s=sc.retroman;s.ambient=0;s.environment_lighting=False;s.mesh_lighting=False;s.reflection_mode='OFF';s.enable_shadows=False;s.pixel_samples=1
  bpy.ops.mesh.primitive_plane_add(size=2);obj=bpy.context.object
  mat=bpy.data.materials.new('White matte');mat.use_nodes=True;bs=mat.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(1,1,1,1);bs.inputs['Specular IOR Level'].default_value=0;obj.data.materials.append(mat)
  bpy.ops.object.camera_add(location=(0,0,3));sc.camera=bpy.context.object;sc.camera.data.type='ORTHO';sc.camera.data.ortho_scale=4
  for kind in ('SUN','POINT','SPOT','AREA'):
   bpy.ops.object.light_add(type=kind,location=(0,0,2));light=bpy.context.object;data=light.data;data.energy=1 if kind=='SUN' else 100
   for nodes in (False,True):
    data.use_nodes=nodes;data.color=(1,1,1) if nodes else (1,0,0)
    if nodes:
     emission=next(n for n in data.node_tree.nodes if n.type=='EMISSION');emission.inputs['Color'].default_value=(0,0,1,1);emission.inputs['Strength'].default_value=1
    bpy.context.view_layer.update();rs,w=extract_scene(bpy.context.evaluated_depsgraph_get(),engine)
    pixels=render_scenes(engine,rs,rs)[0].reshape(32,32,4);rgb=pixels[16,16,:3];expected=2 if nodes else 0
    ok=bool(rgb[expected]>.001 and max(np.delete(rgb,expected))<rgb[expected]*.001)
    out['tests'].append({'type':kind,'nodes':nodes,'rgb':rgb.tolist(),'extracted_color':rs.lights[0].color,'ok':ok})
   bpy.data.objects.remove(light,do_unlink=True)
  from rm_lightqa.scene_adapter import _make_light
  from mathutils import Matrix
  bpy.ops.object.light_add(type='SUN',location=(0,0,2));light=bpy.context.object;data=light.data;data.energy=2;data.color=(.5,1,.25);data.use_nodes=True
  tree=data.node_tree;emission=next(n for n in tree.nodes if n.type=='EMISSION');output=next(n for n in tree.nodes if n.type=='OUTPUT_LIGHT')
  rgb=tree.nodes.new('ShaderNodeRGB');rgb.outputs[0].default_value=(.2,.4,.8,1);tree.links.new(rgb.outputs[0],emission.inputs['Color'])
  math=tree.nodes.new('ShaderNodeMath');math.operation='MULTIPLY';math.inputs[0].default_value=2;math.inputs[1].default_value=3;tree.links.new(math.outputs[0],emission.inputs['Strength'])
  translated=_make_light(light,Matrix.Identity(4));assert np.allclose(translated.color,(.6,2.4,1.2)) and translated.energy==2
  out['tests'].append({'name':'Linked RGB/Math light color, strength and data tint multiply correctly','ok':True})
  tree.links.remove(output.inputs['Surface'].links[0]);assert _make_light(light,Matrix.Identity(4)).color==(0,0,0)
  out['tests'].append({'name':'Disconnected light output emits nothing','ok':True})
  tree.links.new(emission.outputs[0],output.inputs['Surface']);tree.links.remove(emission.inputs['Strength'].links[0]);emission.inputs['Strength'].default_value=1
  tree.links.remove(emission.inputs['Color'].links[0]);emission.inputs['Color'].default_value=(0,0,1,1);data.color=(1,1,1)
  assert hasattr(bpy.types,'RETROMAN_PT_film')
  sc.render.image_settings.file_format='PNG';sc.render.image_settings.color_mode='RGBA'
  for transparent in (False,True):
   sc.render.film_transparent=transparent;bpy.context.view_layer.update()
   rs,w=extract_scene(bpy.context.evaluated_depsgraph_get(),engine)
   pixels=render_scenes(engine,rs,rs)[0].reshape(32,32,4)
   assert pixels[0,0,3]==(0 if transparent else 1) and pixels[16,16,3]==1
   bpy.ops.render.render();path=ROOT/f'blender-tests/film-124-{transparent}.png';bpy.data.images['Render Result'].save_render(str(path))
   saved=bpy.data.images.load(str(path),check_existing=False);arr=np.array(saved.pixels[:]).reshape(32,32,4)
   assert arr[0,0,3]==(0 if transparent else 1) and arr[16,16,3]==1
   assert arr[16,16,2] > arr[16,16,0] + .1
   bpy.data.images.remove(saved)
   out['tests'].append({'name':f'Actual GPU final render blue light and Film Transparent={transparent} PNG alpha','ok':True})
  for _ in range(3):m.unregister();m.register()
  out['tests'].append({'name':'Film panel repeated registration','ok':True})
  out['ok']=all(x['ok'] for x in out['tests'])
 except Exception:out.update(ok=False,error=traceback.format_exc());print(out['error'],flush=True)
 (ROOT/('blender-tests/light-color-'+os.environ.get('RM_LABEL','baseline')+'-124.json')).write_text(json.dumps(out,indent=2));print(out,flush=True);bpy.ops.wm.quit_blender()
bpy.app.timers.register(run,first_interval=2)
