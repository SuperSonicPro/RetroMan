# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolate final GPU ownership from Blender's UI draw-context lock."""
import copy,copyreg,json,math,os,pickle,subprocess,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from .runtime import checkpoint,RenderCancelled

def owned(scene):
    result=copy.copy(scene)
    settings=scene.settings
    result.settings=SimpleNamespace(**{p.identifier:getattr(settings,p.identifier) for p in settings.bl_rna.properties if p.identifier!='rna_type'})
    if hasattr(scene.triangles,'blocks'):
        result.triangles=copy.copy(scene.triangles);result.triangles.cancel=None
    return result

def restore_math(kind,values):
    import mathutils
    return getattr(mathutils,kind)(values)

def dump_request(path,payload):
    from mathutils import Vector,Matrix
    with path.open('wb') as file:
        writer=pickle.Pickler(file,protocol=5);writer.dispatch_table=copyreg.dispatch_table.copy()
        writer.dispatch_table[Vector]=lambda v:(restore_math,('Vector',tuple(v)))
        writer.dispatch_table[Matrix]=lambda m:(restore_math,('Matrix',tuple(tuple(row) for row in m)))
        writer.dump(payload)

def render(engine,depsgraph,initial,warnings):
    import bpy,numpy as np
    from .scene_adapter import extract_scene
    sc=depsgraph.scene;base=sc.frame_current+sc.frame_subframe
    motion=bool(sc.render.use_motion_blur) and bool(getattr(getattr(depsgraph,'view_layer',None),'use_motion_blur',True))
    start=end=owned(initial);all_warnings=set(warnings)
    try:
        if motion:
            shutter=max(0,float(sc.render.motion_blur_shutter));pos=getattr(sc.render,'motion_blur_position','CENTER')
            offset=0 if pos=='START' else (-shutter if pos=='END' else -.5*shutter)
            scenes=[]
            for dt in (offset,offset+shutter):
                checkpoint(engine.test_break);t=base+dt;engine.frame_set(math.floor(t),t-math.floor(t))
                engine._status('Evaluating shutter endpoint')
                rs,w=extract_scene(depsgraph,engine);scenes.append(owned(rs));all_warnings.update(w)
            start,end=scenes
            if start.topology_signature!=end.topology_signature:raise RuntimeError('Motion blur requires stable topology across the shutter')
    finally:
        if motion:engine.frame_set(math.floor(base),base-math.floor(base))
    if motion and hasattr(start.triangles,'matches'):
        engine._status('Checking shutter geometry for reusable shading')
        if start.triangles.matches(end.triangles,engine.test_break):
            end.triangles=start.triangles
            engine._status('Unchanged shutter geometry: reusing shaded grids')
    with tempfile.TemporaryDirectory(prefix='retroman-render-') as folder:
        directory=Path(folder);engine._status('Preparing independent GPU worker')
        dump_request(directory/'input.pickle',(start,end,sorted(all_warnings),motion,__package__))
        addon=Path(__file__).parent
        # The exact running Blender executable preserves Flatpak/runtime libraries.
        command=[bpy.app.binary_path,'--factory-startup','-noaudio']
        if bpy.app.version>=(5,2,0):command+=['--background']
        else:command+=['--window-geometry','0','0','320','240']
        import gpu
        backend=gpu.platform.backend_type_get().lower()
        if backend in ('vulkan','opengl','metal'):command+=['--gpu-backend',backend]
        command+=['--python',str(addon/'worker_boot.py'),'--',str(directory),str(addon)]
        with (directory/'worker.log').open('wb') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:
                cursor=0;last_event=time.monotonic();next_notice=last_event+10
                while True:
                    checkpoint(engine.test_break)
                    events=directory/'events.jsonl'
                    if events.exists():
                        with events.open(encoding='utf-8') as stream:
                            stream.seek(cursor)
                            while True:
                                line=stream.readline()
                                if not line or not line.endswith('\n'):break
                                event=json.loads(line);cursor=stream.tell();last_event=time.monotonic()
                                if event['type']=='status':engine._status(event['text'])
                                elif event['type']=='progress':engine.update_progress(event['value'])
                                elif event['type']=='tile':
                                    tile=directory/event['file'];engine._publish_bucket(event['x'],event['y'],np.load(tile,allow_pickle=False));tile.unlink()
                    if (directory/'complete.json').exists():break
                    if process.poll() is not None:
                        detail=(directory/'worker.log').read_text(encoding='utf-8', errors='replace')[-4000:]
                        raise RuntimeError('GPU worker exited before finishing. '+detail)
                    now=time.monotonic()
                    if now>=next_notice:
                        engine._status(f'GPU worker active; {now-last_event:.0f}s since last progress');next_notice=now+10
                    if now-last_event>120:raise RuntimeError('GPU worker made no progress for 120 seconds; stopped to keep Blender responsive')
                    time.sleep(.025)
                result=json.loads((directory/'complete.json').read_text(encoding='utf-8'))
                if not result['ok']:raise RuntimeError(result['error'])
                pixels=np.load(directory/'pixels.npy',allow_pickle=False)
                summary=SimpleNamespace(**result['renderer'])
                return pixels,summary,initial,result['warnings'],result['samples']
            finally:
                # Cancellation never waits for a stuck GPU command in the UI process.
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=2)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=2)
