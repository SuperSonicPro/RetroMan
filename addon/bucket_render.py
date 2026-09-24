# SPDX-License-Identifier: GPL-3.0-or-later
"""Stream surface grids through resident GPU shading and compute visibility."""
import math
import time
import numpy as np
from .model import TriangleData,VertexData
from .runtime import checkpoint
from .bucket_gpu import Bucket, GPUWorkspace, BucketBudgetExceeded
from .grid_atlas import GridAtlas
from .gpu_renderer import GPURenderer,estimate_subdiv


def projection(scene):return scene.camera.projection @ scene.camera.matrix_world.inverted()

def displacement_bound(material):
    if not material.has_displacement:return 0.0
    if material.displacement_texture is not None:
        data=np.asarray(material.displacement_texture.pixels,dtype=np.float32).reshape(-1,4)[:,0]
        lo=float(data.min());hi=float(data.max())
    else:lo=hi=material.displacement_constant
    return abs(material.displacement_scale)*max(abs(lo-material.displacement_midlevel),abs(hi-material.displacement_midlevel))

def screen_bounds(tri,end,vp,vp_end,width,height,displacement,lens):
    depth_bounds=[]
    for surface,matrix in ((tri,vp),(end,vp_end)):
        near_margin=displacement*sum(abs(matrix[2][k]+matrix[3][k]) for k in range(3))
        far_margin=displacement*sum(abs(matrix[3][k]-matrix[2][k]) for k in range(3))
        for vertex in (surface.a,surface.b,surface.c):
            q=matrix@vertex.p.to_4d();depth_bounds.append((q.z+q.w+near_margin,q.w-q.z+far_margin))
    if all(q[0]<0 for q in depth_bounds) or all(q[1]<0 for q in depth_bounds):return (0,0,0,0),0.0
    points=[];length=0.0
    for surface,matrix in ((tri,vp),(end,vp_end)):
        vertices=[]
        for v in (surface.a,surface.b,surface.c):
            q=matrix @ v.p.to_4d()
            if q.w<=1e-6:return (0,0,width,height),float('inf')
            vertices.append((q.x/q.w*width*.5+width*.5,q.y/q.w*height*.5+height*.5))
            if displacement:
                for sx in (-1,1):
                    for sy in (-1,1):
                        for sz in (-1,1):
                            from mathutils import Vector
                            z=matrix @ (v.p+Vector((sx*displacement,sy*displacement,sz*displacement))).to_4d()
                            if z.w<=1e-6:return (0,0,width,height),float('inf')
                            blurx=abs(lens[0]*(1/lens[2]-1/z.w));blury=abs(lens[1]*(1/lens[2]-1/z.w))
                            x=z.x/z.w*width*.5+width*.5;y=z.y/z.w*height*.5+height*.5
                            points.extend(((x-blurx,y-blury),(x+blurx,y+blury)))
            else:
                x,y=vertices[-1];bx=abs(lens[0]*(1/lens[2]-1/q.w));by=abs(lens[1]*(1/lens[2]-1/q.w))
                points.extend(((x-bx,y-by),(x+bx,y+by)))
        for i,j in ((0,1),(1,2),(2,0)):length=max(length,math.dist(vertices[i],vertices[j]))
    xs,ys=zip(*points)
    return (max(0,math.floor(min(xs))),max(0,math.floor(min(ys))),min(width,math.ceil(max(xs))),min(height,math.ceil(max(ys)))),length


def children(tri):
    def mid(a,b):return VertexData((a.p+b.p)*.5,(a.n+b.n)*.5,tuple((a.uv[k]+b.uv[k])*.5 for k in range(2)))
    a,b,c=tri.a,tri.b,tri.c;ab,bc,ca=mid(a,b),mid(b,c),mid(c,a)
    return [TriangleData(x,y,z,tri.material) for x,y,z in ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))]


def patches(tri,end,vp,vp_end,width,height,rate,cap,bounds,displacement,lens,cancel,adaptive=True,cache=None,root_bound=None):
    stack=[(tri,end,0)];visits=0
    while stack:
        checkpoint(cancel);a,b,depth=stack.pop();visits+=1
        if visits>200000:raise RuntimeError("Eye-plane/displacement splitting exceeds the per-bucket work budget")
        key=(id(a),id(b))
        cached=cache.get(key) if cache is not None else None
        if cached is None:
            box,length=(root_bound[:4],root_bound[4]) if depth==0 and root_bound is not None else screen_bounds(a,b,vp,vp_end,width,height,displacement,lens)
            cached=[a,b,box,length,None]
            if cache is not None and len(cache)<65536:cache[key]=cached
        else:box,length=cached[2:4]
        if box[2]<=bounds[0] or box[0]>=bounds[2] or box[3]<=bounds[1] or box[1]>=bounds[3]:continue
        if not adaptive:
            yield a,b,cap
            continue
        if length>rate*cap and depth<12:
            if cached[4] is None:cached[4]=[(child,child) for child in children(a)] if a is b else list(zip(children(a),children(b)))
            stack.extend((x,y,depth+1) for x,y in cached[4]);continue
        if not math.isfinite(length):
            # The GPU visibility stage clips micropolygons in homogeneous space.
            subdiv=cap
        else:
            subdiv=1
            while subdiv<cap and subdiv<math.ceil(length/rate):subdiv*=2
            subdiv=min(subdiv,cap)
        yield a,b,subdiv


def clip_positions(positions,matrix):
    ones=np.ones((*positions.shape[:2],1),dtype=np.float32)
    return np.concatenate((positions,ones),axis=2) @ np.asarray(matrix,dtype=np.float32).T


def render_buckets(engine,depsgraph,initial_scene,initial_warnings):
    from .scene_adapter import extract_scene
    scene=depsgraph.scene;settings=initial_scene.settings
    motion=bool(scene.render.use_motion_blur) and bool(getattr(getattr(depsgraph,'view_layer',None),'use_motion_blur',True))
    base_time=scene.frame_current+scene.frame_subframe
    start_scene=end_scene=initial_scene;warnings=set(initial_warnings)
    try:
        if motion:
            shutter=max(0.0,float(scene.render.motion_blur_shutter));pos=getattr(scene.render,'motion_blur_position','CENTER')
            start=0.0 if pos=='START' else (-shutter if pos=='END' else -.5*shutter)
            evaluated=[]
            for offset in (start,start+shutter):
                t=base_time+offset;engine.frame_set(math.floor(t),t-math.floor(t))
                engine._status('Evaluating moving micropolygon endpoint')
                rs,w=extract_scene(depsgraph,engine);warnings.update(w);evaluated.append(rs)
            start_scene,end_scene=evaluated
            if start_scene.topology_signature!=end_scene.topology_signature:
                raise RuntimeError('Motion blur requires stable topology across the shutter; this scene changes mesh connectivity or instances')
        return render_scenes(engine,start_scene,end_scene,warnings,motion)
    finally:
        if motion:engine.frame_set(math.floor(base_time),base_time-math.floor(base_time))


def render_scenes(engine,start_scene,end_scene,warnings=(),motion=False,bucket_size=None):
    settings=start_scene.settings;width=start_scene.width;height=start_scene.height
    axis=max(1,int(settings.pixel_samples))
    if motion or start_scene.camera.dof_enabled:axis=max(axis,math.ceil(math.sqrt(max(1,settings.temporal_samples))))
    size=int(bucket_size or getattr(settings,'bucket_size',64));size=max(8,min(size,128))
    camera=start_scene.camera
    lens=(0.0,0.0,max(1e-6,camera.focus_distance))
    if camera.dof_enabled and camera.camera_type=='PERSP':
        lens=(camera.aperture_radius*camera.projection[0][0]*width*.5,
              camera.aperture_radius*camera.projection[1][1]*height*.5/max(.01,camera.aperture_ratio),lens[2])
    vp=projection(start_scene);vp_end=projection(end_scene)
    engine._status('Preparing compact source geometry')
    triangles=start_scene.triangles;end_triangles=end_scene.triangles if motion else triangles
    if len(triangles)!=len(end_triangles):raise RuntimeError('Motion topology mismatch')
    # Coarse GPU source geometry is used for classic auxiliary maps only.
    engine._status('Preparing GPU shadow and reflection maps')
    renderer=GPURenderer(start_scene,viewport=True,cancel=engine.test_break,status=engine._status)
    renderer.viewport=False
    renderer.build_shadow_maps(cancel=engine.test_break);renderer.build_reflection_map(cancel=engine.test_break)
    # Final grids are created and released in bounded batches inside each bucket.
    renderer.reyes_chunks=[];renderer.groups=[];renderer.use_reyes=True
    renderer.framebuffer=None;renderer.color_texture=None;renderer.depth_texture=None
    atlas=GridAtlas(renderer)
    workspace=GPUWorkspace(engine.test_break)
    bins={};displacements={};patch_cache={}
    from itertools import chain
    def materials(source):
        if hasattr(source,'materials'):return source.materials()
        return (tri.material for tri in source)
    for material in chain(materials(triangles),materials(end_triangles)):
        key=id(material)
        if key not in displacements:displacements[key]=displacement_bound(material) if settings.displacement_enabled else 0.0
    engine._status('Bounding source patches on GPU')
    from .patch_bounds import source_bounds
    bounds=source_bounds(triangles,end_triangles,vp,vp_end,width,height,displacements,lens,engine.test_break,engine._status)
    for i,row in enumerate(bounds):
        if i%256==0:checkpoint(engine.test_break)
        if i%65536==0:engine._status(f'Assigning source patches to buckets: {i:,}/{len(bounds):,}')
        box=tuple(map(int,row[:4]))
        if box[2]<=box[0] or box[3]<=box[1]:continue
        for y in range(box[1]//size,(box[3]-1)//size+1):
            for x in range(box[0]//size,(box[2]-1)//size+1):bins.setdefault((x,y),[]).append(i)
    film=start_scene.film_transparent
    world=start_scene.world_color if settings.use_world_color else (0,0,0)
    background=(0,0,0,0) if film else (*world,1)
    pixels=np.empty((height,width,4),dtype=np.float32);pixels[:]=background
    # Environment background remains a map, sampled outside surface visibility.
    if not film and settings.use_world_color and start_scene.environment_texture is not None:
        image=start_scene.environment_texture
        workspace.sky=(renderer._upload_texture(image,precise=True),(1,start_scene.environment_strength*(2**settings.exposure),image.width,image.height))
    total=math.ceil(width/size)*math.ceil(height/size);done=0;peak=0;shaded=0
    atlas_seconds=0.0;visibility_seconds=0.0;resolve_seconds=0.0;started=time.perf_counter()
    rate=max(.5,float(settings.shading_rate));cap=min(32,max(1,int(settings.gpu_max_subdiv)))
    jobs=[(x,y,min(size,width-x),min(size,height-y),bins.get((x//size,y//size),[]))
          for y in range(0,height,size) for x in range(0,width,size)]
    jobs.reverse();splits=0;covered=0
    while jobs:
        checkpoint(engine.test_break);x,y,w,h,indices=jobs.pop()
        engine._status(f'Bucket {done+1}/{total}: ({x}, {y}), {w}x{h}, {axis*axis} stochastic samples')
        try:
            with Bucket(x,y,w,h,width,height,axis,lens,settings.pixel_filter,(camera.aperture_blades,camera.aperture_rotation),workspace=workspace,vp=vp,vp_end=vp_end) as bucket:
                groups={};last_notice=time.monotonic()
                def flush(key):
                    nonlocal peak,shaded,atlas_seconds,visibility_seconds,last_notice
                    entries=groups.pop(key);subdiv=key[1]
                    a=[entry[0] for entry in entries];b=[entry[1] for entry in entries]
                    if time.monotonic()-last_notice>2:
                        engine._status(f'Shading bucket at ({x}, {y}): {shaded:,} micropolygons processed')
                        last_notice=time.monotonic()
                    tick=time.perf_counter()
                    start=atlas.shade(a,subdiv,resident=True)
                    end=start if all(x is y for x,y in zip(a,b)) else atlas.shade(b,subdiv,resident=True)
                    atlas_seconds+=time.perf_counter()-tick
                    tick=time.perf_counter()
                    bucket.push(start,end);visibility_seconds+=time.perf_counter()-tick;peak=max(peak,start[1]);shaded+=start[1]
                for i in indices:
                    box=bounds[i]
                    if box[2]<=x or box[0]>=x+w or box[3]<=y or box[1]>=y+h:continue
                    tri=triangles[i];end_tri=tri if end_triangles is triangles else end_triangles[i]
                    for a,b,n in patches(tri,end_tri,vp,vp_end,width,height,rate,cap,(x,y,x+w,y+h),max(displacements[id(tri.material)],displacements[id(end_tri.material)]),lens,engine.test_break,settings.adaptive_dicing,patch_cache,bounds[i]):
                        key=(id(tri.material),n);groups.setdefault(key,[]).append((a,b))
                        if len(groups[key])*n*n>=16384 or len(groups[key])>=1024:flush(key)
                    # Bound the number of pending materials as well as grid size.
                    if len(groups)>32:
                        for key in list(groups):flush(key)
                for key in list(groups):flush(key)
                tick=time.perf_counter()
                pixels[y:y+h,x:x+w]=bucket.finish(pixels[y:y+h,x:x+w].copy())
                resolve_seconds+=time.perf_counter()-tick
        except BucketBudgetExceeded:
            groups.clear()
            if w==1 and h==1:raise
            # Keep sample locations keyed to full-frame pixels; subdivision must
            # not change transparency order, filtering, or stochastic sequences.
            xs=[(x,w)] if w==1 else [(x,w//2),(x+w//2,w-w//2)]
            ys=[(y,h)] if h==1 else [(y,h//2),(y+h//2,h-h//2)]
            children_jobs=[(xx,yy,ww,hh,indices) for yy,hh in ys for xx,ww in xs]
            jobs.extend(reversed(children_jobs));total+=len(children_jobs)-1;splits+=1
            engine._status(f'Splitting dense bucket at ({x}, {y}) to fit GPU memory')
            continue
        done+=1;covered+=w*h;engine.update_progress(.15+.83*covered/(width*height))
        if hasattr(engine,"_publish_bucket"):engine._publish_bucket(x,y,pixels[y:y+h,x:x+w])
    renderer.bucket_statistics={'buckets':total,'bucket_splits':splits,'peak_grid_micropolygons':peak,'shaded_micropolygons':shaded,'samples_per_pixel':axis*axis,'atlas_submit_seconds':atlas_seconds,'visibility_submit_seconds':visibility_seconds,'resolve_sync_seconds':resolve_seconds,'fragment_capacity':workspace.capacity,'bucket_seconds':time.perf_counter()-started}
    return pixels.reshape(-1,4),renderer,start_scene,list(warnings),axis*axis
