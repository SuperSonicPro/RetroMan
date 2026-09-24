# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU surface shading into a bounded temporary atlas, before visibility."""
import math
import numpy as np
from .gpu_renderer import _create_shader_set,make_barycentric_microgrid,GPUReyesChunk,_texture_option
from .grid_filter import GPUPyramid
from .runtime import checkpoint

class GridAtlas:
    def __init__(self,renderer):
        self.renderer=renderer;self.gpu=renderer.gpu
        self.base,self.light,_=_create_shader_set(reyes=True,atlas=True)
        self.batches={};self.textures={};self.pyramid=GPUPyramid()
    def filtered(self,image,premultiply=False):
        if image is None:return self.renderer.white_texture,(1,1,1)
        key=(id(image),premultiply)
        if key not in self.textures:
            self.textures[key]=self.pyramid.build(image,self.renderer._upload_texture(image,precise=True),premultiply)
        return self.textures[key]
    def batch(self,subdiv):
        if subdiv not in self.batches:
            from gpu_extras.batch import batch_for_shader
            bary=np.asarray(make_barycentric_microgrid(subdiv),dtype=np.float32)
            triples=bary.reshape(-1,3,3)
            attributes={'bary':bary,'shadeBary':np.repeat(triples.mean(axis=1),3,axis=0),
                'atlasA':np.repeat(triples[:,0],3,axis=0),'atlasB':np.repeat(triples[:,1],3,axis=0),'atlasC':np.repeat(triples[:,2],3,axis=0)}
            active={name for name,_ in self.base.attrs_info_get()}
            self.batches[subdiv]=batch_for_shader(self.base,'TRIS',{name:data for name,data in attributes.items() if name in active})
        return self.batches[subdiv]
    def shade(self,triangles,subdiv,*,resident=False):
        r=self.renderer;g=self.gpu;checkpoint(r.cancel)
        count=len(triangles)*subdiv*subdiv;width=min(1024,count);height=math.ceil(count/width)
        if count==0:raise ValueError('Empty shading grid')
        material=triangles[0].material
        filtered=[self.filtered(material.texture,material.use_texture_alpha),self.filtered(material.normal_texture),self.filtered(material.bump_texture)]
        drawable=GPUReyesChunk(material,r._triangle_texture(triangles),len(triangles),subdiv,self.batch(subdiv),
            texture=filtered[0][0],displacement_texture=r._upload_texture(material.displacement_texture),
            normal_texture=filtered[1][0] if material.normal_texture else None,bump_texture=filtered[2][0] if material.bump_texture else None)
        # Preserve whether the material actually has a base texture.
        if material.texture is None:drawable.texture=None
        textures=[g.types.GPUTexture((width,height),format='RGBA32F') for _ in range(4)]
        framebuffer=g.types.GPUFrameBuffer(color_slots=textures)
        colors=g.types.GPUFrameBuffer(color_slots=(textures[0],))
        old_view=g.state.viewport_get();old_blend=g.state.blend_get();old_depth=g.state.depth_test_get();old_mask=g.state.depth_mask_get()
        vp=r.rscene.camera.projection @ r.rscene.camera.matrix_world.inverted();camera=r.rscene.camera.matrix_world.translation
        def bind(shader,base):
            shader.bind();shader.uniform_float('viewProjection',vp);shader.uniform_float('cameraPos',camera)
            shader.uniform_float('atlasLayout',(width,height,subdiv*subdiv))
            shader.uniform_bool('baseFilterPremult',material.use_texture_alpha)
            r._bind_geometry(shader,drawable);r._bind_material(shader,drawable,base_pass=base)
            for name,(_,size) in zip(('baseFilterSize','normalFilterSize','bumpFilterSize'),filtered):shader.uniform_float(name,size)
        try:
            g.state.depth_test_set('NONE');g.state.depth_mask_set(False);g.state.face_culling_set('NONE');g.state.blend_set('NONE')
            with framebuffer.bind():
                g.state.viewport_set(0,0,width,height);framebuffer.clear(color=(0,0,0,0));bind(self.base,True);r._draw_one(drawable,self.base)
            with colors.bind():
                g.state.viewport_set(0,0,width,height);g.state.blend_set('ADDITIVE_PREMULT')
                for light in r.rscene.lights:
                    checkpoint(r.cancel);bind(self.light,False);r._bind_light_parameters(self.light,light);r._draw_one(drawable,self.light)
            if resident:return textures,count,width
            results=[]
            with framebuffer.bind():
                for slot in range(4):
                    buffer=framebuffer.read_color(0,0,width,height,4,slot,'FLOAT');buffer.dimensions=(width*height*4,)
                    try:
                        values=np.frombuffer(buffer,dtype=np.float32,count=width*height*4).reshape(-1,4)
                    except (TypeError,BufferError):
                        values=np.asarray(buffer.to_list(),dtype=np.float32).reshape(-1,4)
                    results.append(values[:count].copy())
            return results[0],np.stack([v[:,:3] for v in results[1:]],axis=1)
        finally:
            g.shader.unbind();g.state.blend_set(old_blend);g.state.depth_test_set(old_depth);g.state.depth_mask_set(old_mask);g.state.viewport_set(*old_view)
