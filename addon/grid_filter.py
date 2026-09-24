# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit surface-footprint mip pyramid, independent of driver mip APIs."""
import numpy as np

def pyramid(pixels, width, height, premultiply=False):
    level=np.asarray(pixels,dtype=np.float32).reshape(height,width,4)
    level=level.copy()
    if premultiply:
        level[:,:,3]=np.clip(level[:,:,3],0,1)
        level[:,:,:3]*=level[:,:,3:4]
    levels=[level]
    while level.shape[0]>1 or level.shape[1]>1:
        h,w=level.shape[:2]
        padded=np.pad(level,((0,h%2),(0,w%2),(0,0)),mode='edge')
        level=(padded[::2,::2]+padded[1::2,::2]+padded[::2,1::2]+padded[1::2,1::2])*.25
        levels.append(level)
    result=np.zeros((sum(p.shape[0] for p in levels),width,4),dtype=np.float32)
    y=0
    for level in levels:
        result[y:y+len(level),:level.shape[1]]=level;y+=len(level)
    return result,(width,height,len(levels))

GLSL=r'''
vec2 rmFilterDu, rmFilterDv;
vec4 rmMip(sampler2D image, vec2 uv, vec3 dimensions, int level) {
    ivec2 size=ivec2(dimensions.xy); int offset=0;
    for(int i=0;i<level;i++){offset+=size.y;size=max(ivec2(1),(size+ivec2(1))/2);}
    vec2 p=fract(uv)*vec2(size)-vec2(0.5);ivec2 a=ivec2(floor(p));vec2 f=fract(p);
    ivec2 b=a+ivec2(1);a=((a%size)+size)%size;b=((b%size)+size)%size;
    return mix(mix(texelFetch(image,ivec2(a.x,a.y+offset),0),texelFetch(image,ivec2(b.x,a.y+offset),0),f.x),
               mix(texelFetch(image,ivec2(a.x,b.y+offset),0),texelFetch(image,ivec2(b.x,b.y+offset),0),f.x),f.y);
}
vec4 rmGridTexture(sampler2D image,vec2 uv,vec3 dimensions) {
    float footprint=max(length(rmFilterDu*dimensions.xy),length(rmFilterDv*dimensions.xy));
    float lod=clamp(log2(max(footprint,1.0)),0.0,dimensions.z-1.0);
    int lower=int(floor(lod)),upper=min(lower+1,int(dimensions.z)-1);
    return mix(rmMip(image,uv,dimensions,lower),rmMip(image,uv,dimensions,upper),fract(lod));
}
'''

# Packed levels are also built on the GPU. The Python pyramid above is retained
# as an independent reference for regression tests and the legacy CPU renderer.
class GPUPyramid:
    def __init__(self):
        from .shader_parameters import ParameterInfo
        info=ParameterInfo();info.push_constant('VEC4','sourceLevel');info.push_constant('VEC4','targetLevel')
        info.sampler(0,'FLOAT_2D','sourceImage');info.image(0,'RGBA32F','FLOAT_2D','levels',qualifiers={'READ','WRITE'})
        info.local_group_size(8,8)
        info.compute_source('''
void main(){
 ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(any(greaterThanEqual(p,ivec2(targetLevel.xy))))return;
 vec4 c;
 if(sourceLevel.w<0){c=texelFetch(sourceImage,p,0);if(targetLevel.w>0){c.a=clamp(c.a,0.0,1.0);c.rgb*=c.a;}}
 else{
  ivec2 a=p*2,b=min(a+1,ivec2(sourceLevel.xy)-1);a=min(a,ivec2(sourceLevel.xy)-1);int y=int(sourceLevel.z);
  c=(imageLoad(levels,ivec2(a.x,a.y+y))+imageLoad(levels,ivec2(b.x,a.y+y))+imageLoad(levels,ivec2(a.x,b.y+y))+imageLoad(levels,ivec2(b.x,b.y+y)))*.25;
 }
 imageStore(levels,p+ivec2(0,int(targetLevel.z)),c);
}''')
        self.shader=info.build()
    def build(self,image,source,premultiply=False):
        import gpu,math
        w,h=image.width,image.height;levels=[];offset=0
        while True:
            levels.append((w,h,offset));offset+=h
            if w==h==1:break
            w=max(1,(w+1)//2);h=max(1,(h+1)//2)
        result=gpu.types.GPUTexture((image.width,offset),format='RGBA32F');s=self.shader
        for i,(w,h,y) in enumerate(levels):
            s.bind();s.uniform_sampler('sourceImage',source);s.image('levels',result)
            s.uniform_float('sourceLevel',(*levels[i-1],0) if i else (0,0,0,-1))
            s.uniform_float('targetLevel',(w,h,y,int(premultiply)))
            gpu.compute.dispatch(s.prepare_draw(),math.ceil(w/8),math.ceil(h/8),1)
        return result,(image.width,image.height,len(levels))
