# SPDX-License-Identifier: GPL-3.0-or-later
"""Conservative GPU bounds for source patches, before CPU bucket scheduling."""
import math
import numpy as np
from .shader_parameters import ParameterInfo
from .runtime import checkpoint

SOURCE=r'''
void main(){
 uint index=gl_GlobalInvocationID.x;if(index>=uint(frame.z))return;
 vec2 lo=vec2(3.402823e38),hi=vec2(-3.402823e38);float longest=0;bool eye=false,nearOutside=true,farOutside=true;
 for(int end=0;end<2;end++){
  mat4 m=end==0?vp:vpEnd;vec2 pxy[3];
  for(int j=0;j<3;j++){
   int offset=int(index)*6+end*3+j;vec4 source=texelFetch(points,ivec2(offset%1024,offset/1024),0);
   float d=source.w;vec4 p=m*vec4(source.xyz,1);
   float nm=d*(abs(m[0][2]+m[0][3])+abs(m[1][2]+m[1][3])+abs(m[2][2]+m[2][3]));
   float fm=d*(abs(m[0][3]-m[0][2])+abs(m[1][3]-m[1][2])+abs(m[2][3]-m[2][2]));
   nearOutside=nearOutside&&(p.z+p.w+nm<0);farOutside=farOutside&&(p.w-p.z+fm<0);
   if(p.w<=1e-6)eye=true;
   pxy[j]=(p.xy/p.w*.5+.5)*frame.xy;
   for(int k=0;k<(d>0?8:1);k++){
    vec3 corner=vec3(k%2==0?-1:1,(k/2)%2==0?-1:1,k/4==0?-1:1)*d;
    vec4 q=d>0?m*vec4(source.xyz+corner,1):p;
    if(q.w<=1e-6){eye=true;continue;}
    vec2 pixel=(q.xy/q.w*.5+.5)*frame.xy,blur=abs(lens.xy*(1.0/lens.z-1.0/q.w));
    lo=min(lo,pixel-blur);hi=max(hi,pixel+blur);
   }
  }
  longest=max(longest,max(length(pxy[0]-pxy[1]),max(length(pxy[1]-pxy[2]),length(pxy[2]-pxy[0]))));
 }
 vec4 bounds=vec4(max(vec2(0),floor(lo-.0005)),min(frame.xy,ceil(hi+.0005)));
 if(eye){bounds=vec4(0,0,frame.xy);longest=3.402823e38;}
 if(nearOutside||farOutside)bounds=vec4(0);
 ivec2 address=ivec2(int(index)%1024,int(index)/1024);imageStore(boxes,address,bounds);imageStore(lengths,address,vec4(longest));
}
'''

def source_bounds(triangles,end_triangles,vp,vp_end,width,height,displacements,lens,cancel,status=None):
    import gpu
    info=ParameterInfo()
    for name in ('vp','vpEnd'):info.push_constant('MAT4',name)
    for name in ('frame','lens'):info.push_constant('VEC3',name)
    info.sampler(0,'FLOAT_2D','points')
    info.image(0,'RGBA32F','FLOAT_2D','boxes',qualifiers={'WRITE'});info.image(1,'R32F','FLOAT_2D','lengths',qualifiers={'WRITE'})
    info.local_group_size(64);info.compute_source(SOURCE);s=info.build()
    result=np.empty((len(triangles),5),np.float32)
    try:
        for first in range(0,len(triangles),16384):
            checkpoint(cancel);n=min(16384,len(triangles)-first)
            if status:status(f'Bounding source patches {first+n:,}/{len(triangles):,}')
            data=np.zeros((math.ceil(n*6/1024)*1024,4),np.float32)
            if hasattr(triangles,'bound_points') and hasattr(end_triangles,'bound_points'):
                pairs=data[:n*6].reshape(n,2,3,4)
                pairs[:,0]=triangles.bound_points(first,first+n,displacements)
                pairs[:,1]=pairs[:,0] if end_triangles is triangles else end_triangles.bound_points(first,first+n,displacements)
            else:
                a=triangles[first:first+n];b=end_triangles[first:first+n]
                data[:n*6]=[(v.p.x,v.p.y,v.p.z,displacements[id(t.material)]) for pair in zip(a,b) for t in pair for v in (t.a,t.b,t.c)]
            source=gpu.types.GPUTexture((1024,len(data)//1024),format='RGBA32F',data=gpu.types.Buffer('FLOAT',data.size,data.reshape(-1)))
            dims=(1024,math.ceil(n/1024));boxes=gpu.types.GPUTexture(dims,format='RGBA32F');lengths=gpu.types.GPUTexture(dims,format='R32F')
            s.bind();s.uniform_float('vp',vp);s.uniform_float('vpEnd',vp_end);s.uniform_float('frame',(width,height,n));s.uniform_float('lens',lens)
            s.uniform_sampler('points',source);s.image('boxes',boxes);s.image('lengths',lengths)
            gpu.compute.dispatch(s.prepare_draw(),math.ceil(n/64),1,1)
            buf=boxes.read();buf.dimensions=(dims[0]*dims[1]*4,);result[first:first+n,:4]=np.frombuffer(buf,np.float32).reshape(-1,4)[:n]
            buf=lengths.read();buf.dimensions=(dims[0]*dims[1],);result[first:first+n,4]=np.frombuffer(buf,np.float32)[:n]
        return result
    finally:gpu.shader.unbind()
