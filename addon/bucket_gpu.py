# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU REYES sample lists. No grid readback, fixed transparency-layer cap or rays.

Shading atlases stay resident. Compute scatters covered micropolygons into sample
lists, then each sample independently sorts its list and resolves premultiplied
coverage. Pool exhaustion is detected and reported, never silently truncated.
"""
import math
import numpy as np
from .shader_parameters import ParameterInfo
from .runtime import checkpoint

COMMON = r'''
ivec2 address(uint i){return ivec2(int(i%1024u),int(i/1024u));}
uint hash32(uint x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;}
float random01(uint x){return float(hash32(x)>>8)*(1.0/16777216.0);}
vec4 samplePoint(ivec2 p,int i){
 uint key=hash32(uint(p.x)+hash32(uint(p.y))),seed=hash32(key+uint(i)*0x9e3779b9u);
 int axis=int(frame.z),n=axis*axis;
 vec2 j=(vec2(i%axis,i/axis)+vec2(random01(seed+1u),random01(seed+2u)))/float(axis);
 if(n==1)j=vec2(.5);
 float t=(float((uint(i)+key%uint(n))%uint(n))+random01(seed+3u))/float(n);
 return vec4(vec2(p)+j,t,uintBitsToFloat(seed));
}
vec2 lensPoint(uint seed){
 float r=sqrt(random01(seed+4u)),theta=6.28318530718*random01(seed+5u);
 if(lens.w>=3.0){float sector=6.28318530718/lens.w,local=mod(theta+sector*.5,sector)-sector*.5;r*=cos(sector*.5)/cos(local);}
 theta+=frame.w;return r*vec2(cos(theta),sin(theta));
}
float edge(vec3 a,vec3 b,vec2 p){return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x);}
bool inside(float e,vec3 a,vec3 b,float signv){e*=signv;if(e>0)return true;if(e<0)return false;vec2 d=(b.xy-a.xy)*signv;return d.y>0||(d.y==0&&d.x<0);}
float plane(vec4 p,int k){return p.w+(k%2==1?-p[k/2]:p[k/2]);}
'''
SCATTER = COMMON+r'''
void main(){
 uint ti=gl_GlobalInvocationID.x;if(ti>=uint(gridLayout.x))return;
 ivec2 uv=ivec2(int(ti)%int(gridLayout.y),int(ti)/int(gridLayout.y));
 vec4 color=texelFetch(colors,uv,0);color.a=clamp(color.a,0.0,1.0);if(color.a<=0)return;
 vec4 a[3],b[3];
 a[0]=vp*vec4(texelFetch(p0,uv,0).xyz,1);a[1]=vp*vec4(texelFetch(p1,uv,0).xyz,1);a[2]=vp*vec4(texelFetch(p2,uv,0).xyz,1);
 b[0]=vpEnd*vec4(texelFetch(q0,uv,0).xyz,1);b[1]=vpEnd*vec4(texelFetch(q1,uv,0).xyz,1);b[2]=vpEnd*vec4(texelFetch(q2,uv,0).xyz,1);
 vec2 lo=vec2(3.402823e38),hi=vec2(-3.402823e38);bool eye=false;
 for(int k=0;k<6;k++){vec4 p=k<3?a[k]:b[k-3];if(p.w<=1e-7){eye=true;break;}
 vec2 pos=(p.xy/p.w*.5+.5)*frame.xy,blur=abs(lens.xy*(1.0/lens.z-1.0/p.w));lo=min(lo,pos-blur);hi=max(hi,pos+blur);}
 ivec2 lower=ivec2(bucket.xy),upper=lower+ivec2(bucket.zw)-1;
 if(!eye){lower=max(lower,ivec2(floor(lo)));upper=min(upper,ivec2(floor(hi)));}
 int n=int(frame.z)*int(frame.z);
 for(int y=lower.y;y<=upper.y;y++)for(int x=lower.x;x<=upper.x;x++)for(int si=0;si<n;si++){
  vec4 sp=samplePoint(ivec2(x,y),si);vec2 lp=lensPoint(floatBitsToUint(sp.w));
  vec4 poly[12],tmp[12];int count=3;
  for(int j=0;j<3;j++){poly[j]=a[j]*(1.0-sp.z)+b[j]*sp.z;poly[j].xy+=lp*lens.xy*(poly[j].w/lens.z-1.0)*2.0/frame.xy;}
  for(int k=0;k<6&&count>0;k++){
   int nextCount=0;vec4 prev=poly[count-1];float dp=plane(prev,k);
   for(int j=0;j<count;j++){vec4 cur=poly[j];float dc=plane(cur,k);if((dp>=0)!=(dc>=0)){float t=dp/(dp-dc);tmp[nextCount++]=prev+t*(cur-prev);}if(dc>=0)tmp[nextCount++]=cur;prev=cur;dp=dc;}
   count=nextCount;for(int j=0;j<count;j++)poly[j]=tmp[j];
  }
  if(count<3)continue;
  vec3 projected[12];for(int j=0;j<count;j++)projected[j]=vec3((poly[j].xy/poly[j].w*.5+.5)*frame.xy,poly[j].z/poly[j].w*.5+.5);
  for(int j=1;j+1<count;j++){
   vec3 v0=projected[0],v1=projected[j],v2=projected[j+1];float ar=edge(v0,v1,v2.xy);if(abs(ar)<1e-12)continue;float sg=ar>0?1.0:-1.0;
   float e0=edge(v1,v2,sp.xy),e1=edge(v2,v0,sp.xy),e2=edge(v0,v1,sp.xy);
   if(!inside(e0,v1,v2,sg)||!inside(e1,v2,v0,sg)||!inside(e2,v0,v1,sg))continue;
   float z=(e0*v0.z+e1*v1.z+e2*v2.z)/ar;
   uint slot=imageAtomicAdd(counter,ivec2(0),1u);
   if(slot>=uint(gridLayout.z))return; // Stop this primitive; replay grows the pool.
   {
    uint sampleIndex=uint(((y-int(bucket.y))*int(bucket.z)+x-int(bucket.x))*n+si);
    uint previous=imageAtomicExchange(heads,address(sampleIndex),slot+1u);
    imageStore(fragmentColor,address(slot),color);imageStore(fragmentData,address(slot),vec4(z,uintBitsToFloat(previous),0,0));
   }
   break;
  }
 }
}
'''
RESOLVE = COMMON+r'''
bool precedes(uint a,uint b){
 vec4 da=imageLoad(fragmentData,address(a-1u)),db=imageLoad(fragmentData,address(b-1u));
 if(da.x!=db.x)return da.x<db.x;
 vec4 ca=imageLoad(fragmentColor,address(a-1u)),cb=imageLoad(fragmentColor,address(b-1u));
 for(int j=0;j<4;j++)if(ca[j]!=cb[j])return ca[j]<cb[j];return false;
}
uint nextOf(uint a){return floatBitsToUint(imageLoad(fragmentData,address(a-1u)).y);}
void linkTo(uint a,uint b){vec4 d=imageLoad(fragmentData,address(a-1u));d.y=uintBitsToFloat(b);imageStore(fragmentData,address(a-1u),d);}
vec4 backdrop(ivec2 p){
 if(skyOptions.x==0)return texelFetch(background,p,0);
 vec2 ndc=(vec2(p+ivec2(bucket.xy))+.5)/frame.xy*2.0-1.0;
 vec4 farp=inverseVP*vec4(ndc,1,1),nearp=inverseVP*vec4(ndc,-1,1);
 vec3 ray=normalize(farp.xyz/farp.w-nearp.xyz/nearp.w);
 vec2 uv=vec2(atan(ray.y,ray.x)/6.28318530718+.5,asin(clamp(ray.z,-1.0,1.0))/3.14159265359+.5);
 ivec2 size=ivec2(skyOptions.zw);vec2 t=uv*vec2(size)-.5;ivec2 base=ivec2(floor(t));vec2 f=fract(t);
 int xa=(base.x%size.x+size.x)%size.x,xb=(xa+1)%size.x,ya=clamp(base.y,0,size.y-1),yb=clamp(base.y+1,0,size.y-1);
 vec3 c=mix(mix(texelFetch(sky,ivec2(xa,ya),0).rgb,texelFetch(sky,ivec2(xb,ya),0).rgb,f.x),mix(texelFetch(sky,ivec2(xa,yb),0).rgb,texelFetch(sky,ivec2(xb,yb),0).rgb,f.x),f.y);
 return vec4(c*skyOptions.y,1);
}
float mitchell(float value){float z=abs(value)*2.0;return z<1?((7*z-12)*z*z+16.0/3)/6:(((-7.0/3*z+12)*z-20)*z+32.0/3)/6;}
void main(){
 ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(any(greaterThanEqual(p,ivec2(bucket.zw))))return;
 int n=int(frame.z)*int(frame.z);vec3 rgb=vec3(0);float alpha=0,weights=0;
 for(int si=0;si<n;si++){
  uint index=uint((p.y*int(bucket.z)+p.x)*n+si);uint node=imageLoad(heads,address(index)).x,sorted=0u;
  // Bottom-up merge sort has no per-sample layer array or quadratic insertion
  // cost on deep transparency. Each invocation exclusively owns these links.
  uint listHead=node;
  for(uint run=1u;listHead!=0u;run*=2u){
   uint pnode=listHead,tail=0u,merges=0u;sorted=0u;
   while(pnode!=0u){
    merges++;uint qnode=pnode,psize=0u;
    for(uint j=0u;j<run&&qnode!=0u;j++){psize++;qnode=nextOf(qnode);}
    uint qsize=run;
    while(psize>0u||(qsize>0u&&qnode!=0u)){
     uint chosen;
     if(psize==0u){chosen=qnode;qnode=nextOf(qnode);qsize--;}
     else if(qsize==0u||qnode==0u||!precedes(qnode,pnode)){chosen=pnode;pnode=nextOf(pnode);psize--;}
     else{chosen=qnode;qnode=nextOf(qnode);qsize--;}
     if(tail!=0u)linkTo(tail,chosen);else sorted=chosen;tail=chosen;
    }
    pnode=qnode;
   }
   linkTo(tail,0u);listHead=sorted;if(merges<=1u)break;
  }
  vec3 c=vec3(0);float tr=1,coverage=0;node=sorted;
  while(node!=0u){vec4 v=imageLoad(fragmentColor,address(node-1u));coverage+=tr*v.a;c+=tr*v.a*v.rgb;tr*=1.0-v.a;if(tr==0)break;node=nextOf(node);}
  vec4 bg=backdrop(p);c+=tr*bg.a*bg.rgb;float aa=coverage+tr*bg.a;
  vec2 d=samplePoint(p+ivec2(bucket.xy),si).xy-vec2(p+ivec2(bucket.xy))-.5;
  float weight=gridLayout.w==0?1.0:(gridLayout.w==2?mitchell(d.x)*mitchell(d.y):exp(-2.0*dot(d,d)));
  rgb+=weight*c;alpha+=weight*aa;weights+=weight;
 }
 imageStore(outputImage,p,vec4(alpha>0?rgb/alpha:vec3(0),alpha/weights));
}
'''

def _shader(source,resolve=False):
    info=ParameterInfo()
    for name in ('bucket','frame','lens','gridLayout'):info.push_constant('VEC4',name)
    if not resolve:
        for name in ('vp','vpEnd'):info.push_constant('MAT4',name)
        for i,name in enumerate(('colors','p0','p1','p2','q0','q1','q2')):info.sampler(i,'FLOAT_2D',name)
    else:
        info.sampler(0,'FLOAT_2D','background');info.sampler(1,'FLOAT_2D','sky')
        info.push_constant('VEC4','skyOptions');info.push_constant('MAT4','inverseVP')
    for slot,name,fmt,kind in ((0,'heads','R32UI','UINT_2D'),(1,'counter','R32UI','UINT_2D'),(2,'fragmentColor','RGBA32F','FLOAT_2D'),(3,'fragmentData','RGBA32F','FLOAT_2D')):
        info.image(slot,fmt,kind,name,qualifiers={'READ','WRITE'})
    if resolve:info.image(4,'RGBA32F','FLOAT_2D','outputImage',qualifiers={'WRITE'})
    info.local_group_size(8,8) if resolve else info.local_group_size(64)
    info.compute_source(source)
    return info.build()

class BucketBudgetExceeded(RuntimeError):
    """Retry this region as smaller buckets without changing sampling."""

class GPUWorkspace:
    """Reusable shaders and pool; one live bucket at a time."""
    def __init__(self,cancel=None):
        import gpu
        self.gpu=gpu;self.cancel=cancel
        self.scatter=_shader(SCATTER);self.resolve=_shader(RESOLVE,True)
        self.capacity=0;self.allocate(262144);self.sky=None
    def allocate(self,capacity):
        g=self.gpu;self.capacity=capacity
        self.colors=g.types.GPUTexture((1024,math.ceil(capacity/1024)),format='RGBA32F')
        self.data=g.types.GPUTexture((1024,math.ceil(capacity/1024)),format='RGBA32F')

class Bucket:
    def __init__(self,x,y,w,h,fw,fh,axis,lens=(0,0,1),pixel_filter='GAUSSIAN',aperture=(0,0),*,workspace,vp,vp_end):
        self.ws=workspace;self.g=workspace.gpu;self.w=w;self.h=h;self.vp=vp;self.vp_end=vp_end
        self.params={'bucket':(x,y,w,h),'frame':(fw,fh,axis,aperture[1]),'lens':(*lens,aperture[0])}
        self.filter={'BOX':0,'GAUSSIAN':1,'MITCHELL':2}.get(pixel_filter,1)
        self.heads=self.g.types.GPUTexture((1024,math.ceil(w*h*axis*axis/1024)),format='R32UI')
        self.counter=self.g.types.GPUTexture((1,1),format='R32UI')
        self.pending=[];self.pending_bytes=0;self.clear()
    def clear(self):
        self.heads.clear(format='UINT',value=(0,));self.counter.clear(format='UINT',value=(0,))
    def bind(self,shader):
        shader.bind()
        for name,value in self.params.items():shader.uniform_float(name,value)
        for name,tex in (('heads',self.heads),('fragmentColor',self.ws.colors),('fragmentData',self.ws.data)):shader.image(name,tex)
        shader.image('counter',self.counter)
    def push(self,start,end):
        self.pending_bytes+=start[2]*math.ceil(start[1]/start[2])*64*(1 if start is end else 2)
        if self.pending_bytes>256*1024*1024:raise BucketBudgetExceeded('GPU bucket shading exceeds 256 MiB; reduce bucket size')
        self.pending.append((start,end));self._dispatch(start,end)
    def _dispatch(self,start,end):
        checkpoint(self.ws.cancel);textures,count,width=start;other=end[0];s=self.ws.scatter;self.bind(s)
        s.uniform_float('vp',self.vp);s.uniform_float('vpEnd',self.vp_end);s.uniform_float('gridLayout',(count,width,self.ws.capacity,self.filter))
        for name,tex in zip(('colors','p0','p1','p2','q0','q1','q2'),[textures[0],*textures[1:],*other[1:]]):s.uniform_sampler(name,tex)
        self.g.compute.dispatch(s.prepare_draw(),math.ceil(count/64),1,1)
    def finish(self,background):
        # One four-byte overflow check per bucket. Replay at larger capacity;
        # never resolve incomplete visibility or cap transparent layer count.
        while True:
            checkpoint(self.ws.cancel);buf=self.counter.read();buf.dimensions=(1,);needed=int(buf[0])
            if needed<=self.ws.capacity:break
            if needed>8*1024*1024:raise BucketBudgetExceeded('GPU bucket exceeds 8 million fragments; reduce bucket size or samples')
            self.ws.allocate(min(8*1024*1024,2**math.ceil(math.log2(needed))));self.clear()
            for a,b in self.pending:self._dispatch(a,b)
        bg=np.ascontiguousarray(background,dtype=np.float32)
        tex=self.g.types.GPUTexture((self.w,self.h),format='RGBA32F',data=self.g.types.Buffer('FLOAT',bg.size,bg.reshape(-1)))
        output=self.g.types.GPUTexture((self.w,self.h),format='RGBA32F');s=self.ws.resolve;self.bind(s)
        s.uniform_float('gridLayout',(0,0,self.ws.capacity,self.filter));s.uniform_sampler('background',tex);s.image('outputImage',output)
        if self.ws.sky:
            sky,options=self.ws.sky;s.uniform_sampler('sky',sky);s.uniform_float('skyOptions',options)
        else:s.uniform_sampler('sky',tex);s.uniform_float('skyOptions',(0,0,1,1))
        s.uniform_float('inverseVP',self.vp.inverted())
        self.g.compute.dispatch(s.prepare_draw(),math.ceil(self.w/8),math.ceil(self.h/8),1)
        buf=output.read();buf.dimensions=(self.w*self.h*4,)
        return np.frombuffer(buf,dtype=np.float32).reshape(self.h,self.w,4).copy()
    def __enter__(self):return self
    def __exit__(self,*args):self.pending.clear();self.g.shader.unbind()
