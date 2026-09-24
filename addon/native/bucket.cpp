// SPDX-License-Identifier: GPL-3.0-or-later
// Bounded bucket visibility for already shaded moving micropolygons.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#include <new>
#include <memory>
#if defined(_WIN32)
#define RM_API extern "C" __declspec(dllexport)
#else
#define RM_API extern "C" __attribute__((visibility("default")))
#endif

struct Fragment { float z,r,g,b,a; };
struct Sample { float x,y,t,lx,ly,weight; std::vector<Fragment> fragments; };
struct Bucket {
 int x,y,w,h,fw,fh,n,filter; size_t hits=0; bool failed=false;
 float lensx,lensy,focus;
 std::vector<Sample> samples;
};
static uint32_t hash32(uint32_t x) { x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return x; }
static float random01(uint32_t x) {return (hash32(x)>>8)*(1.0f/16777216.0f);}
RM_API void* rm_bucket_create(int x,int y,int w,int h,int fw,int fh,int axis,float lensx,float lensy,float focus,int filter,int blades,float rotation) {
 try {
  std::unique_ptr<Bucket> b(new Bucket{x,y,w,h,fw,fh,axis*axis,filter,0,false,lensx,lensy,focus,{}});
  b->samples.resize(size_t(w)*h*b->n);
  for(int py=0;py<h;py++) for(int px=0;px<w;px++) for(int i=0;i<b->n;i++) {
   uint32_t key=hash32(uint32_t(x+px)+hash32(uint32_t(y+py))); uint32_t seed=hash32(key+uint32_t(i)*0x9e3779b9U);
   float jx=(i%axis+random01(seed+1))/axis, jy=(i/axis+random01(seed+2))/axis;
   if(b->n==1){jx=.5f;jy=.5f;}
   float t=((i+key%uint32_t(b->n))%b->n+random01(seed+3))/b->n;
   float r=sqrtf(random01(seed+4)), theta=6.28318530718f*random01(seed+5);
   if(blades>=3){float sector=6.28318530718f/blades;float local=fmodf(theta+sector*.5f,sector)-sector*.5f;r*=cosf(sector*.5f)/cosf(local);}theta+=rotation;
   float dx=jx-.5f,dy=jy-.5f;
   auto mitchell=[](float value){float z=fabsf(value)*2;return z<1 ? ((7*z-12)*z*z+16.0f/3)/6 : (((-7.0f/3*z+12)*z-20)*z+32.0f/3)/6;};
   float weight=filter==0?1.0f:(filter==2?mitchell(dx)*mitchell(dy):expf(-2*(dx*dx+dy*dy)));
   b->samples[(py*w+px)*b->n+i]={x+px+jx,y+py+jy,t,r*cosf(theta),r*sinf(theta),weight,{}};
  } return b.release();
 } catch(...) {return nullptr;}
}
using P=std::array<float,4>;
static float plane(const P&p,int k){return p[3]+(k%2? -p[k/2]:p[k/2]);}
static std::vector<P> clip(std::vector<P> points){
 for(int k=0;k<6 && !points.empty();k++) {
  std::vector<P> out; P a=points.back(); float da=plane(a,k);
  for(P c:points) {float dc=plane(c,k); if((da>=0)!=(dc>=0)){float t=da/(da-dc);P q;for(int j=0;j<4;j++)q[j]=a[j]+t*(c[j]-a[j]);out.push_back(q);} if(dc>=0)out.push_back(c);a=c;da=dc;} points=std::move(out);
 } return points;
}
static float edge(const P&a,const P&b,float x,float y){return (b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]);}
static bool inside(float e,const P&a,const P&b,float sign){
 e*=sign;if(e>0)return true;if(e<0)return false;
 float dy=(b[1]-a[1])*sign,dx=(b[0]-a[0])*sign;return dy>0 || (dy==0 && dx<0);
}
// 28 floats/microtriangle: start clip vertices, end clip vertices, straight RGBA.
RM_API int rm_bucket_push(void* handle,const float* triangles,int count) {
 auto& b=*static_cast<Bucket*>(handle); if(b.failed)return -1;
 try {
  for(int ti=0;ti<count;ti++) {
   const float* q=triangles+size_t(ti)*28; float alpha=std::clamp(q[27],0.0f,1.0f);if(alpha<=0)continue;
   float minx=b.fw,maxx=0,miny=b.fh,maxy=0;bool eye=false;
   for(int k=0;k<6;k++){const float* p=q+k*4;if(p[3]<=1e-7f){eye=true;break;}float x=(p[0]/p[3]*.5f+.5f)*b.fw,y=(p[1]/p[3]*.5f+.5f)*b.fh;
    float blurx=fabsf(b.lensx*(1/b.focus-1/p[3])),blury=fabsf(b.lensy*(1/b.focus-1/p[3]));
    minx=std::min(minx,x-blurx);maxx=std::max(maxx,x+blurx);miny=std::min(miny,y-blury);maxy=std::max(maxy,y+blury);}
   int x0=eye?b.x:std::max(b.x,int(floorf(minx))), x1=eye?b.x+b.w-1:std::min(b.x+b.w-1,int(floorf(maxx)));
   int y0=eye?b.y:std::max(b.y,int(floorf(miny))), y1=eye?b.y+b.h-1:std::min(b.y+b.h-1,int(floorf(maxy)));
   bool depthInside=true;for(int v=0;v<6;v++)if(q[v*4+3]<=0 || q[v*4+2]<-q[v*4+3] || q[v*4+2]>q[v*4+3])depthInside=false;
   P fixed[3];bool simple=(b.lensx==0 && b.lensy==0 && std::equal(q,q+12,q+12));
   if(simple){for(int v=0;v<3;v++){for(int c=0;c<4;c++)fixed[v][c]=q[v*4+c];for(int k=0;k<6;k++)if(plane(fixed[v],k)<0)simple=false;}}
   if(simple)for(auto& p:fixed){float iw=1/p[3];p[0]=(p[0]*iw*.5f+.5f)*b.fw;p[1]=(p[1]*iw*.5f+.5f)*b.fh;p[2]=p[2]*iw*.5f+.5f;}
   for(int y=y0;y<=y1;y++)for(int x=x0;x<=x1;x++)for(int si=0;si<b.n;si++) {
    Sample& s=b.samples[((y-b.y)*b.w+x-b.x)*b.n+si];
    std::vector<P> dynamic;P moving[3];const P* poly=fixed;size_t vertices=3;
    if(!simple){
     for(int v=0;v<3;v++){for(int c=0;c<4;c++)moving[v][c]=q[v*4+c]*(1-s.t)+q[12+v*4+c]*s.t;
      moving[v][0]+=s.lx*b.lensx*(moving[v][3]/b.focus-1)*2/b.fw;
      moving[v][1]+=s.ly*b.lensy*(moving[v][3]/b.focus-1)*2/b.fh;}
     P* projected=moving;
     if(!depthInside){dynamic.assign(moving,moving+3);dynamic=clip(dynamic);if(dynamic.size()<3)continue;projected=dynamic.data();vertices=dynamic.size();}
     for(size_t v=0;v<vertices;v++){auto&p=projected[v];float iw=1/p[3];p[0]=(p[0]*iw*.5f+.5f)*b.fw;p[1]=(p[1]*iw*.5f+.5f)*b.fh;p[2]=p[2]*iw*.5f+.5f;}
     poly=projected;
    }
    for(size_t j=1;j+1<vertices;j++) {
     const P&a=poly[0],&c=poly[j],&d=poly[j+1];float area=edge(a,c,d[0],d[1]);if(fabsf(area)<1e-12f)continue;float sign=area>0?1:-1;
     float e0=edge(c,d,s.x,s.y),e1=edge(d,a,s.x,s.y),e2=edge(a,c,s.x,s.y);
     if(!inside(e0,c,d,sign)||!inside(e1,d,a,sign)||!inside(e2,a,c,sign))continue;
     float z=(e0*a[2]+e1*c[2]+e2*d[2])/area;
     if(++b.hits>8*1024*1024){b.failed=true;return -2;}
     s.fragments.push_back({z,q[24],q[25],q[26],alpha});break;
    }
   }
  }return 0;
 } catch(...) {b.failed=true;return -1;}
}
RM_API int rm_bucket_finish(void* handle,const float* background,float* pixels) {
 auto& b=*static_cast<Bucket*>(handle);if(b.failed)return -1;
 for(int p=0;p<b.w*b.h;p++) {
  double rgb[3]={0,0,0},alpha=0,weights=0;
  for(int i=0;i<b.n;i++) {
   auto&s=b.samples[p*b.n+i];auto&f=s.fragments;
   std::sort(f.begin(),f.end(),[](const Fragment&a,const Fragment&c){
    if(a.z!=c.z)return a.z<c.z; if(a.r!=c.r)return a.r<c.r;if(a.g!=c.g)return a.g<c.g;if(a.b!=c.b)return a.b<c.b;return a.a<c.a;});
   double tr=1,rr=0,gg=0,bb=0,coverage=0;
   for(auto& v:f){coverage+=tr*v.a;rr+=tr*v.a*v.r;gg+=tr*v.a*v.g;bb+=tr*v.a*v.b;tr*=1.0-double(v.a);if(tr==0)break;}
   const float*bg=background+p*4;rr+=tr*bg[3]*bg[0];gg+=tr*bg[3]*bg[1];bb+=tr*bg[3]*bg[2];double aa=coverage+tr*bg[3];
   rgb[0]+=s.weight*rr;rgb[1]+=s.weight*gg;rgb[2]+=s.weight*bb;alpha+=s.weight*aa;weights+=s.weight;
  }
  for(int c=0;c<3;c++)pixels[p*4+c]=alpha>0?float(rgb[c]/alpha):0;pixels[p*4+3]=float(alpha/weights);
 }return 0;
}
RM_API void rm_bucket_destroy(void* handle){delete static_cast<Bucket*>(handle);}
