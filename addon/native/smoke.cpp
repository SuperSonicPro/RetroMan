// SPDX-License-Identifier: GPL-3.0-or-later
// Load the actual packaged library and verify coverage/transparency through its C ABI.
#include <cmath>
#include <cstdio>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
int main(int argc, char** argv) {
 if(argc != 2) return 2;
#ifdef _WIN32
 auto lib=LoadLibraryA(argv[1]);
 auto symbol=[&](const char* name){return (void*)GetProcAddress(lib,name);};
#else
 auto lib=dlopen(argv[1],RTLD_NOW);
 auto symbol=[&](const char* name){return dlsym(lib,name);};
#endif
 if(!lib){std::puts("Library load failed");return 3;}
 auto create=(void*(*)(int,int,int,int,int,int,int,float,float,float,int,int,float))symbol("rm_bucket_create");
 auto push=(int(*)(void*,const float*,int))symbol("rm_bucket_push");
 auto finish=(int(*)(void*,const float*,float*))symbol("rm_bucket_finish");
 auto destroy=(void(*)(void*))symbol("rm_bucket_destroy");
 if(!create||!push||!finish||!destroy)return 4;
 float records[112]={};
 float vertices[24]={-1,-1,0,1,1,-1,0,1,1,1,0,1,-1,-1,0,1,1,1,0,1,-1,1,0,1};
 for(int i=0;i<4;i++){
  float* q=records+i*28;
  std::copy(vertices+(i%2)*12,vertices+(i%2)*12+12,q);std::copy(q,q+12,q+12);
  for(int k=0;k<6;k++)q[k*4+2]=i<2?-.5f:.5f;
  q[24]=i<2?1:0;q[26]=i<2?0:1;q[27]=.5f;
 }
 for(int axis: {1,2,4})for(int reverse: {0,1}) {
  void* b=create(0,0,4,4,4,4,axis,0,0,1,0,0,0);
  if(!b)return 5;
  if(push(b,records+(reverse?56:0),2)<0||push(b,records+(reverse?0:56),2)<0)return 6;
  float bg[64]={},out[64]={};if(finish(b,bg,out)<0)return 7;destroy(b);
  for(int p=0;p<16;p++) {
   float expected[4]={2.0f/3.0f,0,1.0f/3.0f,.75f};
   for(int c=0;c<4;c++)if(std::abs(out[p*4+c]-expected[c])>1e-5f){std::printf("Mismatch sample %d pixel %d channel %d: %g\n",axis,p,c,out[p*4+c]);return 8;}
  }
 }
 std::puts("PASS: actual native library loaded, four C exports, 6 coverage/transparency cases");return 0;
}
