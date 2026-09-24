# SPDX-License-Identifier: GPL-3.0-or-later
import ctypes as C
import platform
from pathlib import Path
import numpy as np

_lib=None

def library():
    global _lib
    if _lib is None:
        system, machine = platform.system(), platform.machine().lower()
        if machine in {'amd64', 'x86_64'}: machine = 'x86_64'
        elif machine in {'arm64', 'aarch64'}: machine = 'arm64'
        filenames = {
            ('Linux', 'x86_64'): 'bucket_linux_x86_64.so',
            ('Windows', 'x86_64'): 'bucket_windows_x86_64.dll',
            ('Darwin', 'arm64'): 'bucket_macos_arm64.dylib',
        }
        filename = filenames.get((system, machine))
        if filename is None:
            raise RuntimeError(f'RetroMan native sampler does not support {system}/{machine}')
        path = Path(__file__).resolve().parent / 'native' / filename
        if not path.is_file():
            raise RuntimeError(f'Missing {filename}; install the RetroMan ZIP for {system}/{machine}, or run native/build.py on this system')
        try:
            lib = C.CDLL(str(path))
        except OSError as exc:
            raise RuntimeError(f'Cannot load RetroMan sampler {filename}: {exc}') from exc
        ptr=C.POINTER(C.c_float)
        lib.rm_bucket_create.argtypes=[C.c_int]*7+[C.c_float]*3+[C.c_int,C.c_int,C.c_float];lib.rm_bucket_create.restype=C.c_void_p
        lib.rm_bucket_push.argtypes=[C.c_void_p,ptr,C.c_int];lib.rm_bucket_push.restype=C.c_int
        lib.rm_bucket_finish.argtypes=[C.c_void_p,ptr,ptr];lib.rm_bucket_finish.restype=C.c_int
        lib.rm_bucket_destroy.argtypes=[C.c_void_p];lib.rm_bucket_destroy.restype=None
        _lib=lib
    return _lib

def pointer(array):return array.ctypes.data_as(C.POINTER(C.c_float))

class Bucket:
    def __init__(self,x,y,w,h,fw,fh,axis,lens=(0,0,1),filter_name='GAUSSIAN',aperture=(0,0.0)):
        if not(0<=x<fw and 0<=y<fh and 1<=w<=fw-x and 1<=h<=fh-y and 1<=axis<=32):
            raise ValueError('Invalid bucket bounds or sample count')
        self.lib=library();self.w=w;self.h=h
        self.handle=self.lib.rm_bucket_create(x,y,w,h,fw,fh,axis,*lens,{'BOX':0,'GAUSSIAN':1,'MITCHELL':2}.get(filter_name,1),*aperture)
        if not self.handle:raise MemoryError('Could not allocate bucket samples')
    def push(self,triangles):
        a=np.ascontiguousarray(triangles,dtype=np.float32).reshape(-1,28)
        if not np.isfinite(a).all():raise ValueError('Non-finite micropolygon data')
        code=self.lib.rm_bucket_push(self.handle,pointer(a),len(a))
        if code:raise RuntimeError(f'Bucket visibility memory limit/allocation failed ({code}); reduce bucket size or sample count')
    def finish(self,background):
        bg=np.ascontiguousarray(background,dtype=np.float32).reshape(self.h*self.w,4)
        out=np.empty_like(bg)
        if self.lib.rm_bucket_finish(self.handle,pointer(bg),pointer(out)):raise RuntimeError('Bucket resolve failed')
        return out.reshape(self.h,self.w,4)
    def close(self):
        if self.handle:self.lib.rm_bucket_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
