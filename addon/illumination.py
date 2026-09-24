# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded classic direct-emitter sampling and modern-world to classic ambient translation."""
import math
import numpy as np
from mathutils import Vector
from .model import LightData
from .runtime import checkpoint


def mesh_emitters(block, name, cancel=None):
    """Area-weighted patches per material and dominant normal direction.

    Uses owned mesh arrays rather than re-expanding viewport geometry to Python
    triangles. Each patch preserves area and radiance, not exact emitter shape.
    """
    lights=[]
    for mi in np.unique(block.material_indices):
        checkpoint(cancel)
        mat=block.materials[int(mi)]
        strength=max(0.0,mat.emission_strength)*max(0.0,min(1.0,mat.alpha*mat.base_color[3]))
        if strength<=0 or max(mat.emission_color)<=0:continue
        accum={}
        for start in range(0,len(block.loops),8192):
            checkpoint(cancel)
            loops=block.loops[start:start+8192]
            loops=loops[block.material_indices[start:start+8192]==mi]
            if not len(loops):continue
            points=block.positions[block.indices[loops]]
            cross=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
            twice_area=np.linalg.norm(cross,axis=1)
            valid=twice_area>1e-12
            cross=cross[valid];twice_area=twice_area[valid];points=points[valid]
            if not len(points):continue
            normals=cross/twice_area[:,None];area=twice_area*.5
            center=points.mean(axis=1)
            axis=np.argmax(np.abs(normals),axis=1)
            keys=axis*2+(normals[np.arange(len(normals)),axis]<0)
            for key in np.unique(keys):
                mask=keys==key;a=area[mask];total=float(a.sum())
                old=accum.setdefault(int(key),[0.0,np.zeros(3),np.zeros(3)])
                old[0]+=total;old[1]+=(center[mask]*a[:,None]).sum(axis=0);old[2]+=(normals[mask]*a[:,None]).sum(axis=0)
        for key,(area,position,normal) in sorted(accum.items()):
            n=Vector(normal.tolist()).normalized()
            matrix=n.to_track_quat('-Z','Y').to_matrix().to_4x4()
            matrix.translation=Vector((position/area).tolist())
            lights.append(LightData(name=f'{name}: {mat.name} [{key}]',type='MESH',
                matrix_world=matrix,color=mat.emission_color,energy=strength,
                use_shadow=True,area_factor=area,size=math.sqrt(area/math.pi)))
    return lights


def environment_diffuse(texture, strength, color, cancel=None):
    """Translate a modern world to a classic constant ambient light.

    Equal-solid-angle averaging is scene translation, not a directional
    irradiance shader. The original image remains available for reflection maps.
    """
    checkpoint(cancel)
    if texture is None:
        return tuple(max(0.0, float(c)) for c in color)
    pixels=np.asarray(texture.pixels,dtype=np.float32).reshape(texture.height,texture.width,4)
    average=np.zeros(3,dtype=np.float64)
    for j in range(32):
        checkpoint(cancel)
        z=-1+(j+.5)*2/32
        v=math.asin(z)/math.pi+.5
        iy=min(texture.height-1,int(v*texture.height))
        for i in range(64):
            u=(i+.5)/64
            average+=np.maximum(pixels[iy,min(texture.width-1,int(u*texture.width)),:3],0)
    return tuple(float(v)*max(0,strength)/(64*32) for v in average)


def evaluate_environment(color, normal):
    """Classic ambient light has no directional response."""
    return tuple(color)
