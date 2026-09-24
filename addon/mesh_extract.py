# SPDX-License-Identifier: GPL-3.0-or-later
"""Bulk evaluated mesh transfer; owned vectors survive to_mesh_clear()."""
import numpy as np
from mathutils import Vector
from .model import VertexData, TriangleData
from .runtime import checkpoint

def mesh_arrays(mesh, world, normal_matrix, materials, uv_map, cancel):
    def read(collection, field, components, dtype=np.float32):
        data=np.empty(len(collection)*components,dtype=dtype)
        collection.foreach_get(field,data)
        return data.reshape(-1,components) if components>1 else data
    checkpoint(cancel)
    positions=read(mesh.vertices,'co',3)
    matrix=np.asarray(world,dtype=np.float32)
    positions=positions @ matrix[:3,:3].T + matrix[:3,3]
    indices=read(mesh.loops,'vertex_index',1,np.int32)
    normals=read(mesh.corner_normals,'vector',3)
    normals=normals @ np.asarray(normal_matrix,dtype=np.float32).T
    normals/=np.maximum(np.linalg.norm(normals,axis=1),1e-20)[:,None]
    uv=np.zeros((len(mesh.loops),2),dtype=np.float32)
    if uv_map is not None:
        uv=read(uv_map.uv,'vector',2) if hasattr(uv_map,'uv') else read(uv_map.data,'uv',2)
    loops=read(mesh.loop_triangles,'loops',3,np.int32)
    material_indices=read(mesh.loop_triangles,'material_index',1,np.int32)
    material_indices=np.minimum(material_indices,len(materials)-1)
    checkpoint(cancel)
    return MeshArrays(positions,indices,normals,uv,loops,material_indices,materials)


class MeshArrays:
    """Owned arrays; never retains evaluated Blender RNA or dependency-graph items."""
    def __init__(self, positions, indices, normals, uv, loops, material_indices, materials):
        self.positions=positions; self.indices=indices; self.normals=normals
        self.uv=uv; self.loops=loops; self.material_indices=material_indices
        self.materials=materials

    def triangles(self, cancel=None):
        vertex_positions=[]
        for start in range(0,len(self.positions),4096):
            checkpoint(cancel)
            vertex_positions.extend(map(Vector,self.positions[start:start+4096].tolist()))
        corners=[]
        for start in range(0,len(self.indices),4096):
            checkpoint(cancel);block=slice(start,start+4096)
            corners.extend(VertexData(vertex_positions[vi],Vector(n),tuple(st))
                           for vi,n,st in zip(self.indices[block].tolist(),self.normals[block].tolist(),self.uv[block].tolist()))
        for start in range(0,len(self.loops),4096):
            checkpoint(cancel);block=slice(start,start+4096)
            for (a,b,c),mi in zip(self.loops[block].tolist(),self.material_indices[block].tolist()):
                yield TriangleData(corners[a],corners[b],corners[c],self.materials[mi])


def mesh_triangles(mesh, world, normal_matrix, materials, uv_map, cancel):
    return list(mesh_arrays(mesh,world,normal_matrix,materials,uv_map,cancel).triangles(cancel))


class PackedTriangles:
    """Compact viewport geometry with a compatibility iterator for REYES/export.

    The fast GPU path consumes arrays directly; CPU/compatibility paths can still
    iterate owned TriangleData records without changing their rendering code.
    """
    def __init__(self, cancel=None):
        self.blocks=[];self.count=0;self.cancel=cancel

    def append(self, block):
        self.blocks.append(block);self.count+=len(block.loops)

    def __len__(self):return self.count

    def __iter__(self):
        for block in self.blocks:yield from block.triangles(self.cancel)

    def __getitem__(self, index):
        # Index owned arrays directly: never walk/expand preceding triangles.
        from bisect import bisect_right
        if isinstance(index,slice):
            return [self[i] for i in range(*index.indices(self.count))]
        if index<0:index+=self.count
        if not 0<=index<self.count:raise IndexError(index)
        if getattr(self,'_index_count',None)!=self.count:
            self._ends=np.cumsum([len(b.loops) for b in self.blocks]).tolist()
            self._index_count=self.count
        from collections import OrderedDict
        if not hasattr(self,'_triangle_cache'):self._triangle_cache=OrderedDict()
        cached=self._triangle_cache.get(index)
        if cached is not None:
            self._triangle_cache.move_to_end(index)
            return cached
        bi=bisect_right(self._ends,index);block=self.blocks[bi]
        local=index-(self._ends[bi-1] if bi else 0)
        corners=block.loops[local]
        vertices=[VertexData(Vector(block.positions[block.indices[c]].tolist()),
                             Vector(block.normals[c].tolist()),tuple(block.uv[c])) for c in corners]
        triangle=TriangleData(*vertices,block.materials[int(block.material_indices[local])])
        self._triangle_cache[index]=triangle
        if len(self._triangle_cache)>16384:self._triangle_cache.popitem(last=False)
        return triangle

    def matches(self, other, cancel=None):
        """Exact shutter geometry equality; never merge animated materials."""
        if self is other:return True
        if not isinstance(other,PackedTriangles) or len(self.blocks)!=len(other.blocks):return False
        compared=set()
        for a,b in zip(self.blocks,other.blocks):
            checkpoint(cancel)
            for name in ('positions','indices','normals','uv','loops','material_indices'):
                if not np.array_equal(getattr(a,name),getattr(b,name)):return False
            if len(a.materials)!=len(b.materials):return False
            for ma,mb in zip(a.materials,b.materials):
                pair=(id(ma),id(mb))
                if pair in compared:continue
                compared.add(pair)
                if vars(ma).keys()!=vars(mb).keys():return False
                for name,va in vars(ma).items():
                    vb=getattr(mb,name)
                    if va is vb:continue
                    if hasattr(va,'pixels') and hasattr(vb,'pixels'):
                        for field,value in vars(va).items():
                            if field=='pixels':
                                if not np.array_equal(value,vb.pixels):return False
                            elif value!=getattr(vb,field):return False
                    elif va!=vb:return False
        return True

    def bound_points(self, first, stop, displacements):
        """A bounded source-position upload, without Python vertex objects."""
        stop=min(stop,self.count)
        result=np.empty((stop-first,3,4),dtype=np.float32);offset=0
        for block in self.blocks:
            end=offset+len(block.loops)
            lo=max(first,offset);hi=min(stop,end)
            if lo<hi:
                selection=slice(lo-offset,hi-offset)
                target=result[lo-first:hi-first]
                target[:,:,:3]=block.positions[block.indices[block.loops[selection]]]
                amounts=np.asarray([displacements.get(id(m),0.0) for m in block.materials],np.float32)
                target[:,:,3]=amounts[block.material_indices[selection],None]
            offset=end
            if offset>=stop:break
        return result

    def materials(self):
        for block in self.blocks:
            for i in np.unique(block.material_indices):yield block.materials[int(i)]

    def bounds(self, cancel=None):
        if not self.count:return Vector((-1,-1,-1)),Vector((1,1,1))
        lo=np.full(3,np.inf);hi=np.full(3,-np.inf)
        for block in self.blocks:
            for start in range(0,len(block.loops),4096):
                checkpoint(cancel)
                points=block.positions[block.indices[block.loops[start:start+4096].reshape(-1)]]
                lo=np.minimum(lo,points.min(axis=0));hi=np.maximum(hi,points.max(axis=0))
        return Vector(lo.tolist()),Vector(hi.tolist())

    def material_arrays(self, cancel=None):
        grouped={}
        for block in self.blocks:
            checkpoint(cancel)
            slots,first=np.unique(block.material_indices,return_index=True)
            for slot in slots[np.argsort(first)]:
                material=block.materials[int(slot)]
                chunks=grouped.setdefault(id(material),(material,[]))[1]
                # Limit intermediate advanced-indexing allocations and cancellation latency.
                for start in range(0,len(block.loops),32768):
                    checkpoint(cancel)
                    loops=block.loops[start:start+32768]
                    mask=block.material_indices[start:start+32768]==slot
                    corners=loops[mask].reshape(-1)
                    if len(corners):
                        chunks.append((block.positions[block.indices[corners]],block.normals[corners],block.uv[corners]))
        for material,chunks in grouped.values():
            checkpoint(cancel)
            yield material,tuple(np.concatenate([c[i] for c in chunks]) for i in range(3))
