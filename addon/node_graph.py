# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only shader socket views with per-instance group context."""
from types import SimpleNamespace


def matching(sockets, socket):
    return next((s for s in sockets if s.identifier == socket.identifier), None)


def source(socket, groups):
    """Resolve group boundaries and reroutes without losing instance inputs."""
    seen = set()
    for _ in range(64):
        if socket is None:return None
        key = (socket.as_pointer(), tuple(g.as_pointer() for g in groups))
        if key in seen:return None
        seen.add(key)
        if not socket.is_output:
            if not socket.is_linked:return None
            socket = socket.links[0].from_socket
            continue
        node = socket.node
        kind = node.bl_idname
        if node.mute:
            link = next((l for l in node.internal_links if l.to_socket == socket), None)
            socket = link.from_socket if link else None
        elif kind == 'NodeReroute':socket = node.inputs[0]
        elif kind == 'ShaderNodeGroup':
            if node.node_tree is None:return None
            output = next((n for n in node.node_tree.nodes if n.bl_idname == 'NodeGroupOutput' and n.is_active_output), None)
            if output is None:return None
            socket = matching(output.inputs, socket)
            groups = groups+(node,)
        elif kind == 'NodeGroupInput':
            if not groups:return None
            socket = matching(groups[-1].inputs, socket)
            groups = groups[:-1]
        else:return Socket(socket, groups)
    return None


class Socket:
    def __init__(self, raw, groups=()):self.raw=raw;self.groups=groups
    def __getattr__(self, name):return getattr(self.raw, name)
    @property
    def links(self):
        s=source(self.raw,self.groups)
        return [SimpleNamespace(from_socket=s,from_node=Node(s.raw.node,s.groups))] if s else []
    @property
    def is_linked(self):return bool(self.links) if self.raw.is_linked else False


class Sockets:
    def __init__(self, raw, groups):self.raw=raw;self.groups=groups
    def __iter__(self):return (Socket(s,self.groups) for s in self.raw)
    def __getitem__(self,key):return Socket(self.raw[key],self.groups)
    def get(self,key):
        s=self.raw.get(key)
        return Socket(s,self.groups) if s is not None else None


class Node:
    def __init__(self, raw, groups=()):self.raw=raw;self.groups=groups
    def __getattr__(self,name):return getattr(self.raw,name)
    def as_pointer(self):return (self.raw.as_pointer(),tuple(g.as_pointer() for g in self.groups))
    @property
    def inputs(self):return Sockets(self.raw.inputs,self.groups)
