# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded evaluation of spatially constant Blender shader inputs.

Unknown (spatial, texture or unsupported) expressions stay unknown. Never treat
an unused linked input default as the evaluated value of its upstream node.
"""
import math

class UnsupportedValue(ValueError):
    pass


def constant(socket):
    groups = getattr(socket, "groups", ())
    socket = getattr(socket, "raw", socket)
    budget = [256]
    active = set()

    def scalar(value):
        if isinstance(value, tuple):
            # Blender's color-to-value conversion uses average RGB.
            return sum(value[:3]) / 3.0
        return float(value)

    def convert(value, target):
        kind = getattr(target, 'type', '')
        if kind == 'VALUE':return scalar(value)
        if kind == 'RGBA' and not isinstance(value, tuple):return (value, value, value, 1.0)
        return value

    def match(sockets, socket):
        return next((s for s in sockets if s.identifier == socket.identifier), None)

    def read(sock, groups=()):
        if sock is None:raise UnsupportedValue('missing socket')
        budget[0] -= 1
        key = (sock.as_pointer(), tuple(g.as_pointer() for g in groups))
        if budget[0] < 0 or len(active) >= 48 or key in active:
            raise UnsupportedValue('graph traversal limit or cycle')
        active.add(key)
        try:
            if not sock.is_output:
                value = read(sock.links[0].from_socket, groups) if sock.is_linked else sock.default_value
            else:
                n = sock.node; kind = n.bl_idname
                if n.mute:
                    link = next((l for l in n.internal_links if l.to_socket == sock), None)
                    if link is None:raise UnsupportedValue('muted node has no bypass')
                    value = read(link.from_socket, groups)
                elif kind in {'ShaderNodeValue','ShaderNodeRGB'}:value = sock.default_value
                elif kind == 'NodeReroute':value = read(n.inputs[0], groups)
                elif kind == 'ShaderNodeGroup':
                    if n.node_tree is None:raise UnsupportedValue('missing node group')
                    output = next((x for x in n.node_tree.nodes if x.bl_idname == 'NodeGroupOutput' and x.is_active_output), None)
                    if output is None:raise UnsupportedValue('missing active group output')
                    value = read(match(output.inputs, sock), groups + (n,))
                elif kind == 'NodeGroupInput':
                    if not groups:raise UnsupportedValue('group context unavailable')
                    value = read(match(groups[-1].inputs, sock), groups[:-1])
                elif kind == 'ShaderNodeMath':
                    op = n.operation
                    unary = {'ABSOLUTE':abs,'FLOOR':math.floor,'CEIL':math.ceil,
                             'SINE':math.sin,'COSINE':math.cos,'SQRT':lambda x:math.sqrt(max(0,x))}
                    a = scalar(read(n.inputs[0], groups))
                    if op in unary:value = unary[op](a)
                    else:
                        b = scalar(read(n.inputs[1], groups))
                        binary = {'ADD':lambda:a+b,'SUBTRACT':lambda:a-b,'MULTIPLY':lambda:a*b,
                                  'DIVIDE':lambda:a/b if b != 0 else 0.0,
                                  'MINIMUM':lambda:min(a,b),'MAXIMUM':lambda:max(a,b),
                                  'LESS_THAN':lambda:float(a<b),'GREATER_THAN':lambda:float(a>b)}
                        if op == 'MULTIPLY_ADD':value = a*b+scalar(read(n.inputs[2], groups))
                        elif op in binary:value = binary[op]()
                        else:raise UnsupportedValue('unsupported math operation '+op)
                    if n.use_clamp:value = min(1.0,max(0.0,value))
                elif kind == 'ShaderNodeMixRGB' and n.blend_type == 'MIX':
                    f = min(1.0,max(0.0,scalar(read(n.inputs[0], groups))))
                    a = read(n.inputs[1],groups); b = read(n.inputs[2],groups)
                    value = tuple(a[i]*(1-f)+b[i]*f for i in range(3))+(a[3],)
                    if n.use_clamp:value=tuple(min(1.0,max(0.0,x)) for x in value)
                else:raise UnsupportedValue('unsupported node '+kind)
            if not isinstance(value,(float,int,tuple)):value=tuple(value)
            value = convert(value,sock)
            values = value if isinstance(value,tuple) else (value,)
            if not all(math.isfinite(x) for x in values):raise UnsupportedValue('non-finite value')
            return value
        except (AttributeError,TypeError,OverflowError,ZeroDivisionError) as exc:
            raise UnsupportedValue(str(exc)) from exc
        finally:active.remove(key)
    return read(socket, groups)
