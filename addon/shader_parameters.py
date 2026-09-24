# SPDX-License-Identifier: GPL-3.0-or-later
"""std140 parameters: vec4 slots and column-major mat4, without large push blocks."""
from array import array
import re

class ParameterInfo:
    def __init__(self):
        import gpu
        self.info = gpu.types.GPUShaderCreateInfo()
        self.fields = {}
        self.sources = {}

    def __getattr__(self, name):
        return getattr(self.info, name)

    def push_constant(self, kind, name):
        if kind not in {'FLOAT', 'INT', 'BOOL', 'VEC3', 'VEC4', 'MAT4'}:
            raise ValueError(f'Unsupported parameter: {kind}')
        if name in self.fields:
            raise ValueError(f'Duplicate parameter: {name}')
        self.fields[name] = kind

    def compute_source(self, source): self.sources['compute_source'] = source
    def vertex_source(self, source): self.sources['vertex_source'] = source
    def fragment_source(self, source): self.sources['fragment_source'] = source

    def build(self):
        import gpu
        declarations = []
        replacements = {}
        for name, kind in self.fields.items():
            declarations.append(f"{'mat4' if kind == 'MAT4' else 'vec4'} {name};")
            member = f'rmParams.{name}'
            replacements[name] = {
                'FLOAT': f'({member}.x)', 'INT': f'int({member}.x)',
                'BOOL': f'({member}.x != 0.0)', 'VEC3': f'({member}.xyz)',
                'VEC4': f'({member})', 'MAT4': f'({member})',
            }[kind]
        self.info.typedef_source('struct RetroManParameters {\n' + '\n'.join(declarations) + '\n};')
        self.info.uniform_buf(0, 'RetroManParameters', 'rmParams')
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, self.fields)) + r')\b')
        for method, source in self.sources.items():
            getattr(self.info, method)(pattern.sub(lambda m: replacements[m.group()], source))
        return ParameterShader(gpu.shader.create_from_info(self.info), self.fields)

class ParameterShader:
    def __init__(self, shader, fields):
        import gpu
        self.shader = shader
        self.fields = {}
        count = 0
        for name, kind in fields.items():
            self.fields[name] = (kind, count)
            count += 16 if kind == 'MAT4' else 4
        self.data = array('f', [0.0] * count)
        self.ubo = gpu.types.GPUUniformBuf(self.data)
        self.dirty = True
        self._packed_values = {}

    def __getattr__(self, name): return getattr(self.shader, name)

    def uniform_float(self, name, value):
        kind, offset = self.fields[name]
        if kind == 'MAT4':
            values = [float(value[row][col]) for col in range(4) for row in range(4)]
        elif kind in {'FLOAT', 'INT', 'BOOL'}:
            values = [float(value)]
        else:
            values = [float(v) for v in value]
            if len(values) != (3 if kind == 'VEC3' else 4):
                raise ValueError(f'Wrong parameter length: {name}')
        packed = array('f', values)
        # Compare uploaded float32 bytes, not mutable caller objects. Consecutive
        # batches often share all parameters; retain bindings without resending.
        key = packed.tobytes()
        if self._packed_values.get(name) == key:
            return
        self._packed_values[name] = key
        self.data[offset:offset+len(values)] = packed
        self.dirty = True

    uniform_int = uniform_float
    uniform_bool = uniform_float

    def prepare_draw(self):
        if self.dirty:
            self.ubo.update(self.data)
            self.dirty = False
        self.shader.uniform_block('rmParams', self.ubo)
        return self.shader
