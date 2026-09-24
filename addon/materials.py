# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender material -> RetroMan 1995 shader translation.

Normal Blender materials are the primary workflow. RetroSL is an optional,
small compatibility layer that reads classic parameters from a Blender Text
block; it is deliberately not an arbitrary code-execution language.
"""
import re
from array import array
from .model import MaterialData, TextureData
from .node_values import constant, UnsupportedValue
from .node_graph import Node


def _socket(node, name, fallback=None):
    try:
        return node.inputs.get(name) or fallback
    except Exception:
        return fallback


def _rgba(value, default=(0.8, 0.8, 0.8, 1.0)):
    try:
        if len(value) >= 4:
            return tuple(float(value[i]) for i in range(4))
    except Exception:
        pass
    return default


def _rgb(value, default=(0.0, 0.0, 0.0)):
    try:
        if len(value) >= 3:
            return tuple(float(value[i]) for i in range(3))
    except Exception:
        pass
    return default


def _value(sock, default):
    if sock is None:
        return default
    try:
        return float(constant(sock))
    except Exception:
        return default


def _color_value(sock, default):
    try:
        return constant(sock)
    except UnsupportedValue:
        return getattr(sock, "default_value", default)


def freeze_image(image, texture_cache):
    if image is None or image.size[0] <= 0 or image.size[1] <= 0:
        return None
    key = image.as_pointer()
    if key in texture_cache:
        return texture_cache[key]
    try:
        # bpy_prop_array.foreach_get() avoids materializing millions of Python
        # float objects for large images.  array('f') also exposes the buffer
        # protocol accepted by Blender's GPU Buffer API.
        count = len(image.pixels)
        pixels = array("f", [0.0]) * count
        try:
            image.pixels.foreach_get(pixels)
        except Exception:
            # Keep compatibility with unusual/generated image buffers that do
            # not expose the fast path.
            pixels = array("f", image.pixels[:])
        tex = TextureData(
            image.name, int(image.size[0]), int(image.size[1]), pixels,
            str(getattr(image, "filepath", "") or ""),
            bool(int(getattr(image, "channels", 4)) >= 4 and str(getattr(image, "alpha_mode", "STRAIGHT")) != "NONE"),
        )
        texture_cache[key] = tex
        return tex
    except Exception:
        return None


def find_image_upstream(sock, texture_cache, *, max_depth=8):
    if sock is None or not getattr(sock, "is_linked", False):
        return None
    seen = set()
    stack = [(link.from_node, 0) for link in sock.links]
    while stack:
        node, depth = stack.pop(0)
        try:
            key = node.as_pointer()
        except Exception:
            key = id(node)
        if key in seen or depth > max_depth:
            continue
        seen.add(key)
        if getattr(node, "bl_idname", "") == "ShaderNodeTexImage":
            return freeze_image(getattr(node, "image", None), texture_cache)
        try:
            for inp in node.inputs:
                if getattr(inp, "is_linked", False):
                    for link in inp.links:
                        stack.append((link.from_node, depth + 1))
        except Exception:
            pass
    return None


def _base_texture(sock, result, texture_cache):
    result.texture = find_image_upstream(sock, texture_cache)
    # A directly linked image replaces the socket default. Group/reroute views
    # preserve that same connection; do not tint it with an unused default.
    if result.texture is not None and sock.is_linked:
        link = sock.links[0]
        if link.from_node.bl_idname == 'ShaderNodeTexImage' and link.from_socket.name == 'Color':
            result.base_color = (1.0, 1.0, 1.0, 1.0)


def _surface_node(surface, warnings=None, name="Material"):
    seen = set()
    for _ in range(32):
        if surface is None or not getattr(surface, "is_linked", False):
            return None
        node = surface.links[0].from_node
        key = node.as_pointer()
        if key in seen:
            if warnings is not None:warnings.add(f"{name}: cyclic surface graph was ignored")
            return None
        seen.add(key)
        kind = getattr(node, "bl_idname", "")
        if kind == "NodeReroute":
            surface = node.inputs[0]
            continue
        if kind == "ShaderNodeMixShader":
            fac_socket = node.inputs[0]
            fac = _value(fac_socket, 0.5)
            try:
                constant(fac_socket)
                linked_unknown = False
            except UnsupportedValue:
                linked_unknown = True
            if warnings is not None and (linked_unknown or 0.0 < fac < 1.0):
                warnings.add(f"{name}: Mix Shader is approximated by one branch; linked factors and blended lobes are not evaluated")
            surface = node.inputs[2 if fac >= 0.5 else 1]
            continue
        return node
    if warnings is not None:warnings.add(f"{name}: surface graph exceeds the traversal limit")
    return None


def _translate_displacement(output, result, texture_cache, warnings):
    disp_socket = output.inputs.get("Displacement") if output is not None else None
    if disp_socket is None or not getattr(disp_socket, "is_linked", False):
        return
    node = disp_socket.links[0].from_node
    node_type = getattr(node, "bl_idname", "")
    if node_type == "ShaderNodeDisplacement":
        height = _socket(node, "Height")
        result.displacement_texture = find_image_upstream(height, texture_cache)
        result.displacement_constant = _value(height, 0.5)
        result.displacement_midlevel = _value(_socket(node, "Midlevel"), 0.5)
        result.displacement_scale = _value(_socket(node, "Scale"), 1.0)
        if height is not None and getattr(height, "is_linked", False) and result.displacement_texture is None:
            # A linked socket's default is not the procedural result. Applying
            # it invents a constant offset and needlessly forces GPU dicing.
            try:
                result.displacement_constant = float(constant(height))
            except (UnsupportedValue, TypeError, ValueError):
                result.displacement_scale = 0.0
                warnings.add(f"{result.name}: unsupported procedural displacement was ignored; use an image or a constant Height graph")
        return
    if node_type == "ShaderNodeVectorDisplacement":
        warnings.add(f"{result.name}: vector displacement is unsupported; scalar Blender Displacement is supported")
        return
    tex = find_image_upstream(disp_socket, texture_cache)
    if tex is not None:
        result.displacement_texture = tex
        result.displacement_midlevel = 0.5
        result.displacement_scale = 0.1
        warnings.add(f"{result.name}: direct displacement image uses RetroMan scale 0.1; add Blender Displacement for exact scale")


def _translate_normal_input(shader, result, texture_cache, warnings):
    normal = _socket(shader, "Normal")
    if normal is None or not getattr(normal, "is_linked", False):
        return
    node = normal.links[0].from_node
    kind = getattr(node, "bl_idname", "")
    if kind == "ShaderNodeNormalMap":
        color = _socket(node, "Color")
        result.normal_texture = find_image_upstream(color, texture_cache)
        result.normal_strength = max(0.0, _value(_socket(node, "Strength"), 1.0))
        result.normal_y_sign = -1.0 if str(getattr(node, "convention", "OPENGL")) == "DIRECTX" else 1.0
        space = str(getattr(node, "space", "TANGENT"))
        if space != "TANGENT":
            warnings.add(f"{result.name}: Normal Map space {space} is not supported; tangent space is used")
        uv_name = str(getattr(node, "uv_map", "") or "")
        if uv_name:
            warnings.add(f"{result.name}: Normal Map requests UV map '{uv_name}'; RetroMan currently uses the mesh Active Render UV")
        if str(getattr(node, "base", "DISPLACED")) == "ORIGINAL" and result.has_displacement:
            warnings.add(f"{result.name}: Normal Map Original Base is approximated relative to the displaced surface")
        if result.normal_texture is None:
            warnings.add(f"{result.name}: Normal Map has no image RetroMan can translate")
    elif kind == "ShaderNodeBump":
        height = _socket(node, "Height")
        result.bump_texture = find_image_upstream(height, texture_cache)
        result.bump_strength = max(0.0, _value(_socket(node, "Strength"), 1.0))
        result.bump_sign = -1.0 if bool(getattr(node, "invert", False)) else 1.0
        result.bump_distance = max(1.0e-6, _value(_socket(node, "Distance"), 0.1))
        if result.bump_texture is None:
            warnings.add(f"{result.name}: procedural Bump is ignored unless an image can be found upstream")
    else:
        tex = find_image_upstream(normal, texture_cache)
        if tex is not None:
            result.normal_texture = tex
            result.normal_strength = 1.0
        else:
            warnings.add(f"{result.name}: unsupported Normal input graph was ignored")


def _parse_number(source, name, default=None):
    m = re.search(r"\b" + re.escape(name) + r"\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", source)
    return float(m.group(1)) if m else default


def _parse_color(source, name):
    patterns = [
        r"\b" + re.escape(name) + r"\s*=\s*color\s*\(\s*([^\)]+)\)",
        r"\b" + re.escape(name) + r"\s*=\s*\[\s*([^\]]+)\]",
    ]
    for pat in patterns:
        m = re.search(pat, source, flags=re.I)
        if m:
            nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", m.group(1))
            if len(nums) >= 3:
                return tuple(float(x) for x in nums[:3])
    return None


def _apply_retrosl(material, result, warnings):
    settings = getattr(material, "retroman", None)
    if settings is None or not getattr(settings, "retrosl_enabled", False):
        return
    text = getattr(settings, "retrosl_text", None)
    if text is None:
        warnings.add(f"{result.name}: RetroSL is enabled but no Blender Text block is selected")
        return
    try:
        source = text.as_string()
    except Exception:
        source = ""
    result.retrosl_source = source
    result.shader_name = "retrosl"
    c = _parse_color(source, "Cs")
    if c is not None:
        result.base_color = (c[0], c[1], c[2], result.base_color[3])
    for attr, token in (("ka", "Ka"), ("kd", "Kd"), ("ks", "Ks"), ("kr", "Kr"), ("roughness", "roughness"), ("alpha", "opacity")):
        val = _parse_number(source, token, None)
        if val is not None:
            setattr(result, attr, float(val))
    ec = _parse_color(source, "emission")
    if ec is not None:
        result.emission_color = ec
        result.emission_strength = max(1.0, _parse_number(source, "emissionStrength", 1.0))


def missing_shader_libraries(tree):
    """Name missing linked group assets instead of blaming graph complexity."""
    stack=[tree];seen=set();missing=set()
    while stack and len(seen)<256:
        current=stack.pop()
        if current is None:continue
        current=getattr(current,'original',current)
        key=current.as_pointer()
        if key in seen:continue
        seen.add(key)
        if getattr(current,'is_missing',False):
            library=getattr(current,'library',None)
            missing.add(str(getattr(library,'filepath',current.name)))
        else:
            for node in current.nodes:
                if node.bl_idname=='ShaderNodeGroup':stack.append(node.node_tree)
    return sorted(missing)


def translate_material(material, texture_cache, warnings):
    if material is None:
        return MaterialData(name="Default")

    result = MaterialData(
        name=material.name,
        base_color=_rgba(getattr(material, "diffuse_color", (0.8, 0.8, 0.8, 1.0))),
        metallic=float(getattr(material, "metallic", 0.0)),
        roughness=float(getattr(material, "roughness", 0.5)),
    )
    result.kd = 1.0 - max(0.0, min(1.0, result.metallic))
    result.ks = 0.5

    if not getattr(material, "use_nodes", False) or material.node_tree is None:
        _apply_retrosl(material, result, warnings)
        return result

    for path in missing_shader_libraries(material.node_tree):
        warnings.add(f"Missing shader library: {path}. Relink it in Blender to restore its materials")

    output = next((n for n in material.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial" and getattr(n, "is_active_output", False)), None)
    if output is None:
        _apply_retrosl(material, result, warnings)
        return result

    output = Node(output)
    _translate_displacement(output, result, texture_cache, warnings)
    shader = _surface_node(output.inputs.get("Surface"), warnings, material.name)
    if shader is None:
        surface = output.inputs.get("Surface")
        if surface is not None and surface.raw.is_linked:
            warnings.add(f"{material.name}: surface graph could not be resolved (empty, cyclic or over traversal limit); using viewport values")
        _apply_retrosl(material, result, warnings)
        return result

    # Diagnose unsupported linked scalar inputs instead of silently presenting
    # their unused defaults as correctly translated Blender shader values.
    for inp in shader.inputs:
        if inp.is_linked and inp.type == 'VALUE':
            try:
                constant(inp)
            except UnsupportedValue:
                if inp.name != 'Alpha' or find_image_upstream(inp, texture_cache) is None:
                    warnings.add(f"{material.name}: linked {inp.name} graph is unsupported; RetroMan fallback is used")
    kind = getattr(shader, "bl_idname", "")
    if kind == "ShaderNodeBsdfPrincipled":
        base = _socket(shader, "Base Color")
        rough = _socket(shader, "Roughness")
        metal = _socket(shader, "Metallic")
        alpha = _socket(shader, "Alpha")
        spec = _socket(shader, "Specular IOR Level") or _socket(shader, "Specular")
        emission = _socket(shader, "Emission Color") or _socket(shader, "Emission")
        emission_strength = _socket(shader, "Emission Strength")
        if base is not None:
            result.base_color = (*_rgba(_color_value(base, result.base_color), result.base_color)[:3], 1.0)
            _base_texture(base, result, texture_cache)
        # A color image having an alpha channel does not mean Blender uses that
        # channel for surface opacity. Honor it only when Principled Alpha is
        # explicitly linked to the same image graph.
        alpha_tex = find_image_upstream(alpha, texture_cache) if alpha is not None and getattr(alpha, "is_linked", False) else None
        result.use_texture_alpha = bool(alpha_tex is not None and alpha_tex is result.texture and getattr(alpha_tex, "has_alpha", False))
        if alpha_tex is not None and alpha_tex is not result.texture:
            warnings.add(f"{material.name}: separate Alpha texture graph is not yet sampled independently; constant Alpha is used")
        result.roughness = max(0.001, min(1.0, _value(rough, result.roughness)))
        result.metallic = max(0.0, min(1.0, _value(metal, result.metallic)))
        result.alpha = max(0.0, min(1.0, _value(alpha, result.base_color[3])))
        result.specular = max(0.0, _value(spec, 0.5))
        result.kd = 1.0 - result.metallic
        result.ks = result.specular
        if emission is not None:
            result.emission_color = _rgb(_color_value(emission, (0.0, 0.0, 0.0)))
        result.emission_strength = max(0.0, _value(emission_strength, 0.0))
        _translate_normal_input(shader, result, texture_cache, warnings)
        for socket_name in ("Coat Weight", "Iridescence Weight", "Subsurface Weight", "Transmission Weight"):
            s = _socket(shader, socket_name)
            if s is not None and _value(s, 0.0) > 0.0001:
                warnings.add(f"{material.name}: {socket_name} is approximated by the 1995 shader model")
    elif kind == "ShaderNodeBsdfDiffuse":
        color = _socket(shader, "Color")
        if color is not None:
            result.base_color = _rgba(_color_value(color, result.base_color), result.base_color)
            _base_texture(color, result, texture_cache)
        result.roughness = max(0.001, min(1.0, _value(_socket(shader, "Roughness"), 0.5)))
        result.metallic = 0.0; result.specular = 0.0; result.kd = 1.0; result.ks = 0.0
        _translate_normal_input(shader, result, texture_cache, warnings)
    elif kind in {"ShaderNodeBsdfGlossy", "ShaderNodeBsdfAnisotropic"}:
        color = _socket(shader, "Color")
        if color is not None:
            result.base_color = _rgba(_color_value(color, result.base_color), result.base_color)
            _base_texture(color, result, texture_cache)
        result.roughness = max(0.001, min(1.0, _value(_socket(shader, "Roughness"), 0.2)))
        result.metallic = 1.0; result.specular = 1.0
        result.kd = 0.0; result.ks = 1.0; result.kr = 1.0
        _translate_normal_input(shader, result, texture_cache, warnings)
        if abs(_value(_socket(shader, "Anisotropy"), 0.0)) > 0.0001:
            warnings.add(f"{material.name}: anisotropy is approximated with an isotropic glossy lobe")
    elif kind == "ShaderNodeBsdfTransparent":
        color = _socket(shader, "Color")
        result.base_color = (1.0, 1.0, 1.0, 1.0)
        result.alpha = 0.0
        result.ka = result.kd = result.ks = result.kr = 0.0
        if color is not None and (color.is_linked or any(abs(v-1.0)>0.0001 for v in color.default_value[:3])):
            warnings.add(f"{material.name}: colored Transparent BSDF is approximated as clear transparency")
    elif kind == "ShaderNodeEmission":
        color = _socket(shader, "Color")
        result.base_color = (0.0, 0.0, 0.0, 1.0)
        result.kd = result.ks = 0.0
        if color is not None:
            result.emission_color = _rgb(_color_value(color, (1.0, 1.0, 1.0)), (1.0, 1.0, 1.0))
            result.texture = find_image_upstream(color, texture_cache)
        result.emission_strength = max(0.0, _value(_socket(shader, "Strength"), 1.0))
    elif kind == "ShaderNodeBsdfGlass":
        color = _socket(shader, "Color")
        if color is not None:
            result.base_color = _rgba(_color_value(color, result.base_color), result.base_color)
        result.alpha = 0.18
        result.kd = 0.08; result.ks = 1.0; result.kr = 0.6
        result.roughness = max(0.001, _value(_socket(shader, "Roughness"), 0.05))
        warnings.add(f"{material.name}: Glass is approximated with opacity + reflection maps; no ray-traced refraction")
    else:
        warnings.add(f"{material.name}: unsupported surface node {kind or shader.name}; using material viewport values")

    _apply_retrosl(material, result, warnings)
    return result
