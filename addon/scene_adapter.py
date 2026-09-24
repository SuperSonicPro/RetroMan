from .runtime import checkpoint
# SPDX-License-Identifier: GPL-3.0-or-later
from mathutils import Vector

from .model import CameraData, LightData, ReflectionProbeData, RenderScene, TriangleData, VertexData
from .materials import freeze_image, translate_material, _value, _color_value
from .node_graph import Node
from .illumination import mesh_emitters, environment_diffuse


def _active_world_background(scene):
    world = scene.world
    if world is None or not getattr(world, "use_nodes", False) or world.node_tree is None:
        return None
    try:
        output = next((
            n for n in world.node_tree.nodes
            if getattr(n, "bl_idname", "") == "ShaderNodeOutputWorld"
            and getattr(n, "is_active_output", False)
        ), None)
        if output is None:
            return None
        surface = Node(output).inputs.get("Surface")
        if surface is None or not surface.is_linked:
            return None
        node = surface.links[0].from_node
        return node if getattr(node, "bl_idname", "") == "ShaderNodeBackground" else None
    except Exception:
        return None


def _world_color(scene):
    world = scene.world
    if world is None:
        return (0.05, 0.05, 0.05)
    background = _active_world_background(scene)
    if background is not None:
        try:
            color_socket = background.inputs.get("Color")
            strength_socket = background.inputs.get("Strength")
            if color_socket is not None:
                from .node_values import constant
                c = constant(color_socket)
                strength = _value(strength_socket, 1.0)
                return tuple(max(0.0, float(c[i]) * strength) for i in range(3))
        except Exception:
            pass
    try:
        c = world.color
        return (float(c[0]), float(c[1]), float(c[2]))
    except Exception:
        return (0.05, 0.05, 0.05)


def _find_environment_upstream(sock, texture_cache, max_depth=8):
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
        if getattr(node, "bl_idname", "") == "ShaderNodeTexEnvironment":
            return freeze_image(getattr(node, "image", None), texture_cache)
        try:
            for inp in node.inputs:
                if getattr(inp, "is_linked", False):
                    for link in inp.links:
                        stack.append((link.from_node, depth + 1))
        except Exception:
            pass
    return None


def _world_environment(scene, texture_cache):
    background = _active_world_background(scene)
    if background is None:
        return None, 1.0
    try:
        color_socket = background.inputs.get("Color")
        strength_socket = background.inputs.get("Strength")
        texture = _find_environment_upstream(color_socket, texture_cache)
        strength = _value(strength_socket, 1.0)
        return texture, max(0.0, strength)
    except Exception:
        return None, 1.0


def extract_world_color(scene):
    return _world_color(scene)


def extract_world_state(scene, texture_cache=None):
    cache = texture_cache if texture_cache is not None else {}
    env, strength = _world_environment(scene, cache)
    return _world_color(scene), env, strength


def _light_emission(data, warnings=None):
    """Translate constant light shader graphs into classic RGB light intensity."""
    from .node_graph import Node
    from .node_values import constant, UnsupportedValue
    if not getattr(data, "use_nodes", False) or not data.node_tree:
        return (1.0, 1.0, 1.0)
    outputs = [n for n in data.node_tree.nodes if n.type == "OUTPUT_LIGHT"]
    output = next((n for n in outputs if n.is_active_output), None)
    if output is None:
        return (0.0, 0.0, 0.0)
    active = set()
    budget = [64]

    def evaluate(socket):
        budget[0] -= 1
        if budget[0] < 0:
            raise UnsupportedValue("light graph traversal limit")
        links = socket.links if socket else []
        if not links:
            return (0.0, 0.0, 0.0)
        node = links[0].from_node
        key = node.as_pointer()
        if key in active:
            raise UnsupportedValue("cyclic light graph")
        active.add(key)
        try:
            if node.type == "EMISSION":
                color = constant(node.inputs.get("Color"))
                strength = max(0.0, float(constant(node.inputs.get("Strength"))))
                return tuple(max(0.0, float(c)) * strength for c in color[:3])
            if node.type in {"ADD_SHADER", "MIX_SHADER"}:
                offset = 1 if node.type == "MIX_SHADER" else 0
                factor = max(0.0, min(1.0, float(constant(node.inputs[0])))) if offset else 1.0
                left = evaluate(node.inputs[offset])
                right = evaluate(node.inputs[offset + 1])
                return tuple(a * (1.0-factor if offset else 1.0) + b * factor for a,b in zip(left,right))
            raise UnsupportedValue("unsupported light shader " + node.bl_idname)
        finally:
            active.remove(key)
    try:
        return evaluate(Node(output).inputs.get("Surface"))
    except (UnsupportedValue, TypeError, ValueError, OverflowError) as exc:
        if warnings is not None:
            warnings.add(f"Light '{data.name}': {exc}; using light data color and power")
        return (1.0, 1.0, 1.0)


def _make_light(obj, matrix_world, warnings=None):
    data = obj.data
    exposure = float(getattr(data, "exposure", 0.0))
    energy = float(data.energy) * (2.0 ** exposure)

    # Blender only applies cutoff_distance when Custom Distance is enabled.
    # Keep 0 as RetroMan's sentinel for unlimited range.  Older RetroMan builds
    # accidentally hard-clipped every point/spot/area light at Blender's
    # default 40-unit cutoff even when the option was disabled.
    use_custom_distance = bool(getattr(data, "use_custom_distance", False))
    cutoff = float(getattr(data, "cutoff_distance", 40.0)) if use_custom_distance else 0.0

    emission = _light_emission(data, warnings)
    color = [float(data.color[i]) * emission[i] for i in range(3)]
    if bool(getattr(data, "use_temperature", False)):
        try:
            temp = data.temperature_color
            color = [color[i] * float(temp[i]) for i in range(3)]
        except Exception:
            pass

    normalize = bool(getattr(data, "normalize", True))
    area_factor = 1.0
    if data.type == "AREA" and not normalize:
        try:
            # Blender exposes the transformed emitter area directly.  When
            # Normalize is disabled, a larger area light emits proportionally
            # more total light; with Normalize enabled its total output stays
            # approximately constant as size changes.
            area_factor = max(1.0e-8, float(data.area(matrix_world=matrix_world)))
        except Exception:
            # Conservative approximation for older/partial APIs.
            size = float(getattr(data, "size", 1.0) or 1.0)
            size_y = float(getattr(data, "size_y", size) or size)
            area_factor = max(1.0e-8, abs(size * size_y))

    return LightData(
        name=obj.name,
        type=data.type,
        matrix_world=matrix_world.copy(),
        color=tuple(color),
        energy=energy,
        use_shadow=bool(getattr(data, "use_shadow", True)),
        spot_size=float(getattr(data, "spot_size", 0.785398)),
        spot_blend=float(getattr(data, "spot_blend", 0.15)),
        cutoff_distance=max(0.0, cutoff),
        size=float(getattr(data, "shadow_soft_size", getattr(data, "size", 0.25))),
        diffuse_factor=max(0.0, float(getattr(data, "diffuse_factor", 1.0))),
        specular_factor=max(0.0, float(getattr(data, "specular_factor", 1.0))),
        area_factor=area_factor,
        normalize=normalize,
    )


def extract_lights(depsgraph, *, viewport=False):
    lights = []
    for inst in depsgraph.object_instances:
        obj = inst.object
        original = getattr(obj, "original", obj)
        if obj.type != "LIGHT":
            continue
        # The viewport dependency graph already reflects viewport visibility.
        # Do not incorrectly hide an object just because its Render (camera)
        # toggle is disabled; that toggle applies to F12, not Rendered View.
        if (not viewport) and getattr(original, "hide_render", False):
            continue
        lights.append(_make_light(obj, inst.matrix_world))
    return lights


def _is_sphere_probe(obj):
    data = getattr(obj, "data", None)
    ident = getattr(getattr(data, "bl_rna", None), "identifier", "")
    return ident == "LightProbeSphere" or data.__class__.__name__ == "LightProbeSphere"


def extract_reflection_probes(depsgraph):
    probes = []
    for inst in depsgraph.object_instances:
        obj = inst.object
        original = getattr(obj, "original", obj)
        if getattr(original, "hide_render", False) or not _is_sphere_probe(obj):
            continue
        data = obj.data
        probes.append(ReflectionProbeData(
            name=obj.name,
            position=inst.matrix_world.translation.copy(),
            clip_start=float(getattr(data, "clip_start", 0.1)),
            clip_end=float(getattr(data, "clip_end", 50.0)),
            influence_distance=float(getattr(data, "influence_distance", 2.5)),
        ))
    return probes


def viewport_camera(context, width, height):
    r3d = context.region_data
    if r3d is None:
        raise RuntimeError("RetroMan Rendered viewport needs a 3D View region")
    view = r3d.view_matrix.copy()
    projection = r3d.window_matrix.copy()
    view3d = context.space_data
    return CameraData(
        matrix_world=view.inverted(),
        projection=projection,
        clip_start=float(getattr(view3d, "clip_start", 0.01)),
        clip_end=float(getattr(view3d, "clip_end", 1000.0)),
        dof_enabled=False,
        camera_type="VIEWPORT",
    )


def _camera_from_blender(scene, depsgraph, width, height):
    camera_obj = scene.camera
    if camera_obj is None:
        raise RuntimeError("RetroMan needs an active Blender camera for F12 rendering")
    cam_eval = camera_obj.evaluated_get(depsgraph)
    # Preserve Blender's non-square-pixel camera semantics as well as lens
    # shift/sensor fit. calc_matrix_camera exposes pixel-aspect scale explicitly.
    render = scene.render
    projection = cam_eval.calc_matrix_camera(
        depsgraph, x=width, y=height,
        scale_x=max(1.0e-6, float(getattr(render, "pixel_aspect_x", 1.0))),
        scale_y=max(1.0e-6, float(getattr(render, "pixel_aspect_y", 1.0))),
    )
    cam_data = camera_obj.data
    camera_type = str(getattr(cam_data, "type", "PERSP"))
    if camera_type not in {"PERSP", "ORTHO"}:
        raise RuntimeError(
            f"RetroMan supports Blender Perspective and Orthographic cameras; "
            f"camera '{camera_obj.name}' uses {camera_type}."
        )
    matrix_world = cam_eval.matrix_world.copy()

    dof = getattr(cam_data, "dof", None)
    # Thin-lens DOF is meaningful for the perspective camera model. Blender
    # exposes the property on camera data generally, so avoid silently applying
    # a perspective lens shift to an orthographic projection.
    dof_enabled = bool(camera_type == "PERSP" and dof and getattr(dof, "use_dof", False))
    focus_distance = float(getattr(dof, "focus_distance", 10.0)) if dof else 10.0
    focus_object = getattr(dof, "focus_object", None) if dof else None
    if dof_enabled and focus_object is not None:
        try:
            focus_eval = focus_object.evaluated_get(depsgraph)
            forward = (matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
            delta = focus_eval.matrix_world.translation - matrix_world.translation
            projected = abs(float(delta.dot(forward)))
            if projected > 1.0e-6:
                focus_distance = projected
        except Exception:
            pass

    fstop = max(0.01, float(getattr(dof, "aperture_fstop", 2.8))) if dof else 2.8
    lens_mm = max(0.01, float(getattr(cam_data, "lens", 50.0)))
    scale_length = max(1.0e-9, float(getattr(scene.unit_settings, "scale_length", 1.0) or 1.0))
    # aperture diameter = focal length / f-number; convert mm to Blender units.
    aperture_radius = (lens_mm / fstop) * 0.0005 / scale_length if dof_enabled else 0.0

    return CameraData(
        matrix_world=matrix_world,
        projection=projection.copy(),
        clip_start=float(cam_data.clip_start),
        clip_end=float(cam_data.clip_end),
        dof_enabled=dof_enabled,
        focus_distance=max(1.0e-5, focus_distance),
        aperture_radius=max(0.0, aperture_radius),
        aperture_blades=int(getattr(dof, "aperture_blades", 0)) if dof else 0,
        aperture_ratio=max(0.01, float(getattr(dof, "aperture_ratio", 1.0))) if dof else 1.0,
        aperture_rotation=float(getattr(dof, "aperture_rotation", 0.0)) if dof else 0.0,
        camera_type=camera_type,
    )


def extract_scene(depsgraph, engine, *, width=None, height=None, camera_override=None, progress_start=0.02, progress_end=0.15):
    scene = depsgraph.scene
    if width is None or height is None:
        scale = scene.render.resolution_percentage / 100.0
        width = max(1, int(scene.render.resolution_x * scale))
        height = max(1, int(scene.render.resolution_y * scale))
    else:
        width = max(1, int(width))
        height = max(1, int(height))

    material_cache = {}
    texture_cache = {}
    warnings = set()

    is_viewport = camera_override is not None
    camera = camera_override if is_viewport else _camera_from_blender(scene, depsgraph, width, height)
    world_color, environment_texture, environment_strength = extract_world_state(scene, texture_cache)

    rscene = RenderScene(
        width=width,
        height=height,
        camera=camera,
        world_color=world_color,
        environment_texture=environment_texture,
        environment_strength=environment_strength,
        film_transparent=bool(getattr(scene.render, "film_transparent", False)),
        settings=scene.retroman,
    )

    if getattr(scene.retroman, "environment_lighting", True):
        rscene.environment_diffuse = environment_diffuse(environment_texture, environment_strength, world_color, engine.test_break)
        if environment_texture is not None:
            warnings.add("World image translated to classic constant ambient fill and a reflection map; directional environment illumination is not reproduced")

    # Keep bulk evaluated arrays for auxiliary GPU maps as well as the viewport.
    # Bucket splitting expands only actively processed source triangles.
    packed_geometry = is_viewport or getattr(scene.retroman, "render_device", "AUTO") != "CPU"
    if packed_geometry:
        from .mesh_extract import PackedTriangles
        rscene.triangles = PackedTriangles(engine.test_break)

    # Blender documents object_instances as an iterator whose items must not be
    # retained. Stream it directly; keeping DepsgraphObjectInstance references
    # in a list can leave dangling evaluated-data references after updates.
    total_hint = max(1, len(getattr(depsgraph, "objects", ())))
    for idx, inst in enumerate(depsgraph.object_instances):
        checkpoint(engine.test_break)
        if hasattr(engine, "update_progress"):
            p = progress_start + (progress_end - progress_start) * (min(idx, total_hint) / total_hint)
            engine.update_progress(p)
        obj = inst.object
        original = getattr(obj, "original", obj)
        # Render visibility and viewport visibility are distinct in Blender.
        # For a Rendered viewport, depsgraph.object_instances has already been
        # filtered for viewport visibility, so respect hide_render only for F12.
        if (not is_viewport) and getattr(original, "hide_render", False):
            continue

        if obj.type == "LIGHT":
            rscene.lights.append(_make_light(obj, inst.matrix_world, warnings))
            continue

        if _is_sphere_probe(obj):
            data = obj.data
            rscene.reflection_probes.append(ReflectionProbeData(
                name=obj.name,
                position=inst.matrix_world.translation.copy(),
                clip_start=float(getattr(data, "clip_start", 0.1)),
                clip_end=float(getattr(data, "clip_end", 50.0)),
                influence_distance=float(getattr(data, "influence_distance", 2.5)),
            ))
            continue

        # Blender evaluates modifiers/Geometry Nodes before RetroMan sees the
        # object. Mesh-convertible curve/surface/text/meta objects therefore use
        # the same final triangle path as ordinary meshes.
        if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
            continue

        # DepsgraphObjectInstance.object is already evaluated. Calling
        # evaluated_get() again is redundant and can force unnecessary graph work.
        mesh_obj = obj
        mesh = None
        try:
            if not is_viewport and hasattr(engine, "_status"):
                engine._status(f"Translating {obj.name}")
            checkpoint(engine.test_break)
            mesh = mesh_obj.to_mesh()
            if mesh is None:
                continue
            mesh.calc_loop_triangles()
            world = inst.matrix_world.copy()
            try:
                normal_matrix = world.to_3x3().inverted().transposed()
            except Exception:
                normal_matrix = world.to_3x3()
            # Prefer the UV layer explicitly marked for rendering.  Blender's
            # editor-active layer can differ from the render-active one.
            uv_map = None
            try:
                uv_map = next((layer for layer in mesh.uv_layers if getattr(layer, "active_render", False)), None)
            except Exception:
                uv_map = None
            if uv_map is None:
                uv_map = mesh.uv_layers.active if mesh.uv_layers else None
            uv_layer = uv_map.uv if uv_map is not None and hasattr(uv_map, "uv") else None
            uv_layer_legacy = uv_map.data if uv_map is not None and uv_layer is None else None

            mats = []
            for slot in mesh_obj.material_slots:
                mat = slot.material
                key = mat.as_pointer() if mat else 0
                if key not in material_cache:
                    material_cache[key] = translate_material(mat, texture_cache, warnings)
                mats.append(material_cache[key])
            if not mats:
                mats = [translate_material(None, texture_cache, warnings)]

            from .mesh_extract import mesh_arrays
            block = mesh_arrays(mesh, world, normal_matrix, mats, uv_map, engine.test_break)
            import hashlib
            rscene.topology_signature.append((obj.name, hashlib.sha256(block.indices.tobytes()+block.loops.tobytes()+block.material_indices.tobytes()).hexdigest()))
            if getattr(scene.retroman, "mesh_lighting", True):
                rscene.mesh_lights.extend(mesh_emitters(block, obj.name, engine.test_break))
            if packed_geometry:
                rscene.triangles.append(block)
            else:
                rscene.triangles.extend(block.triangles(engine.test_break))
        finally:
            if mesh is not None:
                try:
                    mesh_obj.to_mesh_clear()
                except Exception:
                    pass

    budget = max(0, int(getattr(scene.retroman, "max_mesh_lights", 16)))
    if len(rscene.mesh_lights) > budget:
        warnings.add(f"Mesh lighting uses the {budget} brightest patches of {len(rscene.mesh_lights)}; raise Max Mesh Lights to include more")
        rscene.mesh_lights.sort(key=lambda l:l.energy*l.area_factor*max(l.color), reverse=True)
        rscene.mesh_lights = rscene.mesh_lights[:budget]
    rscene.lights.extend(rscene.mesh_lights)
    if not rscene.lights and not rscene.environment_diffuse:
        warnings.add("No direct lights or diffuse environment lighting; enable Mesh Lighting/Environment Lighting or add Blender lights")
    if rscene.mesh_lights:
        warnings.add("Mesh lighting uses area-weighted patches, not exact emitter shapes; indirect surface bounces are not traced")
    return rscene, sorted(warnings)
