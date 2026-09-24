# SPDX-License-Identifier: GPL-3.0-or-later
import bpy


class RETROMAN_OT_gpu_self_test(bpy.types.Operator):
    bl_idname = "retroman.gpu_self_test"
    bl_label = "Run RetroMan GPU Self-Test"
    bl_description = "Compile RetroMan's geometry, environment and accumulation GPU shaders in Blender's active graphics context"

    def execute(self, context):
        try:
            from .gpu_renderer import _create_shader_set, _create_background_shader, _create_accum_shader, _create_safe_viewport_shader, gpu_info, make_barycentric_microgrid
            info = gpu_info()
            if info.get("backend") in {"NONE", "UNKNOWN"}:
                self.report({"ERROR"}, "Blender did not expose a hardware GPU context")
                return {"CANCELLED"}
            if tuple(getattr(bpy.app, "version", (0, 0, 0))) < (5, 0, 0):
                self.report({"ERROR"}, f"RetroMan requires Blender 5.0+; this is {bpy.app.version_string}")
                return {"CANCELLED"}
            max_textures = int(info.get("max_textures", 0) or 0)
            try:
                import gpu
                max_frag = int(gpu.capabilities.max_textures_frag_get())
            except Exception:
                max_frag = max_textures
            # RetroMan's highest numbered binding is 11 (12 combined slots).
            # The fragment stage uses 10 samplers in the richest base pass.
            if max_textures and max_textures < 12:
                self.report({"ERROR"}, f"GPU exposes only {max_textures} combined texture units; RetroMan needs 12")
                return {"CANCELLED"}
            if max_frag and max_frag < 10:
                self.report({"ERROR"}, f"GPU exposes only {max_frag} fragment texture units; RetroMan needs 10")
                return {"CANCELLED"}

            fast = _create_shader_set(reyes=False)
            reyes = _create_shader_set(reyes=True)
            background = _create_background_shader()
            accum = _create_accum_shader()
            safe = _create_safe_viewport_shader()
            grid = make_barycentric_microgrid(8)

            # Exercise the same texture/framebuffer primitives RetroMan uses for
            # final color and depth targets. Shader compilation alone can pass on
            # a driver that later rejects the framebuffer formats.
            import gpu
            old_viewport = gpu.state.viewport_get()
            color = gpu.types.GPUTexture((8, 8), format="RGBA16F")
            depth = gpu.types.GPUTexture((8, 8), format="DEPTH_COMPONENT32F")
            framebuffer = gpu.types.GPUFrameBuffer(depth_slot=depth, color_slots=(color,))
            try:
                with framebuffer.bind():
                    gpu.state.viewport_set(0, 0, 8, 8)
                    framebuffer.clear(color=(0.125, 0.25, 0.5, 1.0), depth=1.0)
            finally:
                try:
                    gpu.state.viewport_set(*old_viewport)
                except Exception:
                    pass

            # Exercise actual uniform binding, draw submission and shaped readback.
            _render_smoke_test(context)
            _bucket_smoke_test()

            # Drop references immediately; Blender owns the GPU resource lifetime.
            del fast, reyes, background, accum, safe, framebuffer, color, depth
            if len(grid) != 8 * 8 * 3:
                raise RuntimeError("micropolygon grid generator failed its 8x8 invariant")
            self.report(
                {"INFO"},
                f"RetroMan GPU self-test passed: {info.get('renderer', 'GPU')} / {info.get('backend', '?')}",
            )
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, f"RetroMan GPU self-test failed: {exc}")
            return {"CANCELLED"}


def _render_smoke_test(context):
    """Small independent scene: no edits to the user's objects or settings."""
    import math
    import gpu
    from types import SimpleNamespace
    from mathutils import Matrix, Vector
    from .model import RenderScene, CameraData, MaterialData, VertexData, TriangleData, LightData
    from .gpu_renderer import GPURenderer, GPUAccumulator
    source = context.scene.retroman
    values = {prop.identifier: getattr(source, prop.identifier)
              for prop in source.bl_rna.properties if prop.identifier != "rna_type"}
    settings = SimpleNamespace(**values)
    settings.exposure = 0.0
    settings.use_world_color = True
    settings.reflection_mode = "OFF"
    settings.enable_shadows = True
    settings.shadow_resolution = "64"
    settings.max_shadow_lights = 1
    settings.gpu_max_subdiv = 2
    settings.adaptive_dicing = False
    settings.auxiliary_min_subdiv = 1
    material = MaterialData("Self-test", emission_color=(0.2, 0.1, 0.05), emission_strength=1.0)
    normal = Vector((0, 0, 1))
    vertices = [VertexData(Vector(pos), normal.copy())
                for pos in [(-0.8, -0.8, 0), (0.8, -0.8, 0), (0, 0.8, 0)]]
    camera = CameraData(Matrix.Translation((0, 0, 2)), Matrix.Identity(4), 0.01, 100)
    scene = RenderScene(16, 16, camera, settings=settings)
    scene.triangles = [TriangleData(*vertices, material)]
    scene.lights = [LightData("Self-test Sun", "SUN", Matrix.Identity(4), (1, 1, 1), 1, True)]
    viewport = gpu.state.viewport_get()
    try:
        for mode in ("TRIANGLES", "MICROPOLYGON"):
            settings.geometry_mode = mode
            renderer = GPURenderer(scene)
            renderer.render_to_texture(view_projection=Matrix.Identity(4), camera_pos=Vector((0, 0, 2)))
            pixels = renderer.read_pixels()
            if len(pixels) != 256 or not all(math.isfinite(v) for pixel in pixels for v in pixel):
                raise RuntimeError(f"{mode}: invalid readback")
            if max(pixel[0] for pixel in pixels) <= 0.1:
                raise RuntimeError(f"{mode}: triangle did not appear")
            accumulator = GPUAccumulator(16, 16)
            accumulator.add(renderer.color_texture, 0.25)
            accumulator.add(renderer.color_texture, 0.75)
            accumulated = accumulator.read_pixels()
            if any(abs(a-b) > 0.01 for pa, pb in zip(pixels, accumulated) for a, b in zip(pa, pb)):
                raise RuntimeError(f"{mode}: accumulation mismatch")
            # Verify actual scene-light response, not just emission/ambient geometry.
            settings.enable_shadows = False
            light = scene.lights[0]
            light.use_shadow = False
            light.matrix_world = Matrix.Translation((0, 0, 2))
            for light_type in ("SUN", "POINT", "SPOT", "AREA"):
                light.type = light_type
                light.color = (1.0, 0.0, 0.0)
                energies = (0.0, 1.0, 2.0) if light_type == "SUN" else (0.0, 1000.0, 2000.0)
                images = []
                for energy in energies:
                    light.energy = energy
                    renderer.render_to_texture(view_projection=Matrix.Identity(4),
                        camera_pos=Vector((0, 0, 2)), rebuild_shadows=False, rebuild_reflections=False)
                    images.append(renderer.read_pixels())
                delta = sum(b[0]-a[0] for a,b in zip(images[0],images[1]))
                double_delta = sum(b[0]-a[0] for a,b in zip(images[0],images[2]))
                if delta <= 0.01 or abs(double_delta-2*delta) > max(0.05, delta*0.05):
                    raise RuntimeError(f"{mode}/{light_type}: light power response failed ({delta}, {double_delta})")
                if max(abs(a[3]-b[3]) for a,b in zip(images[0],images[2])) > 0.01:
                    raise RuntimeError(f"{mode}/{light_type}: lighting changed coverage alpha")
                print(f"RetroMan self-test {mode}/{light_type}: power response passed", flush=True)
            # Restore baseline light for the next geometry pipeline.
            light.type = "SUN"
            light.energy = 1.0
            light.color = (1.0, 1.0, 1.0)
            light.matrix_world = Matrix.Identity(4)
            light.use_shadow = True
            settings.enable_shadows = True
    finally:
        gpu.state.viewport_set(*viewport)


def _bucket_smoke_test():
    """Exercise compute atomics, sorting and readback without changing the scene."""
    import gpu
    import numpy as np
    from mathutils import Matrix
    from .bucket_gpu import Bucket, GPUWorkspace
    workspace=GPUWorkspace()
    arrays=[((1,0,0,.5),),((-1,-1,0,1),),((1,-1,0,1),),((0,1,0,1),)]
    textures=[gpu.types.GPUTexture((1,1),format='RGBA32F',data=gpu.types.Buffer('FLOAT',4,np.asarray(v,dtype=np.float32).reshape(-1))) for v in arrays]
    atlas=(textures,1,1)
    with Bucket(0,0,4,4,4,4,1,pixel_filter='BOX',workspace=workspace,vp=Matrix.Identity(4),vp_end=Matrix.Identity(4)) as bucket:
        bucket.push(atlas,atlas);image=bucket.finish(np.zeros((4,4,4),np.float32))
    np.testing.assert_allclose(image[1,1],(1,0,0,.5),atol=1e-6)
