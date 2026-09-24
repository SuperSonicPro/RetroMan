# SPDX-License-Identifier: GPL-3.0-or-later
import bpy
import math
import time
import traceback
from mathutils import Vector

from .scene_adapter import extract_scene, extract_lights, viewport_camera
from .raster import build_shadow_maps, render_scene
from .runtime import RenderCancelled, checkpoint


class RETROMAN_RenderEngine(bpy.types.RenderEngine):
    bl_idname = "RETROMAN"
    bl_label = "RetroMan"
    bl_use_preview = False
    bl_use_postprocess = True
    bl_use_shading_nodes_custom = False
    # A worker owns the final GPU context; holding it here blocks UI drawing.
    bl_use_gpu_context = False
    bl_use_eevee_viewport = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warnings = []
        self._active_result = None
        self._stage_started = time.perf_counter()
        self._viewport_renderer = None
        self._viewport_scene = None
        self._viewport_needs_shadow = True
        self._viewport_needs_reflection = True
        self._viewport_error = None
        self._viewport_auto_scale = 1.0
        self._viewport_frame_ema = None
        self._viewport_frame_count = 0
        self._viewport_auto_subdiv = None
        self._viewport_safe_mode = False
        self._viewport_traceback = None
        self._viewport_gpu_rebuild = True
        self._viewport_light_dirty = False
        self._viewport_force_update = False
        self._viewport_settings_signature = None
        self._viewport_draw_key = None

    def __del__(self):
        self._viewport_renderer = None
        self._viewport_scene = None
        try:
            super().__del__()
        except Exception:
            pass

    def _status(self, message):
        elapsed = time.perf_counter() - self._stage_started
        self.update_stats("RetroMan 1.0.24", f"{elapsed:.1f}s | {message} | Esc to cancel")
        print(f"[RetroMan 1.0.24] {elapsed:.1f}s {message}", flush=True)

    def _publish_pixels(self, pixels):
        checkpoint(self.test_break)
        if self._active_result is not None:
            if hasattr(pixels,"tolist"):pixels=pixels.tolist()
            self._active_result.layers[0].passes["Combined"].rect = pixels

    def _publish_bucket(self,x,y,pixels):
        checkpoint(self.test_break)
        height,width=pixels.shape[:2]
        result=self.begin_result(x,y,width,height)
        if result is None:raise RuntimeError("Blender could not allocate a bucket result")
        try:
            result.layers[0].passes["Combined"].rect=pixels.reshape(-1,4).tolist()
        finally:
            self.end_result(result,cancel=self.test_break())

    def _report_translation_warnings(self, warnings, settings):
        if warnings and settings.show_translation_warnings:
            for warning in warnings[:20]:
                self.report({"WARNING"}, warning)

    def _render_cpu(self, rscene):
        self.update_stats("RetroMan CPU", "Baking automatic shadow maps")
        build_shadow_maps(
            rscene,
            progress=lambda p: self.update_progress(0.15 + p * 0.35),
            cancel=self.test_break,
        )
        if self.test_break():
            return None
        self.update_stats("RetroMan CPU", "Reference dicing, shading and sampling")
        return render_scene(
            rscene,
            progress=lambda p: self.update_progress(0.50 + p * 0.49),
            cancel=self.test_break,
        )

    def _render_gpu(self, rscene):
        from .gpu_renderer import GPURenderer, gpu_info

        info = gpu_info()
        name = info.get("renderer", "GPU")
        self.update_stats("RetroMan GPU", f"Uploading evaluated Blender scene to {name}")
        renderer = GPURenderer(rscene, viewport=False, cancel=self.test_break, status=self._status)
        if self.test_break():
            return None, renderer

        self.update_stats("RetroMan GPU", f"{renderer.geometry_label}: maps, shading and rasterization")
        renderer.render_to_texture(
            progress=lambda p: self.update_progress(0.15 + p * 0.80),
            cancel=self.test_break,
            rebuild_shadows=True,
            rebuild_reflections=True,
        )
        if self.test_break():
            return None, renderer
        self.update_stats("RetroMan GPU", "Reading final image into Blender")
        pixels = renderer.read_pixels(cancel=self.test_break, progress=lambda p: self.update_progress(0.95 + 0.04 * p))
        self.update_progress(0.99)
        return pixels, renderer

    def _render_gpu_effects(self, depsgraph, initial_scene, initial_warnings):
        """Keep final GPU context ownership outside Blender's UI process."""
        from .render_worker import render
        return render(self,depsgraph,initial_scene,initial_warnings)

    def _render_gpu_effects_legacy(self,depsgraph,initial_scene,initial_warnings):
        from .gpu_renderer import GPURenderer, GPUAccumulator, gpu_info
        from .sampling import camera_sample, pixel_jitter, reconstruction_weight, shutter_offsets

        scene = depsgraph.scene
        settings = scene.retroman
        view_layer = getattr(depsgraph, "view_layer", None)
        motion = bool(getattr(scene.render, "use_motion_blur", False))
        if view_layer is not None:
            motion = motion and bool(getattr(view_layer, "use_motion_blur", True))
        dof = bool(getattr(initial_scene.camera, "dof_enabled", False))
        pixel_axis = max(1, int(getattr(settings, "pixel_samples", 2)))
        aa_samples = pixel_axis * pixel_axis
        temporal = max(1, int(getattr(settings, "temporal_samples", 4))) if (motion or dof) else 1
        sample_count = max(aa_samples, temporal)

        if sample_count <= 1:
            pixels, renderer = self._render_gpu(initial_scene)
            return pixels, renderer, initial_scene, initial_warnings, 1

        info = gpu_info()
        self.update_stats("RetroMan GPU", f"{sample_count} unified stochastic samples on {info.get('renderer', 'GPU')}")
        accumulator = GPUAccumulator(initial_scene.width, initial_scene.height)
        offsets = shutter_offsets(scene.render, sample_count) if motion else [0.0] * sample_count
        jitters = [pixel_jitter(i, sample_count) for i in range(sample_count)]
        raw_weights = [reconstruction_weight(j, getattr(settings, "pixel_filter", "GAUSSIAN")) for j in jitters]
        wsum = max(1.0e-12, sum(raw_weights))
        weights = [w / wsum for w in raw_weights]
        base_time = float(scene.frame_current) + float(getattr(scene, "frame_subframe", 0.0))
        all_warnings = set(initial_warnings)
        last_renderer = None
        last_scene = initial_scene
        shared_reflection = None

        try:
            static_renderer = None
            if not motion:
                static_renderer = GPURenderer(initial_scene, viewport=False, cancel=self.test_break, status=self._status)

            for i in range(sample_count):
                if self.test_break():
                    return None, last_renderer or static_renderer, last_scene, sorted(all_warnings), sample_count

                if motion:
                    sample_time = base_time + offsets[i]
                    sample_frame = math.floor(sample_time)
                    sample_subframe = sample_time - sample_frame
                    # RenderEngine.frame_set() is Blender's render-engine API for
                    # evaluating another time sample (notably motion blur).  It
                    # keeps evaluation owned by the render engine rather than
                    # mutating Scene.frame_set() behind Blender's back.
                    self.frame_set(int(sample_frame), float(sample_subframe))
                    self.update_stats("RetroMan GPU", f"Evaluating shutter sample {i + 1}/{sample_count}")
                    rscene, warnings = extract_scene(
                        depsgraph, self,
                        progress_start=0.16 + 0.78 * (i / sample_count),
                        progress_end=0.16 + 0.78 * (i / sample_count),
                    )
                    all_warnings.update(warnings)
                    renderer = GPURenderer(rscene, viewport=False, cancel=self.test_break, status=self._status)
                    if shared_reflection is not None:
                        renderer.reflection_map = shared_reflection
                else:
                    rscene = initial_scene
                    renderer = static_renderer

                vp, camera_pos, _ = camera_sample(
                    rscene.camera, i, sample_count, rscene.width, rscene.height,
                    use_dof=dof, use_aa=(aa_samples > 1),
                )
                labels = ["AA"]
                if motion: labels.append("motion")
                if dof: labels.append("DOF")
                self.update_stats("RetroMan GPU", f"Sample {i + 1}/{sample_count}: {' + '.join(labels)}")
                renderer.render_to_texture(
                    view_projection=vp,
                    camera_pos=camera_pos,
                    progress=lambda p, ii=i: self.update_progress(0.16 + 0.78 * ((ii + p) / sample_count)),
                    cancel=self.test_break,
                    rebuild_shadows=(motion or i == 0),
                    rebuild_reflections=(i == 0),
                )
                if i == 0:
                    shared_reflection = renderer.reflection_map
                checkpoint(self.test_break)
                if i == 0:
                    self._status("Publishing first sample preview")
                    self._publish_pixels(renderer.read_pixels(cancel=self.test_break))
                accumulator.add(renderer.color_texture, weights[i])
                last_renderer = renderer
                last_scene = rscene

            self.update_stats("RetroMan GPU", "Reading reconstructed stochastic result")
            pixels = accumulator.read_pixels(cancel=self.test_break, progress=lambda p: self.update_progress(0.96 + 0.03 * p))
            self.update_progress(0.99)
            return pixels, last_renderer, last_scene, sorted(all_warnings), sample_count
        finally:
            if motion:
                restore_frame = math.floor(base_time)
                restore_subframe = base_time - restore_frame
                self.frame_set(int(restore_frame), float(restore_subframe))

    def render(self, depsgraph):
        self._stage_started = time.perf_counter()
        self._active_result = None
        completed = False
        self.update_progress(0.0)
        self._status("Starting final render")
        try:
            self.update_stats("RetroMan", "Translating evaluated Blender scene")
            rscene, warnings = extract_scene(depsgraph, self)
            self._warnings = warnings
            self._viewport_settings_signature = self._viewport_settings_key(depsgraph.scene.retroman)
            if self.test_break():
                return
            self._report_translation_warnings(warnings, rscene.settings)

            # GPU buckets own separate, short-lived results. Do not keep a
            # full-frame result open while the main window draws those tiles.
            if rscene.settings.render_device == 'CPU':
                self._active_result = self.begin_result(0, 0, rscene.width, rscene.height)
                if self._active_result is None:raise RuntimeError("Blender could not allocate Render Result")
            settings = rscene.settings
            mode = settings.render_device
            pixels = None
            backend_label = "CPU"
            gpu_renderer = None
            sample_count = 1

            if mode != "CPU":
                from .gpu_renderer import gpu_available, gpu_info
                available = gpu_available()
                if not available:
                    info = gpu_info()
                    raise RuntimeError(
                        f"GPU mode was requested but Blender reports no hardware GPU "
                        f"(backend={info.get('backend')}, device={info.get('device_type')})."
                    )
                if available:
                    try:
                        pixels, gpu_renderer, rscene, warnings, sample_count = self._render_gpu_effects(depsgraph, rscene, warnings)
                        backend_label = "GPU bucket shading and sampling" if hasattr(gpu_renderer,"bucket_statistics") else "GPU"
                        self._warnings = warnings
                        for warning in getattr(gpu_renderer, "shadow_warnings", []):
                            self.report({"WARNING"}, warning)
                        for warning in getattr(gpu_renderer, "reflection_warnings", []):
                            self.report({"WARNING"}, warning)
                        for warning in getattr(gpu_renderer, "resource_warnings", []):
                            self.report({"WARNING"}, warning)
                    except RenderCancelled:
                        raise
                    except Exception as exc:
                        raise RuntimeError(
                            f"GPU rendering failed: {exc}. Run GPU Self-Test; "
                            "CPU Reference is available only by explicitly selecting that device."
                        ) from exc

            if mode == "CPU" and not self.test_break():
                if getattr(depsgraph.scene.render, "use_motion_blur", False):
                    self.report({"WARNING"}, "CPU reference renderer does not implement multi-sampled motion blur")
                if getattr(rscene.camera, "dof_enabled", False):
                    self.report({"WARNING"}, "CPU reference renderer does not implement lens depth of field")
                if any(t.material.has_displacement for t in rscene.triangles):
                    self.report({"WARNING"}, "RetroMan CPU reference renderer does not apply Blender displacement; use GPU mode for diced displacement")
                if any(getattr(t.material, "normal_texture", None) is not None or getattr(t.material, "bump_texture", None) is not None for t in rscene.triangles):
                    self.report({"WARNING"}, "CPU reference renderer ignores image normal/bump maps; GPU mode is the complete 1.0 path")
                if int(getattr(rscene.settings, "pixel_samples", 1)) > 1:
                    self.report({"WARNING"}, "CPU reference renderer renders one pixel sample; GPU mode provides RetroMan stochastic AA")
                pixels = self._render_cpu(rscene)
                backend_label = "CPU"

            if pixels is None or self.test_break():
                return

            self._status("Publishing final image")
            if self._active_result is None:
                self._active_result=self.begin_result(0,0,rscene.width,rscene.height)
                if self._active_result is None:raise RuntimeError("Blender could not allocate final result")
            self._publish_pixels(pixels)
            completed = True
            self.update_progress(1.0)
            self.update_stats(
                "RetroMan",
                f"Done on {backend_label} - {len(rscene.triangles):,} source triangles, "
                f"{getattr(gpu_renderer, 'micropolygon_count', len(rscene.triangles)):,} raster primitives, "
                f"{len(rscene.lights)} lights, {sample_count} sample{'s' if sample_count != 1 else ''}",
            )
        except RenderCancelled:
            self._status("Cancelled")
        except Exception as exc:
            self._status(f"FAILED: {exc}")
            print(traceback.format_exc(), flush=True)
            self.error_set(f"RetroMan: {exc}")
            self.report({"ERROR"}, f"RetroMan render failed: {exc}")
        finally:
            if self._active_result is not None:
                result, self._active_result = self._active_result, None
                self.end_result(result, cancel=not completed)
            if not completed and self.test_break():
                self._status("Cancelled")

    def _viewport_scale_value(self, settings):
        if settings.viewport_scale == "AUTO_60":
            return self._viewport_auto_scale
        try:
            return float(settings.viewport_scale)
        except Exception:
            return 1.0

    def _viewport_settings_key(self, settings):
        # Scene PropertyGroup edits do not consistently arrive as depsgraph ID
        # updates on every Blender/backend combination. Track the renderer knobs
        # explicitly and request a view_update when one changes.
        names = (
            "quality", "geometry_mode", "viewport_subdiv", "displacement_enabled",
            "mesh_lighting", "max_mesh_lights", "environment_lighting",
            "ambient", "exposure", "viewport_shadow_resolution", "max_shadow_lights",
            "shadow_bias", "shadow_softness", "enable_shadows", "viewport_shadows",
            "viewport_scale", "viewport_target_fps", "reflection_mode",
            "viewport_reflections", "viewport_reflection_resolution", "reflection_strength",
            "texture_filter", "auxiliary_min_subdiv", "use_world_color",
            "clamp_fireflies", "max_specular",
        )
        return tuple(getattr(settings, name, None) for name in names)

    # ----- Rendered viewport -------------------------------------------------
    # Geometry/material extraction happens only when the dependency graph says
    # scene data changed. Camera orbit/pan/zoom therefore stays GPU-resident.
    def view_update(self, context, depsgraph):
        """Translate Blender data only; defer every GPU allocation to view_draw().

        Blender's viewport callback contract calls view_update for data changes
        and view_draw for actual GPU drawing.  Creating shaders/framebuffers here
        is unreliable on some backends because this callback is not the GPU draw
        phase.
        """
        try:
            forced = bool(self._viewport_force_update)
            self._viewport_force_update = False
            changed = self._viewport_scene is None or forced
            updates = []
            if not changed:
                try:
                    updates = list(depsgraph.updates)
                    changed = bool(updates)
                except Exception:
                    changed = True

            if not changed:
                return

            if updates and not forced and all(
                isinstance(u.id, bpy.types.Camera) or
                (isinstance(u.id, bpy.types.Object) and u.id.type == "CAMERA")
                for u in updates
            ):
                # Geometry driven by a camera also emits non-camera ID updates.
                return

            # Light-only edits can keep all uploaded geometry.  GPU shadow cache
            # invalidation is deliberately postponed to view_draw(), where a GPU
            # context is active.
            if (not forced) and self._viewport_scene is not None and updates:
                ids = [getattr(u, "id", None) for u in updates]
                lightish = True
                for datablock in ids:
                    if datablock is None:
                        lightish = False
                        break
                    is_light_data = isinstance(datablock, bpy.types.Light)
                    is_light_obj = isinstance(datablock, bpy.types.Object) and getattr(datablock, "type", "") == "LIGHT"
                    lightish = lightish and (is_light_data or is_light_obj)

                if lightish:
                    self._viewport_scene.lights = extract_lights(depsgraph, viewport=True) + self._viewport_scene.mesh_lights
                    self._viewport_light_dirty = True
                    self._viewport_needs_shadow = True
                    self._viewport_needs_reflection = True
                    return

            region = context.region
            scale = self._viewport_scale_value(depsgraph.scene.retroman)
            width = max(1, int(region.width * scale))
            height = max(1, int(region.height * scale))
            cam = viewport_camera(context, width, height)
            rscene, warnings = extract_scene(
                depsgraph,
                self,
                width=width,
                height=height,
                camera_override=cam,
                progress_start=0.0,
                progress_end=0.0,
            )
            self._viewport_scene = rscene
            # Initialize/update the explicit settings signature from the viewport
            # build itself. Previously this was only initialized by F12, so a
            # viewport-only session could miss PropertyGroup-only changes.
            self._viewport_settings_signature = self._viewport_settings_key(depsgraph.scene.retroman)
            self._viewport_gpu_rebuild = True
            self._viewport_light_dirty = False
            self._viewport_safe_mode = False
            self._viewport_error = None
            self._viewport_traceback = None
            self._viewport_auto_subdiv = int(getattr(depsgraph.scene.retroman, "viewport_subdiv", 2))
            self._viewport_needs_shadow = True
            self._viewport_needs_reflection = True
            self._warnings = warnings
        except Exception as exc:
            self._viewport_scene = None
            self._viewport_gpu_rebuild = True
            self._viewport_safe_mode = False
            self._viewport_error = f"Viewport scene translation failed: {exc}"
            self._viewport_traceback = traceback.format_exc()
            print("RetroMan viewport scene translation failed:\n" + self._viewport_traceback)

    def _draw_viewport_message(self, context, text, *, error=False):
        try:
            import blf
            font_id = 0
            x = 18.0
            y = max(24.0, float(context.region.height) - 34.0)
            blf.size(font_id, 13.0)
            blf.color(font_id, 1.0, 0.35 if error else 0.85, 0.22 if error else 0.35, 1.0)
            blf.enable(font_id, blf.SHADOW)
            blf.shadow(font_id, 3, 0.0, 0.0, 0.0, 0.85)
            blf.shadow_offset(font_id, 1, -1)
            for line in str(text).splitlines()[:3]:
                blf.position(font_id, x, y, 0.0)
                blf.draw(font_id, line[:180])
                y -= 18.0
            blf.disable(font_id, blf.SHADOW)
        except Exception:
            pass

    def _activate_safe_viewport(self, reason, trace=None):
        if self._viewport_scene is None:
            return False
        try:
            from .gpu_renderer import GPUViewportSafeRenderer
            self._viewport_renderer = GPUViewportSafeRenderer(self._viewport_scene)
            self._viewport_safe_mode = True
            self._viewport_error = f"Compatibility viewport active: {reason}"
            self._viewport_traceback = trace or str(reason)
            self._viewport_needs_shadow = False
            self._viewport_needs_reflection = False
            self._viewport_gpu_rebuild = False
            print("RetroMan advanced viewport failed; compatibility viewport activated:\n" + self._viewport_traceback)
            return True
        except Exception as fallback_exc:
            self._viewport_renderer = None
            self._viewport_safe_mode = False
            self._viewport_error = f"RetroMan viewport failed: {reason}; fallback also failed: {fallback_exc}"
            self._viewport_traceback = traceback.format_exc()
            print("RetroMan compatibility viewport also failed:\n" + self._viewport_traceback)
            return False

    def view_draw(self, context, depsgraph):
        import gpu
        inherited_viewport = gpu.state.viewport_get()
        try:
            self._view_draw_impl(context, depsgraph)
        except Exception as exc:
            self._viewport_error = f"RetroMan presentation failed: {exc}"
            print(traceback.format_exc(), flush=True)
            self._draw_viewport_message(context, self._viewport_error, error=True)
        finally:
            gpu.state.viewport_set(*inherited_viewport)

    def _view_draw_impl(self, context, depsgraph):
        # GPU objects are created lazily here, not in view_update().  Blender's
        # own external-renderer example treats view_draw as the GPU phase.
        import gpu
        from gpu_extras.presets import draw_texture_2d

        current_settings_key = self._viewport_settings_key(depsgraph.scene.retroman)
        if self._viewport_settings_signature is not None and current_settings_key != self._viewport_settings_signature:
            self._viewport_settings_signature = current_settings_key
            self._viewport_force_update = True
            # Blender documents tag_update/tag_redraw for custom RenderEngines.
            # Keep drawing the previous valid texture for this callback; the
            # requested view_update will rebuild the scene/resources next.
            self.tag_update()
            self.tag_redraw()

        if self._viewport_scene is None:
            if self._viewport_error:
                self._draw_viewport_message(context, self._viewport_error, error=True)
            else:
                self._draw_viewport_message(context, "RetroMan is waiting for the first scene update")
            return

        if self._viewport_gpu_rebuild or self._viewport_renderer is None:
            try:
                from .gpu_renderer import GPURenderer, GPUViewportSafeRenderer, gpu_available
                if not gpu_available():
                    raise RuntimeError("Blender did not expose a hardware GPU viewport context")
                try:
                    candidate = GPURenderer(self._viewport_scene, viewport=True)
                    self._viewport_renderer = candidate
                    self._viewport_safe_mode = False
                    self._viewport_error = None
                    self._viewport_traceback = None
                    self._viewport_auto_subdiv = int(getattr(depsgraph.scene.retroman, "viewport_subdiv", 2))
                    candidate.set_viewport_subdiv(self._viewport_auto_subdiv)
                except Exception as exc:
                    trace = traceback.format_exc()
                    self._viewport_renderer = GPUViewportSafeRenderer(self._viewport_scene)
                    self._viewport_safe_mode = True
                    self._viewport_error = f"Compatibility viewport active: {exc}"
                    self._viewport_traceback = trace
                    print("RetroMan full viewport initialization failed; compatibility viewport activated:\n" + trace)
                self._viewport_gpu_rebuild = False
                self._viewport_light_dirty = False
                self._viewport_needs_shadow = True
                self._viewport_needs_reflection = True
            except Exception as exc:
                self._viewport_renderer = None
                self._viewport_safe_mode = False
                self._viewport_error = f"RetroMan GPU viewport initialization failed: {exc}"
                self._viewport_traceback = traceback.format_exc()
                print("RetroMan GPU viewport initialization failed:\n" + self._viewport_traceback)
                self._draw_viewport_message(context, self._viewport_error, error=True)
                return

        renderer = self._viewport_renderer
        if self._viewport_light_dirty and not self._viewport_safe_mode:
            # Clear GPU-owned map objects only while the viewport GPU context is active.
            renderer.rscene.lights = self._viewport_scene.lights
            renderer.shadow_cache.clear()
            self._viewport_light_dirty = False

        region = context.region
        r3d = context.region_data
        if r3d is None:
            self._draw_viewport_message(context, "RetroMan needs a 3D View region", error=True)
            return
        scale = self._viewport_scale_value(depsgraph.scene.retroman)
        width = max(1, int(region.width * scale))
        height = max(1, int(region.height * scale))
        renderer.resize(width, height)

        view_projection = r3d.perspective_matrix.copy()
        try:
            camera_pos = r3d.view_matrix.inverted().translation
        except Exception:
            camera_pos = Vector((0.0, 0.0, 0.0))

        t0 = time.perf_counter()
        # Framebuffer.bind() restores the framebuffer object, but Blender's GPU
        # API explicitly does not guarantee viewport state restoration across
        # framebuffer rebinds. Preserve the viewport Blender handed us so an
        # internally scaled RetroMan target cannot leak its dimensions into
        # overlays or the final texture blit.
        try:
            inherited_viewport = gpu.state.viewport_get()
        except Exception:
            inherited_viewport = (0, 0, int(region.width), int(region.height))
        draw_key = (id(renderer), width, height,
                    tuple(v for row in view_projection for v in row), tuple(camera_pos))
        rendered_frame = (self._viewport_draw_key != draw_key or
                          self._viewport_needs_shadow or self._viewport_needs_reflection)
        try:
            texture = renderer.color_texture
            if rendered_frame:
                texture = renderer.render_to_texture(
                    view_projection=view_projection,
                    camera_pos=camera_pos,
                    rebuild_shadows=(self._viewport_needs_shadow and not self._viewport_safe_mode),
                    rebuild_reflections=(self._viewport_needs_reflection and not self._viewport_safe_mode),
                )
            try:
                gpu.state.viewport_set(*inherited_viewport)
            except Exception:
                pass
        except Exception as exc:
            try:
                gpu.state.viewport_set(*inherited_viewport)
            except Exception:
                pass
            reason = str(exc)
            trace = traceback.format_exc()
            if not self._viewport_safe_mode and self._activate_safe_viewport(reason, trace):
                renderer = self._viewport_renderer
                renderer.resize(width, height)
                try:
                    texture = renderer.render_to_texture(
                        view_projection=view_projection,
                        camera_pos=camera_pos,
                        rebuild_shadows=False,
                        rebuild_reflections=False,
                    )
                    try:
                        gpu.state.viewport_set(*inherited_viewport)
                    except Exception:
                        pass
                except Exception as fallback_exc:
                    self._viewport_error = f"RetroMan compatibility viewport draw failed: {fallback_exc}"
                    self._viewport_traceback = traceback.format_exc()
                    print("RetroMan compatibility viewport draw failed:\n" + self._viewport_traceback)
                    self._draw_viewport_message(context, self._viewport_error, error=True)
                    return
            else:
                self._viewport_error = f"RetroMan viewport draw failed: {reason}"
                self._viewport_traceback = traceback.format_exc()
                print("RetroMan viewport draw failed:\n" + self._viewport_traceback)
                self._draw_viewport_message(context, self._viewport_error, error=True)
                return

        self._viewport_draw_key = draw_key
        self._viewport_needs_shadow = False
        self._viewport_needs_reflection = False

        display_bound = False
        # Preserve the viewport state Blender handed us instead of guessing what
        # overlays want after RetroMan returns. The getters are part of gpu.state
        # in Blender 5.x; keep defensive fallbacks for unusual backends.
        try:
            old_blend = gpu.state.blend_get()
        except Exception:
            old_blend = "NONE"
        try:
            old_depth_test = gpu.state.depth_test_get()
        except Exception:
            old_depth_test = "NONE"
        try:
            old_depth_mask = gpu.state.depth_mask_get()
        except Exception:
            old_depth_mask = True
        try:
            gpu.state.depth_test_set("NONE")
            gpu.state.depth_mask_set(False)
            gpu.state.face_culling_set("NONE")
            gpu.state.blend_set("ALPHA_PREMULT")
            if self.support_display_space_shader(depsgraph.scene):
                self.bind_display_space_shader(depsgraph.scene)
                display_bound = True
            draw_texture_2d(texture, (0, 0), region.width, region.height)
        finally:
            if display_bound:
                # Blender explicitly requires every bind to be paired with unbind.
                self.unbind_display_space_shader()
            gpu.state.blend_set(old_blend)
            gpu.state.depth_test_set(old_depth_test)
            gpu.state.depth_mask_set(old_depth_mask)
            # There is no portable face-culling getter in Blender's Python GPU
            # API, so leave the conventional viewport-safe mode explicitly.
            gpu.state.face_culling_set("NONE")

        elapsed = max(1.0e-6, time.perf_counter() - t0)
        self._viewport_frame_count += int(rendered_frame)
        if self._viewport_frame_ema is None:
            self._viewport_frame_ema = elapsed
        elif rendered_frame:
            self._viewport_frame_ema = self._viewport_frame_ema * 0.90 + elapsed * 0.10
        fps = 1.0 / max(1.0e-6, self._viewport_frame_ema)

        settings = depsgraph.scene.retroman
        if rendered_frame and not self._viewport_safe_mode and settings.viewport_scale == "AUTO_60" and (self._viewport_frame_count > 8 or elapsed > 0.15):
            target = float(settings.viewport_target_fps)
            levels = [0.33, 0.5, 0.75, 1.0]
            idx = min(range(len(levels)), key=lambda i: abs(levels[i] - self._viewport_auto_scale))
            wanted_subdiv = max(1, int(getattr(settings, "viewport_subdiv", 2)))
            current_subdiv = int(getattr(renderer, "current_subdiv", 1))
            if fps < target * 0.82:
                if renderer.use_reyes and current_subdiv > 1:
                    new_subdiv = max(1, current_subdiv // 2)
                    renderer.set_viewport_subdiv(new_subdiv)
                    self._viewport_auto_subdiv = new_subdiv
                    self._viewport_needs_shadow = True
                    self._viewport_needs_reflection = True
                    self._viewport_frame_count = 0
                    self._viewport_frame_ema = None
                    self.tag_redraw()
                elif idx > 0:
                    self._viewport_auto_scale = levels[idx - 1]
                    self._viewport_frame_count = 0
                    self._viewport_frame_ema = None
                    self.tag_redraw()
            elif fps > target * 1.35:
                if idx < len(levels) - 1:
                    self._viewport_auto_scale = levels[idx + 1]
                    self._viewport_frame_count = 0
                    self._viewport_frame_ema = None
                    self.tag_redraw()
                elif renderer.use_reyes and current_subdiv < wanted_subdiv:
                    new_subdiv = min(wanted_subdiv, current_subdiv * 2)
                    renderer.set_viewport_subdiv(new_subdiv)
                    self._viewport_auto_subdiv = new_subdiv
                    self._viewport_needs_shadow = True
                    self._viewport_needs_reflection = True
                    self._viewport_frame_count = 0
                    self._viewport_frame_ema = None
                    self.tag_redraw()

        if self._viewport_safe_mode and self._viewport_error:
            self._draw_viewport_message(context, "RETROMAN 1.0.24 — DEGRADED DIAGNOSTIC VIEW\n" + self._viewport_error, error=True)
        else:
            self._draw_viewport_message(context,
                f"RETROMAN 1.0.24 | {renderer.geometry_label} | {renderer.micropolygon_count:,} primitives\n"
                f"Scene lights: {len(renderer.rscene.lights)} | {int(scale * 100)}% resolution")

        self.update_stats(
            "RetroMan GPU Viewport",
            f"{1000.0 / fps:.1f} ms submission | {renderer.geometry_label} | {renderer.micropolygon_count:,} prims | "
            f"{len(renderer.rscene.lights)} lights | {int(scale * 100)}%" +
            (" | COMPATIBILITY MODE" if self._viewport_safe_mode else ""),
        )
