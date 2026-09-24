# SPDX-License-Identifier: GPL-3.0-or-later
import bpy


class RETROMAN_OT_activate_viewport(bpy.types.Operator):
    bl_idname = "retroman.activate_viewport"
    bl_label = "Use RetroMan Rendered View"
    bl_description = "Select RetroMan and switch this screen's 3D views to Rendered shading"

    def execute(self, context):
        areas = [a for a in context.screen.areas if a.type == "VIEW_3D"] if context.screen else []
        if not areas:
            self.report({"WARNING"}, "Open a 3D View on this screen first")
            return {"CANCELLED"}
        context.scene.render.engine = "RETROMAN"
        for area in areas:
            area.spaces.active.shading.type = "RENDERED"
            area.tag_redraw()
        self.report({"INFO"}, "RetroMan Rendered view active; check the RETROMAN label in the viewport")
        return {"FINISHED"}


class _RetroManPanel:
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    COMPAT_ENGINES = {"RETROMAN"}

    @classmethod
    def poll(cls, context):
        return context.scene and context.scene.render.engine == "RETROMAN"


def controls(panel):
    layout = panel.layout
    layout.use_property_split = True
    layout.use_property_decorate = False
    return layout.column(align=True)


class RETROMAN_PT_main(_RetroManPanel, bpy.types.Panel):
    bl_label = "RetroMan"

    def draw(self, context):
        col = controls(self)
        col.prop(context.scene.retroman, "render_device", text="Device")
        col.operator("retroman.activate_viewport", icon="SHADING_RENDERED")


class RETROMAN_PT_sampling(_RetroManPanel, bpy.types.Panel):
    bl_label = "Sampling"

    def draw(self, context):
        controls(self).prop(context.scene.retroman, "quality", text="Quality Preset")


class RETROMAN_PT_performance(_RetroManPanel, bpy.types.Panel):
    bl_label = "Viewport"
    bl_parent_id = "RETROMAN_PT_sampling"

    def draw(self, context):
        s = context.scene.retroman
        col = controls(self)
        col.prop(s, "viewport_scale", text="Resolution")
        if s.viewport_scale == "AUTO_60":
            col.prop(s, "viewport_target_fps", text="Target FPS")
        col.prop(s, "viewport_subdiv", text="Max Subdivision")
        col.prop(s, "viewport_shadows", text="Shadows")
        sub = col.column(align=True)
        sub.enabled = s.viewport_shadows and s.enable_shadows
        sub.prop(s, "viewport_shadow_resolution", text="Shadow Resolution")
        col.prop(s, "viewport_reflections", text="Reflection Maps")
        sub = col.column(align=True)
        sub.enabled = s.viewport_reflections and s.reflection_mode in {"AUTO", "PROBE"}
        sub.prop(s, "viewport_reflection_resolution", text="Map Resolution")


class RETROMAN_PT_render_sampling(_RetroManPanel, bpy.types.Panel):
    bl_label = "Render"
    bl_parent_id = "RETROMAN_PT_sampling"

    def draw(self, context):
        col = controls(self)
        s = context.scene.retroman
        col.prop(s, "pixel_samples", text="Pixel Samples")
        col.prop(s, "temporal_samples", text="Motion / Lens Samples")
        col.prop(s, "pixel_filter", text="Pixel Filter")


class RETROMAN_PT_reyes(_RetroManPanel, bpy.types.Panel):
    bl_label = "Geometry"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.retroman
        col = controls(self)
        col.prop(s, "geometry_mode", text="Method")
        col.prop(s, "adaptive_dicing")
        sub = col.column(align=True)
        sub.enabled = s.adaptive_dicing
        sub.prop(s, "shading_rate")
        col.prop(s, "gpu_max_subdiv", text="Render Max Subdivision")
        col.prop(s, "bucket_size")
        col.prop(s, "displacement_enabled", text="Displacement")
        col.prop(s, "auxiliary_min_subdiv", text="Map Min Subdivision")


class RETROMAN_PT_lighting(_RetroManPanel, bpy.types.Panel):
    bl_label = "Lighting"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        col = controls(self)
        s = context.scene.retroman
        col.prop(s, "mesh_lighting")
        row = col.row()
        row.enabled = s.mesh_lighting
        row.prop(s, "max_mesh_lights")
        col.prop(s, "environment_lighting")
        col.prop(s, "ambient")
        col.prop(s, "exposure")
        col.prop(s, "use_world_color")


class RETROMAN_PT_shadow(_RetroManPanel, bpy.types.Panel):
    bl_label = "Shadows"
    bl_parent_id = "RETROMAN_PT_lighting"

    def draw_header(self, context):
        self.layout.prop(context.scene.retroman, "enable_shadows", text="")

    def draw(self, context):
        s = context.scene.retroman
        col = controls(self)
        col.enabled = s.enable_shadows
        col.prop(s, "shadow_resolution", text="Render Resolution")
        col.prop(s, "shadow_bias", text="Bias")
        col.prop(s, "shadow_softness", text="Softness")
        col.prop(s, "max_shadow_lights", text="Max Shadow Lights")
        col.prop(s, "displacement_shadow_maps")


class RETROMAN_PT_reflections(_RetroManPanel, bpy.types.Panel):
    bl_label = "Reflections"
    bl_parent_id = "RETROMAN_PT_lighting"

    def draw(self, context):
        s = context.scene.retroman
        col = controls(self)
        col.prop(s, "reflection_mode", text="Method")
        sub = col.column(align=True)
        sub.enabled = s.reflection_mode != "OFF"
        sub.prop(s, "reflection_strength", text="Strength")
        if s.reflection_mode in {"AUTO", "PROBE"}:
            sub.prop(s, "reflection_resolution", text="Render Resolution")


class RETROMAN_PT_camera_effects(_RetroManPanel, bpy.types.Panel):
    bl_label = "Motion Blur"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.prop(context.scene.render, "use_motion_blur", text="")

    def draw(self, context):
        col = controls(self)
        col.enabled = context.scene.render.use_motion_blur
        col.prop(context.scene.render, "motion_blur_shutter", text="Shutter")
        col.prop(context.scene.render, "motion_blur_position", text="Position")


class RETROMAN_PT_dof(_RetroManPanel, bpy.types.Panel):
    bl_label = "Depth of Field"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        col = controls(self)
        camera = context.scene.camera
        if camera is None:
            col.label(text="Set an active camera.", icon="INFO")
            return
        dof = camera.data.dof
        col.prop(dof, "use_dof", text="Enable")
        sub = col.column(align=True)
        sub.enabled = dof.use_dof
        sub.prop(dof, "focus_object")
        if dof.focus_object is None:
            sub.prop(dof, "focus_distance")
        sub.prop(dof, "aperture_fstop")
        sub.prop(dof, "aperture_blades")
        sub.prop(dof, "aperture_ratio")
        sub.prop(dof, "aperture_rotation")


class RETROMAN_PT_translation(_RetroManPanel, bpy.types.Panel):
    bl_label = "Shading"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.retroman
        col = controls(self)
        col.prop(s, "texture_filter")
        col.prop(s, "clamp_fireflies")
        if s.clamp_fireflies:
            col.prop(s, "max_specular")
        col.prop(s, "show_translation_warnings")


class RETROMAN_PT_cpu(_RetroManPanel, bpy.types.Panel):
    bl_label = "CPU Reference"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return super().poll(context) and context.scene.retroman.render_device == "CPU"

    def draw(self, context):
        controls(self).prop(context.scene.retroman, "max_dice_depth")


class RETROMAN_PT_rib(_RetroManPanel, bpy.types.Panel):
    bl_label = "Export"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        controls(self).operator("retroman.export_rib", icon="EXPORT")


class RETROMAN_PT_diagnostics(_RetroManPanel, bpy.types.Panel):
    bl_label = "System & Diagnostics"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        col = controls(self)
        col.prop(context.preferences.system, "gpu_backend", text="Graphics Backend")
        col.label(text="Restart Blender after changing backend.", icon="INFO")
        from .gpu_renderer import gpu_info
        info = gpu_info()
        col.label(text="RetroMan 1.0.24")
        col.label(text=f"Backend: {info.get('backend', 'UNKNOWN')}")
        col.label(text=f"GPU: {info.get('renderer', 'Unavailable')}")
        col.operator("retroman.gpu_self_test", icon="CHECKMARK")


class RETROMAN_PT_material(bpy.types.Panel):
    bl_label = "RetroMan 1995 Shader"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"
    COMPAT_ENGINES = {"RETROMAN"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and context.scene.render.engine == "RETROMAN" and context.material)

    def draw(self, context):
        m = context.material.retroman
        col = self.layout.column(align=True)
        col.label(text="Normal Blender nodes remain the default workflow.")
        col.prop(m, "retrosl_enabled")
        sub = col.column(align=True)
        sub.enabled = m.retrosl_enabled
        sub.prop(m, "retrosl_text")
        box = col.box()
        box.label(text="Optional RetroSL parameters:")
        box.label(text="Cs, Ka, Kd, Ks, Kr, roughness, opacity")


class RETROMAN_PT_film(_RetroManPanel, bpy.types.Panel):
    bl_label = "Film"

    def draw(self, context):
        controls(self).prop(context.scene.render, "film_transparent", text="Transparent")
