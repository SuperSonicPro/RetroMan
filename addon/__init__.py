# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "RetroMan",
    "author": "RetroMan Project",
    "version": (1, 0, 24),
    "blender": (5, 0, 0),
    "location": "Render Properties > Render Engine > RetroMan",
    "description": "1995-style REYES renderer for modern Blender workflows",
    "category": "Render",
}

import bpy
from bpy.props import PointerProperty

from .engine import RETROMAN_RenderEngine
from .diagnostics import RETROMAN_OT_gpu_self_test
from .rib_export import RETROMAN_OT_export_rib
from .properties import RetroManSceneSettings, RetroManMaterialSettings
from .texture_presets import (RetroManTextureSettings, RETROMAN_OT_apply_pixar_texture,
    RETROMAN_OT_pixar_credits, RETROMAN_PT_pixar_textures, clear_previews)
from .ui import (
    RETROMAN_OT_activate_viewport,
    RETROMAN_PT_main,
    RETROMAN_PT_film,
    RETROMAN_PT_sampling,
    RETROMAN_PT_render_sampling,
    RETROMAN_PT_lighting,
    RETROMAN_PT_dof,
    RETROMAN_PT_camera_effects,
    RETROMAN_PT_performance,
    RETROMAN_PT_shadow,
    RETROMAN_PT_reflections,
    RETROMAN_PT_reyes,
    RETROMAN_PT_translation,
    RETROMAN_PT_rib,
    RETROMAN_PT_cpu,
    RETROMAN_PT_diagnostics,
    RETROMAN_PT_material,
)

_CLASSES = (
    RetroManTextureSettings, RETROMAN_OT_apply_pixar_texture,
    RETROMAN_OT_pixar_credits, RETROMAN_PT_pixar_textures,
    RetroManMaterialSettings,
    RetroManSceneSettings,
    RETROMAN_OT_gpu_self_test,
    RETROMAN_OT_activate_viewport,
    RETROMAN_OT_export_rib,
    RETROMAN_RenderEngine,
    RETROMAN_PT_main,
    RETROMAN_PT_film,
    RETROMAN_PT_sampling,
    RETROMAN_PT_performance,
    RETROMAN_PT_render_sampling,
    RETROMAN_PT_reyes,
    RETROMAN_PT_lighting,
    RETROMAN_PT_shadow,
    RETROMAN_PT_reflections,
    RETROMAN_PT_camera_effects,
    RETROMAN_PT_dof,
    RETROMAN_PT_translation,
    RETROMAN_PT_rib,
    RETROMAN_PT_cpu,
    RETROMAN_PT_diagnostics,
    RETROMAN_PT_material,
)

_EXCLUDE_PANELS = {
    "VIEWLAYER_PT_filter",
    "VIEWLAYER_PT_layer_passes",
}


def _compatible_panels():
    for panel in bpy.types.Panel.__subclasses__():
        engines = getattr(panel, "COMPAT_ENGINES", None)
        if engines is None:
            continue
        if "BLENDER_RENDER" in engines and panel.__name__ not in _EXCLUDE_PANELS:
            yield panel


_REGISTERED_CLASSES = []
_MODIFIED_PANELS = []


def register():
    # Extension updates can fail part-way through registration (for example if
    # Blender changed an RNA symbol). Roll back what succeeded so users do not
    # need to restart Blender just to recover from a half-loaded RetroMan.
    try:
        for cls in _CLASSES:
            bpy.utils.register_class(cls)
            _REGISTERED_CLASSES.append(cls)
        bpy.types.Scene.retroman_textures = PointerProperty(type=RetroManTextureSettings)
        bpy.types.Scene.retroman = PointerProperty(type=RetroManSceneSettings)
        bpy.types.Material.retroman = PointerProperty(type=RetroManMaterialSettings)
        for panel in _compatible_panels():
            panel.COMPAT_ENGINES.add("RETROMAN")
            _MODIFIED_PANELS.append(panel)
    except Exception:
        unregister()
        raise


def unregister():
    clear_previews()
    if hasattr(bpy.types.Scene, "retroman_textures"):
        del bpy.types.Scene.retroman_textures
    for panel in list(_MODIFIED_PANELS) or list(_compatible_panels()):
        try:
            panel.COMPAT_ENGINES.discard("RETROMAN")
        except Exception:
            pass
    _MODIFIED_PANELS.clear()
    if hasattr(bpy.types.Material, "retroman"):
        try:
            del bpy.types.Material.retroman
        except Exception:
            pass
    if hasattr(bpy.types.Scene, "retroman"):
        try:
            del bpy.types.Scene.retroman
        except Exception:
            pass
    classes = list(reversed(_REGISTERED_CLASSES)) if _REGISTERED_CLASSES else list(reversed(_CLASSES))
    for cls in classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    _REGISTERED_CLASSES.clear()
