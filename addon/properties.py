# SPDX-License-Identifier: GPL-3.0-or-later
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty


class RetroManMaterialSettings(bpy.types.PropertyGroup):
    retrosl_enabled: BoolProperty(
        name="Use RetroSL Override",
        description="Read classic Ka/Kd/Ks/Kr/roughness/Cs/opacity values from a Blender Text block",
        default=False,
    )
    retrosl_text: PointerProperty(
        name="RetroSL Text",
        description="Optional Blender Text block containing a small RenderMan-1995-style parameter declaration",
        type=bpy.types.Text,
    )

class RetroManSceneSettings(bpy.types.PropertyGroup):
    render_device: EnumProperty(
        name="Device",
        description="Choose the renderer used for F12; Auto uses Blender's active GPU; errors stop visibly. CPU Reference must be selected explicitly",
        items=[
            ("AUTO", "Auto (GPU preferred)", "Use the GPU; report unavailable hardware or rendering errors without a slow fallback"),
            ("GPU", "GPU", "Require Blender's GPU context and hardware rasterization"),
            ("CPU", "Legacy CPU Reference", "Use the slow Python reference renderer"),
        ],
        default="AUTO",
    )
    quality: EnumProperty(
        name="Quality",
        description="RetroMan quality preset",
        items=[
            ("DRAFT", "Draft", "Fast triangle-oriented preview"),
            ("PRODUCTION", "Production", "GPU adaptive dicing with moderate limits"),
            ("AUTHENTIC", "Fine Micropolygons", "Fine GPU micropolygon dicing and displacement with classic shading"),
        ],
        default="PRODUCTION",
    )
    geometry_mode: EnumProperty(
        name="Geometry Pipeline",
        description="Choose whether the GPU renders the evaluated triangles directly or dices them into REYES-style micropolygons",
        items=[
            ("AUTO", "Auto", "Draft uses source triangles; Production/Authentic use GPU micropolygons"),
            ("MICROPOLYGON", "GPU Micropolygons", "Dice source triangles into barycentric micro-grids on the GPU"),
            ("TRIANGLES", "Fast Triangles", "Render Blender's evaluated triangles directly"),
        ],
        default="AUTO",
    )
    bucket_size: IntProperty(name="Bucket Size", default=64, min=8, max=128,
        description="Final image tile size; changing it preserves the sample pattern and result")
    adaptive_dicing: BoolProperty(
        name="Adaptive Final Dicing",
        description="Choose micropolygon subdivision from projected triangle size for F12 renders",
        default=True,
    )
    shading_rate: FloatProperty(
        name="Shading Rate",
        description="Approximate maximum micropolygon edge length in pixels for adaptive final dicing",
        default=1.0,
        min=0.5,
        max=64.0,
    )
    gpu_max_subdiv: IntProperty(
        name="Final Max Subdivision",
        description="Maximum subdivisions per source triangle edge in the GPU micropolygon path; 8 means up to 64 micropolygons per triangle",
        default=16,
        min=1,
        max=32,
    )
    viewport_subdiv: IntProperty(
        name="Viewport Subdivision",
        description="Fixed micropolygon subdivisions per source triangle edge in Rendered view; keep this modest for GTX 1650-class real-time work",
        default=2,
        min=1,
        max=8,
    )
    max_dice_depth: IntProperty(
        name="CPU Max Dice Depth",
        description="Legacy CPU reference dicer recursion limit",
        default=4,
        min=0,
        max=8,
    )
    displacement_enabled: BoolProperty(
        name="Micropolygon Displacement",
        description="Translate Blender's Displacement node and move diced vertices along their interpolated normals",
        default=True,
    )
    displacement_shadow_maps: BoolProperty(
        name="Displace Shadow Maps",
        description="Use displaced micropolygon geometry when baking automatic shadow maps",
        default=True,
    )
    mesh_lighting: BoolProperty(name="Mesh Lighting", default=True,
        description="Illuminate surfaces from emissive meshes using bounded area-weighted patches")
    max_mesh_lights: IntProperty(name="Max Mesh Lights", default=16, min=0, max=128,
        description="Maximum emissive patches; the brightest are retained when over budget")
    environment_lighting: BoolProperty(name="World to Ambient", default=True,
        description="Translate the modern world to a classic ambient light; environment images remain reflection maps")
    ambient: FloatProperty(
        name="Ambient",
        description="Period-style ambient light contribution",
        default=0.12,
        min=0.0,
        max=2.0,
    )
    exposure: FloatProperty(
        name="Exposure",
        description="Simple photographic exposure multiplier in stops",
        default=0.0,
        min=-10.0,
        max=10.0,
    )
    shadow_resolution: EnumProperty(
        name="Final Shadow Map Size",
        items=[
            ("128", "128", "128 x 128"),
            ("256", "256", "256 x 256"),
            ("512", "512", "512 x 512"),
            ("1024", "1024", "1024 x 1024"),
            ("2048", "2048", "2048 x 2048"),
            ("4096", "4096", "4096 x 4096; memory intensive for point lights"),
        ],
        default="1024",
    )
    viewport_shadow_resolution: EnumProperty(
        name="Viewport Shadow Map Size",
        items=[
            ("128", "128", "Fastest"),
            ("256", "256", "Fast"),
            ("512", "512", "Balanced for GTX 1650-class GPUs"),
            ("1024", "1024", "Sharper but heavier"),
        ],
        default="512",
    )
    max_shadow_lights: IntProperty(
        name="GPU Shadow Light Budget",
        description="Maximum shadow-casting lights kept in GPU memory at once; extra lights still illuminate but do not cast shadows",
        default=8,
        min=0,
        max=32,
    )
    shadow_bias: FloatProperty(name="Shadow Bias", default=0.002, min=0.00001, max=0.1, precision=5)
    shadow_softness: IntProperty(
        name="Shadow Filter",
        description="PCF radius used to soften classic shadow-map edges",
        default=1,
        min=0,
        max=4,
    )
    enable_shadows: BoolProperty(name="Automatic Shadow Maps", default=True)
    viewport_shadows: BoolProperty(
        name="Viewport Shadow Maps",
        description="Bake classic shadow maps in Rendered viewport mode",
        default=True,
    )
    viewport_scale: EnumProperty(
        name="Viewport Resolution",
        description="Internal Rendered-viewport resolution; Auto uses a draw-time heuristic without changing scene settings",
        items=[
            ("AUTO_60", "Auto resolution", "Dynamically choose 100/75/50/33% to target real-time interaction"),
            ("1.0", "100%", "Full viewport resolution"),
            ("0.75", "75%", "Useful for heavier scenes"),
            ("0.5", "50%", "High-speed interactive preview"),
            ("0.33", "33%", "Emergency speed mode"),
        ],
        default="AUTO_60",
    )
    viewport_target_fps: IntProperty(
        name="Viewport Target FPS",
        description="Target used by Auto viewport resolution",
        default=60,
        min=24,
        max=240,
    )
    pixel_samples: IntProperty(
        name="Pixel Samples",
        description="Stochastic samples per pixel axis; unified with motion blur and DOF rather than multiplied blindly",
        default=2, min=1, max=4,
    )
    pixel_filter: EnumProperty(
        name="Pixel Filter",
        description="1990s-style reconstruction weighting for final stochastic samples",
        items=[
            ("BOX", "Box", "Equal sample weights"),
            ("GAUSSIAN", "Gaussian", "Soft Gaussian reconstruction"),
            ("MITCHELL", "Mitchell", "Sharper cubic-style reconstruction approximation"),
        ],
        default="GAUSSIAN",
    )
    temporal_samples: IntProperty(
        name="Motion/DOF Samples",
        description="Unified stochastic samples used for Blender motion blur and camera depth of field in F12 renders",
        default=4,
        min=1,
        max=32,
    )
    reflection_mode: EnumProperty(
        name="Reflection Maps",
        description="1995-style map-based reflections; no ray tracing is used",
        items=[
            ("AUTO", "Automatic", "Use a Blender Sphere Light Probe when present; otherwise use an internal scene-center probe, falling back to the World environment"),
            ("PROBE", "Scene Probe", "Bake a six-face reflection map from a Blender Sphere Light Probe or internal scene-center probe"),
            ("WORLD", "World Environment Only", "Use the Blender World Environment Texture directly as the reflection map"),
            ("OFF", "Off", "Disable environment/reflection-map contribution"),
        ],
        default="AUTO",
    )
    reflection_resolution: EnumProperty(
        name="Final Reflection Map Size",
        items=[
            ("64", "64", "Very fast / visibly retro"),
            ("128", "128", "Fast"),
            ("256", "256", "Balanced"),
            ("512", "512", "High quality for 1995-style maps"),
            ("1024", "1024", "Heavy; six faces"),
        ],
        default="256",
    )
    viewport_reflections: BoolProperty(
        name="Viewport Reflection Maps",
        description="Bake/reuse a low-resolution six-face scene reflection map in Rendered view",
        default=True,
    )
    viewport_reflection_resolution: EnumProperty(
        name="Viewport Reflection Map Size",
        items=[
            ("64", "64", "Fastest"),
            ("128", "128", "Recommended for GTX 1650-class GPUs"),
            ("256", "256", "Sharper"),
        ],
        default="128",
    )
    reflection_strength: FloatProperty(
        name="Reflection Strength",
        description="Global multiplier for period-style environment/reflection maps",
        default=1.0,
        min=0.0,
        max=4.0,
    )
    texture_filter: EnumProperty(
        name="Preview / Map Filtering",
        description="Interactive preview and auxiliary-map sampling; bucket finals use automatic grid-footprint filtering",
        items=[
            ("NEAREST", "Nearest", "Point sampled textures"),
            ("BILINEAR", "Bilinear", "Linear texture filtering"),
            ("MIPMAP", "Mipmapped", "Linear filtering with mipmaps"),
            ("ANISO", "Mipmapped + Anisotropic", "Mipmaps plus anisotropic filtering on supported GPUs"),
        ],
        default="MIPMAP",
    )
    texture_filtering: BoolProperty(
        name="Legacy Bilinear Compatibility",
        description="Compatibility flag for scenes saved with RetroMan 0.x; Texture Filtering takes precedence",
        default=True, options={"HIDDEN"},
    )
    auxiliary_min_subdiv: IntProperty(
        name="Auxiliary Map Min Subdivision",
        description="Minimum micropolygon subdivision used while baking shadow/reflection maps",
        default=2, min=1, max=16,
    )
    use_world_color: BoolProperty(name="Use World Color", default=True)
    show_translation_warnings: BoolProperty(name="Material Translation Warnings", default=True)
    clamp_fireflies: BoolProperty(
        name="Clamp Specular",
        description="Clamp unusually bright legacy Phong highlights",
        default=True,
    )
    max_specular: FloatProperty(name="Specular Clamp", default=8.0, min=1.0, max=100.0)
