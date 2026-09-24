# SPDX-License-Identifier: GPL-3.0-or-later
"""Run from Blender's Scripting workspace to create a RetroMan 1.0 demo scene."""
import bpy
import math
from mathutils import Vector

# Clear scene.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)


def point_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


# Floor.
bpy.ops.mesh.primitive_plane_add(size=16, location=(0, 0, 0))
floor = bpy.context.object
mat_floor = bpy.data.materials.new('Retro Floor')
mat_floor.diffuse_color = (0.22, 0.28, 0.32, 1)
mat_floor.use_nodes = True
bsdf = mat_floor.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value = (0.22, 0.28, 0.32, 1)
bsdf.inputs['Roughness'].default_value = 0.72
floor.data.materials.append(mat_floor)

# Suzanne: focus target and normal plastic.
bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 1.25))
monkey = bpy.context.object
bpy.ops.object.shade_smooth()
subd = monkey.modifiers.new('Subdivision', 'SUBSURF')
subd.levels = 2
subd.render_levels = 2
mat = bpy.data.materials.new('Retro Plastic')
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value = (0.72, 0.11, 0.045, 1)
bsdf.inputs['Roughness'].default_value = 0.22
bsdf.inputs['Metallic'].default_value = 0.0
monkey.data.materials.append(mat)

# Animated metallic cube: exercises Blender motion blur + reflection maps.
bpy.ops.mesh.primitive_cube_add(size=1.5, location=(-2.2, 0.4, 0.75))
cube = bpy.context.object
cube.rotation_euler = (0.15, 0.2, 0.45)
mat2 = bpy.data.materials.new('Retro Chrome Blue')
mat2.use_nodes = True
b2 = mat2.node_tree.nodes.get('Principled BSDF')
b2.inputs['Base Color'].default_value = (0.04, 0.22, 0.75, 1)
b2.inputs['Roughness'].default_value = 0.14
b2.inputs['Metallic'].default_value = 0.92
cube.data.materials.append(mat2)
cube.location.x = -3.0
cube.keyframe_insert(data_path='location', frame=1)
cube.location.x = -1.1
cube.keyframe_insert(data_path='location', frame=24)

# Image-driven displacement patch.
bpy.ops.mesh.primitive_plane_add(size=3.0, location=(2.7, 1.6, 0.18))
patch = bpy.context.object
patch.name = 'Micropolygon Displacement Patch'
sub = patch.modifiers.new('Coarse Subdivision', 'SUBSURF')
sub.subdivision_type = 'SIMPLE'
sub.levels = 2
sub.render_levels = 2

height = bpy.data.images.new('RetroMan Height', width=64, height=64, float_buffer=True)
pixels = []
for y in range(64):
    for x in range(64):
        u = x / 63.0
        v = y / 63.0
        h = 0.5 + 0.22 * math.sin(u * math.tau * 4.0) * math.cos(v * math.tau * 4.0)
        pixels.extend((h, h, h, 1.0))
height.pixels.foreach_set(pixels)
height.update()

mat_disp = bpy.data.materials.new('Retro Displaced Brass')
mat_disp.use_nodes = True
nt = mat_disp.node_tree
bs = nt.nodes.get('Principled BSDF')
bs.inputs['Base Color'].default_value = (0.52, 0.24, 0.045, 1.0)
bs.inputs['Metallic'].default_value = 0.75
bs.inputs['Roughness'].default_value = 0.32
out = nt.nodes.get('Material Output')
tex = nt.nodes.new('ShaderNodeTexImage')
tex.image = height
disp = nt.nodes.new('ShaderNodeDisplacement')
disp.inputs['Midlevel'].default_value = 0.5
disp.inputs['Scale'].default_value = 0.42
nt.links.new(tex.outputs['Color'], disp.inputs['Height'])
nt.links.new(disp.outputs['Displacement'], out.inputs['Displacement'])
patch.data.materials.append(mat_disp)

# Normal Blender lights: RetroMan generates their shadow maps automatically.
bpy.ops.object.light_add(type='SPOT', location=(4.5, -4.5, 6.0))
key = bpy.context.object
key.name = 'Key Spot'
key.data.energy = 1600
key.data.spot_size = math.radians(55)
key.data.spot_blend = 0.3
point_at(key, (0, 0, 1))

bpy.ops.object.light_add(type='AREA', location=(-3.5, -1.5, 4.0))
fill = bpy.context.object
fill.name = 'Fill Area'
fill.data.energy = 700
fill.data.shape = 'DISK'
fill.data.size = 3.0
point_at(fill, (0, 0, 1))

# Blender Sphere Light Probe: RetroMan treats this as a 1995 six-face reflection-map origin.
bpy.ops.object.lightprobe_add(type='SPHERE', radius=4.0, location=(0.0, 0.0, 1.3))
probe = bpy.context.object
probe.name = 'RetroMan Reflection Probe'
probe.data.clip_start = 0.05
probe.data.clip_end = 30.0

# Camera using normal Blender DOF controls.
bpy.ops.object.camera_add(location=(7.5, -9.0, 5.8))
cam = bpy.context.object
point_at(cam, (0, 0, 1.0))
cam.data.lens = 52
cam.data.dof.use_dof = True
cam.data.dof.focus_object = monkey
cam.data.dof.aperture_fstop = 3.2
cam.data.dof.aperture_blades = 6
bpy.context.scene.camera = cam

scene = bpy.context.scene
scene.render.engine = 'RETROMAN'
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.resolution_percentage = 100
scene.frame_start = 1
scene.frame_end = 24
scene.frame_set(12)
scene.render.use_motion_blur = True
scene.render.motion_blur_shutter = 0.5
scene.render.motion_blur_position = 'CENTER'
scene.retroman.pixel_samples = 2
scene.retroman.pixel_filter = 'GAUSSIAN'
scene.retroman.temporal_samples = 4
scene.retroman.texture_filter = 'MIPMAP'
scene.retroman.auxiliary_min_subdiv = 2
scene.retroman.quality = 'AUTHENTIC'
scene.retroman.geometry_mode = 'MICROPOLYGON'
scene.retroman.gpu_max_subdiv = 8
scene.retroman.viewport_subdiv = 2
scene.retroman.displacement_enabled = True
scene.retroman.shadow_resolution = '512'
scene.retroman.reflection_mode = 'AUTO'
scene.retroman.reflection_resolution = '256'
scene.retroman.viewport_reflection_resolution = '128'
scene.retroman.shading_rate = 5.0

# Generated latitude/longitude World Environment Texture. RetroMan uses this as
# the actual background and as the fallback environment reflection map.
world_img = bpy.data.images.new('RetroMan 1995 Environment', width=256, height=128, float_buffer=True)
wpix = []
for y in range(128):
    v = y / 127.0
    for x in range(256):
        u = x / 255.0
        horizon = math.exp(-((v - 0.48) / 0.13) ** 2)
        r = 0.025 + 0.18 * horizon + 0.05 * u
        g = 0.035 + 0.14 * horizon + 0.03 * u
        b = 0.075 + 0.22 * horizon + 0.08 * (1.0 - v)
        wpix.extend((r, g, b, 1.0))
world_img.pixels.foreach_set(wpix)
world_img.update()

scene.world.use_nodes = True
wnt = scene.world.node_tree
bg = wnt.nodes.get('Background')
env = wnt.nodes.new('ShaderNodeTexEnvironment')
env.image = world_img
wnt.links.new(env.outputs['Color'], bg.inputs['Color'])
bg.inputs['Strength'].default_value = 1.0

print('RetroMan 1.0 demo ready. Frame 12 shows motion blur, camera DOF, environment background, shadow maps, reflections and micropolygon displacement.')
print('Press F12 or switch the 3D View to Rendered.')
