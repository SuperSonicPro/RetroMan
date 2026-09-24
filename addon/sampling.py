# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic 1990s-style pixel, shutter and thin-lens sampling."""
import math
from mathutils import Matrix, Vector

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def _halton(index, base):
    f = 1.0
    r = 0.0
    i = int(index)
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def pixel_jitter(index, count):
    if count <= 1:
        return (0.0, 0.0)
    i = int(index) + 1
    return (_halton(i, 2) - 0.5, _halton(i, 3) - 0.5)


def _mitchell_1d(x, B=1.0/3.0, C=1.0/3.0):
    x = abs(float(x)) * 2.0
    if x < 1.0:
        return ((12 - 9*B - 6*C)*x**3 + (-18 + 12*B + 6*C)*x**2 + (6 - 2*B)) / 6.0
    if x < 2.0:
        return ((-B - 6*C)*x**3 + (6*B + 30*C)*x**2 + (-12*B - 48*C)*x + (8*B + 24*C)) / 6.0
    return 0.0


def reconstruction_weight(jitter, filter_name):
    x, y = jitter
    kind = str(filter_name or "GAUSSIAN").upper()
    if kind == "BOX":
        return 1.0
    if kind == "MITCHELL":
        return max(1.0e-6, _mitchell_1d(x) * _mitchell_1d(y))
    return math.exp(-2.0 * (x*x + y*y))


def shutter_offsets(render_settings, count):
    n = max(1, int(count))
    if n == 1:
        return [0.0]
    shutter = max(0.0, float(getattr(render_settings, "motion_blur_shutter", 0.5)))
    position = getattr(render_settings, "motion_blur_position", "CENTER")
    if position == "START":
        start = 0.0
    elif position == "END":
        start = -shutter
    else:
        start = -0.5 * shutter
    return [start + shutter * ((i + 0.5) / n) for i in range(n)]


def lens_disk_sample(index, count, blades=0, rotation=0.0, ratio=1.0):
    n = max(1, int(count))
    if n <= 1:
        return (0.0, 0.0)
    if index == 0:
        return (0.0, 0.0)
    k = index - 1
    m = max(1, n - 1)
    r = math.sqrt((k + 0.5) / m)
    theta = k * _GOLDEN_ANGLE + float(rotation)
    sides = int(blades)
    if sides >= 3:
        sector = 2.0 * math.pi / sides
        local = (theta + 0.5 * sector) % sector - 0.5 * sector
        r *= math.cos(math.pi / sides) / max(1.0e-6, math.cos(local))
    x = r * math.cos(theta)
    y = r * math.sin(theta) / max(0.01, float(ratio))
    return (x, y)


def camera_sample(camera, index, count, width, height, *, use_dof=True, use_aa=True):
    """Return (viewProjection, cameraPosition, pixelJitter).

    DOF uses a parallel shifted camera plus an off-axis projection correction,
    so the original focal plane stays registered without a toe-in camera error.
    Pixel AA adds a subpixel clip-space offset to the same projection.
    """
    base_world = camera.matrix_world.copy()
    proj = camera.projection.copy()
    sample_world = base_world.copy()

    if use_dof and camera.dof_enabled and camera.aperture_radius > 0.0 and count > 1:
        sx, sy = lens_disk_sample(index, count, camera.aperture_blades, camera.aperture_rotation, camera.aperture_ratio)
        radius = float(camera.aperture_radius)
        rot = base_world.to_3x3()
        right = (rot @ Vector((1.0, 0.0, 0.0))).normalized()
        up = (rot @ Vector((0.0, 1.0, 0.0))).normalized()
        dx = sx * radius
        dy = sy * radius
        sample_world.translation = base_world.translation + right * dx + up * dy
        focus = max(1.0e-6, float(camera.focus_distance))
        # clip-space translation by +m00*dx/focus and +m11*dy/focus
        # compensates for the parallel camera shift at the focus plane.
        lens_shift = Matrix.Identity(4)
        lens_shift[0][3] = float(proj[0][0]) * dx / focus
        lens_shift[1][3] = float(proj[1][1]) * dy / focus
        proj = lens_shift @ proj

    jitter = pixel_jitter(index, count) if use_aa else (0.0, 0.0)
    if use_aa and (jitter[0] != 0.0 or jitter[1] != 0.0):
        j = Matrix.Identity(4)
        j[0][3] = 2.0 * jitter[0] / max(1.0, float(width))
        j[1][3] = 2.0 * jitter[1] / max(1.0, float(height))
        proj = j @ proj

    return proj @ sample_world.inverted(), sample_world.translation.copy(), jitter


def dof_camera_sample(camera, index, count):
    # Backward-compatible helper used by older scene files/tests.
    vp, pos, _ = camera_sample(camera, index, count, 1, 1, use_dof=True, use_aa=False)
    return vp, pos


def set_scene_subframe(scene, absolute_frame):
    whole = math.floor(float(absolute_frame))
    sub = float(absolute_frame) - whole
    scene.frame_set(int(whole), subframe=sub)
