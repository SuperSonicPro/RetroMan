from .runtime import checkpoint
# SPDX-License-Identifier: GPL-3.0-or-later
import math
from dataclasses import dataclass
from mathutils import Matrix, Vector

from .model import TriangleData, VertexData

_INF = 1.0e30
_EPS = 1.0e-8


@dataclass
class ProjectedVertex:
    x: float
    y: float
    depth: float
    inv_w: float
    p: object
    n: object
    uv: tuple


@dataclass
class ShadowMap:
    size: int
    matrix: object
    depth: list
    kind: str = "2D"
    view: object = None
    cube: object = None


def _edge(ax, ay, bx, by, px, py):
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


def _clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _lerp_vertex(a, b):
    p = (a.p + b.p) * 0.5
    n = (a.n + b.n)
    if n.length_squared > _EPS:
        n.normalize()
    uv = ((a.uv[0] + b.uv[0]) * 0.5, (a.uv[1] + b.uv[1]) * 0.5)
    return VertexData(p, n, uv)


def _project_vertex(v, view_proj, view, width, height):
    co4 = view_proj @ v.p.to_4d()
    w = float(co4.w)
    if abs(w) < _EPS or w <= 0.0:
        return None
    inv_w = 1.0 / w
    ndc_x = co4.x * inv_w
    ndc_y = co4.y * inv_w
    vc = view @ v.p.to_4d()
    depth = -float(vc.z)
    if depth <= 0.0:
        return None
    return ProjectedVertex(
        (ndc_x * 0.5 + 0.5) * (width - 1),
        (ndc_y * 0.5 + 0.5) * (height - 1),
        depth,
        inv_w,
        v.p,
        v.n,
        v.uv,
    )


def _project_triangle(tri, view_proj, view, width, height):
    a = _project_vertex(tri.a, view_proj, view, width, height)
    b = _project_vertex(tri.b, view_proj, view, width, height)
    c = _project_vertex(tri.c, view_proj, view, width, height)
    if a is None or b is None or c is None:
        return None
    return a, b, c


def _max_screen_edge(ptri):
    a, b, c = ptri
    return max(
        math.hypot(a.x - b.x, a.y - b.y),
        math.hypot(b.x - c.x, b.y - c.y),
        math.hypot(c.x - a.x, c.y - a.y),
    )


def _dice_triangle(tri, view_proj, view, width, height, target, max_depth):
    stack = [(tri, 0)]
    while stack:
        cur, depth = stack.pop()
        projected = _project_triangle(cur, view_proj, view, width, height)
        if projected is None:
            continue
        if depth >= max_depth or _max_screen_edge(projected) <= target:
            yield cur, projected
            continue
        # Split the longest world-space edge. This is a triangular approximation
        # to classic screen-space dicing and keeps the reference implementation small.
        va, vb, vc = cur.a, cur.b, cur.c
        pa, pb, pc = projected
        edges = [
            ((pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2, 0),
            ((pb.x - pc.x) ** 2 + (pb.y - pc.y) ** 2, 1),
            ((pc.x - pa.x) ** 2 + (pc.y - pa.y) ** 2, 2),
        ]
        _, edge_id = max(edges)
        if edge_id == 0:
            m = _lerp_vertex(va, vb)
            stack.append((TriangleData(m, vb, vc, cur.material), depth + 1))
            stack.append((TriangleData(va, m, vc, cur.material), depth + 1))
        elif edge_id == 1:
            m = _lerp_vertex(vb, vc)
            stack.append((TriangleData(va, m, vc, cur.material), depth + 1))
            stack.append((TriangleData(va, vb, m, cur.material), depth + 1))
        else:
            m = _lerp_vertex(vc, va)
            stack.append((TriangleData(m, vb, vc, cur.material), depth + 1))
            stack.append((TriangleData(va, vb, m, cur.material), depth + 1))


def _perspective_weights(p0, p1, p2, w0, w1, w2):
    q0 = w0 * p0.inv_w
    q1 = w1 * p1.inv_w
    q2 = w2 * p2.inv_w
    s = q0 + q1 + q2
    if abs(s) < _EPS:
        return w0, w1, w2
    return q0 / s, q1 / s, q2 / s


def _sample_texture(tex, uv, bilinear):
    if tex is None or tex.width <= 0 or tex.height <= 0:
        return (1.0, 1.0, 1.0, 1.0)
    u = uv[0] - math.floor(uv[0])
    v = uv[1] - math.floor(uv[1])
    x = u * (tex.width - 1)
    y = v * (tex.height - 1)

    def px(ix, iy):
        ix %= tex.width
        iy %= tex.height
        base = (iy * tex.width + ix) * 4
        p = tex.pixels
        return (p[base], p[base + 1], p[base + 2], p[base + 3])

    if not bilinear:
        return px(int(round(x)), int(round(y)))
    x0 = int(math.floor(x)); x1 = min(tex.width - 1, x0 + 1)
    y0 = int(math.floor(y)); y1 = min(tex.height - 1, y0 + 1)
    tx = x - x0; ty = y - y0
    c00 = px(x0, y0); c10 = px(x1, y0); c01 = px(x0, y1); c11 = px(x1, y1)
    out = []
    for i in range(4):
        a = c00[i] * (1.0 - tx) + c10[i] * tx
        b = c01[i] * (1.0 - tx) + c11[i] * tx
        out.append(a * (1.0 - ty) + b * ty)
    return tuple(out)


def _look_at_matrix(position, direction, up_hint=Vector((0.0, 1.0, 0.0))):
    # Blender cameras/lights look down local -Z. Return world->view.
    f = direction.normalized()
    if abs(f.dot(up_hint)) > 0.98:
        up_hint = Vector((1.0, 0.0, 0.0))
    r = f.cross(up_hint).normalized()
    u = r.cross(f).normalized()
    world = Matrix((
        (r.x, u.x, -f.x, position.x),
        (r.y, u.y, -f.y, position.y),
        (r.z, u.z, -f.z, position.z),
        (0.0, 0.0, 0.0, 1.0),
    ))
    return world.inverted()


def _perspective_matrix(fov, aspect, near, far):
    f = 1.0 / math.tan(max(0.001, fov) * 0.5)
    return Matrix((
        (f / aspect, 0.0, 0.0, 0.0),
        (0.0, f, 0.0, 0.0),
        (0.0, 0.0, -(far + near) / (far - near), -(2.0 * far * near) / (far - near)),
        (0.0, 0.0, -1.0, 0.0),
    ))


def _ortho_matrix(left, right, bottom, top, near, far):
    return Matrix((
        (2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)),
        (0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)),
        (0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)),
        (0.0, 0.0, 0.0, 1.0),
    ))


def _scene_bounds(triangles, cancel=None):
    """Exact world-space source bounds with bounded temporary memory."""
    if hasattr(triangles, "bounds"):
        return triangles.bounds(cancel)
    if not triangles:
        return Vector((-1, -1, -1)), Vector((1, 1, 1))
    import numpy as np
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for start in range(0, len(triangles), 4096):
        checkpoint(cancel)
        points = np.asarray([tuple(v.p) for tri in triangles[start:start + 4096]
                             for v in (tri.a, tri.b, tri.c)], dtype=np.float64)
        lo = np.minimum(lo, points.min(axis=0))
        hi = np.maximum(hi, points.max(axis=0))
    return Vector(lo.tolist()), Vector(hi.tolist())


def _shadow_far_for_light(light, triangles, bounds=None):
    """Choose a finite map far plane without inventing a lighting cutoff.

    Blender's Custom Distance can be disabled, but perspective shadow maps still
    require a finite clip plane.  In that case encompass the translated scene
    bounds instead of using the old accidental 1-unit far plane.
    """
    cutoff = float(getattr(light, "cutoff_distance", 0.0) or 0.0)
    if cutoff > 0.0:
        return max(1.0, cutoff)
    if not triangles:
        return 1000.0
    lo, hi = bounds if bounds is not None else _scene_bounds(triangles)
    pos = light.matrix_world.translation
    far = 1.0
    for x in (lo.x, hi.x):
        for y in (lo.y, hi.y):
            for z in (lo.z, hi.z):
                far = max(far, (Vector((x, y, z)) - pos).length)
    return max(1.0, far * 1.05)

def _shadow_camera_for_light(light, triangles, size, bounds=None):
    pos = light.matrix_world.translation.copy()
    direction = -(light.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0)))
    direction.normalize()
    near = 0.02
    if light.type == "SPOT":
        far = _shadow_far_for_light(light, triangles, bounds)
        view = _look_at_matrix(pos, direction)
        proj = _perspective_matrix(min(math.pi - 0.01, light.spot_size * 1.03), 1.0, near, far)
        return view, proj
    if light.type == "SUN":
        lo, hi = bounds if bounds is not None else _scene_bounds(triangles)
        center = (lo + hi) * 0.5
        radius = max((hi - lo).length * 0.6, 1.0)
        sun_pos = center - direction * radius * 2.0
        view = _look_at_matrix(sun_pos, direction)
        proj = _ortho_matrix(-radius, radius, -radius, radius, 0.01, radius * 5.0)
        return view, proj
    return None, None


def _raster_depth_triangle(depthbuf, ptri, size, cancel=None):
    p0, p1, p2 = ptri
    area = _edge(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y)
    if abs(area) < _EPS:
        return
    minx = max(0, int(math.floor(min(p0.x, p1.x, p2.x))))
    maxx = min(size - 1, int(math.ceil(max(p0.x, p1.x, p2.x))))
    miny = max(0, int(math.floor(min(p0.y, p1.y, p2.y))))
    maxy = min(size - 1, int(math.ceil(max(p0.y, p1.y, p2.y))))
    inv_area = 1.0 / area
    for y in range(miny, maxy + 1):
        if y % 8 == 0:
            checkpoint(cancel)
        py = y + 0.5
        row = y * size
        for x in range(minx, maxx + 1):
            px = x + 0.5
            w0 = _edge(p1.x, p1.y, p2.x, p2.y, px, py) * inv_area
            w1 = _edge(p2.x, p2.y, p0.x, p0.y, px, py) * inv_area
            w2 = 1.0 - w0 - w1
            if w0 < -_EPS or w1 < -_EPS or w2 < -_EPS:
                continue
            bw0, bw1, bw2 = _perspective_weights(p0, p1, p2, w0, w1, w2)
            d = p0.depth * bw0 + p1.depth * bw1 + p2.depth * bw2
            i = row + x
            if d < depthbuf[i]:
                depthbuf[i] = d


def build_shadow_maps(rscene, progress=None, cancel=None):
    if not rscene.settings.enable_shadows:
        return
    size = int(rscene.settings.shadow_resolution)
    shadow_lights = [l for l in rscene.lights if l.use_shadow]
    total = max(1, len(shadow_lights))
    for li, light in enumerate(shadow_lights):
        if cancel and cancel():
            return
        if progress:
            progress(li / total)
        if light.type in {"SPOT", "SUN"}:
            view, proj = _shadow_camera_for_light(light, rscene.triangles, size)
            if view is None:
                continue
            vp = proj @ view
            depth = [_INF] * (size * size)
            for tri in rscene.triangles:
                if tri.material.alpha * tri.material.base_color[3] <= 0.003:
                    continue
                ptri = _project_triangle(tri, vp, view, size, size)
                if ptri:
                    _raster_depth_triangle(depth, ptri, size, cancel=cancel)
            light.shadow = ShadowMap(size=size, matrix=vp, depth=depth, kind="2D", view=view)
        elif light.type in {"POINT", "AREA", "MESH"}:
            # Classic point-light shadow maps are represented as six 90-degree faces.
            pos = light.matrix_world.translation.copy()
            dirs = [
                (Vector((1, 0, 0)), Vector((0, 0, 1))),
                (Vector((-1, 0, 0)), Vector((0, 0, 1))),
                (Vector((0, 1, 0)), Vector((0, 0, 1))),
                (Vector((0, -1, 0)), Vector((0, 0, 1))),
                (Vector((0, 0, 1)), Vector((0, 1, 0))),
                (Vector((0, 0, -1)), Vector((0, 1, 0))),
            ]
            faces = []
            proj = _perspective_matrix(math.radians(90.0), 1.0, 0.02, _shadow_far_for_light(light, rscene.triangles))
            for direction, up in dirs:
                view = _look_at_matrix(pos, direction, up)
                vp = proj @ view
                depth = [_INF] * (size * size)
                for tri in rscene.triangles:
                    if tri.material.alpha * tri.material.base_color[3] <= 0.003:
                        continue
                    ptri = _project_triangle(tri, vp, view, size, size)
                    if ptri:
                        _raster_depth_triangle(depth, ptri, size, cancel=cancel)
                faces.append((vp, view, depth))
            light.shadow = ShadowMap(size=size, matrix=None, depth=None, kind="CUBE", cube=faces)
    if progress:
        progress(1.0)


def _shadow_lookup_2d(shadow, point, bias, radius):
    co = shadow.matrix @ point.to_4d()
    if abs(co.w) < _EPS or co.w <= 0:
        return 1.0
    invw = 1.0 / co.w
    nx = co.x * invw; ny = co.y * invw
    if nx < -1 or nx > 1 or ny < -1 or ny > 1:
        return 1.0
    # Shadow maps store positive light-view distance, so compare against the same
    # quantity for both perspective (spot) and orthographic (sun) maps.
    current = -(shadow.view @ point.to_4d()).z if shadow.view is not None else abs(float(co.w))
    if current <= 0.0:
        return 1.0
    sx = int((nx * 0.5 + 0.5) * (shadow.size - 1))
    sy = int((ny * 0.5 + 0.5) * (shadow.size - 1))
    lit = 0; samples = 0
    for oy in range(-radius, radius + 1):
        yy = min(shadow.size - 1, max(0, sy + oy))
        for ox in range(-radius, radius + 1):
            xx = min(shadow.size - 1, max(0, sx + ox))
            stored = shadow.depth[yy * shadow.size + xx]
            lit += 1 if current <= stored + bias * max(1.0, current) else 0
            samples += 1
    return lit / max(1, samples)


def _shadow_lookup(light, point, settings):
    shadow = light.shadow
    if shadow is None:
        return 1.0
    bias = settings.shadow_bias
    radius = settings.shadow_softness
    if shadow.kind == "2D":
        return _shadow_lookup_2d(shadow, point, bias, radius)
    # Cube: choose the face whose projected point is valid and has the largest clip w.
    best = None
    for vp, view, depth in shadow.cube:
        co = vp @ point.to_4d()
        if abs(co.w) < _EPS or co.w <= 0:
            continue
        nx = co.x / co.w; ny = co.y / co.w
        if -1.0 <= nx <= 1.0 and -1.0 <= ny <= 1.0:
            current = -(view @ point.to_4d()).z
            if current <= 0:
                continue
            best = (vp, depth, current, nx, ny)
            break
    if best is None:
        return 1.0
    vp, depth, current, nx, ny = best
    size = shadow.size
    sx = int((nx * 0.5 + 0.5) * (size - 1)); sy = int((ny * 0.5 + 0.5) * (size - 1))
    lit = 0; samples = 0
    for oy in range(-radius, radius + 1):
        yy = min(size - 1, max(0, sy + oy))
        for ox in range(-radius, radius + 1):
            xx = min(size - 1, max(0, sx + ox))
            stored = depth[yy * size + xx]
            lit += 1 if current <= stored + bias * max(1.0, current) else 0
            samples += 1
    return lit / max(1, samples)


def _shade(rscene, material, p, n, uv, view_dir):
    settings = rscene.settings
    tex = _sample_texture(material.texture, uv, str(getattr(settings, "texture_filter", "BILINEAR")) != "NEAREST")
    base = [material.base_color[i] * tex[i] for i in range(3)]
    tex_alpha = tex[3] if getattr(material, "use_texture_alpha", False) else 1.0
    alpha = _clamp01(material.alpha * tex_alpha * material.base_color[3])
    if n.length_squared < _EPS:
        n = Vector((0.0, 0.0, 1.0))
    else:
        n = n.normalized()
    v = view_dir.normalized() if view_dir.length_squared > _EPS else Vector((0, 0, 1))
    world = rscene.world_color if settings.use_world_color else (1.0, 1.0, 1.0)
    rgb = [base[i] * settings.ambient * world[i] * max(0.0, float(getattr(material, "ka", 1.0))) for i in range(3)]
    if getattr(rscene, "environment_diffuse", ()):
        from .illumination import evaluate_environment
        env = evaluate_environment(rscene.environment_diffuse, n)
        rgb = [rgb[i] + base[i] * env[i] * max(0.0, material.ka) for i in range(3)]
    for light in rscene.lights:
        if light.type == "SUN":
            L = -(light.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0)))
            if L.length_squared < _EPS:
                continue
            L.normalize()
            attenuation = max(0.0, light.energy) / math.pi
        else:
            lp = light.matrix_world.translation
            delta = lp - p
            dist2 = max(0.01, delta.length_squared)
            L = delta.normalized()
            # Point/spot flux -> inverse-square irradiance -> Lambertian radiance.
            attenuation = max(0.0, light.energy) / (4.0 * math.pi * math.pi * dist2)
            if light.type == "MESH":
                forward = (light.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
                area = max(0.0, light.area_factor)
                attenuation = max(0.0, light.energy) * area * abs(forward.dot(-L)) / (math.pi * (dist2 + area / math.pi))
            if light.type == "AREA":
                forward = (light.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
                attenuation *= 4.0 * max(1.0e-8, float(getattr(light, "area_factor", 1.0))) * max(0.0, forward.dot(-L))
            if light.cutoff_distance > 0 and math.sqrt(dist2) > light.cutoff_distance:
                continue
        if light.type == "SPOT":
            forward = -(light.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
            to_p = (p - light.matrix_world.translation).normalized()
            cosang = forward.dot(to_p)
            outer = math.cos(light.spot_size * 0.5)
            if cosang <= outer:
                continue
            blend = max(0.001, light.spot_blend)
            inner = math.cos(light.spot_size * 0.5 * (1.0 - min(1.0, blend)))
            cone = _clamp01((cosang - outer) / max(_EPS, inner - outer))
            attenuation *= cone * cone
        ndotl = max(0.0, n.dot(L))
        if ndotl <= 0.0:
            continue
        shadow = _shadow_lookup(light, p + n * (settings.shadow_bias * 2.0), settings) if settings.enable_shadows and light.use_shadow else 1.0
        if shadow <= 0.0:
            continue
        lc = light.color
        diffuse_strength = max(0.0, float(getattr(material, "kd", 1.0 - material.metallic))) * ndotl
        # Blinn-Phong approximation: roughness maps to a classic exponent.
        h = L + v
        specular = 0.0
        if h.length_squared > _EPS:
            h.normalize()
            exponent = max(1.0, min(1000.0, 2.0 / (material.roughness * material.roughness) - 2.0))
            specular = max(0.0, n.dot(h)) ** exponent
            specular *= max(0.0, float(getattr(material, "ks", material.specular))) * (0.25 + 0.75 * material.metallic)
            if settings.clamp_fireflies:
                specular = min(specular, settings.max_specular)
        for i in range(3):
            metal_spec = base[i] if material.metallic > 0.0 else 1.0
            diffuse_light = max(0.0, float(getattr(light, "diffuse_factor", 1.0)))
            specular_light = max(0.0, float(getattr(light, "specular_factor", 1.0)))
            rgb[i] += shadow * attenuation * lc[i] * (
                base[i] * diffuse_strength * diffuse_light + metal_spec * specular * specular_light
            )
    if material.emission_strength > 0.0:
        for i in range(3):
            rgb[i] += material.emission_color[i] * material.emission_strength
    exposure = 2.0 ** settings.exposure
    return (max(0.0, rgb[0] * exposure), max(0.0, rgb[1] * exposure), max(0.0, rgb[2] * exposure), alpha)


def render_scene(rscene, progress=None, cancel=None):
    width, height = rscene.width, rscene.height
    world = rscene.world_color if rscene.settings.use_world_color else (0.0, 0.0, 0.0)
    clear_alpha = 0.0 if getattr(rscene, "film_transparent", False) else 1.0
    clear_rgb = (0.0, 0.0, 0.0) if clear_alpha == 0.0 else world
    pixels = [[clear_rgb[0], clear_rgb[1], clear_rgb[2], clear_alpha] for _ in range(width * height)]
    zbuf = [_INF] * (width * height)
    cam_view = rscene.camera.matrix_world.inverted()
    vp = rscene.camera.projection @ cam_view
    camera_pos = rscene.camera.matrix_world.translation
    settings = rscene.settings
    target = settings.shading_rate
    max_depth = settings.max_dice_depth
    if settings.quality == "DRAFT":
        target = max(target, 16.0); max_depth = min(max_depth, 2)
    elif settings.quality == "AUTHENTIC":
        target = min(target, 2.0); max_depth = max(max_depth, 5)

    total = max(1, len(rscene.triangles))
    for ti, tri in enumerate(rscene.triangles):
        checkpoint(cancel)
        if progress and (ti & 15) == 0:
            progress(ti / total)
        for micro, ptri in _dice_triangle(tri, vp, cam_view, width, height, target, max_depth):
            checkpoint(cancel)
            p0, p1, p2 = ptri
            area = _edge(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y)
            if abs(area) < _EPS:
                continue
            minx = max(0, int(math.floor(min(p0.x, p1.x, p2.x))))
            maxx = min(width - 1, int(math.ceil(max(p0.x, p1.x, p2.x))))
            miny = max(0, int(math.floor(min(p0.y, p1.y, p2.y))))
            maxy = min(height - 1, int(math.ceil(max(p0.y, p1.y, p2.y))))
            if maxx < minx or maxy < miny:
                continue
            p = (p0.p + p1.p + p2.p) / 3.0
            n = (p0.n + p1.n + p2.n) / 3.0
            uv = tuple((p0.uv[k] + p1.uv[k] + p2.uv[k]) / 3.0 for k in range(2))
            rgba = _shade(rscene, micro.material, p, n, uv, camera_pos - p)
            inv_area = 1.0 / area
            for y in range(miny, maxy + 1):
                if y % 8 == 0:
                    checkpoint(cancel)
                py = y + 0.5; row = y * width
                for x in range(minx, maxx + 1):
                    px = x + 0.5
                    w0 = _edge(p1.x, p1.y, p2.x, p2.y, px, py) * inv_area
                    w1 = _edge(p2.x, p2.y, p0.x, p0.y, px, py) * inv_area
                    w2 = 1.0 - w0 - w1
                    if w0 < -_EPS or w1 < -_EPS or w2 < -_EPS:
                        continue
                    b0, b1, b2 = _perspective_weights(p0, p1, p2, w0, w1, w2)
                    d = p0.depth * b0 + p1.depth * b1 + p2.depth * b2
                    idx = row + x
                    if d >= zbuf[idx]:
                        continue
                    if rgba[3] >= 0.999:
                        pixels[idx] = list(rgba)
                        zbuf[idx] = d
                    elif rgba[3] > 0.001:
                        dst = pixels[idx]; a = rgba[3]; da = dst[3]
                        out_a = a + da * (1.0 - a)
                        if out_a > 1.0e-8:
                            rgb = [(rgba[i] * a + dst[i] * da * (1.0 - a)) / out_a for i in range(3)]
                        else:
                            rgb = [0.0, 0.0, 0.0]
                        pixels[idx] = rgb + [out_a]
                        zbuf[idx] = d
    if progress:
        progress(1.0)
    return pixels
