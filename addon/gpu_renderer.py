# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU renderer for RetroMan 1.0.

RetroMan deliberately uses Blender's public ``gpu`` abstraction rather than
CUDA/OpenGL calls.  The fast path renders evaluated Blender triangles directly.
The REYES path uploads the *source* triangles once, then repeatedly instances a
small barycentric grid over them.  Position/normal/UV interpolation and scalar
normal displacement happen in the GPU vertex stage, producing real micro-
triangles for hardware rasterization.

This is a clean-room, 1995-inspired REYES approximation, not Pixar PRMan code.
"""

from dataclasses import dataclass, field
from .runtime import checkpoint, read_framebuffer
from .shader_parameters import ParameterInfo
from .model import TriangleData, VertexData
import math
from mathutils import Matrix, Vector

from .raster import _look_at_matrix, _perspective_matrix, _shadow_camera_for_light, _shadow_far_for_light


@dataclass
class GPUShadow:
    kind: str = "NONE"  # NONE, MAP2D, CUBE6
    textures: list = field(default_factory=list)
    matrices: list = field(default_factory=list)


@dataclass
class GPUReflection:
    kind: str = "NONE"  # NONE, ENV2D, CUBE6
    textures: list = field(default_factory=list)
    matrices: list = field(default_factory=list)
    probe_position: object = None


@dataclass
class GPUBatchGroup:
    material: object
    batch: object
    triangle_count: int
    texture: object = None
    displacement_texture: object = None
    normal_texture: object = None
    bump_texture: object = None
    center: object = None
    bounds: object = None


@dataclass
class GPUReyesChunk:
    material: object
    triangle_texture: object
    triangle_count: int
    subdiv: int
    batch: object
    texture: object = None
    displacement_texture: object = None
    normal_texture: object = None
    bump_texture: object = None
    center: object = None


# ---------------------------------------------------------------------------
# Public diagnostics

def _texture_option(texture, name, value):
    # Blender 5.0 lacks sampler-state setters exposed in newer versions.
    method = getattr(texture, name, None)
    if method is not None:
        method(value)



def gpu_info():
    import gpu
    info = {}
    queries = {
        "backend": (gpu.platform, "backend_type_get", "UNKNOWN"),
        "device_type": (gpu.platform, "device_type_get", "UNKNOWN"),
        "renderer": (gpu.platform, "renderer_get", "Unavailable"),
        "vendor": (gpu.platform, "vendor_get", ""),
        "version": (gpu.platform, "version_get", ""),
        "compute": (gpu.capabilities, "compute_shader_support_get", False),
        "max_texture_size": (gpu.capabilities, "max_texture_size_get", 0),
        "max_textures": (gpu.capabilities, "max_textures_get", 0),
    }
    for key, (owner, name, default) in queries.items():
        try:
            info[key] = getattr(owner, name)()
        except Exception as exc:
            info[key] = default
            info.setdefault("warnings", []).append(f"{name}: {exc}")
    return info


def gpu_available():
    info = gpu_info()
    return info.get("backend") not in {"NONE", "UNKNOWN"} and info.get("device_type") != "SOFTWARE"


# ---------------------------------------------------------------------------
# Micropolygon grid generation / adaptive dicing


def make_barycentric_microgrid(subdiv):
    """Return 3 barycentric vertices for each of ``subdiv**2`` micro-triangles."""
    n = max(1, int(subdiv))
    inv = 1.0 / n
    out = []
    for i in range(n):
        for j in range(n - i):
            a = (i * inv, j * inv, 1.0 - (i + j) * inv)
            b = ((i + 1) * inv, j * inv, 1.0 - (i + 1 + j) * inv)
            c = (i * inv, (j + 1) * inv, 1.0 - (i + j + 1) * inv)
            out.extend((a, b, c))
            if j < n - i - 1:
                d = ((i + 1) * inv, (j + 1) * inv, 1.0 - (i + j + 2) * inv)
                out.extend((b, d, c))
    return out


def _snap_subdiv(desired, cap):
    desired = max(1, int(desired))
    cap = max(1, int(cap))
    level = 1
    while level < desired and level < cap:
        level *= 2
    return min(level, cap)


def _project_to_pixel(vp, p, width, height):
    q = vp @ p.to_4d()
    if abs(q.w) < 1.0e-8 or q.w <= 0.0:
        return None
    x = (q.x / q.w * 0.5 + 0.5) * width
    y = (q.y / q.w * 0.5 + 0.5) * height
    return (x, y)


def estimate_subdiv(triangle, vp, width, height, shading_rate, cap):
    """Approximate REYES screen-space dicing and snap to reusable grid levels."""
    pts = [_project_to_pixel(vp, v.p, width, height) for v in (triangle.a, triangle.b, triangle.c)]
    if any(p is None for p in pts):
        # A triangle crossing the eye plane is exactly where under-dicing is most
        # visible.  Use the cap rather than returning a coarse primitive.
        return max(1, int(cap))
    lengths = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        dx = pts[i][0] - pts[j][0]
        dy = pts[i][1] - pts[j][1]
        lengths.append(math.sqrt(dx * dx + dy * dy))
    target = max(0.5, float(shading_rate))
    desired = int(math.ceil(max(lengths) / target))
    return _snap_subdiv(desired, cap)


def split_for_dicing(triangle, vp, width, height, rate, cap, cancel=None):
    """Split large visible linear patches before creating bounded-size grids.

    Offscreen patches are retained for shadow/reflection maps. Near-eye patches
    stop at a bounded depth; the renderer reports that limitation separately.
    """
    stack = [(triangle, 0)]
    while stack:
        checkpoint(cancel)
        tri, depth = stack.pop()
        pts = [_project_to_pixel(vp, v.p, width, height) for v in (tri.a, tri.b, tri.c)]
        visible = all(p is not None for p in pts)
        if visible:
            xs, ys = zip(*pts)
            visible = not (max(xs) < 0 or min(xs) > width or max(ys) < 0 or min(ys) > height)
        large = visible and estimate_subdiv(tri, vp, width, height, rate, cap * 2) > cap
        if not large or depth >= 12:
            yield tri, bool(large or any(p is None for p in pts))
            continue
        def mid(a, b):
            # Preserve the original linear normal field through patch splitting;
            # normalize only when the GPU evaluates the surface.
            return VertexData((a.p+b.p)*0.5, (a.n+b.n)*0.5,
                              tuple((a.uv[k]+b.uv[k])*0.5 for k in range(2)))
        ab, bc, ca = mid(tri.a, tri.b), mid(tri.b, tri.c), mid(tri.c, tri.a)
        for a,b,c in ((tri.a,ab,ca),(ab,tri.b,bc),(ca,bc,tri.c),(ab,bc,ca)):
            stack.append((TriangleData(a,b,c,tri.material), depth+1))


# ---------------------------------------------------------------------------
# Shader construction


_LIGHT_FRAGMENT = r'''

vec3 rm_surface_normal(vec3 Nin)
{
    vec3 N = normalize(Nin);
    if (displacementNormal) {
        vec3 Ng = normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)));
        if (dot(Ng, N) < 0.0) Ng = -Ng;
        N = Ng;
    }
    if (!useNormalTexture && !useBumpTexture) return N;

    vec3 dpdx = dFdx(vWorldPos);
    vec3 dpdy = dFdy(vWorldPos);
    vec2 duvdx = dFdx(vUV);
    vec2 duvdy = dFdy(vUV);
    float det = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
    if (abs(det) < 1.0e-10) return N;
    vec3 T = normalize((dpdx * duvdy.y - dpdy * duvdx.y) / det);
    T = normalize(T - N * dot(N, T));
    vec3 B = normalize(cross(N, T));
    if (det < 0.0) B = -B;

    if (useNormalTexture) {
        vec3 tn = texture(normalTex, vUV).xyz * 2.0 - 1.0;
        tn.x *= normalStrength;
        tn.y *= normalStrength * normalYSign;
        N = normalize(T * tn.x + B * tn.y + N * max(0.001, tn.z));
    }
    if (useBumpTexture) {
        vec2 texel = 1.0 / vec2(textureSize(bumpTex, 0));
        float h = texture(bumpTex, vUV).r;
        float hx = texture(bumpTex, vUV + vec2(texel.x, 0.0)).r;
        float hy = texture(bumpTex, vUV + vec2(0.0, texel.y)).r;
        vec2 slope = vec2(hx - h, hy - h) * bumpStrength * bumpSign * max(0.0001, bumpDistance) * 10.0;
        N = normalize(N - T * slope.x - B * slope.y);
    }
    return N;
}
float shadow_sample(sampler2D tex, mat4 sm, vec3 p, vec3 n)
{
    vec4 q = sm * vec4(p + n * shadowBias * 2.0, 1.0);
    if (q.w <= 0.0) return 1.0;
    vec3 ndc = q.xyz / q.w;
    vec2 suv = ndc.xy * 0.5 + 0.5;
    float current = ndc.z * 0.5 + 0.5;
    if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0 || current < 0.0 || current > 1.0)
        return 1.0;

    vec2 texel = 1.0 / vec2(textureSize(tex, 0));
    float lit = 0.0;
    float count = 0.0;
    for (int y = -4; y <= 4; ++y) {
        for (int x = -4; x <= 4; ++x) {
            if (abs(x) > shadowRadius || abs(y) > shadowRadius) continue;
            float stored = texture(tex, suv + vec2(float(x), float(y)) * texel).r;
            lit += ((current - shadowBias) <= stored) ? 1.0 : 0.0;
            count += 1.0;
        }
    }
    return count > 0.0 ? lit / count : 1.0;
}

float get_shadow(vec3 p, vec3 n)
{
    if (shadowKind == 0) return 1.0;
    if (shadowKind == 1) return shadow_sample(shadow0, shadowMatrix0, p, n);

    vec3 d = p - lightPos;
    vec3 a = abs(d);
    int face = 0;
    if (a.x >= a.y && a.x >= a.z) face = d.x >= 0.0 ? 0 : 1;
    else if (a.y >= a.x && a.y >= a.z) face = d.y >= 0.0 ? 2 : 3;
    else face = d.z >= 0.0 ? 4 : 5;

    if (face == 0) return shadow_sample(shadow0, shadowMatrix0, p, n);
    if (face == 1) return shadow_sample(shadow1, shadowMatrix1, p, n);
    if (face == 2) return shadow_sample(shadow2, shadowMatrix2, p, n);
    if (face == 3) return shadow_sample(shadow3, shadowMatrix3, p, n);
    if (face == 4) return shadow_sample(shadow4, shadowMatrix4, p, n);
    return shadow_sample(shadow5, shadowMatrix5, p, n);
}

void main()
{
    vec4 texel = useTexture ? texture(baseTex, vUV) : vec4(1.0);
    vec3 base = baseColor.rgb * texel.rgb;
    float surfaceAlpha = clamp(baseColor.a * materialAlpha * (useTextureAlpha ? texel.a : 1.0), 0.0, 1.0);
    if (surfaceAlpha <= 0.003) discard;
    vec3 N = rm_surface_normal(vWorldNormal);
    vec3 V = normalize(cameraPos - vWorldPos);
    vec3 L;
    float attenuation = 0.0;

    if (lightType == 0) {
        L = normalize(lightDir);
        attenuation = max(lightEnergy, 0.0) / 3.141592653589793;
    }
    else {
        vec3 delta = lightPos - vWorldPos;
        float d2 = max(dot(delta, delta), 0.01);
        float dist = sqrt(d2);
        if (lightCutoff > 0.0 && dist > lightCutoff) {
            FragColor = vec4(0.0);
            return;
        }
        L = delta / dist;
        attenuation = max(lightEnergy, 0.0) / (39.47841760435743 * d2);
        if (lightType == 4)
            attenuation = max(lightEnergy, 0.0) * lightAreaFactor * abs(dot(normalize(lightDir), -L)) /
                (3.141592653589793 * (d2 + lightAreaFactor / 3.141592653589793));
        if (lightType == 3)
            attenuation *= 4.0 * max(lightAreaFactor, 0.00000001) * max(dot(normalize(lightDir), -L), 0.0);
        if (lightType == 2) {
            vec3 toP = normalize(vWorldPos - lightPos);
            float coneCos = dot(normalize(lightDir), toP);
            if (coneCos <= spotOuterCos) {
                FragColor = vec4(0.0);
                return;
            }
            float cone = clamp((coneCos - spotOuterCos) / max(0.00001, spotInnerCos - spotOuterCos), 0.0, 1.0);
            attenuation *= cone * cone;
        }
    }

    float ndotl = max(dot(N, L), 0.0);
    if (ndotl <= 0.0) {
        FragColor = vec4(0.0);
        return;
    }

    float visibility = get_shadow(vWorldPos, N);
    if (visibility <= 0.0) {
        FragColor = vec4(0.0);
        return;
    }

    float diffuseStrength = materialKd * ndotl;
    vec3 H = normalize(L + V);
    float exponent = clamp(2.0 / max(roughness * roughness, 0.000001) - 2.0, 1.0, 1000.0);
    float spec = pow(max(dot(N, H), 0.0), exponent);
    spec *= materialKs * (0.25 + 0.75 * metallic);
    if (clampSpecular) spec = min(spec, maxSpecular);

    vec3 metalSpec = mix(vec3(1.0), base, metallic);
    vec3 rgb = visibility * attenuation * lightColor * (
        base * diffuseStrength * lightDiffuseFactor + metalSpec * spec * lightSpecularFactor
    );
    // Lighting is composited additively after the base pass, so multiply it by
    // opacity here; otherwise transparent/Glass materials receive opaque light.
    FragColor = vec4(max(rgb * exposureMul * surfaceAlpha, vec3(0.0)), 0.0);
}
'''


_BASE_FRAGMENT = r'''

vec3 rm_surface_normal(vec3 Nin)
{
    vec3 N = normalize(Nin);
    if (displacementNormal) {
        vec3 Ng = normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)));
        if (dot(Ng, N) < 0.0) Ng = -Ng;
        N = Ng;
    }
    if (!useNormalTexture && !useBumpTexture) return N;

    vec3 dpdx = dFdx(vWorldPos);
    vec3 dpdy = dFdy(vWorldPos);
    vec2 duvdx = dFdx(vUV);
    vec2 duvdy = dFdy(vUV);
    float det = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
    if (abs(det) < 1.0e-10) return N;
    vec3 T = normalize((dpdx * duvdy.y - dpdy * duvdx.y) / det);
    T = normalize(T - N * dot(N, T));
    vec3 B = normalize(cross(N, T));
    if (det < 0.0) B = -B;

    if (useNormalTexture) {
        vec3 tn = texture(normalTex, vUV).xyz * 2.0 - 1.0;
        tn.x *= normalStrength;
        tn.y *= normalStrength * normalYSign;
        N = normalize(T * tn.x + B * tn.y + N * max(0.001, tn.z));
    }
    if (useBumpTexture) {
        vec2 texel = 1.0 / vec2(textureSize(bumpTex, 0));
        float h = texture(bumpTex, vUV).r;
        float hx = texture(bumpTex, vUV + vec2(texel.x, 0.0)).r;
        float hy = texture(bumpTex, vUV + vec2(0.0, texel.y)).r;
        vec2 slope = vec2(hx - h, hy - h) * bumpStrength * bumpSign * max(0.0001, bumpDistance) * 10.0;
        N = normalize(N - T * slope.x - B * slope.y);
    }
    return N;
}

const float RM_PI = 3.14159265358979323846;

vec3 sample_environment(vec3 d)
{
    d = normalize(d);
    float u = atan(d.y, d.x) / (2.0 * RM_PI) + 0.5;
    float v = asin(clamp(d.z, -1.0, 1.0)) / RM_PI + 0.5;
    return texture(environmentTex, vec2(u, v)).rgb * environmentStrength;
}

vec3 sample_reflection_face(sampler2D tex, mat4 m, vec3 d)
{
    vec4 q = m * vec4(reflectionProbePos + normalize(d), 1.0);
    if (q.w <= 0.0) return vec3(0.0);
    vec2 uv = (q.xy / q.w) * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return vec3(0.0);
    return texture(tex, uv).rgb;
}

vec3 sample_reflection(vec3 d)
{
    if (reflectionKind == 1) return sample_environment(d);
    if (reflectionKind != 2) return vec3(0.0);

    vec3 a = abs(d);
    int face = 0;
    if (a.x >= a.y && a.x >= a.z) face = d.x >= 0.0 ? 0 : 1;
    else if (a.y >= a.x && a.y >= a.z) face = d.y >= 0.0 ? 2 : 3;
    else face = d.z >= 0.0 ? 4 : 5;

    if (face == 0) return sample_reflection_face(reflection0, reflectionMatrix0, d);
    if (face == 1) return sample_reflection_face(reflection1, reflectionMatrix1, d);
    if (face == 2) return sample_reflection_face(reflection2, reflectionMatrix2, d);
    if (face == 3) return sample_reflection_face(reflection3, reflectionMatrix3, d);
    if (face == 4) return sample_reflection_face(reflection4, reflectionMatrix4, d);
    return sample_reflection_face(reflection5, reflectionMatrix5, d);
}

void main()
{
    vec4 texel = useTexture ? texture(baseTex, vUV) : vec4(1.0);
    vec3 base = baseColor.rgb * texel.rgb;
    float a = clamp(baseColor.a * materialAlpha * (useTextureAlpha ? texel.a : 1.0), 0.0, 1.0);
    if (a <= 0.003) discard;

    vec3 rgb = base * (ambientColor * materialKa + environmentAmbient * materialKa) + emissionColor * emissionStrength;
    if (reflectionKind != 0 && materialReflectivity > 0.0001) {
        vec3 N = rm_surface_normal(vWorldNormal);
        vec3 V = normalize(cameraPos - vWorldPos);
        vec3 R = reflect(-V, N);
        rgb += sample_reflection(R) * materialReflectivity * reflectionGlobalStrength;
    }

    FragColor = vec4(max(rgb * exposureMul, vec3(0.0)), a);
}
'''


_BACKGROUND_VERTEX = r'''
void main()
{
    vNdc = pos;
    gl_Position = vec4(pos, 0.0, 1.0);
}
'''

_BACKGROUND_FRAGMENT = r'''
const float RM_PI = 3.14159265358979323846;
void main()
{
    vec4 a = invViewProjection * vec4(vNdc, -1.0, 1.0);
    vec4 b = invViewProjection * vec4(vNdc,  1.0, 1.0);
    vec3 pa = a.xyz / max(abs(a.w), 1.0e-8);
    vec3 pb = b.xyz / max(abs(b.w), 1.0e-8);
    vec3 d = normalize(pb - pa);
    float u = atan(d.y, d.x) / (2.0 * RM_PI) + 0.5;
    float v = asin(clamp(d.z, -1.0, 1.0)) / RM_PI + 0.5;
    vec3 rgb = texture(environmentTex, vec2(u, v)).rgb * environmentStrength;
    FragColor = vec4(max(rgb * exposureMul, vec3(0.0)), 1.0);
}
'''

_ACCUM_VERTEX = r'''
void main()
{
    vUV = uv;
    gl_Position = vec4(pos, 0.0, 1.0);
}
'''

_ACCUM_FRAGMENT = r'''
void main()
{
    FragColor = texture(srcTex, vUV) * weight;
}
'''


_FAST_VERTEX = r'''
void main()
{
    vec3 wp = pos;
    vec3 wn = normalize(normal);
    if (useDisplacement) {
        float h = displacementConstant;
        if (useDisplacementTexture)
            h = texture(displacementTex, uv).r;
        wp += wn * ((h - displacementMidlevel) * displacementScale);
    }
    vWorldPos = wp;
    vWorldNormal = wn;
    vUV = uv;
    gl_Position = viewProjection * vec4(wp, 1.0);
}
'''


_REYES_VERTEX = r'''
vec3 rmDpdu, rmDpdv;
vec2 rmDuvdu, rmDuvdv;
void rm_evaluate(vec3 b, out vec3 p, out vec3 n, out vec2 uv)
{
    int tri = int(gl_InstanceID);
    p = texelFetch(triData, ivec2(tri, 0), 0).xyz * b.x
      + texelFetch(triData, ivec2(tri, 1), 0).xyz * b.y
      + texelFetch(triData, ivec2(tri, 2), 0).xyz * b.z;
    n = normalize(texelFetch(triData, ivec2(tri, 3), 0).xyz * b.x
      + texelFetch(triData, ivec2(tri, 4), 0).xyz * b.y
      + texelFetch(triData, ivec2(tri, 5), 0).xyz * b.z);
    vec4 uv01 = texelFetch(triData, ivec2(tri, 6), 0);
    uv = uv01.xy * b.x + uv01.zw * b.y + texelFetch(triData, ivec2(tri, 7), 0).xy * b.z;
    if (useDisplacement) {
        float h = displacementConstant;
        if (useDisplacementTexture) h = textureLod(displacementTex, uv, 0.0).r;
        p += n * ((h - displacementMidlevel) * displacementScale);
    }
}
void main()
{
    vec3 wp, wn;
    vec2 uv;
    rm_evaluate(bary, wp, wn, uv);
    gl_Position = viewProjection * vec4(wp, 1.0);
    // Shade a surface sample shared by all three vertices of this micropolygon.
    // Rasterization only resolves visibility and copies the resulting color.
    rm_evaluate(shadeBary, vWorldPos, vWorldNormal, vUV);
    vec3 pu, pv, unusedN;
    vec2 uvu, uvv;
    const float eps = 0.0001;
    rm_evaluate(shadeBary + vec3(-eps, eps, 0.0), pu, unusedN, uvu);
    rm_evaluate(shadeBary + vec3(-eps, 0.0, eps), pv, unusedN, uvv);
    rmDpdu = (pu - vWorldPos) / eps;
    rmDpdv = (pv - vWorldPos) / eps;
    rmDuvdu = (uvu - vUV) / eps;
    rmDuvdv = (uvv - vUV) / eps;
}
'''


def _grid_shading_sources(surface_source, atlas=False):
    """Move surface evaluation before visibility; no pixel derivatives on grids."""
    shade = surface_source.replace('void main()', 'void rm_shade()')
    shade = shade.replace('FragColor', 'rmGridColor')
    shade = shade.replace('discard;', '{ rmGridColor = vec4(0.0); return; }')
    for original, replacement in {
        'dFdx(vWorldPos)': 'rmDpdu', 'dFdy(vWorldPos)': 'rmDpdv',
        'dFdx(vUV)': 'rmDuvdu', 'dFdy(vUV)': 'rmDuvdv',
    }.items():
        shade = shade.replace(original, replacement)
    if atlas:
        from .grid_filter import GLSL
        shade=GLSL+shade
        shade=shade.replace("a <= 0.003", "a <= 0.0").replace("surfaceAlpha <= 0.003", "surfaceAlpha <= 0.0")
        shade=shade.replace('    vec3 base = baseColor.rgb * texel.rgb;', '    if (baseFilterPremult && texel.a > 0.0) texel.rgb /= texel.a;\n    vec3 base = baseColor.rgb * texel.rgb;')
        shade=shade.replace('rgb * exposureMul * surfaceAlpha','rgb * exposureMul')
        for sampler,dimensions in (("baseTex","baseFilterSize"),("normalTex","normalFilterSize"),("bumpTex","bumpFilterSize")):
            shade=shade.replace(f'texture({sampler}, vUV)',f'rmGridTexture({sampler}, vUV, {dimensions})')
        shade=shade.replace('vec2(textureSize(bumpTex, 0))','bumpFilterSize.xy')
        shade=shade.replace('texture(bumpTex, vUV + vec2(texel.x, 0.0))','rmGridTexture(bumpTex, vUV + vec2(texel.x, 0.0), bumpFilterSize)')
        shade=shade.replace('texture(bumpTex, vUV + vec2(0.0, texel.y))','rmGridTexture(bumpTex, vUV + vec2(0.0, texel.y), bumpFilterSize)')
    geometry = _REYES_VERTEX.replace('void main()', 'void rm_geometry()')
    entry = '\nvoid main() { rm_geometry(); rm_shade(); }\n'
    if atlas:
        entry = r'''
void main() {
    rm_geometry();
    rmFilterDu=rmDuvdu*(atlasB.y-atlasA.y)+rmDuvdv*(atlasB.z-atlasA.z);
    rmFilterDv=rmDuvdu*(atlasC.y-atlasA.y)+rmDuvdv*(atlasC.z-atlasA.z);
    rm_shade();
    vec3 unusedN; vec2 unusedUV;
    rm_evaluate(atlasA, atlasP0, unusedN, unusedUV);
    rm_evaluate(atlasB, atlasP1, unusedN, unusedUV);
    rm_evaluate(atlasC, atlasP2, unusedN, unusedUV);
    int pixel = int(gl_InstanceID)*int(atlasLayout.z)+int(gl_VertexID)/3;
    vec2 center = vec2(float(pixel % int(atlasLayout.x)), float(pixel / int(atlasLayout.x))) + vec2(0.5);
    int corner = int(gl_VertexID)%3;
    vec2 offset = corner==0 ? vec2(-0.5,-0.5) : (corner==1 ? vec2(1.5,-0.5) : vec2(-0.5,1.5));
    gl_Position = vec4((center+offset)/atlasLayout.xy*2.0-1.0,0.0,1.0);
}
'''
    vertex = geometry + shade + entry
    fragment = 'void main() { FragColor = rmGridColor; }'
    return vertex, fragment


def _add_common_vertex_resources(info, *, reyes, disp_slot):
    if reyes:
        info.vertex_in(0, "VEC3", "bary")
        info.vertex_in(1, "VEC3", "shadeBary")
        info.sampler(disp_slot - 1, "FLOAT_2D", "triData")
    else:
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC3", "normal")
        info.vertex_in(2, "VEC2", "uv")
    info.push_constant("MAT4", "viewProjection")
    info.push_constant("BOOL", "useDisplacement")
    info.push_constant("BOOL", "useDisplacementTexture")
    info.push_constant("FLOAT", "displacementScale")
    info.push_constant("FLOAT", "displacementMidlevel")
    info.push_constant("FLOAT", "displacementConstant")
    info.sampler(disp_slot, "FLOAT_2D", "displacementTex")


def _make_interface(gpu, name):
    iface = gpu.types.GPUStageInterfaceInfo(name)
    iface.smooth("VEC3", "vWorldPos")
    iface.smooth("VEC3", "vWorldNormal")
    iface.smooth("VEC2", "vUV")
    return iface


def _create_shader_set(*, reyes=False, atlas=False):
    import gpu
    vertex_source = _REYES_VERTEX if reyes else _FAST_VERTEX
    prefix = "reyes" if reyes else "fast"

    # Base/ambient pass.
    base = ParameterInfo()
    iface = _make_interface(gpu, f"retroman_{prefix}_base_iface")
    if reyes:
        iface.flat("VEC4", "rmGridColor")
        if atlas:
            for name in ("atlasP0", "atlasP1", "atlasP2"):
                iface.flat("VEC3", name)
    base.vertex_out(iface)
    # slot 0 is baseTex. REYES also needs triData before displacementTex.
    disp_slot = 2 if reyes else 1
    _add_common_vertex_resources(base, reyes=reyes, disp_slot=disp_slot)
    if atlas:
        for index, name in enumerate(("atlasA", "atlasB", "atlasC"), 2): base.vertex_in(index,"VEC3",name)
        base.push_constant("VEC3", "atlasLayout")
        base.push_constant("BOOL", "baseFilterPremult")
        for name in ("baseFilterSize","normalFilterSize","bumpFilterSize"): base.push_constant("VEC3",name)
    base.push_constant("VEC3", "environmentAmbient")
    base.push_constant("VEC4", "baseColor")
    base.push_constant("VEC3", "ambientColor")
    base.push_constant("VEC3", "emissionColor")
    base.push_constant("FLOAT", "emissionStrength")
    base.push_constant("FLOAT", "materialAlpha")
    base.push_constant("FLOAT", "exposureMul")
    base.push_constant("BOOL", "useTexture")
    base.push_constant("BOOL", "useTextureAlpha")
    base.push_constant("VEC3", "cameraPos")
    base.push_constant("BOOL", "displacementNormal")
    base.push_constant("BOOL", "useNormalTexture")
    base.push_constant("BOOL", "useBumpTexture")
    base.push_constant("FLOAT", "normalStrength")
    base.push_constant("FLOAT", "normalYSign")
    base.push_constant("FLOAT", "bumpStrength")
    base.push_constant("FLOAT", "bumpSign")
    base.push_constant("FLOAT", "bumpDistance")
    base.push_constant("FLOAT", "materialKa")
    base.push_constant("INT", "reflectionKind")
    base.push_constant("FLOAT", "materialReflectivity")
    base.push_constant("FLOAT", "reflectionGlobalStrength")
    base.push_constant("FLOAT", "environmentStrength")
    base.push_constant("VEC3", "reflectionProbePos")
    for i in range(6):
        base.push_constant("MAT4", f"reflectionMatrix{i}")
    base.sampler(0, "FLOAT_2D", "baseTex")
    env_slot = disp_slot + 1
    base.sampler(env_slot, "FLOAT_2D", "environmentTex")
    for i in range(6):
        base.sampler(env_slot + 1 + i, "FLOAT_2D", f"reflection{i}")
    base.sampler(env_slot + 7, "FLOAT_2D", "normalTex")
    base.sampler(env_slot + 8, "FLOAT_2D", "bumpTex")
    base.fragment_out(0, "VEC4", "FragColor")
    base_vertex, base_fragment = _grid_shading_sources(_BASE_FRAGMENT, atlas=atlas) if reyes else (vertex_source, _BASE_FRAGMENT)
    if atlas:
        for index in range(3): base.fragment_out(index+1,"VEC4",f"Position{index}")
        base_fragment = base_fragment.replace('FragColor = rmGridColor;',
            'FragColor = rmGridColor; Position0=vec4(atlasP0,1.0); Position1=vec4(atlasP1,1.0); Position2=vec4(atlasP2,1.0);')
    base.vertex_source(base_vertex)
    base.fragment_source(base_fragment if atlas else base_fragment.replace("FragColor = rmGridColor;", "if (rmGridColor.a <= 0.003) discard; FragColor = rmGridColor;"))
    base_shader = base.build()
    del iface, base

    # One additive pass per light.
    light = ParameterInfo()
    iface = _make_interface(gpu, f"retroman_{prefix}_light_iface")
    if reyes:
        iface.flat("VEC4", "rmGridColor")
        if atlas:
            for name in ("atlasP0", "atlasP1", "atlasP2"):
                iface.flat("VEC3", name)
    light.vertex_out(iface)
    disp_slot = 8 if reyes else 7
    _add_common_vertex_resources(light, reyes=reyes, disp_slot=disp_slot)
    if atlas:
        for index, name in enumerate(("atlasA", "atlasB", "atlasC"), 2): light.vertex_in(index,"VEC3",name)
        light.push_constant("VEC3", "atlasLayout")
        light.push_constant("BOOL", "baseFilterPremult")
        for name in ("baseFilterSize","normalFilterSize","bumpFilterSize"): light.push_constant("VEC3",name)
    light.push_constant("VEC3", "cameraPos")
    light.push_constant("VEC4", "baseColor")
    light.push_constant("FLOAT", "roughness")
    light.push_constant("FLOAT", "metallic")
    light.push_constant("FLOAT", "materialKd")
    light.push_constant("FLOAT", "materialKs")
    light.push_constant("FLOAT", "materialAlpha")
    light.push_constant("FLOAT", "exposureMul")
    light.push_constant("BOOL", "clampSpecular")
    light.push_constant("FLOAT", "maxSpecular")
    light.push_constant("BOOL", "useTexture")
    light.push_constant("BOOL", "useTextureAlpha")
    light.push_constant("BOOL", "displacementNormal")
    light.push_constant("BOOL", "useNormalTexture")
    light.push_constant("BOOL", "useBumpTexture")
    light.push_constant("FLOAT", "normalStrength")
    light.push_constant("FLOAT", "normalYSign")
    light.push_constant("FLOAT", "bumpStrength")
    light.push_constant("FLOAT", "bumpSign")
    light.push_constant("FLOAT", "bumpDistance")
    light.push_constant("INT", "lightType")
    light.push_constant("VEC3", "lightPos")
    light.push_constant("VEC3", "lightDir")
    light.push_constant("VEC3", "lightColor")
    light.push_constant("FLOAT", "lightEnergy")
    light.push_constant("FLOAT", "lightAreaFactor")
    light.push_constant("FLOAT", "lightDiffuseFactor")
    light.push_constant("FLOAT", "lightSpecularFactor")
    light.push_constant("FLOAT", "lightCutoff")
    light.push_constant("FLOAT", "spotOuterCos")
    light.push_constant("FLOAT", "spotInnerCos")
    light.push_constant("INT", "shadowKind")
    light.push_constant("FLOAT", "shadowBias")
    light.push_constant("INT", "shadowRadius")
    for i in range(6):
        light.push_constant("MAT4", f"shadowMatrix{i}")
    light.sampler(0, "FLOAT_2D", "baseTex")
    for i in range(6):
        light.sampler(i + 1, "DEPTH_2D", f"shadow{i}")
    light.sampler(disp_slot + 1, "FLOAT_2D", "normalTex")
    light.sampler(disp_slot + 2, "FLOAT_2D", "bumpTex")
    light.fragment_out(0, "VEC4", "FragColor")
    light_vertex, light_fragment = _grid_shading_sources(_LIGHT_FRAGMENT, atlas=atlas) if reyes else (vertex_source, _LIGHT_FRAGMENT)
    light.vertex_source(light_vertex)
    light.fragment_source(light_fragment)
    light_shader = light.build()
    del iface, light

    # Depth-only map. Blender still requires fragment source.
    shadow = ParameterInfo()
    iface = _make_interface(gpu, f"retroman_{prefix}_shadow_iface")
    shadow.vertex_out(iface)
    disp_slot = 1 if reyes else 0
    _add_common_vertex_resources(shadow, reyes=reyes, disp_slot=disp_slot)
    shadow.vertex_source(vertex_source)
    shadow.fragment_source("void main() {}")
    shadow_shader = shadow.build()
    del iface, shadow

    return base_shader, light_shader, shadow_shader


def _create_background_shader():
    import gpu
    info = gpu.types.GPUShaderCreateInfo()
    iface = gpu.types.GPUStageInterfaceInfo("retroman_background_iface")
    iface.smooth("VEC2", "vNdc")
    info.vertex_in(0, "VEC2", "pos")
    info.vertex_out(iface)
    info.push_constant("MAT4", "invViewProjection")
    info.push_constant("FLOAT", "environmentStrength")
    info.push_constant("FLOAT", "exposureMul")
    info.sampler(0, "FLOAT_2D", "environmentTex")
    info.fragment_out(0, "VEC4", "FragColor")
    info.vertex_source(_BACKGROUND_VERTEX)
    info.fragment_source(_BACKGROUND_FRAGMENT)
    shader = gpu.shader.create_from_info(info)
    del iface, info
    return shader


def _create_accum_shader():
    import gpu
    info = gpu.types.GPUShaderCreateInfo()
    iface = gpu.types.GPUStageInterfaceInfo("retroman_accum_iface")
    iface.smooth("VEC2", "vUV")
    info.vertex_in(0, "VEC2", "pos")
    info.vertex_in(1, "VEC2", "uv")
    info.vertex_out(iface)
    info.push_constant("FLOAT", "weight")
    info.sampler(0, "FLOAT_2D", "srcTex")
    info.fragment_out(0, "VEC4", "FragColor")
    info.vertex_source(_ACCUM_VERTEX)
    info.fragment_source(_ACCUM_FRAGMENT)
    shader = gpu.shader.create_from_info(info)
    del iface, info
    return shader




def _create_safe_viewport_shader():
    '''Very small Blender-GPU shader used when the full RetroMan viewport fails.'''
    import gpu
    info = gpu.types.GPUShaderCreateInfo()
    iface = gpu.types.GPUStageInterfaceInfo("retroman_safe_view_iface")
    iface.smooth("VEC3", "vNormal")
    info.vertex_in(0, "VEC3", "pos")
    info.vertex_in(1, "VEC3", "normal")
    info.vertex_out(iface)
    info.push_constant("MAT4", "viewProjection")
    info.push_constant("VEC4", "baseColor")
    info.push_constant("VEC3", "emissionColor")
    info.push_constant("FLOAT", "emissionStrength")
    info.push_constant("FLOAT", "exposureMul")
    info.fragment_out(0, "VEC4", "FragColor")
    info.vertex_source(r'''
void main()
{
    vNormal = normalize(normal);
    gl_Position = viewProjection * vec4(pos, 1.0);
}
''')
    info.fragment_source(r'''
void main()
{
    vec3 N = normalize(vNormal);
    vec3 L1 = normalize(vec3(0.35, 0.55, 0.75));
    vec3 L2 = normalize(vec3(-0.55, -0.20, 0.45));
    float key = max(dot(N, L1), 0.0);
    float fill = max(dot(N, L2), 0.0);
    float shade = 0.20 + key * 0.68 + fill * 0.12;
    vec3 rgb = baseColor.rgb * shade + emissionColor * emissionStrength;
    FragColor = vec4(max(rgb * exposureMul, vec3(0.0)), max(baseColor.a, 0.05));
}
''')
    shader = gpu.shader.create_from_info(info)
    del iface, info
    return shader


class GPUViewportSafeRenderer:
    '''Minimal fail-safe Rendered viewport used if the full GPU path fails.'''
    def __init__(self, rscene):
        import gpu
        from gpu_extras.batch import batch_for_shader
        self.gpu = gpu
        self.rscene = rscene
        self.viewport = True
        self.width = max(1, int(rscene.width))
        self.height = max(1, int(rscene.height))
        self.use_reyes = False
        self.current_subdiv = 1
        self.geometry_label = "Compatibility triangles"
        self.shader = _create_safe_viewport_shader()
        self.groups = []
        grouped = {}
        for tri in rscene.triangles:
            grouped.setdefault(id(tri.material), [tri.material, []])[1].append(tri)
        try:
            max_vertices = int(gpu.capabilities.max_batch_vertices_get())
        except Exception:
            max_vertices = 393216
        # Avoid one monolithic Python/GPU batch for large scenes. Keep a sane
        # cap even if a driver advertises a much larger theoretical limit.
        tri_chunk = max(1, min(max_vertices // 3, 8192))
        for material, triangles in grouped.values():
            for start in range(0, len(triangles), tri_chunk):
                block = triangles[start:start + tri_chunk]
                pos, normal = [], []
                for tri in block:
                    for v in (tri.a, tri.b, tri.c):
                        pos.append(tuple(v.p))
                        normal.append(tuple(v.n))
                batch = batch_for_shader(self.shader, "TRIS", {"pos": pos, "normal": normal})
                self.groups.append((material, batch, len(block)))
        self.color_texture = None
        self.depth_texture = None
        self.framebuffer = None
        self.resize(self.width, self.height)

    def resize(self, width, height):
        width = max(1, int(width)); height = max(1, int(height))
        if self.color_texture is not None and self.width == width and self.height == height:
            return
        self.width, self.height = width, height
        self.color_texture = self.gpu.types.GPUTexture((width, height), format="RGBA16F")
        self.depth_texture = self.gpu.types.GPUTexture((width, height), format="DEPTH_COMPONENT32F")
        self.framebuffer = self.gpu.types.GPUFrameBuffer(depth_slot=self.depth_texture, color_slots=(self.color_texture,))

    def set_viewport_subdiv(self, subdiv):
        self.current_subdiv = 1
        return False

    def render_to_texture(self, *, view_projection=None, camera_pos=None, progress=None, cancel=None,
                          rebuild_shadows=True, rebuild_reflections=True):
        if view_projection is None:
            view_projection = self.rscene.camera.projection @ self.rscene.camera.matrix_world.inverted()
        world = self.rscene.world_color if getattr(self.rscene.settings, "use_world_color", True) else (0.035, 0.035, 0.035)
        clear = (float(world[0]), float(world[1]), float(world[2]), 1.0)
        try:
            old_blend = self.gpu.state.blend_get()
        except Exception:
            old_blend = "NONE"
        try:
            old_depth_test = self.gpu.state.depth_test_get()
        except Exception:
            old_depth_test = "NONE"
        try:
            old_depth_mask = self.gpu.state.depth_mask_get()
        except Exception:
            old_depth_mask = True
        try:
            with self.framebuffer.bind():
                self.gpu.state.viewport_set(0, 0, self.width, self.height)
                # OpenGL depth clears respect the inherited depth write mask.
                self.gpu.state.depth_mask_set(True)
                self.framebuffer.clear(color=clear, depth=1.0)
                self.gpu.state.face_culling_set("NONE")
                self.gpu.state.depth_test_set("LESS_EQUAL")
                self.gpu.state.depth_mask_set(True)
                self.gpu.state.blend_set("NONE")
                self.shader.bind()
                self.shader.uniform_float("viewProjection", view_projection)
                self.shader.uniform_float("exposureMul", 2.0 ** float(getattr(self.rscene.settings, "exposure", 0.0)))
                for material, batch, _count in self.groups:
                    self.shader.uniform_float("baseColor", material.base_color)
                    self.shader.uniform_float("emissionColor", material.emission_color)
                    self.shader.uniform_float("emissionStrength", float(material.emission_strength))
                    batch.draw(self.shader)
        finally:
            self.gpu.shader.unbind()
            self.gpu.state.depth_mask_set(old_depth_mask)
            self.gpu.state.depth_test_set(old_depth_test)
            self.gpu.state.blend_set(old_blend)
            self.gpu.state.face_culling_set("NONE")
        if progress:
            progress(1.0)
        return self.color_texture

    @property
    def micropolygon_count(self):
        # Compatibility viewport renders source triangles only.
        return sum(int(count) for _material, _batch, count in self.groups)


class GPUAccumulator:
    """GPU-side weighted accumulation so motion blur/DOF need one final readback."""
    def __init__(self, width, height):
        import gpu
        from gpu_extras.batch import batch_for_shader
        self.gpu = gpu
        self.width = max(1, int(width)); self.height = max(1, int(height))
        self.shader = _create_accum_shader()
        self.texture = gpu.types.GPUTexture((self.width, self.height), format="RGBA16F")
        self.framebuffer = gpu.types.GPUFrameBuffer(color_slots=(self.texture,))
        pos = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
        uv = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        self.batch = batch_for_shader(self.shader, "TRIS", {"pos": pos, "uv": uv})
        with self.framebuffer.bind():
            self.gpu.state.viewport_set(0, 0, self.width, self.height)
            self.framebuffer.clear(color=(0.0, 0.0, 0.0, 0.0))

    def add(self, texture, weight):
        try:
            old_blend = self.gpu.state.blend_get()
        except Exception:
            old_blend = "NONE"
        try:
            old_depth_test = self.gpu.state.depth_test_get()
        except Exception:
            old_depth_test = "NONE"
        try:
            old_depth_mask = self.gpu.state.depth_mask_get()
        except Exception:
            old_depth_mask = True
        try:
            with self.framebuffer.bind():
                self.gpu.state.viewport_set(0, 0, self.width, self.height)
                self.gpu.state.depth_test_set("NONE")
                self.gpu.state.depth_mask_set(False)
                # Source RGB is already weighted; ADDITIVE would multiply it by alpha again.
                self.gpu.state.blend_set("ADDITIVE_PREMULT")
                self.shader.bind()
                self.shader.uniform_float("weight", float(weight))
                self.shader.uniform_sampler("srcTex", texture)
                self.batch.draw(self.shader)
        finally:
            self.gpu.shader.unbind()
            self.gpu.state.blend_set(old_blend)
            self.gpu.state.depth_test_set(old_depth_test)
            self.gpu.state.depth_mask_set(old_depth_mask)

    def read_pixels(self, *, cancel=None, progress=None):
        return read_framebuffer(self.framebuffer, self.width, self.height,
                                cancel=cancel, progress=progress)


# ---------------------------------------------------------------------------
# Renderer


class GPURenderer:
    def __init__(self, rscene, *, viewport=False, cancel=None, status=None):
        import gpu
        from gpu_extras.batch import batch_for_shader

        self.gpu = gpu
        self.rscene = rscene
        self.viewport = viewport
        self.cancel = cancel
        self.status = status or (lambda message: None)
        self.status("Allocating GPU textures")
        checkpoint(self.cancel)
        self.width = max(1, int(rscene.width))
        self.height = max(1, int(rscene.height))
        self.texture_cache = {}
        self.resource_warnings = []
        self.shadow_cache = {}
        self.shadow_warnings = []
        self.reflection_map = GPUReflection()
        self.reflection_warnings = []
        self.color_texture = None
        self.depth_texture = None
        self.framebuffer = None
        self._white_depth = None
        self.white_texture = self._make_white_texture()
        self.environment_gpu = self._upload_texture(getattr(rscene, "environment_texture", None))
        if self.environment_gpu is not None and getattr(rscene.settings, "reflection_mode", "AUTO") != "OFF":
            self.reflection_map = GPUReflection(kind="ENV2D")
        self.background_shader = _create_background_shader()
        bg_pos = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
        self.background_batch = batch_for_shader(self.background_shader, "TRIS", {"pos": bg_pos})
        self.groups = []
        self.reyes_chunks = []
        self.microgrid_cache = {}

        settings = rscene.settings
        requested = getattr(settings, "geometry_mode", "AUTO")
        if requested == "MICROPOLYGON":
            self.use_reyes = True
        elif requested == "TRIANGLES":
            self.use_reyes = False
        else:
            self.use_reyes = getattr(settings, "quality", "PRODUCTION") != "DRAFT"
            if getattr(settings, "displacement_enabled", True):
                self.use_reyes = self.use_reyes or self._has_displacement()

        self.status("Compiling RetroMan shaders")
        checkpoint(self.cancel)
        self.base_shader, self.light_shader, self.shadow_shader = _create_shader_set(reyes=self.use_reyes)

        self.status("Uploading geometry")
        checkpoint(self.cancel)
        if self.use_reyes:
            self._build_reyes_geometry(batch_for_shader)
        else:
            self._build_fast_geometry(batch_for_shader)

        self.resize(self.width, self.height)

    # ----- GPU resources ----------------------------------------------------
    def _make_white_texture(self):
        gpu = self.gpu
        buf = gpu.types.Buffer("FLOAT", 4, [1.0, 1.0, 1.0, 1.0])
        tex = gpu.types.GPUTexture((1, 1), format="RGBA16F", data=buf)
        _texture_option(tex, "filter_mode", False)
        return tex

    def _upload_texture(self, texture, *, precise=False):
        if texture is None:
            return None
        key = (id(texture),precise)
        cached = self.texture_cache.get(key)
        if cached is not None:
            return cached
        try:
            buf = self.gpu.types.Buffer("FLOAT", len(texture.pixels), texture.pixels)
            tex = self.gpu.types.GPUTexture((texture.width, texture.height), format="RGBA32F" if precise else "RGBA16F", data=buf)
            mode = str(getattr(self.rscene.settings, "texture_filter", "BILINEAR")).upper()
            if mode == "NEAREST":
                _texture_option(tex, "filter_mode", False)
            elif mode == "BILINEAR":
                _texture_option(tex, "filter_mode", True)
            else:
                _texture_option(tex, "filter_mode", True)
                try:
                    tex.mipmap_mode(True, True)
                except Exception:
                    pass
                if mode == "ANISO":
                    try:
                        tex.anisotropic_filter(True)
                    except Exception:
                        pass
            _texture_option(tex, "extend_mode", "REPEAT")
            self.texture_cache[key] = tex
            return tex
        except Exception as exc:
            name = getattr(texture, "name", "<texture>")
            msg = f"GPU upload failed for {name}: {exc}"
            if msg not in self.resource_warnings:
                self.resource_warnings.append(msg)
            return None

    @staticmethod
    def _triangle_center(triangles):
        if not triangles:
            return Vector((0.0, 0.0, 0.0))
        acc = Vector((0.0, 0.0, 0.0))
        count = 0
        for tri in triangles:
            for v in (tri.a, tri.b, tri.c):
                acc += v.p
                count += 1
        return acc / max(1, count)

    def _materials(self):
        triangles = self.rscene.triangles
        return triangles.materials() if hasattr(triangles, "materials") else (t.material for t in triangles)

    def _has_displacement(self):
        return any(m.has_displacement for m in self._materials())

    def _build_packed_geometry(self, batch_for_shader):
        try:
            transparent_chunk = max(1, min(int(self.gpu.capabilities.max_batch_vertices_get()) // 3, 8192))
        except Exception:
            transparent_chunk = 8192
        for material, (positions, normals, uv) in self.rscene.triangles.material_arrays(self.cancel):
            transparent = material.alpha < 0.999 or material.use_texture_alpha
            chunk_vertices = 3 * (transparent_chunk if transparent else 32768)
            textures = [self._upload_texture(t) for t in (material.texture,material.displacement_texture,material.normal_texture,material.bump_texture)]
            for start in range(0,len(positions),chunk_vertices):
                checkpoint(self.cancel)
                end=start+chunk_vertices
                p,n,st=positions[start:end],normals[start:end],uv[start:end]
                import numpy as np
                # Deduplicate exact attribute bytes, preserving triangle order
                # through indices. UV seams and split normals remain distinct.
                attributes=np.empty((len(p),8),dtype=np.float32)
                attributes[:,:3]=p;attributes[:,3:6]=n;attributes[:,6:]=st
                _,first,inverse=np.unique(attributes.view('V32').reshape(-1),return_index=True,return_inverse=True)
                compact=attributes[first]
                batch=batch_for_shader(self.base_shader,"TRIS",
                    {"pos":np.ascontiguousarray(compact[:,:3]),"normal":np.ascontiguousarray(compact[:,3:6]),"uv":np.ascontiguousarray(compact[:,6:])},
                    indices=inverse.astype(np.int32).reshape(-1,3))
                center=None
                if transparent:
                    # Preserve the previous float32 accumulation/order exactly.
                    center=Vector((0,0,0))
                    for point in p.tolist():center+=Vector(point)
                    center/=len(p)
                self.groups.append(GPUBatchGroup(material=material,batch=batch,triangle_count=len(p)//3,
                    texture=textures[0],displacement_texture=textures[1],normal_texture=textures[2],bump_texture=textures[3],center=center,bounds=self._batch_bounds(p)))

    def _build_fast_geometry(self, batch_for_shader):
        if hasattr(self.rscene.triangles, "material_arrays"):
            self._build_packed_geometry(batch_for_shader)
            return
        grouped = {}
        for tri_index, tri in enumerate(self.rscene.triangles):
            if tri_index % 256 == 0:
                checkpoint(self.cancel)
            grouped.setdefault(id(tri.material), [tri.material, []])[1].append(tri)
        try:
            max_vertices = int(self.gpu.capabilities.max_batch_vertices_get())
        except Exception:
            max_vertices = 393216
        tri_chunk = max(1, min(max_vertices // 3, 8192))
        for material, triangles in grouped.values():
            base_tex = self._upload_texture(material.texture)
            disp_tex = self._upload_texture(material.displacement_texture)
            normal_tex = self._upload_texture(material.normal_texture)
            bump_tex = self._upload_texture(material.bump_texture)
            # GL_MAX_ELEMENTS_VERTICES is a draw-range recommendation, not a
            # buffer limit. Bound allocations ourselves for opaque geometry.
            # Keep transparent chunks unchanged: their centers define sorting.
            transparent = material.alpha < 0.999 or material.use_texture_alpha
            chunk_size = tri_chunk if transparent else 32768
            for start in range(0, len(triangles), chunk_size):
                checkpoint(self.cancel)
                block = triangles[start:start + chunk_size]
                pos, normal, uv = [], [], []
                for tri in block:
                    for v in (tri.a, tri.b, tri.c):
                        pos.append(tuple(v.p))
                        normal.append(tuple(v.n))
                        uv.append(v.uv)
                batch = batch_for_shader(self.base_shader, "TRIS", {"pos": pos, "normal": normal, "uv": uv})
                self.groups.append(GPUBatchGroup(
                    material=material,
                    batch=batch,
                    triangle_count=len(block),
                    texture=base_tex,
                    displacement_texture=disp_tex,
                    normal_texture=normal_tex,
                    bump_texture=bump_tex,
                    center=self._triangle_center(block) if transparent else None,
                    bounds=self._batch_bounds(pos),
                ))

    def _microgrid_batch(self, subdiv, batch_for_shader):
        subdiv = max(1, int(subdiv))
        cached = self.microgrid_cache.get(subdiv)
        if cached is not None:
            return cached
        bary = make_barycentric_microgrid(subdiv)
        centers = []
        for i in range(0, len(bary), 3):
            center = tuple(sum(bary[i+j][k] for j in range(3)) / 3.0 for k in range(3))
            centers.extend((center,) * 3)
        batch = batch_for_shader(self.base_shader, "TRIS", {"bary": bary, "shadeBary": centers})
        self.microgrid_cache[subdiv] = batch
        return batch

    def _final_vp(self):
        return self.rscene.camera.projection @ self.rscene.camera.matrix_world.inverted()

    def _subdiv_for_triangle(self, tri):
        settings = self.rscene.settings
        if self.viewport:
            return max(1, int(getattr(settings, "viewport_subdiv", 2)))
        cap = max(1, int(getattr(settings, "gpu_max_subdiv", 8)))
        if not getattr(settings, "adaptive_dicing", True):
            return cap
        return estimate_subdiv(
            tri,
            self._final_vp(),
            self.width,
            self.height,
            float(getattr(settings, "shading_rate", 4.0)),
            cap,
        )

    @staticmethod
    def _pack_triangle_row(tri, row):
        if row == 0:
            p = tri.a.p; return (float(p.x), float(p.y), float(p.z), 1.0)
        if row == 1:
            p = tri.b.p; return (float(p.x), float(p.y), float(p.z), 1.0)
        if row == 2:
            p = tri.c.p; return (float(p.x), float(p.y), float(p.z), 1.0)
        if row == 3:
            n = tri.a.n; return (float(n.x), float(n.y), float(n.z), 0.0)
        if row == 4:
            n = tri.b.n; return (float(n.x), float(n.y), float(n.z), 0.0)
        if row == 5:
            n = tri.c.n; return (float(n.x), float(n.y), float(n.z), 0.0)
        if row == 6:
            return (float(tri.a.uv[0]), float(tri.a.uv[1]), float(tri.b.uv[0]), float(tri.b.uv[1]))
        return (float(tri.c.uv[0]), float(tri.c.uv[1]), 0.0, 0.0)

    def _triangle_texture(self, triangles):
        count = len(triangles)
        flat = []
        for row in range(8):
            for tri in triangles:
                flat.extend(self._pack_triangle_row(tri, row))
        buf = self.gpu.types.Buffer("FLOAT", len(flat), flat)
        tex = self.gpu.types.GPUTexture((count, 8), format="RGBA32F", data=buf)
        _texture_option(tex, "filter_mode", False)
        _texture_option(tex, "extend_mode", "EXTEND")
        return tex

    def _build_packed_reyes_geometry(self, batch_for_shader):
        """Upload viewport patch arrays without allocating Python triangles."""
        import numpy as np
        subdiv = max(1, int(getattr(self.rscene.settings, "viewport_subdiv", 2)))
        batch = self._microgrid_batch(subdiv, batch_for_shader)
        width = min(4096, int(self.gpu.capabilities.max_texture_size_get()))
        for material, (positions, normals, uv) in self.rscene.triangles.material_arrays(self.cancel):
            textures = [self._upload_texture(t) for t in (material.texture, material.displacement_texture,
                                                         material.normal_texture, material.bump_texture)]
            for start in range(0, len(positions), width * 3):
                checkpoint(self.cancel)
                p = positions[start:start+width*3].reshape(-1,3,3)
                n = normals[start:start+width*3].reshape(-1,3,3)
                st = uv[start:start+width*3].reshape(-1,3,2)
                count = len(p)
                data = np.zeros((8,count,4),dtype=np.float32)
                data[0:3,:,:3] = p.transpose(1,0,2)
                data[0:3,:,3] = 1.0
                data[3:6,:,:3] = n.transpose(1,0,2)
                data[6,:,:2] = st[:,0]; data[6,:,2:] = st[:,1]; data[7,:,:2] = st[:,2]
                buf = self.gpu.types.Buffer("FLOAT",data.size,data.reshape(-1))
                tex = self.gpu.types.GPUTexture((count,8),format="RGBA32F",data=buf)
                _texture_option(tex,"filter_mode",False)
                _texture_option(tex,"extend_mode","EXTEND")
                self.reyes_chunks.append(GPUReyesChunk(material=material,triangle_texture=tex,
                    triangle_count=count,subdiv=subdiv,batch=batch,texture=textures[0],
                    displacement_texture=textures[1],normal_texture=textures[2],bump_texture=textures[3],
                    center=Vector(p.mean(axis=(0,1)).tolist())))

    def _build_reyes_geometry(self, batch_for_shader):
        if self.viewport and hasattr(self.rscene.triangles, "material_arrays"):
            self._build_packed_reyes_geometry(batch_for_shader)
            return
        # Group by material and dice level, then chunk to a safe texture width.
        buckets = {}
        settings = self.rscene.settings
        adaptive = not self.viewport and getattr(settings, "adaptive_dicing", True)
        cap = max(1, int(getattr(settings, "gpu_max_subdiv", 8)))
        rate = float(getattr(settings, "shading_rate", 1.0))
        total_patches = 0
        limited = 0
        for tri_index, source_tri in enumerate(self.rscene.triangles):
            if tri_index % 256 == 0:
                checkpoint(self.cancel)
                self.status(f"Splitting and dicing surface {tri_index + 1}/{len(self.rscene.triangles)}")
            patches = split_for_dicing(source_tri, self._final_vp(), self.width, self.height,
                                      rate, cap, self.cancel) if adaptive else [(source_tri, False)]
            for tri, reached_limit in patches:
                total_patches += 1
                if total_patches > max(2000000, len(self.rscene.triangles) * 4):
                    raise RuntimeError("Dicing exceeds the patch memory budget. Increase Shading Rate or reduce output resolution.")
                limited += int(reached_limit)
                subdiv = self._subdiv_for_triangle(tri)
                buckets.setdefault((id(tri.material), subdiv), [tri.material, subdiv, []])[2].append(tri)
        if limited:
            self.resource_warnings.append(f"{limited} patches cross the eye plane or reach the split limit; their shading rate is not guaranteed.")

        try:
            max_tex = int(self.gpu.capabilities.max_texture_size_get())
        except Exception:
            max_tex = 4096
        chunk_width = max(1, min(4096, max_tex))

        for material, subdiv, triangles in buckets.values():
            batch = self._microgrid_batch(subdiv, batch_for_shader)
            base_tex = self._upload_texture(material.texture)
            disp_tex = self._upload_texture(material.displacement_texture)
            for start in range(0, len(triangles), chunk_width):
                checkpoint(self.cancel)
                block = triangles[start:start + chunk_width]
                self.reyes_chunks.append(GPUReyesChunk(
                    material=material,
                    triangle_texture=self._triangle_texture(block),
                    triangle_count=len(block),
                    subdiv=subdiv,
                    batch=batch,
                    texture=base_tex,
                    displacement_texture=disp_tex,
                    normal_texture=self._upload_texture(material.normal_texture),
                    bump_texture=self._upload_texture(material.bump_texture),
                    center=self._triangle_center(block),
                ))

    def set_viewport_subdiv(self, subdiv):
        """Switch a viewport REYES renderer to another uniform grid without re-uploading source triangles."""
        if not self.viewport or not self.use_reyes or not self.reyes_chunks:
            return
        from gpu_extras.batch import batch_for_shader
        subdiv = max(1, min(8, int(subdiv)))
        batch = self._microgrid_batch(subdiv, batch_for_shader)
        for chunk in self.reyes_chunks:
            chunk.subdiv = subdiv
            chunk.batch = batch

    @property
    def current_subdiv(self):
        if not self.use_reyes or not self.reyes_chunks:
            return 1
        return int(self.reyes_chunks[0].subdiv)

    def _begin_auxiliary_dicing(self):
        """Temporarily raise REYES subdivision for shadow/reflection map passes."""
        if not self.use_reyes or not self.reyes_chunks:
            return []
        from gpu_extras.batch import batch_for_shader
        minimum = max(1, int(getattr(self.rscene.settings, "auxiliary_min_subdiv", 2)))
        cap = max(minimum, int(getattr(self.rscene.settings, "gpu_max_subdiv", 8)))
        saved = []
        for chunk in self.reyes_chunks:
            saved.append((chunk, chunk.subdiv, chunk.batch))
            target = min(cap, max(int(chunk.subdiv), minimum))
            if target != chunk.subdiv:
                chunk.subdiv = target
                chunk.batch = self._microgrid_batch(target, batch_for_shader)
        return saved

    @staticmethod
    def _restore_auxiliary_dicing(saved):
        for chunk, subdiv, batch in saved:
            chunk.subdiv = subdiv
            chunk.batch = batch

    def resize(self, width, height):
        width = max(1, int(width)); height = max(1, int(height))
        if self.color_texture is not None and width == self.width and height == self.height:
            return
        self.width, self.height = width, height
        self.color_texture = self.gpu.types.GPUTexture((width, height), format="RGBA16F")
        self.depth_texture = self.gpu.types.GPUTexture((width, height), format="DEPTH_COMPONENT32F")
        self.framebuffer = self.gpu.types.GPUFrameBuffer(depth_slot=self.depth_texture, color_slots=(self.color_texture,))

    # ----- Uniform binding / drawing ---------------------------------------
    @staticmethod
    def _light_direction(light):
        if light.type == "SUN":
            d = light.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
        else:
            d = light.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
        if d.length_squared <= 1.0e-12:
            return Vector((0.0, 0.0, -1.0))
        return d.normalized()

    def _bind_geometry(self, shader, drawable, *, for_shadow=False):
        m = drawable.material
        use_disp = bool(getattr(self.rscene.settings, "displacement_enabled", True))
        if for_shadow and not getattr(self.rscene.settings, "displacement_shadow_maps", True):
            use_disp = False
        use_disp = use_disp and m.has_displacement
        disp_tex = drawable.displacement_texture
        shader.uniform_bool("useDisplacement", use_disp)
        shader.uniform_bool("useDisplacementTexture", use_disp and disp_tex is not None)
        shader.uniform_float("displacementScale", float(m.displacement_scale))
        shader.uniform_float("displacementMidlevel", float(m.displacement_midlevel))
        shader.uniform_float("displacementConstant", float(m.displacement_constant))
        shader.uniform_sampler("displacementTex", disp_tex or self.white_texture)
        if self.use_reyes:
            shader.uniform_sampler("triData", drawable.triangle_texture)

    def _bind_reflection(self, shader, material):
        settings = self.rscene.settings
        mode = getattr(settings, "reflection_mode", "AUTO")
        kind = 0
        reflection = self.reflection_map
        if mode != "OFF" and getattr(self, "_allow_reflections", True):
            if reflection.kind == "CUBE6" and len(reflection.textures) >= 6:
                kind = 2
            elif self.environment_gpu is not None:
                kind = 1
        shader.uniform_int("reflectionKind", kind)
        shader.uniform_float("materialReflectivity", float(material.reflection_amount))
        shader.uniform_float("reflectionGlobalStrength", float(getattr(settings, "reflection_strength", 1.0)))
        shader.uniform_float("environmentStrength", float(getattr(self.rscene, "environment_strength", 1.0)))
        shader.uniform_sampler("environmentTex", self.environment_gpu or self.white_texture)

        identity = Matrix.Identity(4)
        probe_pos = Vector((0.0, 0.0, 0.0))
        if reflection.probe_position is not None:
            probe_pos = reflection.probe_position
        shader.uniform_float("reflectionProbePos", probe_pos)
        for i in range(6):
            tex = self.white_texture
            matrix = identity
            if kind == 2 and i < len(reflection.textures):
                tex = reflection.textures[i]
                matrix = reflection.matrices[i]
            shader.uniform_sampler(f"reflection{i}", tex)
            shader.uniform_float(f"reflectionMatrix{i}", matrix)

    def _bind_material(self, shader, drawable, *, base_pass):
        m = drawable.material
        shader.uniform_float("baseColor", m.base_color)
        shader.uniform_float("exposureMul", 2.0 ** float(self.rscene.settings.exposure))
        use_tex = drawable.texture is not None
        shader.uniform_bool("useTexture", use_tex)
        shader.uniform_bool("useTextureAlpha", bool(use_tex and getattr(m, "use_texture_alpha", False)))
        shader.uniform_sampler("baseTex", drawable.texture or self.white_texture)
        shader.uniform_bool("useNormalTexture", drawable.normal_texture is not None)
        shader.uniform_bool("useBumpTexture", drawable.bump_texture is not None)
        shader.uniform_sampler("normalTex", drawable.normal_texture or self.white_texture)
        shader.uniform_sampler("bumpTex", drawable.bump_texture or self.white_texture)
        shader.uniform_float("normalStrength", float(getattr(m, "normal_strength", 1.0)))
        shader.uniform_float("normalYSign", float(getattr(m, "normal_y_sign", 1.0)))
        shader.uniform_float("bumpStrength", float(getattr(m, "bump_strength", 1.0)))
        shader.uniform_float("bumpSign", float(getattr(m, "bump_sign", 1.0)))
        shader.uniform_float("bumpDistance", float(getattr(m, "bump_distance", 0.1)))
        if base_pass:
            environment = getattr(self.rscene, "environment_diffuse", ())
            shader.uniform_float("environmentAmbient", environment or (0.0, 0.0, 0.0))
            world = self.rscene.world_color if self.rscene.settings.use_world_color else (1.0, 1.0, 1.0)
            ambient = float(self.rscene.settings.ambient)
            shader.uniform_float("ambientColor", (world[0] * ambient, world[1] * ambient, world[2] * ambient))
            shader.uniform_float("emissionColor", m.emission_color)
            shader.uniform_float("emissionStrength", float(m.emission_strength))
            shader.uniform_float("materialAlpha", float(m.alpha))
            shader.uniform_float("materialKa", max(0.0, float(getattr(m, "ka", 1.0))))
            shader.uniform_bool("displacementNormal", bool(getattr(self.rscene.settings, "displacement_enabled", True) and m.has_displacement))
            self._bind_reflection(shader, m)
        else:
            shader.uniform_float("roughness", max(0.001, min(1.0, float(m.roughness))))
            shader.uniform_float("metallic", max(0.0, min(1.0, float(m.metallic))))
            shader.uniform_float("materialKd", max(0.0, float(getattr(m, "kd", 1.0 - m.metallic))))
            shader.uniform_float("materialKs", max(0.0, float(getattr(m, "ks", m.specular))))
            shader.uniform_float("materialAlpha", float(m.alpha))
            shader.uniform_bool("displacementNormal", bool(getattr(self.rscene.settings, "displacement_enabled", True) and m.has_displacement))
            shader.uniform_bool("clampSpecular", bool(self.rscene.settings.clamp_fireflies))
            shader.uniform_float("maxSpecular", float(self.rscene.settings.max_specular))

    def _draw_one(self, drawable, shader):
        checkpoint(self.cancel)
        shader = shader.prepare_draw()
        if self.use_reyes:
            drawable.batch.draw_instanced(shader, instance_count=drawable.triangle_count)
        else:
            drawable.batch.draw(shader)

    def _drawables(self):
        return self.reyes_chunks if self.use_reyes else self.groups

    @staticmethod
    def _is_transparent_drawable(drawable):
        # Material alpha is the reliable translated signal. Image alpha cutouts
        # still use discard in the shader; arbitrary semi-transparent texture
        # alpha is intentionally not guessed at scene-translation time.
        m = drawable.material
        return (
            float(getattr(m, "alpha", 1.0)) < 0.999
            or bool(getattr(m, "use_texture_alpha", False))
        )

    @staticmethod
    def _batch_bounds(positions):
        import numpy as np
        points=np.asarray(positions,dtype=np.float32)
        lo=points.min(axis=0);hi=points.max(axis=0)
        return tuple(Vector((float(x),float(y),float(z),1.0))
                     for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2]))

    @staticmethod
    def _batch_visible(drawable, view_projection):
        corners=getattr(drawable,"bounds",None)
        # Source bounds do not enclose shader displacement; never cull it.
        if corners is None or drawable.material.has_displacement:
            return True
        projected=[view_projection @ p for p in corners]
        # Only lateral clip planes, avoiding backend-specific depth conventions.
        # Tolerance keeps boundary geometry despite float32 projection rounding.
        tolerance=1e-5*max(1.0,*(abs(v) for p in projected for v in p))
        return not any(all(sign*p[axis]+p.w < -tolerance for p in projected)
                       for axis in (0,1) for sign in (-1,1))

    def _ordered_drawables(self, camera_pos, view_projection=None):
        drawables = [d for d in self._drawables() if view_projection is None or self._batch_visible(d,view_projection)]
        opaque = [d for d in drawables if not self._is_transparent_drawable(d)]
        transparent = [d for d in drawables if self._is_transparent_drawable(d)]
        # Approximate classic painter ordering at material/chunk granularity.
        transparent.sort(
            key=lambda d: ((d.center or Vector((0.0, 0.0, 0.0))) - camera_pos).length_squared,
            reverse=True,
        )
        return opaque, transparent

    def _bind_light_parameters(self, shader, light):
        shader.uniform_int("lightType", {"SUN":0,"POINT":1,"SPOT":2,"AREA":3,"MESH":4}.get(light.type,1))
        shader.uniform_float("lightPos",light.matrix_world.translation)
        shader.uniform_float("lightDir",self._light_direction(light))
        shader.uniform_float("lightColor",light.color)
        for uniform, value in (("lightEnergy",light.energy),("lightAreaFactor",light.area_factor),
            ("lightDiffuseFactor",light.diffuse_factor),("lightSpecularFactor",light.specular_factor),
            ("lightCutoff",light.cutoff_distance)):
            shader.uniform_float(uniform,float(value))
        shader.uniform_float("spotOuterCos",math.cos(light.spot_size*.5))
        shader.uniform_float("spotInnerCos",math.cos(light.spot_size*.5*(1-max(0,min(1,light.spot_blend)))))
        self._bind_shadow(shader,light)

    def _draw_material_pass(self, drawables, shader, *, base_pass):
        previous = None
        for drawable in drawables:
            material = drawable.material
            changed = material is not previous
            # A pass begins with fresh bindings. Every batch of a translated
            # material shares its uploaded textures; REYES triangle textures
            # are per chunk and must still be rebound on every draw.
            if changed or self.use_reyes:
                self._bind_geometry(shader, drawable)
            if changed:
                self._bind_material(shader, drawable, base_pass=base_pass)
            self._draw_one(drawable, shader)
            previous = material

    def _draw_depth_geometry(self, shader, view_projection=None):
        previous = None
        for drawable in self._drawables():
            if drawable.material.alpha * drawable.material.base_color[3] <= 0.003:
                continue
            if view_projection is not None and not self._batch_visible(drawable,view_projection):
                continue
            if drawable.material is not previous or self.use_reyes:
                self._bind_geometry(shader, drawable, for_shadow=True)
            self._draw_one(drawable, shader)
            previous = drawable.material

    # ----- Automatic period-style shadow maps -------------------------------
    def _geometry_bounds(self):
        # A renderer owns one geometry snapshot. Light/view changes reuse it;
        # geometry changes create a new renderer, so this cache cannot go stale.
        bounds = getattr(self, "_cached_geometry_bounds", None)
        if bounds is None:
            from .raster import _scene_bounds
            bounds = _scene_bounds(self.rscene.triangles, self.cancel)
            self._cached_geometry_bounds = bounds
        return bounds

    def _shadow_resolution(self):
        settings = self.rscene.settings
        if self.viewport:
            return int(getattr(settings, "viewport_shadow_resolution", 512))
        return int(settings.shadow_resolution)

    def _build_shadow(self, light):
        size = self._shadow_resolution()
        settings = self.rscene.settings
        if not settings.enable_shadows or (self.viewport and not getattr(settings, "viewport_shadows", True)) or not light.use_shadow:
            return GPUShadow()

        result = GPUShadow()
        self.gpu.state.blend_set("NONE")
        self.gpu.state.depth_test_set("LESS_EQUAL")
        self.gpu.state.depth_mask_set(True)
        self.gpu.state.face_culling_set("NONE")

        def render_depth(vp):
            checkpoint(self.cancel)
            depth = self.gpu.types.GPUTexture((size, size), format="DEPTH_COMPONENT32F")
            _texture_option(depth, "filter_mode", False)
            _texture_option(depth, "extend_mode", "CLAMP_TO_BORDER")
            fb = self.gpu.types.GPUFrameBuffer(depth_slot=depth)
            with fb.bind():
                self.gpu.state.viewport_set(0, 0, size, size)
                fb.clear(depth=1.0)
                self.shadow_shader.bind()
                self.shadow_shader.uniform_float("viewProjection", vp)
                self._draw_depth_geometry(self.shadow_shader, vp)
            return depth

        if light.type in {"SPOT", "SUN"}:
            view, proj = _shadow_camera_for_light(light, self.rscene.triangles, size, self._geometry_bounds())
            if view is None:
                return result
            vp = proj @ view
            result.kind = "MAP2D"
            result.textures = [render_depth(vp)]
            result.matrices = [vp]
            return result

        if light.type in {"POINT", "AREA", "MESH"}:
            pos = light.matrix_world.translation.copy()
            dirs = [
                (Vector((1, 0, 0)), Vector((0, 0, 1))),
                (Vector((-1, 0, 0)), Vector((0, 0, 1))),
                (Vector((0, 1, 0)), Vector((0, 0, 1))),
                (Vector((0, -1, 0)), Vector((0, 0, 1))),
                (Vector((0, 0, 1)), Vector((0, 1, 0))),
                (Vector((0, 0, -1)), Vector((0, 1, 0))),
            ]
            proj = _perspective_matrix(math.radians(90.0), 1.0, 0.02, _shadow_far_for_light(light, self.rscene.triangles, self._geometry_bounds()))
            result.kind = "CUBE6"
            for face, (direction, up) in enumerate(dirs):
                self.status(f"Shadow {light.name}: face {face + 1}/6")
                view = _look_at_matrix(pos, direction, up)
                vp = proj @ view
                result.matrices.append(vp)
                result.textures.append(render_depth(vp))
            return result
        return result

    def build_shadow_maps(self, progress=None, cancel=None):
        settings = self.rscene.settings
        if not settings.enable_shadows or (self.viewport and not getattr(settings, "viewport_shadows", True)):
            return
        shadow_lights = [light for light in self.rscene.lights if light.use_shadow]
        max_lights = max(0, int(getattr(settings, "max_shadow_lights", 8)))
        if len(shadow_lights) > max_lights:
            self.shadow_warnings.append(
                f"{len(shadow_lights)} shadow-casting lights; GPU budget is {max_lights}. Extra lights render unshadowed."
            )
        shadow_lights = shadow_lights[:max_lights]
        total = max(1, len(shadow_lights))
        try:
            old_blend = self.gpu.state.blend_get()
        except Exception:
            old_blend = "NONE"
        try:
            old_depth_test = self.gpu.state.depth_test_get()
        except Exception:
            old_depth_test = "NONE"
        try:
            old_depth_mask = self.gpu.state.depth_mask_get()
        except Exception:
            old_depth_mask = True
        saved_dicing = self._begin_auxiliary_dicing()
        try:
            for index, light in enumerate(shadow_lights):
                if cancel and cancel():
                    return
                if progress:
                    progress(index / total)
                self.status(f"Shadow light {index + 1}/{len(shadow_lights)}: {light.name}")
                self.shadow_cache[id(light)] = self._build_shadow(light)
            if progress:
                progress(1.0)
        finally:
            self.gpu.shader.unbind()
            self._restore_auxiliary_dicing(saved_dicing)
            self.gpu.state.blend_set(old_blend)
            self.gpu.state.depth_test_set(old_depth_test)
            self.gpu.state.depth_mask_set(old_depth_mask)
            self.gpu.state.face_culling_set("NONE")

    def _bind_shadow(self, shader, light):
        shadow = self.shadow_cache.get(id(light))
        kind = 0
        if shadow is not None:
            if shadow.kind == "MAP2D": kind = 1
            elif shadow.kind == "CUBE6": kind = 2
        shader.uniform_int("shadowKind", kind)
        shader.uniform_float("shadowBias", float(self.rscene.settings.shadow_bias))
        shader.uniform_int("shadowRadius", int(self.rscene.settings.shadow_softness))

        if self._white_depth is None:
            self._white_depth = self.gpu.types.GPUTexture((1, 1), format="DEPTH_COMPONENT32F")
            self._white_depth.clear(format="FLOAT", value=(1.0,))
            _texture_option(self._white_depth, "filter_mode", False)

        identity = Matrix.Identity(4)
        for i in range(6):
            tex = self._white_depth
            matrix = identity
            if shadow is not None and i < len(shadow.textures):
                tex = shadow.textures[i]
                matrix = shadow.matrices[i]
            shader.uniform_sampler(f"shadow{i}", tex)
            shader.uniform_float(f"shadowMatrix{i}", matrix)

    # ----- Automatic period-style environment / reflection maps ------------
    def _reflection_resolution(self):
        settings = self.rscene.settings
        if self.viewport:
            return int(getattr(settings, "viewport_reflection_resolution", 128))
        return int(getattr(settings, "reflection_resolution", 256))

    def _auto_probe(self):
        if not self.rscene.triangles:
            return self.rscene.reflection_probes[0] if self.rscene.reflection_probes else None
        lo, hi = self._geometry_bounds()
        center = (lo + hi) * 0.5
        extent = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 1.0)
        if self.rscene.reflection_probes:
            return min(self.rscene.reflection_probes, key=lambda p: (p.position - center).length_squared)
        from .model import ReflectionProbeData
        return ReflectionProbeData("RetroMan Auto Probe", center, 0.02, max(10.0, extent * 2.5), extent)

    def _snapshot_color_texture(self):
        with self.framebuffer.bind():
            buf = self.framebuffer.read_color(0, 0, self.width, self.height, 4, 0, "FLOAT")
        tex = self.gpu.types.GPUTexture((self.width, self.height), format="RGBA16F", data=buf)
        _texture_option(tex, "filter_mode", True)
        _texture_option(tex, "extend_mode", "EXTEND")
        return tex

    def _draw_background(self, view_projection, *, force=False):
        if (getattr(self.rscene, "film_transparent", False) and not force) or self.environment_gpu is None or not getattr(self.rscene.settings, "use_world_color", True):
            return
        try:
            inv = view_projection.inverted()
        except Exception:
            return
        self.gpu.state.depth_test_set("NONE")
        self.gpu.state.depth_mask_set(False)
        self.gpu.state.blend_set("NONE")
        self.background_shader.bind()
        self.background_shader.uniform_float("invViewProjection", inv)
        self.background_shader.uniform_float("environmentStrength", float(getattr(self.rscene, "environment_strength", 1.0)))
        self.background_shader.uniform_float("exposureMul", 2.0 ** float(self.rscene.settings.exposure))
        self.background_shader.uniform_sampler("environmentTex", self.environment_gpu)
        self.background_batch.draw(self.background_shader)

    def _render_scene_pass(self, view_projection, camera_pos, *, allow_reflections=True, respect_film=True, progress=None, cancel=None):
        world = self.rscene.world_color if self.rscene.settings.use_world_color else (0.0, 0.0, 0.0)
        old_allow = getattr(self, "_allow_reflections", True)
        self._allow_reflections = bool(allow_reflections)
        opaque, transparent_drawables = self._ordered_drawables(camera_pos, view_projection)
        try:
            old_blend = self.gpu.state.blend_get()
        except Exception:
            old_blend = "NONE"
        try:
            old_depth_test = self.gpu.state.depth_test_get()
        except Exception:
            old_depth_test = "NONE"
        try:
            old_depth_mask = self.gpu.state.depth_mask_get()
        except Exception:
            old_depth_mask = True
        try:
            with self.framebuffer.bind():
                self.gpu.state.viewport_set(0, 0, self.width, self.height)
                transparent_film = bool(getattr(self.rscene, "film_transparent", False) and respect_film)
                clear = (0.0, 0.0, 0.0, 0.0) if transparent_film else (world[0], world[1], world[2], 1.0)
                # OpenGL depth clears respect the inherited depth write mask.
                self.gpu.state.depth_mask_set(True)
                self.framebuffer.clear(color=clear, depth=1.0)
                self.gpu.state.face_culling_set("NONE")
                self._draw_background(view_projection, force=not respect_film)

                # Base/ambient pass: opaque first, then alpha materials back-to-front.
                self.gpu.state.depth_test_set("LESS_EQUAL")
                self.gpu.state.depth_mask_set(True)
                self.gpu.state.blend_set("NONE")
                self.base_shader.bind()
                self.base_shader.uniform_float("viewProjection", view_projection)
                self.base_shader.uniform_float("cameraPos", camera_pos)
                self._draw_material_pass(opaque, self.base_shader, base_pass=True)

                if transparent_drawables:
                    self.gpu.state.depth_mask_set(False)
                    self.gpu.state.blend_set("ALPHA")
                    self._draw_material_pass(transparent_drawables, self.base_shader, base_pass=True)

                # Direct lighting is additive. Keep depth writes disabled, and the
                # light shader itself scales RGB by material/texture opacity.
                self.gpu.state.depth_mask_set(False)
                self.gpu.state.depth_test_set("LESS_EQUAL")
                # Source RGB is already weighted; ADDITIVE would multiply it by alpha again.
                self.gpu.state.blend_set("ADDITIVE_PREMULT")

                lights = self.rscene.lights
                total = max(1, len(lights))
                ordered = opaque + transparent_drawables
                for li, light in enumerate(lights):
                    if cancel and cancel():
                        break
                    if progress:
                        progress(li / total)
                    self.status(f"Lighting {li + 1}/{len(lights)}: {light.name}")
                    shader = self.light_shader
                    shader.bind()
                    shader.uniform_float("viewProjection", view_projection)
                    shader.uniform_float("cameraPos", camera_pos)
                    ltype = {"SUN": 0, "POINT": 1, "SPOT": 2, "AREA": 3, "MESH": 4}.get(light.type, 1)
                    shader.uniform_int("lightType", ltype)
                    shader.uniform_float("lightPos", light.matrix_world.translation)
                    shader.uniform_float("lightDir", self._light_direction(light))
                    shader.uniform_float("lightColor", light.color)
                    shader.uniform_float("lightEnergy", float(light.energy))
                    shader.uniform_float("lightAreaFactor", float(getattr(light, "area_factor", 1.0)))
                    shader.uniform_float("lightDiffuseFactor", float(getattr(light, "diffuse_factor", 1.0)))
                    shader.uniform_float("lightSpecularFactor", float(getattr(light, "specular_factor", 1.0)))
                    shader.uniform_float("lightCutoff", float(light.cutoff_distance))
                    outer = math.cos(float(light.spot_size) * 0.5)
                    inner = math.cos(float(light.spot_size) * 0.5 * (1.0 - max(0.0, min(1.0, float(light.spot_blend)))))
                    shader.uniform_float("spotOuterCos", outer)
                    shader.uniform_float("spotInnerCos", inner)
                    self._bind_shadow(shader, light)

                    self._draw_material_pass(ordered, shader, base_pass=False)
        finally:
            # Blender 5.0 can retain stale state if a bound shader is freed.
            self.gpu.shader.unbind()
            self._allow_reflections = old_allow
            # A custom RenderEngine shares Blender's GPU context. Restore the
            # state we inherited rather than assuming Blender's next draw starts
            # from a particular blend/depth configuration.
            try:
                self.gpu.state.blend_set(old_blend)
                self.gpu.state.depth_mask_set(old_depth_mask)
                self.gpu.state.depth_test_set(old_depth_test)
                self.gpu.state.face_culling_set("NONE")
            except Exception:
                pass
        return self.color_texture

    def build_reflection_map(self, progress=None, cancel=None):
        settings = self.rscene.settings
        mode = getattr(settings, "reflection_mode", "AUTO")
        if mode == "OFF":
            self.reflection_map = GPUReflection()
            return
        if mode == "WORLD":
            self.reflection_map = GPUReflection(kind="ENV2D" if self.environment_gpu is not None else "NONE")
            return
        if self.viewport and not getattr(settings, "viewport_reflections", True):
            self.reflection_map = GPUReflection(kind="ENV2D" if self.environment_gpu is not None else "NONE")
            return
        if not any(m.reflection_amount > 0.001 for m in self._materials()):
            self.reflection_map = GPUReflection(kind="ENV2D" if self.environment_gpu is not None else "NONE")
            return

        probe = self._auto_probe()
        if probe is None:
            self.reflection_map = GPUReflection(kind="ENV2D" if self.environment_gpu is not None else "NONE")
            return

        size = self._reflection_resolution()
        original_size = (self.width, self.height)
        dirs = [
            (Vector((1, 0, 0)), Vector((0, 0, 1))),
            (Vector((-1, 0, 0)), Vector((0, 0, 1))),
            (Vector((0, 1, 0)), Vector((0, 0, 1))),
            (Vector((0, -1, 0)), Vector((0, 0, 1))),
            (Vector((0, 0, 1)), Vector((0, 1, 0))),
            (Vector((0, 0, -1)), Vector((0, 1, 0))),
        ]
        proj = _perspective_matrix(math.radians(90.0), 1.0, max(0.001, probe.clip_start), max(probe.clip_start + 0.01, probe.clip_end))
        result = GPUReflection(kind="CUBE6", probe_position=probe.position.copy())
        saved_dicing = self._begin_auxiliary_dicing()
        try:
            self.resize(size, size)
            for face, (direction, up) in enumerate(dirs):
                if cancel and cancel():
                    return
                self.status(f"Reflection map: face {face + 1}/6")
                view = _look_at_matrix(probe.position, direction, up)
                vp = proj @ view
                result.matrices.append(vp)
                self._render_scene_pass(vp, probe.position, allow_reflections=False, respect_film=False, cancel=cancel)
                result.textures.append(self._snapshot_color_texture())
                if progress:
                    progress((face + 1) / 6.0)
        finally:
            self.resize(*original_size)
            self._restore_auxiliary_dicing(saved_dicing)
        self.reflection_map = result

    # ----- Frame rendering --------------------------------------------------
    def render_to_texture(self, *, view_projection=None, camera_pos=None, progress=None, cancel=None,
                          rebuild_shadows=True, rebuild_reflections=True):
        if view_projection is None:
            cam_view = self.rscene.camera.matrix_world.inverted()
            view_projection = self.rscene.camera.projection @ cam_view
            camera_pos = self.rscene.camera.matrix_world.translation
        elif camera_pos is None:
            camera_pos = Vector((0.0, 0.0, 0.0))

        checkpoint(self.cancel)
        if rebuild_shadows:
            self.status("Baking shadow maps")
            self.build_shadow_maps(progress=(lambda p: progress(p * 0.28) if progress else None), cancel=cancel)
        if cancel and cancel():
            return self.color_texture

        if rebuild_reflections:
            self.status("Baking reflection maps")
            self.build_reflection_map(progress=(lambda p: progress(0.28 + p * 0.34) if progress else None), cancel=cancel)
        if cancel and cancel():
            return self.color_texture

        self.status("Drawing scene lights and materials")
        self._render_scene_pass(
            view_projection, camera_pos,
            allow_reflections=True,
            progress=(lambda p: progress(0.62 + p * 0.37) if progress else None),
            cancel=cancel,
        )
        if progress:
            progress(1.0)
        return self.color_texture

    def read_pixels(self, *, cancel=None, progress=None):
        return read_framebuffer(self.framebuffer, self.width, self.height,
                                cancel=cancel, progress=progress)

    @property
    def triangle_count(self):
        return len(self.rscene.triangles)

    @property
    def micropolygon_count(self):
        if hasattr(self,"bucket_statistics"):return self.bucket_statistics["shaded_micropolygons"]
        if not self.use_reyes:
            return self.triangle_count
        return sum(c.triangle_count * c.subdiv * c.subdiv for c in self.reyes_chunks)

    @property
    def geometry_label(self):
        if hasattr(self,"bucket_statistics"):return "Bucket REYES (GPU shading / GPU sampling)"
        return "GPU surface-shaded micropolygons" if self.use_reyes else "GPU triangles (preview approximation)"
