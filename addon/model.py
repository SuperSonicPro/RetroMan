# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextureData:
    name: str
    width: int
    height: int
    pixels: object
    filepath: str = ""
    has_alpha: bool = False


@dataclass
class MaterialData:
    name: str
    base_color: tuple = (0.8, 0.8, 0.8, 1.0)
    roughness: float = 0.5
    metallic: float = 0.0
    alpha: float = 1.0
    specular: float = 0.5
    emission_color: tuple = (0.0, 0.0, 0.0)
    emission_strength: float = 0.0
    texture: Optional[TextureData] = None
    normal_texture: Optional[TextureData] = None
    normal_strength: float = 1.0
    normal_y_sign: float = 1.0
    bump_texture: Optional[TextureData] = None
    bump_strength: float = 1.0
    bump_sign: float = 1.0
    bump_distance: float = 0.1
    displacement_texture: Optional[TextureData] = None
    displacement_scale: float = 0.0
    displacement_midlevel: float = 0.5
    displacement_constant: float = 0.5
    # Classic RenderMan-style coefficients. Normal Blender materials are mapped
    # into these automatically; RetroSL can override them explicitly.
    ka: float = 1.0
    kd: float = 1.0
    ks: float = 0.5
    kr: float = -1.0  # negative means derive from Blender metallic/specular
    shader_name: str = "plastic"
    retrosl_source: str = ""
    use_texture_alpha: bool = False

    @property
    def has_displacement(self):
        return self.displacement_texture is not None or abs(self.displacement_scale) > 1.0e-9

    @property
    def reflection_amount(self):
        if self.kr >= 0.0:
            return max(0.0, min(1.0, self.kr))
        mirror = max(0.0, min(1.0, self.metallic))
        dielectric = max(0.0, min(1.0, self.specular)) * 0.18
        gloss = max(0.0, 1.0 - max(0.0, min(1.0, self.roughness)) * 0.72)
        return max(mirror, dielectric) * gloss


@dataclass(slots=True)
class VertexData:
    p: object
    n: object
    uv: tuple = (0.0, 0.0)


@dataclass(slots=True)
class TriangleData:
    a: VertexData
    b: VertexData
    c: VertexData
    material: MaterialData


@dataclass
class LightData:
    name: str
    type: str
    matrix_world: object
    color: tuple
    energy: float
    use_shadow: bool
    spot_size: float = 0.785398
    spot_blend: float = 0.15
    # 0 means Blender's Custom Distance is disabled (no hard cutoff).
    cutoff_distance: float = 0.0
    size: float = 0.25
    diffuse_factor: float = 1.0
    specular_factor: float = 1.0
    area_factor: float = 1.0
    normalize: bool = True
    shadow: object = None


@dataclass
class ReflectionProbeData:
    name: str
    position: object
    clip_start: float = 0.1
    clip_end: float = 50.0
    influence_distance: float = 2.5


@dataclass
class CameraData:
    matrix_world: object
    projection: object
    clip_start: float
    clip_end: float
    dof_enabled: bool = False
    focus_distance: float = 10.0
    aperture_radius: float = 0.0
    aperture_blades: int = 0
    aperture_ratio: float = 1.0
    aperture_rotation: float = 0.0
    camera_type: str = "PERSP"


@dataclass
class RenderScene:
    width: int
    height: int
    camera: CameraData
    triangles: list = field(default_factory=list)
    topology_signature: list = field(default_factory=list)
    lights: list = field(default_factory=list)
    mesh_lights: list = field(default_factory=list)
    environment_diffuse: tuple = ()
    reflection_probes: list = field(default_factory=list)
    world_color: tuple = (0.05, 0.05, 0.05)
    environment_texture: Optional[TextureData] = None
    environment_strength: float = 1.0
    film_transparent: bool = False
    settings: object = None
