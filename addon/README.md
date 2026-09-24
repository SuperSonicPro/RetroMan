# RetroMan 1.0.24

Dense scene fixes: automatic bucket subdivision, compact source geometry, and
progress updates inside long shading passes. Missing linked shader libraries
are identified explicitly; relink those files in Blender to restore the shaders.

F12 now uses an independent local GPU worker so Blender stays responsive.
Completed buckets and status stream back to the render window. Blender 5.2 runs
the helper in the background; Blender 5.0 may show a temporary small window.
A worker that exits or makes no progress for 120 seconds produces a visible error.

Modern Blender authoring with a classic REYES-style final renderer. GPU compute
now resolves the buckets as well as shading the surfaces. Shaded grids remain on
the GPU through stochastic visibility, depth ordering, opacity compositing and
sample reconstruction. The CPU receives completed image tiles rather than every
micropolygon. This preserves the classic rendering model; it does not add modern
path-traced illumination.

## Install and use
Install the matching RetroMan-1.0.24 platform ZIP through Blender Preferences > Get Extensions > Install
from Disk. Select RetroMan. Keep Quality at Production, Geometry at Auto, and
Device at Auto or GPU. Vulkan, OpenGL and Metal use the same bucket algorithm through
Blender's active graphics backend; the addon does not change Blender preferences.
Run RetroMan > Diagnostics > Run RetroMan GPU Self-Test to check your context.

Platform packages: Linux x86-64, Windows x64, and macOS Apple Silicon (13+). Windows/macOS packages are preview builds pending Blender testing on those operating systems. See BUILDING.md for build commands and validation limits. Linux GPU rendering is tested on RTX 3090.
The native library is retained as a numerical reference; GPU finals do not use it.
No compiler or CUDA installation is required. Modern materials, lights, cameras
and supported shader groups are translated automatically to the classic model.

## Pixar texture presets

Select an object and open **Material Properties → Pixar 128 Texture Presets**.
Choose a category and thumbnail, then **Apply Texture Preset**. The object needs
an ordinary Active Render UV map. A new material is placed in the active slot;
existing materials are preserved. The nodes remain editable in the Shader Editor,
and the new material appears among the current file's material assets.

Only selected images are loaded and packed into the scene, making saved files
portable across machines and addon upgrades. Texture credits are visible in the
panel and saved in material/image metadata and a Blender Text datablock named
**RetroMan — Pixar Texture Credits**. Use that text when crediting shared work.

All 128 color TIFFs are unchanged from Pixar's official archive. This is the
1993 collection as re-released in 2018, with later normal/bump maps excluded.
The images are CC BY 4.0, separately from the engine's GPL license. See
**THIRD_PARTY_NOTICES.md** for the creators, source, license and modification notes.

## Rendering controls
Pixel Samples is the square grid's axis count. Motion / Lens Samples can increase
the shared pool when shutter blur or DOF is enabled. Each pixel has deterministic
independent spatial, time and lens samples. Box, Gaussian and Mitchell filters
reconstruct samples within each pixel. The shutter and aperture controls remain
Blender's normal controls.

Shading Rate controls surface detail. Render Max Subdivision bounds an individual
grid. Bucket Size defaults to 64 and dense buckets split automatically on memory
pressure. Even a single-pixel bucket may still report a bucket-memory
limit. Progress and completed tiles appear during F12. Esc cancels cooperatively;
the isolated worker is terminated during cleanup.

The Rendered viewport remains a fast GPU micropolygon preview. Material Preview
is Blender's own preview. Explicit Draft/Fast Triangles and legacy CPU Reference
modes retain their older approximations and are not the production bucket path.

## Scope and limits
GPU stages include source-patch bounds, dicing/displacement, shading, mip-pyramid
construction/filtering, camera sample generation, clipping/visibility, per-sample
transparency sorting, reconstruction and mapped backgrounds. Blender dependency
graph evaluation, material translation, patch splitting, bucket scheduling,
uploads and image publication still require CPU work. This is not literally an
entirely CPU-free application, and a dense scene can remain preparation-bound.

Motion uses linear endpoint geometry and stable topology. Surface shading and
auxiliary maps are evaluated at shutter opening. Filtering is isotropic trilinear,
not EWA. Shadow and reflection maps retain their approximate transparency.
A bucket is limited to 8,388,608 fragments and 256 MiB of retained shading atlases;
errors are explicit rather than silently dropping layers. Source geometry and
textures remain resident. The viewport is an approximation of the final renderer.

This is a clean-room implementation of the historical approach, not Pixar's
original 1995 code, a full RSL implementation, or a bit-exact PRMan reproduction.
See AUDIT_REPORT.md for measured performance and verification evidence.
