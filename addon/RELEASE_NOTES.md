# 1.0.24 — Platform packages

- Separate Linux x64, Windows x64, and macOS Apple Silicon install ZIPs.
- Platform-aware native sampler loading with actionable missing-library diagnostics.
- C++ C ABI exports for Windows; portable native and Zig cross-build commands.
- UTF-8 worker event exchange independent of Windows system language settings.
- Source package contains all three native libraries, sources, licenses, reproducible packaging tools and build instructions.
- Windows/macOS previews require on-device Blender/GPU validation. Windows library numerical checks pass under Wine; macOS compilation/binary/signature structure checks pass only.
- Rendering algorithms, light-color fixes, Film Transparent control and Pixar presets retained.

# 1.0.22 — Pixar 128 texture presets

- Bundle all 128 color textures from Pixar One Twenty Eight, unchanged from the official archive.
- Add a thumbnail browser by category in Material Properties, with ordinary editable Blender material nodes.
- Apply a new material to the active slot while preserving existing/shared materials.
- Pack selected images into the scene and mark created materials as assets.
- Include visible credits, source/license links, per-material/image attribution, an in-scene credit text, full CC BY 4.0 license, and file provenance.

The 1993 collection is supplied through Pixar's 2018 re-release. Later normal/bump additions are excluded from these classic color presets. The renderer retains the 1.0.21 dense-scene fixes; its rendering algorithms are unchanged.
