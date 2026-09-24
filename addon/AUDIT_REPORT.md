# RetroMan 1.0.24 — Platform builds

## Deliverables
Separate Blender extension install ZIPs for Linux x86-64, Windows x64, and macOS Apple Silicon (macOS 13+), plus a common source ZIP and SHA-256 checksums. Each installer restricts Blender's platform manifest to its target and contains only the corresponding native library. The source contains all libraries, C++ source, portable build script, packaging script, licenses and verification tools.

Windows and macOS are preview builds pending actual Blender testing on those operating systems. Cross-compilation, format validation and Wine tests cannot establish Metal GPU behavior or Windows driver compatibility. Intel Macs and Windows ARM64 are not supported by these packages. Official Blender 5.x Mac builds target Apple Silicon.

## Changes
- Native sampler loader selects Linux ELF, Windows DLL or macOS dylib using operating system and architecture; explicit errors identify unsupported targets or missing/wrong libraries.
- Windows C ABI functions are explicitly exported. Native compiler support: GCC/Clang on Linux, MSVC or GCC-compatible compiler on Windows, and Apple Clang on Mac. Zig 0.14.1 cross-compilation is supported.
- Mac build targets macOS 13, sets a portable @rpath install name and carries an ad-hoc signature. This is not Developer ID signing or notarization.
- Build failure leaves existing libraries intact; compiler intermediates remain in a private temporary directory.
- Worker event files and bundled texture catalog/credits use UTF-8 explicitly, avoiding Windows code-page assumptions. Worker invocation already uses the running Blender executable, argument lists and the active Vulkan/OpenGL/Metal backend.
- The GPU rendering algorithm is unchanged. The native sampler is a numerical reference; GPU final renders do not use that library. The Windows/Mac work does not replace the GPU pipeline with CPU sampling.
- Light-color translation, Film Transparent controls and all 128 Pixar texture presets remain included.

## Validation
- 64/64 static, numerical, asset and platform tests passed on Linux, including the rebuilt native sampler.
- Windows DLL cross-compiled with verified official Zig 0.14.1. Inspected PE x64 DLL architecture, all four exported functions and dependencies (Windows kernel/UCRT only). The actual library loaded under an isolated Wine profile and passed six coverage/transparency numerical cases at three sampling rates and both insertion orders. No Windows Blender/GPU test was performed.
- Mac library cross-compiled to ARM64 Mach-O. Inspected all four required exports and its sole external dependency, /usr/lib/libSystem.B.dylib. Verified @rpath identity, ad-hoc signature flag and every signed SHA-256 code page. No Mac runtime or Metal test was possible here.
- Linux Blender 5.2.2 Vulkan on RTX 3090: 13 light/film checks passed, including all four light types, linked node inputs, actual final-render workers, saved PNG alpha and repeated registration.
- Linux installation/upgrade: 6/6 checks passed in an isolated Blender 5.2.2 profile, including upgrade from 1.0.23, repeated disable/enable, packaged native library, GPU graphics/compute self-test, and a Pixar preset final render.
- Install/source ZIP integrity, packaged Python compilation, per-platform native headers and source payload equality are checked by the packaging script.

Large user scenes were not rerendered for this platform release. See BUILDING.md for build commands and target-machine verification steps. The retained self-test exercises graphics, compute, bucket sampling and framebuffer readback.

## Licenses and references
Pixar assets remain CC BY 4.0 with original attribution. Cross-compiled runtime copyright/license notices are included under native/licenses and referenced in THIRD_PARTY_NOTICES.md. No compiler or Apple SDK is included.

- https://ziglang.org/download/
- https://www.blender.org/download/requirements/
- https://docs.blender.org/manual/en/4.2/advanced/extensions/getting_started.html
