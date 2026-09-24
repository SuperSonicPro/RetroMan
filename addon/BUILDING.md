# Building RetroMan 1.0.24

The source ZIP contains an `addon/` directory, the build tools, regression tests and prebuilt libraries for all three targets. Run commands from the extracted source directory. Python 3.11+ is needed for the build/test tools; Blender 5.0+ supplies the addon runtime.

## Installer targets

| Target | Installer | Graphics backend |
| --- | --- | --- |
| Linux x86-64 | `RetroMan-1.0.24-linux-x64.zip` | Blender's selected Vulkan/OpenGL backend |
| Windows x64 | `RetroMan-1.0.24-windows-x64.zip` | Blender's selected Vulkan/OpenGL backend |
| macOS Apple Silicon, macOS 13+ | `RetroMan-1.0.24-macos-arm64.zip` | Blender's Metal backend |

Install the matching ZIP using Blender Preferences → Get Extensions → Install from Disk. Do not unzip it first. Restart Blender after replacing a loaded native library, particularly on Windows. The source ZIP is for development, not installation. Intel Macs and Windows ARM64 are not targets of this release. Blender 5.x official Mac builds target Apple Silicon.

Windows and macOS are preview builds: compilation and binary validation passed; actual Windows Blender and Mac Metal rendering were not tested from this Linux machine. The Windows native sampler passed six numerical cases under Wine. Run RetroMan → Diagnostics → Run RetroMan GPU Self-Test on each target, then verify a final render, Rendered viewport, cancellation, transparent output, and a scene with colored lights before production use.

## Build with the host compiler

Linux (GCC C++17):

```sh
python3 addon/native/build.py --platform linux-x64
python3 build_release.py --platform linux-x64
```

Windows (x64 Native Tools Command Prompt for Visual Studio, Desktop development with C++ installed):

```bat
py addon\native\build.py --platform windows-x64
py build_release.py --platform windows-x64
```

The Windows native build uses `/MT`, so the C++ runtime does not require a separate Visual C++ redistributable. GCC-compatible Windows compilers can be selected with `--compiler`.

Mac (Apple Silicon; install Xcode Command Line Tools):

```sh
python3 addon/native/build.py --platform macos-arm64
python3 build_release.py --platform macos-arm64
```

The Mac build sets a macOS 13 deployment target, an `@rpath` install name, and applies an ad-hoc signature. It is not Apple Developer ID signed or notarized. No Apple SDK is redistributed in the source ZIP.

`CXX` or `--compiler` selects a native compiler executable, not a shell command containing arguments. Failed builds preserve the previous binary. Intermediate object/import/debug files stay in a temporary directory.

## Cross-compile with Zig

The Windows and Mac release libraries were built with the official Zig 0.14.1 Linux x86-64 compiler from https://ziglang.org/download/. Its download SHA-256 was verified against Zig's official index. Zig is not included in the source ZIP. Runtime license notices are included under `addon/native/licenses/`.

```sh
python3 addon/native/build.py --platform windows-x64 --zig /absolute/path/to/zig
python3 addon/native/build.py --platform macos-arm64 --zig /absolute/path/to/zig
python3 build_release.py
```

Or compile and package one target in one step:

```sh
python3 build_release.py --platform windows-x64 --rebuild-native --zig /absolute/path/to/zig
```

Without `--rebuild-native`, packaging uses the included binaries. `--platform all` is the default and generates all three installers and one source ZIP. Every installer contains only its matching native library and a matching Blender platform manifest. `--no-source` skips the source ZIP; `--output DIRECTORY` selects a destination. Archives use stable timestamps and include checksums. A local rebuild with a different compiler may legitimately change the binary and its checksum.

## Verification

```sh
python3 -m pip install numpy
python3 -m unittest discover -s tests
```

On Windows use `py` in place of `python3`. The native numerical tests load the host's actual library. Platform selection tests also verify the three supplied binary formats. The source must therefore retain all three libraries even when only rebuilding one.

`addon/native/smoke.cpp` provides a standalone C ABI load/numerical test. Build it with a C++17 compiler, then pass the absolute library filename. On Linux link with `-ldl`; on Windows compile it as an ordinary console program. It exercises exported functions, full coverage, transparency sorting and three sample rates.

`blender-tests/light_color_124.py` is a Blender UI test harness: use factory startup and `--python` with this script. It checks light colors and final PNG alpha. It deletes the factory objects and quits Blender; use an isolated test session, not an open user scene. Source archives detect `addon/` automatically. Test reports are written under `blender-tests/`.

References: https://www.blender.org/download/requirements/ and https://docs.blender.org/manual/en/4.2/advanced/extensions/getting_started.html
