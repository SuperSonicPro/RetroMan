# SPDX-License-Identifier: GPL-3.0-or-later
"""Build RetroMan's sampler natively or cross-compile with Zig 0.14.1.

Native: python native/build.py (MSVC Developer Prompt on Windows).
Cross:  python native/build.py --platform windows-x64 --zig /path/to/zig
"""
from pathlib import Path
import argparse, os, platform, shutil, subprocess, tempfile

TARGETS = {
    'linux-x64': ('bucket_linux_x86_64.so', 'x86_64-linux-gnu.2.28'),
    'windows-x64': ('bucket_windows_x86_64.dll', 'x86_64-windows-gnu'),
    'macos-arm64': ('bucket_macos_arm64.dylib', 'aarch64-macos.13.0'),
}

def host_platform():
    machine = platform.machine().lower()
    arch = 'x64' if machine in ('amd64', 'x86_64') else 'arm64' if machine in ('arm64', 'aarch64') else machine
    return {'Linux':'linux', 'Windows':'windows', 'Darwin':'macos'}.get(platform.system(), 'unknown') + '-' + arch

def command(target, output, compiler=None, zig=None):
    source = str(Path(__file__).resolve().with_name('bucket.cpp'))
    if zig:
        return [zig, 'c++', '-target', TARGETS[target][1], '-std=c++17', '-O3', '-fvisibility=hidden',
                *(['-dynamiclib', '-Wl,-install_name,@rpath/'+TARGETS[target][0]] if target.startswith('macos') else ['-shared', '-static'] if target.startswith('windows') else ['-shared', '-fPIC']), source, '-o', str(output)]
    if target != host_platform():
        raise ValueError('Cross-compiling requires --zig; native compiler builds must match this host')
    compiler = compiler or os.environ.get('CXX') or ('cl' if target.startswith('windows') else 'clang++' if target.startswith('macos') else 'g++')
    if Path(compiler).name.lower() in ('cl', 'cl.exe'):
        return [compiler, '/nologo', '/std:c++17', '/O2', '/EHsc', '/MT', '/LD', source, '/link', '/OUT:'+str(output)]
    return [compiler, '-std=c++17', '-O3', '-fvisibility=hidden',
            *(['-dynamiclib', '-arch', 'arm64', '-mmacosx-version-min=13.0', '-Wl,-install_name,@rpath/'+TARGETS[target][0]] if target.startswith('macos') else ['-shared', '-static'] if target.startswith('windows') else ['-shared', '-fPIC']), source, '-o', str(output)]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=TARGETS, default=host_platform())
    parser.add_argument('--zig', help='Path to the Zig executable (0.14.1 tested)')
    parser.add_argument('--compiler', help='Native C++ compiler executable')
    args = parser.parse_args()
    if args.platform not in TARGETS: parser.error('Unsupported host; select a supported --platform and --zig')
    if args.zig: args.zig = str(Path(shutil.which(args.zig) or args.zig).resolve())
    if args.compiler: args.compiler = shutil.which(args.compiler) or str(Path(args.compiler).resolve())
    root = Path(__file__).resolve().parent
    # Failed builds never replace the last usable binary. MSVC intermediates stay private.
    with tempfile.TemporaryDirectory(prefix='retroman-build-') as tmp:
        output = Path(tmp)/TARGETS[args.platform][0]
        subprocess.run(command(args.platform, output, args.compiler, args.zig), check=True, cwd=tmp)
        if args.platform.startswith('macos') and platform.system() == 'Darwin':
            subprocess.run(['codesign', '--force', '--sign', '-', str(output)], check=True)
        shutil.copy2(output, root/output.name)
        print(root/output.name)

if __name__ == '__main__': main()
