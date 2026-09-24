"""Build platform-specific RetroMan ZIPs; see BUILDING.md for native compilation."""
from pathlib import Path
import argparse, hashlib, json, re, struct, subprocess, sys, zipfile
ROOT = Path(__file__).resolve().parent
ADDON = ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work/RetroMan-1.0.24'
VERSION = '1.0.24'
TARGETS = {'linux-x64':'bucket_linux_x86_64.so', 'windows-x64':'bucket_windows_x86_64.dll', 'macos-arm64':'bucket_macos_arm64.dylib'}

def payload(directory):
    return sorted(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pdb','.lib','.obj','.o'})

def validate_binary(data, target):
    if target=='linux-x64':
        valid=data[:6]==b'\x7fELF\x02\x01' and struct.unpack_from('<H',data,18)[0]==62
    elif target=='windows-x64':
        offset=struct.unpack_from('<I',data,60)[0]
        valid=data[:2]==b'MZ' and data[offset:offset+4]==b'PE\0\0' and struct.unpack_from('<H',data,offset+4)[0]==0x8664 and bool(struct.unpack_from('<H',data,offset+22)[0]&0x2000)
    elif target=='macos-arm64':
        valid=struct.unpack_from('<IIII',data)==(0xfeedfacf,0x100000c,0,6)
    else:raise ValueError(target)
    if not valid: raise ValueError(f'Wrong native binary format/architecture for {target}')

def write_zip(path, entries):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            info=zipfile.ZipInfo(name,date_time=(2026,9,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644 << 16
            archive.writestr(info,data)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise ValueError('Archive integrity failure')
        for name in archive.namelist():
            if name.endswith('.py'):compile(archive.read(name),name,'exec')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform',choices=['all',*TARGETS],default='all')
    parser.add_argument('--rebuild-native',action='store_true',help='Compile selected platforms before packaging; cross compilation needs --zig')
    parser.add_argument('--zig',help='Zig compiler path for native/cross compilation')
    parser.add_argument('--output',type=Path,default=ROOT/'dist')
    parser.add_argument('--no-source',action='store_true',help='Only emit selected platform installer(s) and checksums')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    targets=list(TARGETS) if args.platform=='all' else [args.platform]
    for target in targets:
        if args.rebuild_native:
            command=[sys.executable,str(ADDON/'native/build.py'),'--platform',target]
            if args.zig:command+=['--zig',args.zig]
            subprocess.run(command,check=True)
        binary=ADDON/'native'/TARGETS[target]
        if not binary.exists():raise SystemExit(f'Missing {binary.name}; run native/build.py --platform {target} first')
        validate_binary(binary.read_bytes(),target)
    files=[];manifest=(ADDON/'blender_manifest.toml').read_text(encoding='utf-8')
    for target in targets:
        entries=[]
        for p in payload(ADDON):
            name=p.relative_to(ADDON).as_posix()
            if p.suffix in {'.so','.dll','.dylib'} and p.name!=TARGETS[target]:continue
            data=p.read_bytes()
            if name=='blender_manifest.toml':data=re.sub(r'^platforms\s*=.*$',f'platforms = ["{target}"]',manifest,flags=re.M).encode('utf-8')
            entries.append((name,data))
        path=out/f'RetroMan-{VERSION}-{target}.zip';write_zip(path,entries);files.append(path)
    report=out/f'RetroMan-{VERSION}-Audit-Report.md';report.write_bytes((ADDON/'AUDIT_REPORT.md').read_bytes());files.append(report)
    if not args.no_source:
        prefix=f'RetroMan-{VERSION}-source/';entries=[]
        for p in payload(ADDON):entries.append((prefix+'addon/'+p.relative_to(ADDON).as_posix(),p.read_bytes()))
        for folder in ('tests','scripts'):
            for p in payload(ROOT/folder):entries.append((prefix+folder+'/'+p.relative_to(ROOT/folder).as_posix(),p.read_bytes()))
        for name in ('build_release.py','BUILDING.md'):
            entries.append((prefix+name,(ROOT/name).read_bytes()))
        for name in ('light_color_124.py','light-color-linux-124.json','windows-native-124.log','portability-124.json','install-qa-124.json'):
            p=ROOT/'blender-tests'/name
            if p.exists():entries.append((prefix+'blender-tests/'+name,p.read_bytes()))
        p=ROOT/'dist/TEST_RESULTS-1.0.24.txt'
        if p.exists():entries.append((prefix+p.name,p.read_bytes()))
        source=out/f'RetroMan-{VERSION}-source.zip';write_zip(source,entries);files.append(source)
        with zipfile.ZipFile(source) as src:
            for target in targets:
                with zipfile.ZipFile(out/f'RetroMan-{VERSION}-{target}.zip') as install:
                    for name in install.namelist():
                        if name!='blender_manifest.toml':assert install.read(name)==src.read(prefix+'addon/'+name)
    checksum=out/f'RetroMan-{VERSION}{"-"+args.platform if args.platform!="all" else ""}.sha256'
    checksum.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files),encoding='utf-8')
    for p in files:print(f'{p.name}: {p.stat().st_size:,} bytes')
    print('Verified native headers, ZIP integrity, Python compilation and matching source payloads.')

if __name__=='__main__':main()
