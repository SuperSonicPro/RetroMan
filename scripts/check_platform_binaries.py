"""Inspect release ABI/dependencies and verify the Mach-O ad-hoc page hashes."""
import hashlib,json,struct,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ADDON=ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work/RetroMan-1.0.24'
root=ADDON/'native';out={}
for filename in ('bucket_linux_x86_64.so','bucket_windows_x86_64.dll','bucket_macos_arm64.dylib'):
 out[filename]={'sha256':hashlib.sha256((root/filename).read_bytes()).hexdigest()}
b=(root/'bucket_macos_arm64.dylib').read_bytes();pos=32;libraries=[];symbols=[];signature=None
for _ in range(struct.unpack_from('<I',b,16)[0]):
 cmd,size=struct.unpack_from('<II',b,pos)
 if cmd in (0xc,0xd):
  offset=struct.unpack_from('<I',b,pos+8)[0];name=b[pos+offset:pos+size].split(b'\0')[0].decode()
  if cmd==0xc:libraries.append(name)
  else:install_name=name
 if cmd==0x2:
  symoff,count,stroff,strsize=struct.unpack_from('<4I',b,pos+8)
  for n in range(count):
   idx,kind,sect,desc,value=struct.unpack_from('<IBBHQ',b,symoff+n*16)
   if kind&1 and kind&0xe==0xe:symbols.append(b[stroff+idx:].split(b'\0')[0].decode())
 if cmd==0x1d:signature=struct.unpack_from('<II',b,pos+8)
 pos+=size
assert libraries==['/usr/lib/libSystem.B.dylib'],libraries
assert install_name=='@rpath/bucket_macos_arm64.dylib',install_name
for fn in ('create','push','finish','destroy'):assert '_rm_bucket_'+fn in symbols
assert signature
start,length=signature;blob=b[start:start+length];magic,size,count=struct.unpack_from('>III',blob);assert magic==0xfade0cc0
verified=False
for i in range(count):
 kind,offset=struct.unpack_from('>II',blob,12+i*8)
 if kind!=0:continue
 cd=blob[offset:];magic,length,version,flags,hashoff,identoff,special,slots,limit=struct.unpack_from('>9I',cd)
 hashsize,hashtype,platform,pageshift=struct.unpack_from('>4B',cd,36)
 assert magic==0xfade0c02 and flags&2 and hashtype==2 and hashsize==32
 assert slots==(limit+(1<<pageshift)-1)//(1<<pageshift)
 for n in range(slots):assert hashlib.sha256(b[n*(1<<pageshift):min((n+1)*(1<<pageshift),limit)]).digest()==cd[hashoff+n*hashsize:hashoff+(n+1)*hashsize]
 verified=True
assert verified
out['bucket_macos_arm64.dylib'].update(system_dependencies=libraries,install_name=install_name,required_exports=True,adhoc_signature_page_hashes_valid=True,macos_runtime_tested=False)
assert 'PASS: actual native library loaded' in (ROOT/'blender-tests/windows-native-124.log').read_text(encoding='utf-8')
pe=subprocess.check_output(['objdump','-p',str(root/'bucket_windows_x86_64.dll')],text=True)
imports=[line.split('DLL Name:')[1].strip() for line in pe.splitlines() if 'DLL Name:' in line]
assert all(n=='KERNEL32.dll' or n.startswith('api-ms-win-crt-') for n in imports),imports
for fn in ('create','push','finish','destroy'):assert 'rm_bucket_'+fn in pe
out['bucket_windows_x86_64.dll'].update(system_dependencies=imports,required_exports=True,windows_blender_tested=False,wine_numerical_cases=6)
(ROOT/'blender-tests/portability-124.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
