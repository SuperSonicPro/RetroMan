"""Platform selection, binary identity and native build contract."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase,mock
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work/RetroMan-1.0.24'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
builder=load('native_build',SRC/'native/build.py')
release=load('release_build',ROOT/'build_release.py')
class Platforms(TestCase):
 def test_bundled_binary_architectures(self):
  for target,name in release.TARGETS.items():release.validate_binary((SRC/'native'/name).read_bytes(),target)
 def test_loader_selects_os_and_architecture(self):
  for system,machine,filename in [('Linux','x86_64','bucket_linux_x86_64.so'),('Windows','AMD64','bucket_windows_x86_64.dll'),('Darwin','arm64','bucket_macos_arm64.dylib')]:
   m=load('sampler',SRC/'bucket_native.py');fake=SimpleNamespace(**{name:SimpleNamespace() for name in ('rm_bucket_create','rm_bucket_push','rm_bucket_finish','rm_bucket_destroy')})
   with mock.patch.object(m.platform,'system',return_value=system),mock.patch.object(m.platform,'machine',return_value=machine),mock.patch.object(m.C,'CDLL',return_value=fake) as loader:
    self.assertIs(m.library(),fake);self.assertEqual(Path(loader.call_args.args[0]).name,filename)
 def test_unsupported_host_rejected(self):
  m=load('sampler',SRC/'bucket_native.py')
  with mock.patch.object(m.platform,'system',return_value='Darwin'),mock.patch.object(m.platform,'machine',return_value='x86_64'):
   with self.assertRaisesRegex(RuntimeError,'does not support'):m.library()
 def test_missing_binary_diagnostic(self):
  m=load('sampler',SRC/'bucket_native.py')
  with mock.patch.object(m.platform,'system',return_value='Windows'),mock.patch.object(m.platform,'machine',return_value='AMD64'),mock.patch.object(m.Path,'is_file',return_value=False):
   with self.assertRaisesRegex(RuntimeError,'install the RetroMan ZIP'):m.library()
 def test_native_msvc_command(self):
  with mock.patch.object(builder,'host_platform',return_value='windows-x64'):
   cmd=builder.command('windows-x64',Path('test.dll'),compiler='cl')
  self.assertIn('/MT',cmd);self.assertIn('/LD',cmd);self.assertIn('/OUT:test.dll',cmd)
 def test_cross_requires_explicit_toolchain(self):
  with mock.patch.object(builder,'host_platform',return_value='linux-x64'):
   with self.assertRaisesRegex(ValueError,'Cross-compiling'):builder.command('macos-arm64',Path('x'))
 def test_zig_target_and_sdk_floor(self):
  cmd=builder.command('macos-arm64',Path('x'),zig='/toolchain/zig')
  self.assertIn('aarch64-macos.13.0',cmd);self.assertIn('-dynamiclib',cmd)
