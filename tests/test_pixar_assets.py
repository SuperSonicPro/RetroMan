import unittest,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'addon' if (ROOT/'addon').exists() else ROOT/'work/RetroMan-1.0.24'
ASSETS=SRC/'assets/pixar128'
class Tests(unittest.TestCase):
 def test_original_color_catalog_integrity(self):
  items=json.loads((ASSETS/'catalog.json').read_text(encoding='utf-8'));self.assertEqual(len(items),128);self.assertEqual(len({x['id'] for x in items}),128)
  for item in items:
   p=ASSETS/item['file'];self.assertTrue(p.resolve().is_relative_to(ASSETS.resolve()))
   self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),item['sha256'])
   self.assertFalse(p.stem.endswith(('_bmp','_normal')))
 def test_asset_licenses_and_attribution_are_separate(self):
  credits=(ASSETS/'CREDITS.txt').read_text(encoding='utf-8')
  for value in ('Pixar Animation Studios','Dylan Sisson','Leif Pedersen','David DiFrancesco','https://creativecommons.org/licenses/by/4.0/','unchanged','2018'):self.assertIn(value,credits)
  self.assertIn('Attribution 4.0 International',(ASSETS/'CC-BY-4.0.txt').read_text(encoding='utf-8'))
  self.assertIn('CC-BY-4.0',(SRC/'blender_manifest.toml').read_text(encoding='utf-8'))
