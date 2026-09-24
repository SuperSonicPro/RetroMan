# SPDX-License-Identifier: GPL-3.0-or-later
"""Local, attributed Pixar color-texture presets using ordinary Blender nodes."""
import json
from pathlib import Path
import bpy
from bpy.props import EnumProperty

ROOT = Path(__file__).parent / 'assets' / 'pixar128'
CATALOG = json.loads((ROOT / 'catalog.json').read_text(encoding='utf-8'))
BY_ID = {item['id']: item for item in CATALOG}
CATEGORIES = tuple((name, name, '') for name in sorted({x['category'] for x in CATALOG}))
SOURCE = 'https://renderman.pixar.com/pixar-one-twenty-eight'
LICENSE = 'https://creativecommons.org/licenses/by/4.0/'
CREDIT = 'Pixar One Twenty Eight — Pixar Animation Studios; Dylan Sisson, updated by Leif Pedersen. CC BY 4.0.'
_previews = None
_items = {}  # Keep enum strings alive for Blender RNA.


def texture_items(self, context):
    global _previews
    category = self.category
    if category not in _items:
        import bpy.utils.previews
        if _previews is None:
            _previews = bpy.utils.previews.new()
        values = []
        for number, entry in enumerate(CATALOG):
            if entry['category'] != category:
                continue
            icon = 0
            if not bpy.app.background:
                try:
                    key = entry['id']
                    if key not in _previews:
                        _previews.load(key, str(ROOT / entry['file']), 'IMAGE')
                    icon = _previews[key].icon_id
                except (OSError, RuntimeError):
                    pass  # Named presets remain usable if a preview cannot load.
            values.append((entry['id'], entry['name'], CREDIT, icon, number))
        _items[category] = values
    return _items[category]


def category_changed(self, context):
    self.texture = next(x['id'] for x in CATALOG if x['category'] == self.category)


class RetroManTextureSettings(bpy.types.PropertyGroup):
    category: EnumProperty(name='Category', items=CATEGORIES, default='Brick', update=category_changed)
    texture: EnumProperty(name='Texture', items=texture_items)


def add_credit_text():
    body = (ROOT / 'CREDITS.txt').read_text(encoding='utf-8')
    # Do not overwrite any user-edited text with the same name.
    for text in bpy.data.texts:
        if text.get('retroman_pixar_credits') and text.as_string() == body:
            return text
    text = bpy.data.texts.new('RetroMan — Pixar Texture Credits')
    text.write(body)
    text['retroman_pixar_credits'] = True
    return text


def create_material(identifier):
    entry = BY_ID[identifier]
    image = bpy.data.images.load(str(ROOT / entry['file']), check_existing=False)
    material = None
    try:
        image.name = 'Pixar 128 — ' + entry['name']
        image.colorspace_settings.name = 'sRGB'
        image.pack()  # Scene remains portable across machines and addon upgrades.
        image['attribution'] = CREDIT
        image['source'] = SOURCE
        image['license'] = LICENSE
        image['source_sha256'] = entry['sha256']
        material = bpy.data.materials.new('Pixar 128 — ' + entry['name'])
        material.use_nodes = True
        shader = material.node_tree.nodes.get('Principled BSDF')
        shader.inputs['Roughness'].default_value = .5
        shader.inputs['Metallic'].default_value = 0
        texture = material.node_tree.nodes.new('ShaderNodeTexImage')
        texture.image = image
        texture.extension = 'REPEAT'
        texture.label = 'Pixar 128 • CC BY 4.0'
        texture.location = (-320, 180)
        material.node_tree.links.new(texture.outputs['Color'], shader.inputs['Base Color'])
        material['attribution'] = CREDIT
        material['source'] = SOURCE
        material['license'] = LICENSE
        material['retroman_pixar_preset'] = identifier
        material.asset_mark()
        material.asset_data.author = 'Pixar Animation Studios; Dylan Sisson; Leif Pedersen'
        material.asset_data.description = CREDIT + ' Color TIFF unchanged; Blender preset by RetroMan. ' + SOURCE
        material.asset_data.tags.new('Pixar 128')
        material.asset_data.tags.new(entry['category'])
        add_credit_text()
        return material
    except Exception:
        if material is not None:
            bpy.data.materials.remove(material)
        bpy.data.images.remove(image)
        raise


class RETROMAN_OT_apply_pixar_texture(bpy.types.Operator):
    bl_idname = 'retroman.apply_pixar_texture'
    bl_label = 'Apply Texture Preset'
    bl_description = 'Create a new attributed material in the active slot; the previous material is preserved'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return bool(obj and obj.mode == 'OBJECT' and getattr(obj, 'is_editable', True)
                    and hasattr(obj.data, 'materials'))

    def execute(self, context):
        settings = context.scene.retroman_textures
        identifier = settings.texture
        if identifier not in BY_ID or BY_ID[identifier]['category'] != settings.category:
            identifier = next(x['id'] for x in CATALOG if x['category'] == settings.category)
        material = None
        try:
            material = create_material(identifier)
            obj = context.object
            if not obj.material_slots:
                if obj.data.users > 1 or obj.data.library:
                    obj.data = obj.data.copy()
                obj.data.materials.append(material)
            else:
                slot = obj.material_slots[obj.active_material_index]
                slot.link = 'OBJECT'
                slot.material = material
            if obj.type == 'MESH' and not obj.data.uv_layers:
                self.report({'WARNING'}, 'Preset applied. This mesh needs a UV map for texture placement.')
            else:
                self.report({'INFO'}, 'Applied ' + material.name)
            return {'FINISHED'}
        except Exception as exc:
            if material is not None and not material.users:
                bpy.data.materials.remove(material)
            self.report({'ERROR'}, 'Could not apply Pixar texture: ' + str(exc))
            return {'CANCELLED'}


class RETROMAN_OT_pixar_credits(bpy.types.Operator):
    bl_idname = 'retroman.pixar_texture_credits'
    bl_label = 'Pixar Texture Credits'
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=520)
    def draw(self, context):
        col = self.layout.column(align=True)
        for line in ('Pixar One Twenty Eight — © Pixar Animation Studios',
                     'Collection credit: Dylan Sisson; updated by Leif Pedersen.',
                     'Repeating-texture technology: David DiFrancesco.',
                     '1993 collection; files from the official 2018 re-release.',
                     '128 color TIFFs unchanged. Blender presets by RetroMan.',
                     'CC BY 4.0: retain credit, link the license, note changes.'):
            col.label(text=line)
        col.operator('wm.url_open', text='Pixar source and authors', icon='URL').url = SOURCE
        col.operator('wm.url_open', text='Creative Commons Attribution 4.0', icon='URL').url = LICENSE
    def execute(self, context):
        add_credit_text()
        return {'FINISHED'}


class RETROMAN_PT_pixar_textures(bpy.types.Panel):
    bl_label = 'Pixar 128 Texture Presets'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'material'
    @classmethod
    def poll(cls, context):
        return bool(context.scene and context.scene.render.engine == 'RETROMAN'
                    and context.object and hasattr(context.object.data, 'materials'))
    def draw(self, context):
        layout = self.layout
        settings = context.scene.retroman_textures
        layout.prop(settings, 'category')
        layout.template_icon_view(settings, 'texture', show_labels=True, scale=5)
        layout.prop(settings, 'texture', text='')
        layout.operator('retroman.apply_pixar_texture', icon='MATERIAL')
        layout.label(text='Uses the mesh’s Active Render UV map.')
        layout.label(text='Textures © Pixar Animation Studios • CC BY 4.0')
        layout.operator('retroman.pixar_texture_credits', icon='INFO')


def clear_previews():
    global _previews
    if _previews is not None:
        import bpy.utils.previews
        bpy.utils.previews.remove(_previews)
        _previews = None
    _items.clear()
