"""Write UGC (custom clothes / face paint) into a Tomodachi Life: Living the Dream save and dress Miis.

Save layout (found from the user's own designs):
  Ugc/UgcCloth%03d.canvas.zs       256x256 RGBA8, Tegra block-linear, zstd      (editor drawing)
  Ugc/UgcCloth%03d.ugctex.zs       512x512 BC1, block-linear, zstd             (texture on the model)
  Ugc/UgcCloth%03d_Thumb.ugctex.zs 256x256 BC3, block-linear, zstd             (menu picture)
  Ugc/UgcFacePaint%03d.*           like cloth, without thumb
  Player.sav  UGC.Cloth.* (300 slots) / UGC.FacePaint.* (72 slots); Hash = (kind << 16) | slot
  Mii.sav     Mii.MiiMisc.#3b098df0.<Slot>.KeyHash = murmur3(cloth row name), #ab277b66 = UGC slot of the top,
              Mii.MiiMisc.#a060c827.#bef8223b = face paint slot, IsFacePaintFront
"""
import os, json, struct
import numpy as np
import zstandard
from PIL import Image
from tlsave import Save, enc_wstr, ELEM
from hashnames import murmur3_32
import ugctex

UGC_KIND = {'Food': 1, 'Goods': 2, 'Cloth': 3, 'MapObject': 4, 'MapFloor': 5, 'Interior': 6, 'Exterior': 7, 'FacePaint': 8}

# UGC cloth type (UGC.Cloth.#8ce692a7) -> (cloth row in RSDB/Cloth, Mii slot)
CLOTH = {
    'EasyH': ('ClothTopsUgcTshirtHalfEasy', 'Tops'),
    'TShirtN': ('ClothTopsUgcTshirtSleeveless', 'Tops'),
    'TShirtH': ('ClothTopsUgcTshirtHalf', 'Tops'),
    'TShirtL': ('ClothTopsUgcTshirtLong', 'Tops'),
    'TopsLongAlineN': ('ClothTopslongUgcAlineSleeveless', 'Topslong'),
    'TopsLongAlineH': ('ClothTopslongUgcAlineHalf', 'Topslong'),
    'TopsLongAlineL': ('ClothTopslongUgcAlineLong', 'Topslong'),
    'Robe': ('ClothTopslongUgcKimonoLong', 'Topslong'),
    'Dress': ('ClothTopslongUgcDressHalf', 'Topslong'),
    'SkirtS': ('ClothBottomsUgcSkirtMini', 'BottomsA'),
    'SkirtL': ('ClothBottomsUgcSkirtLong', 'BottomsA'),
    'PantsM': ('ClothBottomsUgcPantsWide', 'BottomsA'),
    'PantsL': ('ClothBottomsUgcPantsLong', 'BottomsA'),
    'Cap': ('ClothHeadwearHatUgcCap', 'Headwear'),
    'Costume': ('ClothHeadwearHatUgcKnit', 'Headwear'),
    'Hat': ('ClothHeadwearHatUgcBoater', 'Headwear'),
}
MII_SLOTS = ['Tops', 'Topslong', 'BottomsA', 'BottomsB', 'Shoes', 'Accessory', 'Headwear', 'All', 'Coordinate']


def mm(s):
    return murmur3_32(s.encode())


def load_flags(path):
    rows = json.load(open(path, encoding='utf-8'))
    return {r['path']: (int(r['hash'], 16), r['type'], r.get('enum') or []) for r in rows}


class Ed:
    """typed read/write of one save file by GameData path"""

    def __init__(self, path, F):
        self.path, self.F = path, F
        self.sv = Save(path)

    def get(self, p, i=None):
        v = self.sv.get(self.F[p][0])
        return v if i is None else v[i]

    def set(self, p, value, i=None):
        h, t, enum = self.F[p]
        sv = self.sv
        if isinstance(value, str) and t in ('Enum', 'EnumArray'):
            assert value in enum, (p, value, enum)
            value = mm(value)
        if i is None:
            if t == 'Bool':
                sv.set_inline(h, 1 if value else 0)
            elif t in ('Int',):
                sv.set_inline(h, value & 0xFFFFFFFF)
            elif t in ('UInt', 'Enum'):
                sv.set_inline(h, value)
            elif t == 'Float':
                sv.set_inline(h, struct.unpack('<I', struct.pack('<f', value))[0])
            else:
                raise ValueError('scalar type %s not handled (%s)' % (t, p))
            return
        e = sv.by_hash[h]
        cnt = struct.unpack_from('<I', sv.d, e.off)[0]
        assert 0 <= i < cnt, (p, i, cnt)
        if t == 'BoolArray':
            o = e.off + 4 + (i >> 3)
            if value:
                sv.d[o] |= (1 << (i & 7))
            else:
                sv.d[o] &= (~(1 << (i & 7))) & 0xFF
            return
        off = sv.elem_off(h, i)
        if t == 'IntArray':
            struct.pack_into('<i', sv.d, off, value)
        elif t in ('UIntArray', 'EnumArray'):
            struct.pack_into('<I', sv.d, off, value)
        elif t == 'FloatArray':
            struct.pack_into('<f', sv.d, off, value)
        elif t == 'UInt64Array':
            struct.pack_into('<Q', sv.d, off, value)
        elif t == 'Int64Array':
            struct.pack_into('<q', sv.d, off, value)
        elif t == 'Vector3Array':
            struct.pack_into('<3f', sv.d, off, *value)
        elif t == 'Vector2Array':
            struct.pack_into('<2f', sv.d, off, *value)
        elif t.startswith('WString'):
            sv.write_bytes(off, enc_wstr(value, ELEM[t]))
        elif t.startswith('String'):
            b = value.encode('utf-8')[:ELEM[t] - 1]
            sv.write_bytes(off, b + b'\0' * (ELEM[t] - len(b)))
        else:
            raise ValueError('array type %s not handled (%s)' % (t, p))

    def save(self, path=None):
        open(path or self.path, 'wb').write(self.sv.bytes())


# ---------------- files ----------------

def _zs(b):
    return zstandard.ZstdCompressor(level=9).compress(b)


def canvas_files(canvas, thumb=None):
    """canvas: 256x256x4 uint8 -> dict of file suffix -> bytes"""
    canvas = np.ascontiguousarray(canvas, np.uint8)
    assert canvas.shape == (256, 256, 4)
    out = {'.canvas.zs': _zs(ugctex.swizzle(canvas.tobytes(), 256, 256, 4))}
    big = np.repeat(np.repeat(canvas, 2, 0), 2, 1)
    tex = ugctex.write_tex(big, 'bc1', True, size=131072)
    assert len(tex) == 131072
    out['.ugctex.zs'] = _zs(tex)
    if thumb is not None:
        th = np.ascontiguousarray(thumb, np.uint8)
        assert th.shape == (256, 256, 4)
        t = ugctex.write_tex(th, 'bc3', True, size=65536)
        assert len(t) == 65536
        out['_Thumb.ugctex.zs'] = _zs(t)
    return out


def free_slot(ed, kind):
    hs = ed.get('UGC.%s.Hash' % kind)
    for i, h in enumerate(hs):
        if h == 0:
            return i
    raise RuntimeError('no free %s slot' % kind)


def write_cloth(savedir, ed, slot, ctype, canvas, thumb, name, when, gender='Unisex', style='Normal',
                grammar='cNeuter', price=2000, ref=None):
    """ed: Ed on Player.sav. ref: slot to copy unknown language fields from (or None)."""
    assert ctype in CLOTH
    for suf, b in canvas_files(canvas, thumb).items():
        open(os.path.join(savedir, 'Ugc', 'UgcCloth%03d%s' % (slot, suf)), 'wb').write(b)
    P = 'UGC.Cloth.'
    lang_a = ed.get(P + '#9285431c', ref) if ref is not None else mm('JPja')
    lang_b = ed.get(P + '#c93baceb', ref) if ref is not None else mm('JPja')
    vals = [('#2940fa1c', False), ('#2bccbcf0', when), ('#443bd132', 'SaveDataUgcBc1_Bc3SRGB'), ('#7cc5ca74', False),
            ('#81cc0d6d', ''), ('#88513427', 0), ('#8ce692a7', ctype), ('#9285431c', lang_a), ('#c121b84a', False),
            ('#c93baceb', lang_b), ('#d73851ab', when + 2), ('#f2d368bf', style), ('#f4d12244', 0),
            ('BaseColor', (1.0, 1.0, 1.0)), ('ClothGender', gender), ('ClothSeasonType', 'All'),
            ('EmissionIntensity', 0.0), ('EmissionPattern', 0), ('Hash', (UGC_KIND['Cloth'] << 16) | slot),
            ('HowToCallName', ''), ('IsEmissionNightOnly', False), ('IslandName', ''), ('Name', name),
            ('PlayerName', ''), ('Price', price), ('RegionLanguageID', 'EUde'), ('State', 'Obtained'),
            ('WordAttrCount', 'cSingular'), ('WordAttrGrammaticality', grammar)]
    for k, v in vals:
        ed.set(P + k, v, slot)
    return (UGC_KIND['Cloth'] << 16) | slot


def write_facepaint(savedir, ed, slot, canvas, name, when, ref=None, price=500):
    for suf, b in canvas_files(canvas, None).items():
        open(os.path.join(savedir, 'Ugc', 'UgcFacePaint%03d%s' % (slot, suf)), 'wb').write(b)
    P = 'UGC.FacePaint.'
    lang_a = ed.get(P + '#9285431c', ref) if ref is not None else mm('JPja')
    lang_b = ed.get(P + '#c93baceb', ref) if ref is not None else mm('JPja')
    misc = ed.get(P + '#2dbf038a', ref) if ref is not None else 32768
    vals = [('#2940fa1c', False), ('#2bccbcf0', when), ('#2dbf038a', misc), ('#443bd132', 'SaveDataUgcBc1'),
            ('#7cc5ca74', False), ('#81cc0d6d', ''), ('#88513427', 0), ('#9285431c', lang_a), ('#c121b84a', False),
            ('#c93baceb', lang_b), ('#d73851ab', 0), ('#f4d12244', 0), ('EmissionIntensity', 0.0),
            ('EmissionPattern', 0), ('Hash', (UGC_KIND['FacePaint'] << 16) | slot), ('HowToCallName', ''),
            ('IsEmissionNightOnly', False), ('IslandName', ''), ('Name', name), ('PlayerName', ''), ('Price', price),
            ('State', 'New')]
    for k, v in vals:
        ed.set(P + k, v, slot)
    return (UGC_KIND['FacePaint'] << 16) | slot


# UGC slot fields per Mii (which Mii.MiiMisc.#3b098df0 int belongs to which garment group)
MII_UGC_FIELD = {'Tops': '#ab277b66', 'Topslong': '#ab277b66', 'All': '#ab277b66',
                 'BottomsA': '#b90d9b9a', 'Headwear': '#ed14a1bc'}


def dress_mii(med, i, pieces, facepaint=None, front=None, ugc_fields=None):
    """med: Ed on Mii.sav. pieces: {slot: (cloth_row_name or keyhash or 0, color_index, ugc_slot or None)}.
    ugc_fields: {slot: '#field'} overrides MII_UGC_FIELD (for bottoms / headwear UGC)."""
    fields = dict(MII_UGC_FIELD)
    if ugc_fields:
        fields.update(ugc_fields)
    B = 'Mii.MiiMisc.#3b098df0.'
    if pieces:
        med.set(B + 'Coordinate.KeyHash', 0, i)
        med.set(B + 'Coordinate.ColorIndex', 0, i)
    for slot, (key, color, ugc) in pieces.items():
        kh = mm(key) if isinstance(key, str) else key
        med.set(B + slot + '.KeyHash', kh, i)
        med.set(B + slot + '.ColorIndex', color, i)
        if slot in fields and (kh != 0 or ugc is not None):
            med.set(B + fields[slot], -1 if ugc is None else ugc, i)
    if facepaint is not None:
        med.set('Mii.MiiMisc.#a060c827.#bef8223b', facepaint, i)
    if front is not None:
        med.set('Mii.MiiMisc.#a060c827.IsFacePaintFront', bool(front), i)


# ---------------- thumbnails ----------------

def make_thumb(icon_png, canvas, front_box, extra=None):
    """Approximate menu picture: the game's orange garment icon as shading, coloured by the front panel.
    front_box: (x0, y0, x1, y1) of the front panel in canvas coordinates."""
    icon = Image.open(icon_png).convert('RGBA').resize((256, 256), Image.LANCZOS)
    ia = np.asarray(icon).astype(np.float32) / 255.0
    lum = ia[..., :3].mean(-1)
    shade = np.clip(lum / max(1e-3, np.percentile(lum[ia[..., 3] > 0.5], 70)), 0.55, 1.15)
    ys, xs = np.nonzero(ia[..., 3] > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    fx0, fy0, fx1, fy1 = front_box
    panel = Image.fromarray(np.ascontiguousarray(canvas[fy0:fy1, fx0:fx1]), 'RGBA')
    # fill transparent holes of the panel with its mean colour so the silhouette stays solid
    pa = np.asarray(panel).astype(np.float32)
    m = pa[..., 3] > 128
    mean = pa[m][:, :3].mean(0) if m.any() else np.array([200, 200, 200], np.float32)
    pa[~m, :3] = mean
    pa[..., 3] = 255
    panel = Image.fromarray(pa.astype(np.uint8), 'RGBA').resize((x1 - x0, y1 - y0), Image.LANCZOS)
    col = np.zeros((256, 256, 4), np.float32)
    col[..., :3] = mean
    col[y0:y1, x0:x1] = np.asarray(panel).astype(np.float32)
    out = np.zeros((256, 256, 4), np.float32)
    out[..., :3] = np.clip(col[..., :3] * shade[..., None], 0, 255)
    out[..., 3] = ia[..., 3] * 255
    return out.astype(np.uint8)
