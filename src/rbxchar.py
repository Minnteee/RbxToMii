"""Assemble a Roblox R15 character (from roblox/rigs.json + downloaded assets) in Roblox world space.
Every visible thing becomes a Piece: positions/normals/uvs/faces + texture (RGBA array) + how it follows the body
(attach = R15 part name for rigid pieces, or skin = per-vertex R15 bone weights for layered clothing)."""
import os, json, io
import numpy as np
from PIL import Image, ImageDraw

import rbxmesh
import rbx_substitutes

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, '..'))
NET3D = os.path.join(WORK, 'roblox', 'net3d')
NET = os.path.join(WORK, 'roblox', 'net')

BODY = ['Head', 'UpperTorso', 'LowerTorso', 'LeftUpperArm', 'LeftLowerArm', 'LeftHand', 'RightUpperArm',
        'RightLowerArm', 'RightHand', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot', 'RightUpperLeg', 'RightLowerLeg',
        'RightFoot']
GROUP = {'UpperTorso': 'Torso', 'LowerTorso': 'Torso',
         'LeftUpperArm': 'LeftArm', 'LeftLowerArm': 'LeftArm', 'LeftHand': 'LeftArm',
         'RightUpperArm': 'RightArm', 'RightLowerArm': 'RightArm', 'RightHand': 'RightArm',
         'LeftUpperLeg': 'LeftLeg', 'LeftLowerLeg': 'LeftLeg', 'LeftFoot': 'LeftLeg',
         'RightUpperLeg': 'RightLeg', 'RightLowerLeg': 'RightLeg', 'RightFoot': 'RightLeg'}
COLOR_KEY = {'Torso': 'TorsoColor3', 'LeftArm': 'LeftArmColor3', 'RightArm': 'RightArmColor3',
             'LeftLeg': 'LeftLegColor3', 'RightLeg': 'RightLegColor3', 'Head': 'HeadColor3'}
# classic clothing template rectangles (x, y, w, h) per box face, Roblox 585x559 layout
TORSO_RECTS = {'F': (231, 74, 128, 128), 'B': (427, 74, 128, 128), 'R': (165, 74, 64, 128), 'L': (361, 74, 64, 128),
               'U': (231, 8, 128, 64), 'D': (231, 204, 128, 64)}
RIGHT_LIMB = {'L': (19, 355, 64, 128), 'B': (85, 355, 64, 128), 'R': (151, 355, 64, 128), 'F': (217, 355, 64, 128),
              'U': (217, 289, 64, 64), 'D': (217, 485, 64, 64)}
LEFT_LIMB = {'F': (308, 355, 64, 128), 'L': (374, 355, 64, 128), 'B': (440, 355, 64, 128), 'R': (506, 355, 64, 128),
             'U': (308, 289, 64, 64), 'D': (308, 485, 64, 64)}
RECTS = {'Torso': TORSO_RECTS, 'RightArm': RIGHT_LIMB, 'LeftArm': LEFT_LIMB, 'RightLeg': RIGHT_LIMB, 'LeftLeg': LEFT_LIMB}
# which body part an accessory attachment belongs to (fallback when the part has no such attachment)
ATTACH_FALLBACK = {'Hat': 'Head', 'Hair': 'Head', 'FaceFront': 'Head', 'FaceCenter': 'Head', 'Neck': 'UpperTorso',
                   'BodyFront': 'UpperTorso', 'BodyBack': 'UpperTorso', 'LeftCollar': 'UpperTorso',
                   'RightCollar': 'UpperTorso', 'LeftShoulder': 'UpperTorso', 'RightShoulder': 'UpperTorso',
                   'WaistCenter': 'LowerTorso', 'WaistFront': 'LowerTorso', 'WaistBack': 'LowerTorso'}


class Piece:
    def __init__(self, name, pos, nrm, uv, faces, tex=None, color=(200, 200, 200), attach=None, skin=None, kind=''):
        self.name, self.pos, self.nrm, self.uv, self.faces = name, pos, nrm, uv, faces
        self.tex, self.color, self.attach, self.skin, self.kind = tex, color, attach, skin, kind

    def __repr__(self):
        return '<Piece %s %s v=%d f=%d attach=%s skin=%s>' % (self.kind, self.name, len(self.pos), len(self.faces),
                                                            self.attach, self.skin is not None)


def cf_mat(cf):
    M = np.eye(4)
    M[:3, :3] = np.array(cf['r'], float).reshape(3, 3)
    M[:3, 3] = cf['p']
    return M


def load_image(aid):
    p = os.path.join(NET3D, '%d.bin' % aid)
    if not os.path.exists(p):
        return None
    try:
        return np.asarray(Image.open(p).convert('RGBA'))
    except Exception:
        return None


def mesh_to_world(mesh, part):
    """MeshPart rule: the mesh bounding box is centred on the part and stretched to part.size"""
    p = mesh['pos'].astype(np.float64)
    lo, hi = p.min(0), p.max(0)
    c = (lo + hi) / 2
    ext = np.where(hi - lo > 1e-6, hi - lo, 1.0)
    s = np.array(part['size'], float) / ext
    M = cf_mat(part['cf'])
    pw = ((p - c) * s) @ M[:3, :3].T + M[:3, 3]
    n = mesh['nrm'].astype(np.float64) / s
    n = n @ M[:3, :3].T
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-9)[:, None]
    return pw, n, (c, s, M)


# ---------------------------------------------------------------- clothing composite
def template_image(rig, key, name):
    """shirt / pants template as a 585x559 image (Roblox maps the template over the whole picture, so HD uploads,
    e.g. 1219x1165, are scaled down - until 1.2.5 only their top left corner was used and the pants were missing)"""
    aid = rig.get(key)
    if not aid:
        return None
    im = load_image(aid)
    if im is not None:
        im = Image.fromarray(im)
    else:
        alt = [f for f in os.listdir(NET) if f.startswith('%s_%s_%d' % (name, key, aid))]
        if not alt:
            return None
        im = Image.open(os.path.join(NET, alt[0])).convert('RGBA')
    if key in ('shirt', 'pants') and im.size != (585, 559):
        im = im.resize((585, 559), Image.LANCZOS)
    return im


def body_color(rig, group):
    bc = rig.get('body_colors') or {}
    c = bc.get(COLOR_KEY[group])
    return tuple(c) if c else (163, 162, 165)


# Roblox's default "modesty" clothing (teal shirt / grey pants with the Roblox logo) sits under layered clothing
# and is never visible in Roblox; layered AccessoryTypes: TShirt 9, Shirt 10, Pants 11, Jacket 12, Sweater 13,
# Shorts 14, DressSkirt 17
DEFAULT_SHIRTS = range(855_700_000, 855_800_000)
DEFAULT_PANTS = range(867_800_000, 867_900_000)
LAYERED_TOPS, LAYERED_BOTTOMS = {9, 10, 12, 13, 17}, {11, 14, 17}


def composites(rig, name):
    """TOP (torso + arms: skin, pants torso, shirt, t-shirt) and BOTTOM (legs: skin, pants) 585x559 images"""
    shirt, pants, tshirt = template_image(rig, 'shirt', name), template_image(rig, 'pants', name), template_image(rig, 'tshirt', name)
    types = {a.get('type') for a in rig.get('accessories', [])}
    if (rig.get('shirt') or 0) in DEFAULT_SHIRTS and types & LAYERED_TOPS:
        shirt = None
    if (rig.get('pants') or 0) in DEFAULT_PANTS and types & LAYERED_BOTTOMS:
        pants = None
    top = Image.new('RGBA', (585, 559), (0, 0, 0, 255))
    bot = Image.new('RGBA', (585, 559), (0, 0, 0, 255))
    dt, db = ImageDraw.Draw(top), ImageDraw.Draw(bot)
    for x, y, w, h in TORSO_RECTS.values():
        dt.rectangle((x - 2, y - 2, x + w + 1, y + h + 1), fill=body_color(rig, 'Torso') + (255,))
        db.rectangle((x - 2, y - 2, x + w + 1, y + h + 1), fill=body_color(rig, 'Torso') + (255,))
    for grp, rects in (('RightArm', RIGHT_LIMB), ('LeftArm', LEFT_LIMB)):
        for x, y, w, h in rects.values():
            dt.rectangle((x - 2, y - 2, x + w + 1, y + h + 1), fill=body_color(rig, grp) + (255,))
    for grp, rects in (('RightLeg', RIGHT_LIMB), ('LeftLeg', LEFT_LIMB)):
        for x, y, w, h in rects.values():
            db.rectangle((x - 2, y - 2, x + w + 1, y + h + 1), fill=body_color(rig, grp) + (255,))
    if pants is not None:
        torso_part = pants.crop((163, 6, 557, 270))
        top.alpha_composite(torso_part, (163, 6))
        bot.alpha_composite(pants)
    if shirt is not None:
        top.alpha_composite(shirt)
    if tshirt is not None:
        t = tshirt.resize((128, 128), Image.LANCZOS)
        top.alpha_composite(t, (231, 74))
    return np.asarray(top.convert('RGBA')), np.asarray(bot.convert('RGBA'))


def clothing_layers(rig, name):
    """classic clothing alone (transparent elsewhere): TOP (torso + arms) and BOTTOM (legs) 585x559 RGBA, or None"""
    shirt, pants, tshirt = template_image(rig, 'shirt', name), template_image(rig, 'pants', name), template_image(rig, 'tshirt', name)
    # Roblox's automatic default clothing is not drawn over a package's own texture
    if (rig.get('shirt') or 0) in DEFAULT_SHIRTS:
        shirt = None
    if (rig.get('pants') or 0) in DEFAULT_PANTS:
        pants = None
    if shirt is None and pants is None and tshirt is None:
        return None
    top = Image.new('RGBA', (585, 559), (0, 0, 0, 0))
    bot = Image.new('RGBA', (585, 559), (0, 0, 0, 0))
    if pants is not None:
        top.alpha_composite(pants.crop((163, 6, 557, 270)), (163, 6))
        bot.alpha_composite(pants)
    if shirt is not None:
        top.alpha_composite(shirt)
    if tshirt is not None:
        top.alpha_composite(tshirt.resize((128, 128), Image.LANCZOS), (231, 74))
    return np.asarray(top), np.asarray(bot)


def bake_package_texture(tex, colour, tris, layer=None, scale=2):
    """a body part's own texture (packages: Paladin, Rthro, ...) over the body colour, classic clothing on top.
    tris: list of (mesh uv (T,3,2), clothing template uv (T,3,2) in 0..1) of every part using this texture"""
    t = tex.astype(np.float32)
    a = t[..., 3:4] / 255.0
    base = t[..., :3] * a + np.array(colour[:3], np.float32) * (1 - a)
    if layer is None:
        out = np.empty(tex.shape, np.uint8)
        out[..., :3] = np.clip(base, 0, 255).astype(np.uint8)
        out[..., 3] = 255
        return out
    h, w = tex.shape[:2]
    H, W = h * scale, w * scale
    big = np.asarray(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32)
    # which triangle covers which texel: draw every triangle with its number as colour
    idx_img = Image.new('RGB', (W, H), (0, 0, 0))
    d = ImageDraw.Draw(idx_img)
    muv = np.concatenate([m for m, _ in tris])
    cuv = np.concatenate([c for _, c in tris])
    for k, tri in enumerate(muv):
        n = k + 1
        d.polygon([(float(u) * W, float(v) * H) for u, v in tri], fill=(n & 255, (n >> 8) & 255, (n >> 16) & 255))
    ia = np.asarray(idx_img).astype(np.int64)
    tri_id = ia[..., 0] | (ia[..., 1] << 8) | (ia[..., 2] << 16)
    ys, xs = np.nonzero(tri_id)
    if len(ys):
        k = tri_id[ys, xs] - 1
        p = np.stack([(xs + 0.5) / W, (ys + 0.5) / H], 1)
        A, B, C = muv[k, 0], muv[k, 1], muv[k, 2]
        v0, v1, v2 = B - A, C - A, p - A
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        b1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        b2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        b0 = 1 - b1 - b2
        q = b0[:, None] * cuv[k, 0] + b1[:, None] * cuv[k, 1] + b2[:, None] * cuv[k, 2]
        lh, lw = layer.shape[:2]
        qx = np.clip((q[:, 0] * lw).astype(int), 0, lw - 1)
        qy = np.clip((q[:, 1] * lh).astype(int), 0, lh - 1)
        col = layer[qy, qx].astype(np.float32)
        al = col[:, 3:4] / 255.0
        big[ys, xs] = big[ys, xs] * (1 - al) + col[:, :3] * al
    out = np.empty((H, W, 4), np.uint8)
    out[..., :3] = np.clip(big, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def box_uv(group, face, P, box):
    """template pixel coords for points P on box face (character faces -Z, right = +X)"""
    (x0, y0, z0), (x1, y1, z1) = box
    fx, fy, fw, fh = RECTS[group][face]
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    sx = lambda v, a, b: np.clip((v - a) / max(b - a, 1e-6), 0, 1)
    if face == 'F':
        u, v = sx(-X, -x1, -x0), sx(-Y, -y1, -y0)
    elif face == 'B':
        u, v = sx(X, x0, x1), sx(-Y, -y1, -y0)
    elif face == 'R':
        u, v = sx(-Z, -z1, -z0), sx(-Y, -y1, -y0)
    elif face == 'L':
        u, v = sx(Z, z0, z1), sx(-Y, -y1, -y0)
    elif face == 'U':
        u, v = sx(-X, -x1, -x0), sx(-Z, -z1, -z0)
    else:  # D
        u, v = sx(-X, -x1, -x0), sx(Z, z0, z1)
    m = 0.75
    px = fx + m + u * (fw - 2 * m)
    py = fy + m + v * (fh - 2 * m)
    return np.stack([px / 585.0, py / 559.0], 1)


# ---------------------------------------------------------------- head
def classic_head(size, segs=48, rings=28, n=5.0):
    """Roblox classic head: vertical rounded cylinder (superellipse profile), cylindrical UVs (seam at the back)"""
    th = np.linspace(0, np.pi, rings + 1)  # 0 = top
    ys = np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2 / n)
    rs = np.abs(np.sin(th)) ** (2 / n)
    ang = np.linspace(-np.pi, np.pi, segs + 1)  # -pi..pi, angle 0 = front (-Z)
    pos, uv, nrm = [], [], []
    for i in range(rings + 1):
        for j in range(segs + 1):
            a = ang[j]
            x = np.sin(a) * rs[i]
            z = -np.cos(a) * rs[i]
            pos.append((x * size[0] / 2, ys[i] * size[1] / 2, z * size[2] / 2))
            uv.append((j / segs, i / rings))
    pos = np.array(pos)
    faces = []
    W = segs + 1
    for i in range(rings):
        for j in range(segs):
            a, b, c, d = i * W + j, i * W + j + 1, (i + 1) * W + j, (i + 1) * W + j + 1
            faces += [(a, b, c), (b, d, c)]  # counter-clockwise seen from outside (game convention)
    faces = np.array(faces)
    # smooth normals from geometry
    nrm = np.zeros_like(pos)
    A, B, C = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    fn = np.cross(B - A, C - A)
    for k in range(3):
        np.add.at(nrm, faces[:, k], fn)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
    return pos, nrm, np.array(uv), faces


# optional, per character: how the headshot (420x420) maps onto the front of a stand-in head - centre pixel (cx, cy)
# and pixels per stud, from roblox/face_src.json if it exists; without it the stand-in head gets the classic face
def _face_src():
    try:
        return json.load(open(os.path.join(WORK, 'roblox', 'face_src.json'), encoding='utf-8'))
    except (OSError, ValueError):
        return {}


FACE_SRC = _face_src()


def draw_smile(img, cx, cy, w, h, width):
    d = ImageDraw.Draw(img)
    d.arc((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 20, 160, fill=(10, 10, 10, 255), width=width)


def classic_face(n=400):
    """Roblox's classic face (two oval eyes + smile) on an n x n picture of the head front (n px = head width,
    centre = head centre). Layout measured on Roblox's default head with its default "Smile" mood."""
    k = 4                                    # drawn 4x larger, then shrunk: smooth edges
    s = n * k / 400.0
    img = Image.new('RGBA', (n * k, n * k), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ink = (12, 12, 12, 255)
    for ex in (162.1, 236.9):
        d.ellipse(((ex - 13.5) * s, (115.4 - 29) * s, (ex + 13.5) * s, (115.4 + 29) * s), fill=ink)
    # smile: (distance from the middle, upper edge, lower edge) - thick in the middle, pointed at the ends
    prof = [(0, 293, 309), (5, 292.5, 308.5), (10, 292, 307.5), (15, 290, 306), (20, 288.5, 304.5),
            (25, 285.5, 301.5), (30, 282.5, 298.5), (35, 277.5, 295.5), (40, 273.5, 290.5), (45, 266.5, 286),
            (50, 258, 278.5), (55, 248, 271), (60, 241.5, 260), (65, 243, 247)]
    full = [(-dx, a, b) for dx, a, b in prof[:0:-1]] + prof
    upper = [(200 + dx, a) for dx, a, b in full]
    lower = [(200 + dx, b) for dx, a, b in full[::-1]]
    d.polygon([(x * s, y * s) for x, y in upper + lower], fill=ink)
    return img.resize((n, n), Image.LANCZOS)


# FACS control weights of Roblox's default mood (smile) for dynamic heads; per character overrides possible
DEFAULT_MOOD = None


def mood_for(name):
    """FACS weights of the character's mood (net3d/moods.json, made by cache_assets.py) - None = neutral face"""
    p = os.path.join(NET3D, 'moods.json')
    if not os.path.exists(p):
        return DEFAULT_MOOD
    w = (json.load(open(p, encoding='utf-8')).get(name) or {}).get('weights')
    return DEFAULT_MOOD if w is None else w        # {} = the head's own neutral face


def mesh_to_world_fixed(mesh, part, rest_pos):
    """like mesh_to_world, but the box -> part fit comes from another (rest) pose of the same mesh"""
    p = rest_pos.astype(np.float64)
    lo, hi = p.min(0), p.max(0)
    c = (lo + hi) / 2
    ext = np.where(hi - lo > 1e-6, hi - lo, 1.0)
    s = np.array(part['size'], float) / ext
    M = cf_mat(part['cf'])
    pw = ((mesh['pos'].astype(np.float64) - c) * s) @ M[:3, :3].T + M[:3, 3]
    n = mesh['nrm'].astype(np.float64) / s
    n = n @ M[:3, :3].T
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-9)[:, None]
    return pw, n


def skin_under(tex, skin):
    """Roblox draws a head texture over the head colour: transparent texels show the skin"""
    if tex is None:
        return None
    t = tex.astype(np.float32)
    a = t[..., 3:4] / 255.0
    out = np.empty_like(t)
    out[..., :3] = t[..., :3] * a + np.array(skin, np.float32) * (1 - a)
    out[..., 3] = 255
    return np.clip(out, 0, 255).astype(np.uint8)


def head_texture(name, size, skin, tw=1024, th=512):
    """bake the cylindrical head texture: skin + face projected from the avatar headshot"""
    img = np.zeros((th, tw, 4), np.float32)
    img[..., :3] = np.array(skin, np.float32)
    img[..., 3] = 255
    # surface point for each texel
    u = (np.arange(tw) + 0.5) / tw
    v = (np.arange(th) + 0.5) / th
    U, V = np.meshgrid(u, v)
    ang = -np.pi + U * 2 * np.pi
    n = 5.0
    t = V * np.pi
    ys = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2 / n)
    rs = np.abs(np.sin(t)) ** (2 / n)
    X = np.sin(ang) * rs * size[0] / 2
    Y = ys * size[1] / 2
    Z = -np.cos(ang) * rs * size[2] / 2
    front = np.cos(ang)  # 1 at the face centre
    src = FACE_SRC.get(name)
    if src:
        shot = np.asarray(Image.open(os.path.join(NET, '%s_avatar-headshot.png' % name)).convert('RGBA')).astype(np.float32)
        px = src['cx'] - X * src['s']
        py = src['cy'] - Y * src['s']
        ix = np.clip(px.astype(int), 0, 419)
        iy = np.clip(py.astype(int), 0, 419)
        col = shot[iy, ix]
        rx, ry = src['r']
        e = (X / (size[0] * rx)) ** 2 + ((Y - src.get('cshift', 0.0) * size[1]) / (size[1] * ry)) ** 2
        f0 = src.get('front0', 0.15)
        wgt = np.clip((1.0 - e) / 0.18, 0, 1) * np.clip((front - f0) / 0.25, 0, 1) * (col[..., 3] / 255.0)
        if src.get('white_key'):
            # clothes (white collar) next to the face: fade to skin
            wgt = wgt * np.clip((235 - col[..., :3].min(-1)) / 40.0, 0, 1)
        img[..., :3] = img[..., :3] * (1 - wgt[..., None]) + col[..., :3] * wgt[..., None]
    else:
        # Roblox's classic face (eyes + smile): what the default head shows
        f = np.asarray(classic_face(400)).astype(np.float32)
        s = 400 / (size[0] * 1.0)
        px = 200 - X * s
        py = 200 - Y * s
        ix = np.clip(px.astype(int), 0, 399)
        iy = np.clip(py.astype(int), 0, 399)
        col = f[iy, ix]
        wgt = (col[..., 3] / 255.0) * (front > 0.3)
        img[..., :3] = img[..., :3] * (1 - wgt[..., None]) + col[..., :3] * wgt[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)



# ---------------------------------------------------------------- layered clothing (cage fitting)
def uv_keys(uv):
    q = np.round(np.asarray(uv, np.float64) * 10000).astype(np.int64)
    return q[:, 0] * 1000003 + q[:, 1]


def body_cage_map(parts):
    """UV key -> world position of the body's outer cage (all body parts; the head's cage only exists when Roblox's
    local cache had it - the web API gives it out only after a login)"""
    keys, pts = [], []
    for pn in BODY:
        p = parts[pn]
        wt = p.get('wrap_target')
        if not wt or not wt.get('cage'):
            continue
        cp = os.path.join(NET3D, '%d.bin' % wt['cage'])
        if not os.path.exists(cp):
            continue
        cg = rbxmesh.load(cp)
        sc = np.array(p['size'], float) / np.array(p.get('initial_size') or p['size'], float)
        M = cf_mat(p['cf'])
        C = cf_mat(wt['cage_origin']) if wt.get('cage_origin') else np.eye(4)
        C[:3, 3] *= sc
        loc = (cg['pos'] * sc) @ C[:3, :3].T + C[:3, 3]
        keys.append(uv_keys(cg['uv']))
        pts.append(loc @ M[:3, :3].T + M[:3, 3])
    keys = np.concatenate(keys)
    pts = np.concatenate(pts)
    out = {}
    for k in np.unique(keys):
        out[int(k)] = pts[keys == k].mean(0)
    return out


def knn_idw(src, dst_vals, query, k=8):
    """inverse-distance interpolation of dst_vals (defined at src points) at query points"""
    out = np.zeros((len(query), dst_vals.shape[1]))
    for a in range(0, len(query), 1024):
        q = query[a:a + 1024]
        d2 = ((q[:, None, :] - src[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(d2, k, axis=1)[:, :k]
        dd = np.take_along_axis(d2, idx, 1)
        w = 1.0 / (dd + 1e-6)
        w /= w.sum(1, keepdims=True)
        out[a:a + 1024] = (w[..., None] * dst_vals[idx]).sum(1)
    return out


def closest_on_tris(P, A, B, C):
    """closest points on triangles (A,B,C: (n,k,3)) to points P (n,3) -> barycentric (n,k,3), dist2 (n,k)"""
    P = P[:, None, :]
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = P - B
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = P - C
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    v = vb / den
    w = vc / den
    u = 1 - v - w
    bary = np.stack([u, v, w], -1)
    # region tests (vertex / edge regions)
    def setb(mask, b0, b1, b2):
        bary[mask] = np.stack([b0[mask], b1[mask], b2[mask]], -1)
    z = np.zeros_like(d1)
    o = np.ones_like(d1)
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = np.clip((d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1e-20, (d4 - d3) + (d5 - d6)), 0, 1)
    setb(m, z, 1 - t, t)
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = np.clip(d2 / np.where(d2 - d6 == 0, 1e-20, d2 - d6), 0, 1)
    setb(m, 1 - t, z, t)
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = np.clip(d1 / np.where(d1 - d3 == 0, 1e-20, d1 - d3), 0, 1)
    setb(m, 1 - t, t, z)
    m = (d6 >= 0) & (d5 <= d6)
    setb(m, z, z, o)
    m = (d3 >= 0) & (d4 <= d3)
    setb(m, z, o, z)
    m = (d1 <= 0) & (d2 <= 0)
    setb(m, o, z, z)
    Q = bary[..., 0:1] * A + bary[..., 1:2] * B + bary[..., 2:3] * C
    return bary, ((Q - P) ** 2).sum(-1)


def tri_frame(A, B, C):
    e1 = B - A
    n = np.cross(B - A, C - A)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    e1 /= np.maximum(np.linalg.norm(e1, axis=-1, keepdims=True), 1e-12)
    e2 = np.cross(n, e1)
    return np.stack([e1, e2, n], -2)  # rows


def bind(cage_pos, faces, Q, cand=24):
    """bind points Q to the closest cage triangle: (tri, bary, local offset)"""
    cen = cage_pos[faces].mean(1)
    tri_out = np.zeros(len(Q), np.int64)
    bary_out = np.zeros((len(Q), 3))
    for a in range(0, len(Q), 512):
        q = Q[a:a + 512]
        d2 = ((q[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
        k = min(cand, len(faces) - 1)
        idx = np.argpartition(d2, k, axis=1)[:, :k]
        T = faces[idx]  # (n,k,3)
        bary, dist = closest_on_tris(q, cage_pos[T[..., 0]], cage_pos[T[..., 1]], cage_pos[T[..., 2]])
        best = dist.argmin(1)
        r = np.arange(len(q))
        tri_out[a:a + 512] = idx[r, best]
        bary_out[a:a + 512] = bary[r, best]
    T = faces[tri_out]
    A, B, C = cage_pos[T[:, 0]], cage_pos[T[:, 1]], cage_pos[T[:, 2]]
    P = bary_out[:, 0:1] * A + bary_out[:, 1:2] * B + bary_out[:, 2:3] * C
    Fr = tri_frame(A, B, C)
    off = np.einsum('nij,nj->ni', Fr, Q - P)
    return tri_out, bary_out, off


def apply_bind(cage_pos, faces, binding):
    tri, bary, off = binding
    T = faces[tri]
    A, B, C = cage_pos[T[:, 0]], cage_pos[T[:, 1]], cage_pos[T[:, 2]]
    P = bary[:, 0:1] * A + bary[:, 1:2] * B + bary[:, 2:3] * C
    Fr = tri_frame(A, B, C)
    return P + np.einsum('nji,nj->ni', Fr, off)


def head_targets(rw, head_part, n=5.0):
    """map the mannequin head region of a reference cage radially onto our classic head (superellipsoid)"""
    lo, hi = rw.min(0), rw.max(0)
    cm, em = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-6)
    H = cf_mat(head_part['cf'])
    eh = np.array(head_part['size'], float) / 2
    d = (rw - cm) / em
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    rxz = np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2)
    t = (np.abs(d[:, 1]) ** n + rxz ** n) ** (-1.0 / n)
    loc = d * t[:, None] * eh
    return loc @ H[:3, :3].T + H[:3, 3]


def fit_layer(piece, h, target, head_part=None):
    """deform one layered item: its inner cage is pulled onto the target cage (UV key -> world pos), the mesh and
    the outer cage follow by surface binding. Returns the deformed outer cage as target for the next layer."""
    wl = h['wrap_layer']
    ref = rbxmesh.load(os.path.join(NET3D, '%d.bin' % wl['reference']))
    cage = rbxmesh.load(os.path.join(NET3D, '%d.bin' % wl['cage']))
    M = cf_mat(h['cf'])
    Rf = cf_mat(wl['reference_origin']) if wl.get('reference_origin') else np.eye(4)
    Co = cf_mat(wl['cage_origin']) if wl.get('cage_origin') else np.eye(4)
    rw = (ref['pos'] @ Rf[:3, :3].T + Rf[:3, 3]) @ M[:3, :3].T + M[:3, 3]
    cw = (cage['pos'] @ Co[:3, :3].T + Co[:3, 3]) @ M[:3, :3].T + M[:3, 3]
    rk = uv_keys(ref['uv'])
    have = np.array([int(k) in target for k in rk])
    uv = ref['uv']
    head = (~have) & (uv[:, 0] > 3) & (uv[:, 0] < 4) & (uv[:, 1] > 0) & (uv[:, 1] < 1)
    new_ref = rw.copy()
    if have.any():
        new_ref[have] = np.array([target[int(k)] for k in rk[have]])
    if head.any() and head_part is not None:
        new_ref[head] = head_targets(rw[head], head_part)
    rest = ~have & ~head
    if rest.any() and (~rest).any():
        disp = new_ref[~rest] - rw[~rest]
        new_ref[rest] = rw[rest] + knn_idw(rw[~rest], disp, rw[rest], k=4)
    F = ref['faces']
    piece.pos = apply_bind(new_ref, F, bind(rw, F, piece.pos))
    cw2 = apply_bind(new_ref, F, bind(rw, F, cw))
    new = dict(target)
    ck = uv_keys(cage['uv'])
    for k in np.unique(ck):
        new[int(k)] = cw2[ck == k].mean(0)
    return new

def conform(inner, outer, margin=0.015, reach=0.12):
    """push points of an inner layer that stick out through an outer layer back under it"""
    if not len(inner.pos) or not len(outer.faces):
        return 0
    tri, bary, off = bind(outer.pos, outer.faces, inner.pos, cand=16)
    T = outer.faces[tri]
    A, B, C = outer.pos[T[:, 0]], outer.pos[T[:, 1]], outer.pos[T[:, 2]]
    P = bary[:, 0:1] * A + bary[:, 1:2] * B + bary[:, 2:3] * C
    n = np.cross(B - A, C - A)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    # outward = away from the body centre line (outer meshes may have inconsistent winding)
    d = inner.pos - P
    dist = np.linalg.norm(d, axis=1)
    side = (d * n).sum(1)
    out_dir = np.sign(((P - inner.centre) * n).sum(1))
    outside = (side * out_dir > -margin) & (dist < reach)
    inner.pos[outside] = P[outside] - n[outside] * out_dir[outside, None] * margin
    return int(outside.sum())


# ---------------------------------------------------------------- build
def build(name, rigs):
    rig = rigs[name]
    parts = {p['path']: p for p in rig['parts']}
    pieces = []
    pkg = {}            # texture id of package body parts -> parts using it
    top, bot = composites(rig, name)
    # group boxes (default pose = axis aligned)
    gbox = {}
    for pn in BODY[1:]:
        p = parts[pn]
        c = np.array(p['cf']['p'])
        h = np.array(p['size']) / 2
        g = GROUP[pn]
        lo, hi = c - h, c + h
        if g in gbox:
            gbox[g] = (np.minimum(gbox[g][0], lo), np.maximum(gbox[g][1], hi))
        else:
            gbox[g] = (lo, hi)
    for pn in BODY[1:]:
        p = parts[pn]
        mesh = rbxmesh.load(os.path.join(NET3D, '%d.bin' % p['mesh']))
        pw, nw, _ = mesh_to_world(mesh, p)
        F = mesh['faces']
        # unweld: one vertex per triangle corner, uv from box projection of the template
        P = pw[F].reshape(-1, 3)
        N = nw[F].reshape(-1, 3)
        A, B, C = pw[F[:, 0]], pw[F[:, 1]], pw[F[:, 2]]
        fn = np.cross(B - A, C - A)
        fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-12)[:, None]
        ax = np.abs(fn).argmax(1)
        sgn = np.sign(fn[np.arange(len(fn)), ax])
        face = np.where(ax == 2, np.where(sgn < 0, 'F', 'B'), np.where(ax == 0, np.where(sgn > 0, 'R', 'L'), np.where(sgn > 0, 'U', 'D')))
        g = GROUP[pn]
        uv = np.zeros((len(P), 2))
        for fc in 'FBRLUD':
            m = np.repeat(face == fc, 3)
            if m.any():
                uv[m] = box_uv(g, fc, P[m], gbox[g])
        tex = bot if g in ('LeftLeg', 'RightLeg') else top
        # packages (Paladin, Rthro, ...) bring their own texture, laid out on the mesh's own UVs
        own = load_image(p['tex']) if p.get('tex') else None
        if own is not None and mesh.get('uv') is not None and len(mesh['uv']) == len(pw):
            muv = np.asarray(mesh['uv'], np.float64)[F].reshape(-1, 2)
            e = pkg.setdefault(p['tex'], {'img': own, 'colour': body_color(rig, g), 'legs': g in ('LeftLeg', 'RightLeg'),
                                          'items': []})
            e['items'].append((len(pieces), muv, uv))
        pieces.append(Piece(pn, P, N, uv, np.arange(len(P)).reshape(-1, 3), tex=tex, attach=pn, kind='body'))
    if pkg:
        layers = clothing_layers(rig, name)
        for e in pkg.values():
            lay = (layers[1] if e['legs'] else layers[0]) if layers is not None else None
            tris = [(muv.reshape(-1, 3, 2), buv.reshape(-1, 3, 2)) for _, muv, buv in e['items']]
            baked = bake_package_texture(e['img'], e['colour'], tris, lay)
            for i, muv, _ in e['items']:
                pieces[i].uv = muv
                pieces[i].tex = baked
    # head: the real Roblox head (dynamic head / Rthro head) with its own texture over the head colour;
    # only when its mesh is missing a classic head with the face from the avatar picture
    hp = parts['Head']
    M = cf_mat(hp['cf'])
    skin = body_color(rig, 'Head')
    hmp = os.path.join(NET3D, '%d.bin' % hp['mesh']) if hp.get('mesh') else ''
    if hmp and os.path.exists(hmp):
        hm = rbxmesh.load(hmp)
        if hm.get('facs'):
            # Roblox shows dynamic heads with the avatar's mood (default: a smile) - pose the face the same way;
            # the part fit uses the bounding box of the neutral head
            posed = dict(hm)
            posed['pos'], posed['nrm'] = rbxmesh.facs_pose(hm, mood_for(name))
            hw, hnw = mesh_to_world_fixed(posed, hp, hm['pos'])
        else:
            hw, hnw, _ = mesh_to_world(hm, hp)
        # head picture: SurfaceAppearance.ColorMap (newer heads) or TextureID, drawn over the head colour
        htid = (hp.get('surface') or {}).get('ColorMap') or hp.get('tex')
        htex = load_image(htid) if htid else None
        pieces.append(Piece('Head', hw, hnw, hm['uv'].astype(np.float64), hm['faces'], tex=skin_under(htex, skin),
                            color=skin, attach='Head', kind='head'))
        real_head = True
    else:
        hpos, hn, huv, hf = classic_head(hp['size'])
        pieces.append(Piece('Head', hpos @ M[:3, :3].T + M[:3, 3], hn @ M[:3, :3].T, huv, hf,
                            tex=head_texture(name, hp['size'], skin), attach='Head', kind='head'))
        real_head = False
    # accessories (rigid) and layered clothing (skinned)
    body_att = {}
    for pn in BODY:
        for an in (parts[pn].get('attachments') or {}):
            body_att[an] = pn
    for acc in rig['accessories']:
        hpth = acc['handle']
        if not hpth or hpth not in parts:
            continue
        h = parts[hpth]
        if not h.get('mesh') and not h.get('special'):
            continue
        mid = h.get('mesh') or (h.get('special') or {}).get('mesh')
        mp = os.path.join(NET3D, '%d.bin' % mid)
        if not os.path.exists(mp):
            subs = rbx_substitutes.substitutes(acc['name'], h, parts, NET3D)
            for sub in subs:
                nm, spos, snrm, sfaces, scol, satt = sub[:6]
                suv = sub[6] if len(sub) > 6 else np.zeros((len(spos), 2))
                stex = load_image(sub[7]) if len(sub) > 7 else None
                pieces.append(Piece(nm, spos, snrm, np.asarray(suv, float), np.asarray(sfaces), tex=stex,
                                    color=scol or (200, 200, 200), attach=satt, kind='accessory'))
            if not subs:
                pieces.append(Piece(acc['name'], np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2)),
                                    np.zeros((0, 3), int), kind='missing'))
            continue
        mesh = rbxmesh.load(mp)
        pw, nw, (c, s, Mh) = mesh_to_world(mesh, h)
        texid = (h.get('surface') or {}).get('ColorMap') or h.get('tex') or (h.get('special') or {}).get('tex')
        tex = load_image(texid) if texid else None
        color = tuple(h.get('color') or (200, 200, 200))
        att = None
        for an in (h.get('attachments') or {}):
            if an in body_att:
                att = body_att[an]
                break
            base = an.replace('Attachment', '')
            if base in ATTACH_FALLBACK:
                att = ATTACH_FALLBACK[base]
                break
        skin = None
        if 'wrap_layer' in h and mesh.get('skin_idx') is not None:
            bn = [b['name'] for b in mesh['bones']]
            if 'DynamicHead' in bn:
                if not real_head:
                    continue  # eyebrow/face accessory of a dynamic head we could not load - face picture has it
                att = 'Head'   # sits on the real head: rigid on it
            else:
                skin = (bn, mesh['skin_idx'], mesh['skin_w'])
        pc = Piece(acc['name'], pw, nw, mesh['uv'].astype(np.float64), mesh['faces'], tex=tex, color=color,
                   attach=att or 'UpperTorso', skin=skin, kind='layered' if 'wrap_layer' in h else 'accessory')
        pc.handle = h
        pieces.append(pc)
    # layered clothing: fit inner cages to the body, layer by layer (WrapLayer.Order)
    layered = sorted([p for p in pieces if p.kind == 'layered'], key=lambda p: p.handle['wrap_layer'].get('order') or 0)
    if layered:
        target = body_cage_map(parts)
        for pc in layered:
            target = fit_layer(pc, pc.handle, target, head_part=parts['Head'])
        # keep inner layers under outer ones (the crop shirt under the hoodie, ...)
        for i, inner in enumerate(layered):
            for outer in layered[i + 1:]:
                if outer.attach == 'Head' or inner.attach == 'Head':
                    continue
                inner.centre = np.array(parts['UpperTorso']['cf']['p']) * [1, 0, 1] + [0, inner.pos[:, 1].mean(), 0]
                inner.centre = np.where(np.arange(3) == 1, inner.pos[:, 1][:, None].mean(), inner.centre)
                inner.centre = np.broadcast_to(np.array(parts['UpperTorso']['cf']['p']), inner.pos.shape).copy()
                inner.centre[:, 1] = inner.pos[:, 1]
                conform(inner, outer)
    joints = rig_joints(rig, parts)
    return pieces, joints


def rig_joints(rig, parts):
    """world positions of the R15 rig attachments (shoulders, elbows, ...)"""
    out = {}
    for pn in BODY:
        p = parts[pn]
        M = cf_mat(p['cf'])
        for an, cf in (p.get('attachments') or {}).items():
            if an.endswith('RigAttachment'):
                out.setdefault(an, {})[pn] = (M @ cf_mat(cf))[:3, 3]
    return out
