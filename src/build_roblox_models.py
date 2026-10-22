"""Build the Roblox 3D models as Tomodachi Life clothes:
  body  -> new model ClothAllRoblox<Tag> (skinned to the Mii skeleton), used by one redirected 'All' costume row
  head  -> replaces one full mask model (rigid on the head; full masks hide the Mii head)
Used by rbxtomii.py (make_character_files with its ToolCtx); GameCtx reads the same numbers from an extracted game."""
import sys, os, json, struct, time
import numpy as np
import zstandard
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bfres, bfres_write, bntx, bntx_write, byml_write, tlfmt, rbxchar, miimap, restbl, fska, chinfit
from hashnames import murmur3_32

WORK = os.path.abspath(os.path.join(HERE, '..'))
GAME = os.path.join(WORK, 'game', 'romfs')
ZC = zstandard.ZstdCompressor(level=19)


def zs(data):
    return ZC.compress(data)


def load_tex(name):
    d = tlfmt.read(os.path.join(GAME, 'Tex', name + '.bntx.zs'))
    return bntx.read(d)[0]


# ---------------------------------------------------------------- texture atlas
def fill_transparent(img):
    """opaque rendering: give transparent texels the colour of nearby opaque ones (no dark fringes)"""
    a = img[..., 3] > 40
    if a.all() or not a.any():
        out = img.copy()
        out[..., 3] = 255
        return out
    rgb = img[..., :3].astype(np.float32)
    mean = rgb[a].mean(0)
    acc = np.where(a[..., None], rgb, 0)
    cnt = a.astype(np.float32)
    for _ in range(6):  # a few dilation steps
        sa = np.zeros_like(acc)
        sc = np.zeros_like(cnt)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            sa += np.roll(np.roll(acc, dy, 0), dx, 1)
            sc += np.roll(np.roll(cnt, dy, 0), dx, 1)
        grow = (cnt == 0) & (sc > 0)
        acc[grow] = sa[grow] / sc[grow, None]
        cnt[grow] = 1
    acc[cnt == 0] = mean
    out = np.zeros_like(img)
    out[..., :3] = np.clip(acc, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def pack(textures, size, pad=4):
    """textures: list of (key, HxWx4 uint8, cap). Shelf packer; returns atlas, {key: (x, y, w, h)} or None"""
    for shrink in (1.0, 0.85, 0.7, 0.6, 0.5, 0.42, 0.35, 0.3, 0.25, 0.21, 0.18, 0.15, 0.12, 0.1):
        items = []
        for key, img, cap in textures:
            h, w = img.shape[:2]
            s = min(1.0, cap * shrink / max(w, h))
            items.append((key, img, max(4, int(w * s)), max(4, int(h * s))))
        items.sort(key=lambda t: -t[3])
        x = y = shelf = 0
        place = {}
        ok = True
        for key, img, w, h in items:
            if w + 2 * pad > size:
                ok = False
                break
            if x + w + 2 * pad > size:
                x, y = 0, y + shelf
                shelf = 0
            if y + h + 2 * pad > size:
                ok = False
                break
            place[key] = (x + pad, y + pad, w, h)
            x += w + 2 * pad
            shelf = max(shelf, h + 2 * pad)
        if not ok:
            continue
        atlas = np.zeros((size, size, 4), np.uint8)
        atlas[..., 3] = 255
        for key, img, w, h in items:
            px, py, _, _ = place[key]
            im = np.asarray(Image.fromarray(img).resize((w, h), Image.LANCZOS))
            ext = np.pad(im, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
            atlas[py - pad:py + h + pad, px - pad:px + w + pad] = ext
        return atlas, place
    raise RuntimeError('atlas overflow')


def texture_list(entries, caps):
    """entries: list of piece objects -> unique textures with caps; returns list, key per piece"""
    texs, keys, seen = [], [], {}
    for p in entries:
        if p.tex is not None:
            k = id(p.tex)
            if k not in seen:
                seen[k] = len(texs)
                cap = caps.get(p.kind, 512)
                texs.append((k, fill_transparent(p.tex), cap))
            keys.append(k)
        else:
            c = tuple(int(v) for v in (p.color or (200, 200, 200))[:3])
            k = ('col',) + c
            if k not in seen:
                seen[k] = len(texs)
                tile = np.zeros((8, 8, 4), np.uint8)
                tile[..., :3] = c
                tile[..., 3] = 255
                texs.append((k, tile, 8))
            keys.append(k)
    return texs, keys


def remap_uv(uv, rect, size, inset=0.5):
    x, y, w, h = rect
    u = np.clip(uv[:, 0], 0, 1)
    v = np.clip(uv[:, 1], 0, 1)
    return np.stack([(x + inset + u * (w - 2 * inset)) / size, (y + inset + v * (h - 2 * inset)) / size], 1)


def tangents(n):
    up = np.where(np.abs(n[:, 1:2]) > 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t = np.cross(up, n)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return np.concatenate([t, np.ones((len(n), 1))], 1)


def double_sided(p, pos, nrm, faces, uv, extra=None):
    """append mirrored triangles for meshes Roblox draws from both sides"""
    n = len(pos)
    pos2 = np.concatenate([pos, pos])
    nrm2 = np.concatenate([nrm, -nrm])
    uv2 = np.concatenate([uv, uv])
    faces2 = np.concatenate([faces, faces[:, ::-1] + n])
    ex2 = (extra + extra) if extra is not None else None
    return pos2, nrm2, faces2, uv2, ex2


def under_surface(pts, outer_pos, outer_faces, max_dist=0.07, straight_only=True, border=None, border_keep=0.0,
                  pts_nrm=None):
    """points that lie behind (inside) an outer layer and close to it - not next to an open border of it
    (border: border_samples of the outer layer)"""
    tri, bary, off = rbxchar.bind(outer_pos, outer_faces, pts, cand=12)
    T = outer_faces[tri]
    A, B, C = outer_pos[T[:, 0]], outer_pos[T[:, 1]], outer_pos[T[:, 2]]
    P = bary[:, 0:1] * A + bary[:, 1:2] * B + bary[:, 2:3] * C
    n = np.cross(B - A, C - A)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    d = pts - P
    dist = np.linalg.norm(d, axis=1)
    # only points straight under the outer surface count (not ones beside an edge, e.g. in an open jacket)
    straight = (-(d * n).sum(1)) >= 0.85 * np.maximum(dist, 1e-9)
    if not straight_only:   # bare body inside clothing: behind the nearest surface is enough
        res = ((d * n).sum(1) < 0) & (dist < max_dist)
    else:
        res = ((d * n).sum(1) < 0) & (dist < max_dist) & (straight | (dist < 1e-4))
    if pts_nrm is not None:     # covered only by the outside of the outer layer (not by its lining)
        res &= (n * pts_nrm).sum(1) > 0.2
    if border is not None and res.any():
        idx = np.nonzero(res)[0]
        res[idx[near_border(P[idx], border, border_keep)]] = False
    return res


def nearest_vertex(src, query):
    idx = np.zeros(len(query), np.int64)
    dist = np.zeros(len(query))
    for a in range(0, len(query), 2048):
        q = query[a:a + 2048]
        d2 = ((q[:, None, :] - src[None, :, :]) ** 2).sum(-1)
        i = d2.argmin(1)
        idx[a:a + 2048] = i
        dist[a:a + 2048] = np.sqrt(d2[np.arange(len(q)), i])
    return idx, dist


def outward_fraction(pos, faces):
    """share of triangles whose CCW normal points away from the piece centre (closed shapes: ~1, inside-out: ~0)"""
    if not len(faces):        # fully hidden under clothing
        return 1.0
    A, B, C = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    n = np.cross(B - A, C - A)
    c = (A + B + C) / 3
    return float(np.mean(((c - pos.mean(0)) * n).sum(1) > 0))


BONE_GROUP = {}
for _g, _bones in (('torso', ('Spine_1', 'Spine_2', 'Neck', 'Waist')),
                   ('arm_L', ('Clavicle_L', 'Arm_1_L', 'Arm_1_Sub_L', 'Arm_2_L', 'Wrist_L', 'Wrist_Sub_L')),
                   ('arm_R', ('Clavicle_R', 'Arm_1_R', 'Arm_1_Sub_R', 'Arm_2_R', 'Wrist_R', 'Wrist_Sub_R')),
                   ('leg_L', ('Leg_1_L', 'Leg_1_Sub_L', 'Leg_2_L', 'Ankle_L', 'Toe_L')),
                   ('leg_R', ('Leg_1_R', 'Leg_1_Sub_R', 'Leg_2_R', 'Ankle_R', 'Toe_R'))):
    for _b in _bones:
        BONE_GROUP[_b] = _g


def body_group(wd):
    """torso / arm / leg group of a vertex's strongest Mii bone"""
    return BONE_GROUP.get(max(wd.items(), key=lambda t: t[1])[0], 'torso') if wd else 'torso'


def same_group(inner, outer, inner_pos, outer_pos):
    """per inner vertex: does the nearest outer vertex belong to the same body part (no hand <-> trouser hem)?
    Only the bare body is checked; clothing under clothing always follows the outer layer."""
    near, dist = nearest_vertex(outer_pos, inner_pos)
    if inner[0].kind != 'body':
        return np.ones(len(inner_pos), bool), near, dist
    gi = [body_group(w) for w in inner[3]]
    go = [body_group(outer[3][k]) for k in near]
    return np.array([a == b for a, b in zip(gi, go)]), near, dist


def border_samples(pos, faces, q=1e-5):
    """points along the open borders of a mesh (front opening of a jacket, hems, cuffs, neck): edges that belong
    to one triangle only, after welding corners at the same place (UV seams are no borders)"""
    if not len(faces):
        return np.zeros((0, 3))
    key = np.round(np.asarray(pos) / q).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    F = inv[faces]
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = e[e[:, 0] != e[:, 1]]
    e.sort(1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    b = ue[cnt == 1]
    if not len(b):
        return np.zeros((0, 3))
    rep = np.asarray(pos)[first]
    a, c = rep[b[:, 0]], rep[b[:, 1]]
    return np.concatenate([a, c, (a + c) / 2, a * 0.75 + c * 0.25, a * 0.25 + c * 0.75])


def near_border(P, border, dist):
    """for surface points P: closer than dist to an open border of that surface?"""
    out = np.zeros(len(P), bool)
    if not len(border) or dist <= 0:
        return out
    for a in range(0, len(P), 2048):
        q = P[a:a + 2048]
        d2 = ((q[:, None, :] - border[None, :, :]) ** 2).sum(-1).min(1)
        out[a:a + 2048] = d2 < dist * dist
    return out


def conform_mii(inner_pos, outer_pos, outer_faces, margin=0.004, reach=0.045, mask=None, border=None, border_keep=0.0,
                inner_nrm=None):
    """after retargeting: push inner-layer points that stick out through an outer layer back under it
    (outer triangle normals from the CCW winding point outwards); mask: only these points may move.
    Points whose nearest outer point lies next to an open border of the outer layer stay where Roblox has them
    (the sweater in an open jacket is meant to be seen)."""
    tri, bary, off = rbxchar.bind(outer_pos, outer_faces, inner_pos, cand=16)
    T = outer_faces[tri]
    A, B, C = outer_pos[T[:, 0]], outer_pos[T[:, 1]], outer_pos[T[:, 2]]
    P = bary[:, 0:1] * A + bary[:, 1:2] * B + bary[:, 2:3] * C
    n = np.cross(B - A, C - A)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    d = inner_pos - P
    dist = np.linalg.norm(d, axis=1)
    along = np.abs((d * n).sum(1)) >= 0.85 * np.maximum(dist, 1e-9)   # in front of the surface, not beside an edge
    out = ((d * n).sum(1) > -margin) & (dist < reach) & (along | (dist < 2 * margin))
    if mask is not None:
        out &= mask
    if inner_nrm is not None:
        # only the outside of the outer layer counts: a jacket's lining (facing the body) must not pull the
        # sweater that is meant to be seen in the open front
        out &= (n * inner_nrm).sum(1) > 0.2
    if border is not None and out.any():
        idx = np.nonzero(out)[0]
        out[idx[near_border(P[idx], border, border_keep)]] = False
    res = inner_pos.copy()
    res[out] = P[out] - n[out] * margin
    return res, int(out.sum())


# ---------------------------------------------------------------- per character
STEPS = {'conform': True, 'hsr': True}     # layer steps (switchable for checks)
BORDER_KEEP_STUDS = 0.12                   # 1.2.3: inner layers next to an open border of an outer one stay as in Roblox


def build_character(name, rigs, mascot, mask_tpl, palette, rest, ref, skin_names, atlas_body=2048, atlas_head=2048, poses=None,
                    log=print):
    pieces, joints = rbxchar.build(name, rigs)
    # exact Roblox proportions: one scale factor, parts bound in the Mii standing pose (miimap.retarget_uniform)
    body, head, info = miimap.retarget_uniform(pieces, joints, rigs[name], rest, ref)
    rig = rigs[name]
    parts = {p['path']: p for p in rig['parts']}
    # keep layer order after the pose change: inner layers (and the body) stay under outer layers
    # layer distances in Roblox studs, converted with the character's scale
    sc = info['s']
    LAYER_GAP, TRANSFER_BODY, TRANSFER_CLOTH, REACH, MARGIN = 0.35 * sc, 0.15 * sc, 0.35 * sc, 0.22 * sc, 0.02 * sc
    BORDER_KEEP = BORDER_KEEP_STUDS * sc
    lay = [i for i, b in enumerate(body) if b[0].kind == 'layered']
    lay.sort(key=lambda i: body[i][0].handle['wrap_layer'].get('order') or 0)
    inner_all = [i for i, b in enumerate(body) if b[0].kind == 'body'] + lay
    # open borders of every layer (jacket front, hems, cuffs): what lies next to them is meant to be seen
    borders = {oi: border_samples(body[oi][1], body[oi][0].faces) for oi in lay}
    moved = 0
    for oi in lay:
        op = body[oi]
        for ii in inner_all:
            if ii == oi or (ii in lay and lay.index(ii) >= lay.index(oi)):
                continue
            ip = body[ii]
            same, _, _ = same_group(ip, op, ip[1], op[1])
            if STEPS['conform']:
                newpos, cnt = conform_mii(ip[1], op[1], op[0].faces, margin=MARGIN, reach=REACH, mask=same,
                                          border=borders[oi], border_keep=BORDER_KEEP, inner_nrm=ip[2])
            else:
                newpos, cnt = ip[1], 0
            moved += cnt
            # what lies right under an outer layer must bend exactly like it (same bone weights) - for the bare body
            # only within the same body part (a hand hanging next to a jacket hem keeps its own weights)
            W_in = list(ip[3])
            near, dist = nearest_vertex(op[1], newpos)
            if ip[0].kind == 'body' and ip[0].attach in ('LeftHand', 'RightHand', 'LeftFoot', 'RightFoot'):
                close = np.zeros(len(newpos), bool)   # hands and feet stay rigid on their own joint
            elif ip[0].kind == 'body':
                close = same & ((dist < TRANSFER_BODY) | under_surface(newpos, op[1], op[0].faces, max_dist=LAYER_GAP, straight_only=False))
            else:   # clothing under clothing: only where it lies straight under it and hangs on the same bone
                dom_in = [max(w.items(), key=lambda t: t[1])[0] if w else '' for w in ip[3]]
                dom_out = [max(op[3][k].items(), key=lambda t: t[1])[0] if op[3][k] else '' for k in near]
                same_bone = np.array([a == b for a, b in zip(dom_in, dom_out)])
                close = same_bone & under_surface(newpos, op[1], op[0].faces, max_dist=LAYER_GAP)
            for v in np.nonzero(close)[0]:
                W_in[v] = dict(op[3][near[v]])
            body[ii] = (ip[0], newpos, ip[2], W_in)
    # hidden surface removal: drop triangles of the body / inner layers that lie completely under an outer layer
    removed = 0
    for ii in (inner_all if STEPS['hsr'] else []):
        ip = body[ii]
        covered = np.zeros(len(ip[1]), bool)
        for oi in lay:
            if oi == ii or (ii in lay and lay.index(ii) >= lay.index(oi)):
                continue
            op = body[oi]
            same, _, _ = same_group(ip, op, ip[1], op[1])
            straight = under_surface(ip[1], op[1], op[0].faces, max_dist=LAYER_GAP, border=borders[oi], border_keep=BORDER_KEEP,
                                     pts_nrm=ip[2])
            if ip[0].kind == 'body':   # bare body: also everything behind clothing of the same body part
                straight |= under_surface(ip[1], op[1], op[0].faces, max_dist=LAYER_GAP, straight_only=False,
                                          border=borders[oi], border_keep=BORDER_KEEP, pts_nrm=ip[2]) & same
            covered |= straight
        F = ip[0].faces
        keep = ~covered[F].all(1)
        if not keep.all():
            removed += int((~keep).sum())
            ip[0].faces = F[keep]
    log('   hidden triangles removed: %d' % removed)
    ds_names = {p['path'].split('.')[0] for p in rig['parts'] if p.get('double_sided')}
    caps = {'body': 600, 'layered': 1024, 'accessory': 512, 'head': 1024, 'missing': 8}

    # ---- body
    texs, keys = texture_list([b[0] for b in body], caps)
    atlas, place = pack(texs, atlas_body)
    P, N, UV, T, Wl, OCC = [], [], [], [], [], []
    base = 0
    for (p, pos, nrm, W), key in zip(body, keys):
        uv = remap_uv(p.uv, place[key], atlas_body)
        faces = p.faces
        if p.name in ds_names:
            pos, nrm, faces, uv, W = double_sided(p, pos, nrm, faces, uv, W)
        P.append(pos)
        N.append(nrm)
        UV.append(uv)
        Wl += W
        T.append(faces + base)
        OCC.append(np.full(len(pos), p.kind in ('body', 'layered')))
        base += len(pos)
    P, N, UV, T = np.concatenate(P), np.concatenate(N), np.concatenate(UV), np.concatenate(T)
    OCC = np.concatenate(OCC)
    pal = {b: i for i, b in enumerate(palette)}
    I = np.zeros((len(P), 4), np.int32)
    Wt = np.zeros((len(P), 4), np.float64)
    for v, wd in enumerate(Wl):
        items = sorted(((b, w) for b, w in wd.items() if b in pal and w > 1e-4), key=lambda t: -t[1])[:4]
        if not items:
            items = [('Spine_2', 1.0)]
        tot = sum(w for _, w in items)
        q = [int(round(255 * w / tot)) for _, w in items]
        q[0] += 255 - sum(q)
        for s, ((b, _), qq) in enumerate(zip(items, q)):
            I[v, s] = pal[b]
            Wt[v, s] = qq / 255.0
    # standing pose -> bind (rest) pose with exactly the weights the game uses
    P_ref = P
    P, N = miimap.unskin(P_ref, N, I, Wt, palette, rest, ref)
    spheres = miimap.skin_spheres(P, I, Wt, palette, rest, skin_names)
    attrs = {'_p0': P, '_n0': np.concatenate([N, np.ones((len(N), 1))], 1), '_t0': tangents(N), '_u0': UV,
             '_i0': I, '_w0': Wt}
    body_bfres = bfres_write.replace_geometry(mascot, attrs, T, skin_spheres=spheres)
    # ---- head
    texs, keys = texture_list([h[0] for h in head], caps)
    hatlas, hplace = pack(texs, atlas_head)
    P2, N2, UV2, T2, HV = [], [], [], [], []
    base = 0
    for (p, pos, nrm), key in zip(head, keys):
        uv = remap_uv(p.uv, hplace[key], atlas_head)
        faces = p.faces
        if p.name in ds_names:
            pos, nrm, faces, uv, _ = double_sided(p, pos, nrm, faces, uv)
        P2.append(pos)
        N2.append(nrm)
        UV2.append(uv)
        T2.append(faces + base)
        HV.append(np.full(len(pos), p.kind == 'head'))
        base += len(pos)
    P2, N2, UV2, T2 = np.concatenate(P2), np.concatenate(N2), np.concatenate(UV2), np.concatenate(T2)
    HV = np.concatenate(HV)
    # the game bends neck and upper spine in everyday poses: keep a long Roblox jaw in front of the chest
    head_shift, chin = 0.0, {}
    if poses is not None:
        deltas, pivot_rest = poses
        head_shift, chin = chinfit.forward_shift(P, I, Wt, palette, T, OCC, P2, N2, HV, deltas, pivot_rest, log=log)
        P2 = P2 + np.array([0.0, 0.0, head_shift / miimap.MASK_SCALE])
    lo, hi = P2.min(0), P2.max(0)
    attrs2 = {'_p0': P2, '_n0': np.concatenate([N2, np.ones((len(N2), 1))], 1), '_t0': tangents(N2), '_u0': UV2}
    head_bfres = bfres_write.replace_geometry(mask_tpl, attrs2, T2, bounds=((lo + hi) / 2, (hi - lo) / 2))
    flips = []
    for (p, pos, nrm, W) in body:
        fr = outward_fraction(pos, p.faces)
        if fr < 0.35:
            flips.append((p.name, round(fr, 2)))
    for (p, pos, nrm) in head:
        fr = outward_fraction(pos, p.faces)
        if fr < 0.35:
            flips.append((p.name, round(fr, 2)))
    stats = {'inside_out': flips, 'body_verts': len(P), 'body_tris': len(T), 'head_verts': len(P2), 'head_tris': len(T2),
             'scale': info['s'], 'joint_offset': info['joint_offset'], 'bind_pose': info['bind_pose'],
             'arm_mode': info['arm_mode'], 'head_bbox': [lo.tolist(), hi.tolist()], 'body_bbox': [P_ref.min(0).tolist(), P_ref.max(0).tolist()],
             'head_forward_shift': round(head_shift, 4), 'chin_check': chin,
             'missing': [p.name for p in pieces if p.kind == 'missing']}
    arrays = {'P': P, 'Pref': P_ref, 'UV': UV, 'T': T, 'I': I, 'W': Wt, 'P2': P2, 'UV2': UV2, 'T2': T2}
    return body_bfres, atlas, head_bfres, hatlas, stats, arrays


class _Mesh:
    pass


def render_icons(arrays, atlas, hatlas, mascot, size=300):
    """300x300 RGBA pictures like the game's cloth icons (front view, transparent background)"""
    import rrender
    B = bfres.Bfres(mascot)
    b = _Mesh()
    b.pos, b.uv, b.faces, b.tex = arrays['P'], arrays['UV'], arrays['T'], atlas
    b.idx, b.w, b.skel = arrays['I'], arrays['W'], B.models[0]['skeleton']
    if 'Pref' in arrays:
        b.pos = arrays['Pref']   # standing pose = exact Roblox look
    else:
        import preview_game_model as G
        b.pos = G.pose_arms_down(b, 72)
    h = _Mesh()
    h.pos, h.uv, h.faces, h.tex = arrays['P2'], arrays['UV2'], arrays['T2'], hatlas
    out = []
    for parts, yaw, pitch in (([b], 180 + 12, -8), ([h], 180 + 22, -10)):
        im = rrender.render(parts, yaw=yaw, pitch=pitch, size=size * 2, alpha=True, light=(-0.3, 0.55, 0.78), ambient=0.55)
        a = np.asarray(im)
        ys, xs = np.nonzero(a[..., 3])
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        crop = im.crop(box)
        scale = 0.86 * size / max(crop.size)
        crop = crop.resize((max(1, int(crop.size[0] * scale)), max(1, int(crop.size[1] * scale))), Image.LANCZOS)
        canvas = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        canvas.alpha_composite(crop, ((size - crop.size[0]) // 2, (size - crop.size[1]) // 2))
        out.append(np.asarray(canvas))
    return out


def mic_value(t):
    """mean colour of a material (Mic) texture -> constant 8x8 picture"""
    im = np.asarray(t['image'].convert('RGBA')).reshape(-1, 4).mean(0)
    return np.tile(np.round(im).astype(np.uint8), (8, 8, 1))


def mic_constant(tex_name):
    return mic_value(load_tex(tex_name))


class GameCtx:
    """templates and numbers from the extracted game (game/romfs)"""

    def __init__(self, game=GAME):
        self.game = game
        self.mascot = tlfmt.read(os.path.join(game, 'Model', 'ClothAllMascotLong.bfres.zs'))
        _, self.palette = miimap.mii_rig(self.mascot)
        anim_bytes = tlfmt.read(os.path.join(game, 'Model', 'AnimationProduct.bfres.zs'))
        self.rest, self.ref = miimap.reference_pose(self.mascot, anim_bytes)
        mb = bfres.Bfres(self.mascot).models[0]
        self.skel = mb['skeleton']
        self.skin_names = [self.skel['bones'][i]['name'] for i in mb['shapes'][0]['skin_bones']]
        self.deltas, self.mask_pivot_rest = chinfit.pose_deltas(self.skel, fska.Anims(anim_bytes))
        cloth = tlfmt.byml(os.path.join(game, 'RSDB', 'Cloth.Product.100.rstbl.byml.zs'))
        self.model_hash = {c['__RowId']: c['ModelHash'] for c in cloth}
        self.cloth_index = {c['__RowId']: c['ClothIndex'] for c in cloth}
        self.cmp_root, self.cmp_ver = byml_write.parse(tlfmt.read(os.path.join(game, 'RSDB', 'ClothModelParam.Product.100.rstbl.byml.zs')))
        self.rows = {r['__RowId'][1]: r for r in self.cmp_root}

    def row(self, item):
        return self.rows[self.model_hash[item]]

    def ncol(self, item):
        return max(1, len(self.row(item).get('ColorVariation') or []))

    def mask_template(self, mask):
        return tlfmt.read(os.path.join(self.game, 'Model', mask + '.bfres.zs'))

    def mic_body(self, body_item):
        return mic_constant(self.row(body_item)['ModelUnit']['TextureProjectName'][1] + '_Body_Mic')

    def mic_mask(self, mask):
        return mic_constant(mask + '_Body_Mic')

    def has_emm(self, mask):
        return os.path.exists(os.path.join(self.game, 'Tex', mask + '_Body_Emm.bntx.zs'))


def make_character_files(name, rigs, tag, body_item, mask, ctx, atlas_sizes=(2048, 2048), log=print):
    """build one Roblox character as game files -> (files {romfs path: bytes}, stats, extra pictures)"""
    t0 = time.time()
    body_bfres, atlas, head_bfres, hatlas, stats, arrays = build_character(
        name, rigs, ctx.mascot, ctx.mask_template(mask), ctx.palette, ctx.rest, ctx.ref, ctx.skin_names,
        atlas_body=atlas_sizes[0], atlas_head=atlas_sizes[1], poses=(ctx.deltas, ctx.mask_pivot_rest), log=log)
    model = 'ClothAllRoblox' + tag
    texproj = model + 'Tex'
    files = {}
    files['Model/%s.bfres.zs' % model] = zs(bfres_write.rename_model(body_bfres, model))
    files['Model/%s.bfres.zs' % mask] = zs(head_bfres)
    ncol_body, ncol_mask = ctx.ncol(body_item), ctx.ncol(mask)
    mic_body, mic_mask = ctx.mic_body(body_item), ctx.mic_mask(mask)
    # textures: every colour slot of the item gets the same picture
    for c in range(ncol_body):
        nm = '%s_Body_Alb.%02d' % (texproj, c)
        files['Tex/%s.bntx.zs' % nm] = zs(bntx_write.build(nm, atlas, 'bc1_srgb', swz=(2, 3, 4, 1)))
    flat_nrm = np.zeros((64, 64, 4), np.uint8)
    files['Tex/%s_Body_Nrm.bntx.zs' % texproj] = zs(bntx_write.build(texproj + '_Body_Nrm', flat_nrm, 'bc5_snorm_flat'))
    files['Tex/%s_Body_Mic.bntx.zs' % texproj] = zs(bntx_write.build(texproj + '_Body_Mic', mic_body, 'rgba8', swz=(0, 0, 4, 1)))
    for c in range(ncol_mask):
        nm = '%s_Body_Alb.%02d' % (mask, c)
        files['Tex/%s.bntx.zs' % nm] = zs(bntx_write.build(nm, hatlas, 'bc1_srgb', swz=(2, 3, 4, 1)))
    # a template's glow map (e.g. the skull mask's _Emm) would light up the new head with the new UVs: make it black
    black = np.zeros((64, 64, 4), np.uint8)
    black[..., 3] = 255
    mask_emm = ctx.has_emm(mask)
    if mask_emm:
        files['Tex/%s_Body_Emm.bntx.zs' % mask] = zs(bntx_write.build(mask + '_Body_Emm', black, 'bc1_srgb'))
    files['Tex/%s_Body_Nrm.bntx.zs' % mask] = zs(bntx_write.build(mask + '_Body_Nrm', flat_nrm, 'bc5_snorm_flat'))
    files['Tex/%s_Body_Mic.bntx.zs' % mask] = zs(bntx_write.build(mask + '_Body_Mic', mic_mask, 'rgba8', swz=(0, 0, 4, 1)))
    # low resolution copies (Tex/Lo): the game shows these first / at a distance and only knows textures that have them
    for proj, at, ncol, mic, emm in ((texproj, atlas, ncol_body, mic_body, False), (mask, hatlas, ncol_mask, mic_mask, mask_emm)):
        small = np.asarray(Image.fromarray(at).resize((max(64, at.shape[1] // 4), max(64, at.shape[0] // 4)), Image.BOX))
        for c in range(ncol):
            nm = '%s_Body_Alb.%02d' % (proj, c)
            files['Tex/Lo/%s.%02d.bntx.zs' % (proj, c)] = zs(bntx_write.build_multi('%s.%02d' % (proj, c), [(nm, small, 'bc1_srgb', (2, 3, 4, 1))]))
        lo = [(proj + '_Body_Emm', black, 'bc1_srgb', None)] if emm else []
        lo += [(proj + '_Body_Mic', mic, 'rgba8', (0, 0, 4, 1)), (proj + '_Body_Nrm', flat_nrm, 'bc5_snorm_flat', None)]
        files['Tex/Lo/%s.bntx.zs' % proj] = zs(bntx_write.build_multi(proj, lo))
    # wardrobe / shop pictures (Icon/<item>_<colour>.bntx): the Roblox body resp. head
    ib, ih = render_icons(arrays, atlas, hatlas, ctx.mascot)
    for item, img, ncol in ((body_item, ib, ncol_body), (mask, ih, ncol_mask)):
        for c in range(ncol):
            nm = '%s_%02d' % (item, c)
            files['Icon/%s.bntx.zs' % nm] = zs(bntx_write.build(nm, img, 'bc3_srgb'))
    stats.update(model=model, texture_project=texproj, body_item=body_item, mask=mask, seconds=round(time.time() - t0, 1),
                 colors_body=ncol_body, colors_mask=ncol_mask)
    extra = {'icon_body': ib, 'icon_head': ih, 'atlas_body': atlas, 'atlas_head': hatlas, 'arrays': arrays}
    return files, stats, extra


def redirect_body_row(row, model):
    """table: the costume row shows the new model / textures, without its tail etc."""
    mu = row['ModelUnit']
    mu['ModelName'] = ('S', model)
    mu['ModelResName'] = ('S', model)
    mu['TextureProjectName'] = ('S', model + 'Tex')
    row['SubAccessoryHash0'] = ('u', 0)
    row['SubAccessoryHash1'] = ('u', 0)


def update_restbl(R, files):
    """resource size table: every changed or new file needs an entry at least as big as its memory use"""
    changed = []
    for rel, data in sorted(files.items()):
        if not rel.endswith('.zs'):
            continue
        key = rel[:-3]
        need = restbl.needed(key, len(tlfmt.unzs(data)))
        old = R.get(key)
        if old is None or old < need:
            R.set(key, need)
            changed.append((key, old, need))
    return changed


def write_restbl(out_romfs):
    rt_rel = 'System/Resource/ResourceSizeTable.Product.100.Nin_NX_NVN.rsizetable'
    R = restbl.Restbl(tlfmt.read(os.path.join(GAME, rt_rel + '.zs')))
    files = {}
    for sub in ('Model', 'Tex', 'Tex/Lo', 'Icon', 'RSDB'):
        if not os.path.isdir(os.path.join(out_romfs, sub)):
            continue
        for f in sorted(os.listdir(os.path.join(out_romfs, sub))):
            if f.endswith('.zs'):
                files[sub + '/' + f] = open(os.path.join(out_romfs, sub, f), 'rb').read()
    changed = update_restbl(R, files)
    os.makedirs(os.path.dirname(os.path.join(out_romfs, rt_rel)), exist_ok=True)
    open(os.path.join(out_romfs, rt_rel + '.zs'), 'wb').write(zs(R.write()))
    print('size table: %d entries changed/added' % len(changed))
    return changed

