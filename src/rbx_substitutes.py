"""Stand-ins for Roblox accessories whose meshes are private (assetdelivery answers 401):
built from simple shapes at the accessory's real place (handle CFrame / size from the place file)."""
import os
import numpy as np

import rbxmesh


def _torus(R, r, n_major=10, n_minor=6):
    pos, nrm = [], []
    for i in range(n_major):
        a = 2 * np.pi * i / n_major
        ca, sa = np.cos(a), np.sin(a)
        for j in range(n_minor):
            b = 2 * np.pi * j / n_minor
            cb, sb = np.cos(b), np.sin(b)
            pos.append(((R + r * cb) * ca, r * sb, (R + r * cb) * sa))
            nrm.append((cb * ca, sb, cb * sa))
    faces = []
    for i in range(n_major):
        for j in range(n_minor):
            a = i * n_minor + j
            b = ((i + 1) % n_major) * n_minor + j
            c = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            d = i * n_minor + (j + 1) % n_minor
            faces += [(a, d, c), (a, c, b)]
    return np.array(pos), np.array(nrm), np.array(faces)


def _box(lo, hi):
    """axis aligned box, outward CCW faces, per-face normals (24 vertices)"""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    P, N, F = [], [], []
    for axis in range(3):
        for sign in (-1, 1):
            n = np.zeros(3)
            n[axis] = sign
            u, v = [(1, 2), (2, 0), (0, 1)][axis]
            base = len(P)
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = np.where(n > 0, hi, np.where(n < 0, lo, 0.0)).astype(float)
                p[axis] = hi[axis] if sign > 0 else lo[axis]
                p[u] = hi[u] if du else lo[u]
                p[v] = hi[v] if dv else lo[v]
                P.append(p)
                N.append(n)
            # orient CCW around the outward normal
            a, b, c = P[base], P[base + 1], P[base + 2]
            if np.dot(np.cross(b - a, c - a), n) > 0:
                F += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
            else:
                F += [(base, base + 2, base + 1), (base, base + 3, base + 2)]
    return np.array(P), np.array(N), np.array(F)


def _cf(cf):
    M = np.eye(4)
    M[:3, :3] = np.array(cf['r']).reshape(3, 3)
    M[:3, 3] = cf['p']
    return M


def chain(handle, torso):
    """silver chain: rests on the shoulders around the neck and hangs down over the chest"""
    hc = np.array(handle['cf']['p'])
    tc = np.array(torso['cf']['p'])
    tsz = np.array(torso['size'])
    top = tc[1] + tsz[1] / 2
    front = tc[2] - tsz[2] / 2 - 0.06   # Roblox front is -Z
    pts = []
    for t in np.linspace(0, 2 * np.pi, 400, endpoint=False):
        s, c = np.sin(t), np.cos(t)
        x = tc[0] + 0.55 * s
        z = tc[2] + 0.62 * c
        y = top + 0.04
        if c < 0:  # front half: drop down the chest
            w = (-c) ** 1.6
            y = top + 0.04 - 0.85 * w
            z = tc[2] + (front - tc[2]) * (-c) ** 0.35
        pts.append((x, y, z))
    pts = np.array(pts)
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    step = 0.16
    P, N, F = [], [], []
    k = 0
    for d in np.arange(0, L[-1], step):
        i = np.searchsorted(L, d, side='right') - 1
        i = min(i, len(pts) - 1)
        p = pts[i]
        tang = pts[(i + 1) % len(pts)] - pts[i]
        tang /= np.linalg.norm(tang)
        tp, tn, tf = _torus(0.1, 0.032)
        # link plane contains the tangent; alternate links turn 90 degrees
        up = np.array([0, 1.0, 0]) if k % 2 == 0 else np.cross(tang, [0, 1.0, 0])
        if np.linalg.norm(up) < 1e-6:
            up = np.array([1.0, 0, 0])
        side = np.cross(tang, up)
        side /= np.linalg.norm(side)
        up = np.cross(side, tang)
        Rm = np.stack([tang, up, side], 1)  # torus x->tang, y->up (ring normal), z->side
        P.append(tp @ Rm.T + p)
        N.append(tn @ Rm.T)
        F.append(tf + sum(len(q) for q in P[:-1]))
        k += 1
    pos, nrm, faces = np.concatenate(P), np.concatenate(N), np.concatenate(F)
    return [('ChinaChain', pos, nrm, faces, (205, 205, 212), 'UpperTorso')]


def backpack(handle, torso):
    tc = np.array(torso['cf']['p'])
    tsz = np.array(torso['size'])
    hc = np.array(handle['cf']['p'])
    back = tc[2] + tsz[2] / 2  # Roblox back is +Z
    cx, cy = hc[0], tc[1] - 0.05
    out = []
    body_lo, body_hi = (cx - 0.72, cy - 0.85, back), (cx + 0.72, cy + 0.75, back + 0.62)
    out.append(('Backpack', *_box(body_lo, body_hi), (112, 72, 42), 'UpperTorso'))
    out.append(('BackpackFlap', *_box((cx - 0.74, cy + 0.15, back + 0.3), (cx + 0.74, cy + 0.8, back + 0.67)), (84, 52, 30), 'UpperTorso'))
    for sx in (-1, 1):
        out.append(('BackpackPocket', *_box((cx + sx * 0.72 - (0.0 if sx > 0 else 0.22), cy - 0.75, back + 0.12),
                                            (cx + sx * 0.72 + (0.22 if sx > 0 else 0.0), cy - 0.1, back + 0.5)), (98, 62, 36), 'UpperTorso'))
        # shoulder straps: over the shoulder to the front
        x0 = tc[0] + sx * 0.55
        out.append(('BackpackStrapTop', *_box((x0 - 0.12, tc[1] + tsz[1] / 2, tc[2] - tsz[2] / 2 - 0.04),
                                              (x0 + 0.12, tc[1] + tsz[1] / 2 + 0.06, back + 0.05)), (70, 44, 26), 'UpperTorso'))
        out.append(('BackpackStrapFront', *_box((x0 - 0.12, tc[1] - 0.35, tc[2] - tsz[2] / 2 - 0.05),
                                                (x0 + 0.12, tc[1] + tsz[1] / 2 + 0.06, tc[2] - tsz[2] / 2)), (70, 44, 26), 'UpperTorso'))
    for sx in (-0.4, 0.4):
        out.append(('BackpackBuckle', *_box((cx + sx - 0.08, cy + 0.2, back + 0.66), (cx + sx + 0.08, cy + 0.36, back + 0.7)),
                    (205, 180, 90), 'UpperTorso'))
    return out


def waist_camera(handle, net3d, mesh_id=5592637327):
    """the same kind of camcorder as the head camera, hanging at the hip"""
    m = rbxmesh.load(os.path.join(net3d, '%d.bin' % mesh_id))
    p = m['pos'].astype(float)
    lo, hi = p.min(0), p.max(0)
    p = p - (lo + hi) / 2
    s = 0.95 * max(handle['size']) / max(hi - lo)
    rot90 = np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])  # long axis X -> Z
    M = _cf(handle['cf'])
    R = M[:3, :3] @ rot90
    pos = (p * s) @ R.T + M[:3, 3]
    nrm = m['nrm'] @ R.T
    return [('WaistCamera', pos, nrm, m['faces'], None, 'LowerTorso', m['uv'], 5592637398)]


def substitutes(acc_name, handle, parts, net3d):
    if 'ChinaChain' in acc_name:
        return chain(handle, parts['UpperTorso'])
    if 'LeatherAdventureBackpack' in acc_name:
        return backpack(handle, parts['UpperTorso'])
    if acc_name == 'Accessory (MeshPartAccessory)':
        return waist_camera(handle, net3d)
    return []
