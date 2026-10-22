"""Roblox mesh reader (FileMesh versions 1.00, 1.01, 2.00, 3.00, 3.01, 4.00, 4.01, 5.00, 6.00, 7.00).
parse(bytes) -> dict(pos (N,3) f32, nrm (N,3), uv (N,2) [v down = image row], col (N,4) u8,
                     faces (M,3) int  [highest level of detail only],
                     bones [{name, parent, R (3x3), t (3,)}], skin_idx (N,4) global bone index, skin_w (N,4) float)"""
import struct
import numpy as np


def _v1(data):
    text = data.decode('ascii', 'replace')
    lines = text.splitlines()
    ver = lines[0].strip()
    nfaces = int(lines[1].strip())
    nums = []
    import re
    for m in re.finditer(r'\[([^\]]*)\]', '\n'.join(lines[2:])):
        nums.append([float(x) for x in m.group(1).split(',')])
    vec = np.array(nums, np.float32).reshape(nfaces, 3, 3, 3)  # face, corner, (pos, nrm, uv)
    pos = vec[:, :, 0].reshape(-1, 3)
    if ver.endswith('1.00'):
        pos = pos * 0.5
    nrm = vec[:, :, 1].reshape(-1, 3)
    uv = vec[:, :, 2, :2].reshape(-1, 2).copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    n = pos.shape[0]
    return {'pos': pos, 'nrm': nrm, 'uv': uv, 'col': np.full((n, 4), 255, np.uint8),
            'faces': np.arange(n).reshape(-1, 3), 'bones': [], 'skin_idx': None, 'skin_w': None}


def _vert_block(b, off, n, size, v2=False):
    raw = np.frombuffer(b, np.uint8, n * size, off).reshape(n, size)
    pos = raw[:, 0:12].copy().view('<f4').reshape(n, 3)
    nrm = raw[:, 12:24].copy().view('<f4').reshape(n, 3)
    uv = raw[:, 24:32].copy().view('<f4').reshape(n, 2)  # v2: u, v, w (w ignored); v3+: u, v, then tangent
    col = raw[:, 36:40].copy() if size >= 40 else np.full((n, 4), 255, np.uint8)
    return pos, nrm, uv, col


def _bones(b, off, nb):
    bones = []
    for i in range(nb):
        name_off, parent, lod_parent, cull = struct.unpack_from('<IHHf', b, off + i * 60)
        rot = struct.unpack_from('<9f', b, off + i * 60 + 12)
        t = struct.unpack_from('<3f', b, off + i * 60 + 48)
        bones.append({'name_off': name_off, 'parent': -1 if parent == 0xFFFF else parent,
                      'R': np.array(rot, np.float32).reshape(3, 3), 't': np.array(t, np.float32)})
    return bones


def _subset_remap(skin_raw, subsets, n):
    """envelope bone ids are subset-local -> convert to global bone indices"""
    idx = np.zeros((n, 4), np.int32)
    w = skin_raw[:, 4:8].astype(np.float32) / 255.0
    loc = skin_raw[:, 0:4].astype(np.int32)
    for s in subsets:
        vb, vl, table = s['verts_begin'], s['verts_length'], np.array(s['bones'], np.int32)
        sl = slice(vb, vb + vl)
        idx[sl] = table[np.clip(loc[sl], 0, len(table) - 1)]
    return idx, w


def _subsets(b, off, ns):
    out = []
    for i in range(ns):
        fb, fl, vb, vl, nbi = struct.unpack_from('<5I', b, off + i * 72)
        bones = struct.unpack_from('<26H', b, off + i * 72 + 20)[:nbi]
        out.append({'faces_begin': fb, 'faces_length': fl, 'verts_begin': vb, 'verts_length': vl, 'bones': list(bones)})
    return out


def _names(b, off, size, bones):
    blob = b[off:off + size]
    for bo in bones:
        e = blob.find(b'\0', bo['name_off'])
        bo['name'] = blob[bo['name_off']:e].decode('utf-8', 'replace')
        del bo['name_off']


def _lod0(faces, lods):
    if lods is not None and len(lods) >= 2 and lods[1] > lods[0]:
        return faces[lods[0]:lods[1]]
    return faces


def parse(data):
    nl = data.index(b'\n')
    ver = data[8:nl].decode().strip()
    if ver in ('1.00', '1.01'):
        return _v1(data)
    b = data
    o = nl + 1
    if ver == '2.00':
        hs, vs, fs, nv, nf = struct.unpack_from('<HBBII', b, o)
        o += hs
        pos, nrm, uv, col = _vert_block(b, o, nv, vs, v2=True)
        o += nv * vs
        faces = np.frombuffer(b, '<u4', nf * 3, o).reshape(nf, 3).astype(np.int64)
        return {'pos': pos, 'nrm': nrm, 'uv': uv, 'col': col, 'faces': faces, 'bones': [], 'skin_idx': None, 'skin_w': None}
    if ver in ('3.00', '3.01'):
        hs, vs, fs, ls, nl_, nv, nf = struct.unpack_from('<HBBHHII', b, o)
        o += hs
        pos, nrm, uv, col = _vert_block(b, o, nv, vs)
        o += nv * vs
        faces = np.frombuffer(b, '<u4', nf * 3, o).reshape(nf, 3).astype(np.int64)
        o += nf * fs
        lods = np.frombuffer(b, '<u4', nl_, o) if nl_ else None
        return {'pos': pos, 'nrm': nrm, 'uv': uv, 'col': col, 'faces': _lod0(faces, lods), 'bones': [], 'skin_idx': None, 'skin_w': None}
    if ver in ('4.00', '4.01', '5.00'):
        hs = struct.unpack_from('<H', b, o)[0]
        _, lod_type, nv, nf, nlods, nb, names_size, nsub, nhq, _u = struct.unpack_from('<HHIIHHIHBB', b, o)
        o += hs
        pos, nrm, uv, col = _vert_block(b, o, nv, 40)
        o += nv * 40
        skin_raw = None
        if nb:
            skin_raw = np.frombuffer(b, np.uint8, nv * 8, o).reshape(nv, 8)
            o += nv * 8
        faces = np.frombuffer(b, '<u4', nf * 3, o).reshape(nf, 3).astype(np.int64)
        o += nf * 12
        lods = np.frombuffer(b, '<u4', nlods, o) if nlods else None
        o += nlods * 4
        bones = _bones(b, o, nb)
        o += nb * 60
        _names(b, o, names_size, bones)
        o += names_size
        subs = _subsets(b, o, nsub)
        o += nsub * 72
        res = {'pos': pos, 'nrm': nrm, 'uv': uv, 'col': col, 'faces': _lod0(faces, lods), 'bones': bones,
               'skin_idx': None, 'skin_w': None}
        if nb and skin_raw is not None:
            res['skin_idx'], res['skin_w'] = _subset_remap(skin_raw, subs, nv)
        if ver == '5.00' and hs >= 32:
            # 5.00: the face rig (FACS) follows the subsets; same layout as the FACS chunk of 6.00+, minus its size field
            facs_fmt, facs_size = struct.unpack_from('<II', b, nl + 1 + 24)
            if facs_size and o + facs_size <= len(b):
                try:
                    res['facs'] = _facs(struct.pack('<I', facs_size) + b[o:o + facs_size])
                except Exception:
                    pass
        return res
    if ver in ('6.00', '7.00'):
        chunks = {}
        while o + 16 <= len(b):
            name = b[o:o + 8].rstrip(b'\0').decode('ascii', 'replace')
            cver, size = struct.unpack_from('<II', b, o + 8)
            chunks[name] = (cver, b[o + 16:o + 16 + size])
            o += 16 + size
        res = {'bones': [], 'skin_idx': None, 'skin_w': None}
        cver, c = chunks['COREMESH']
        if cver == 1:
            nv = struct.unpack_from('<I', c, 0)[0]
            pos, nrm, uv, col = _vert_block(c, 4, nv, 40)
            p = 4 + nv * 40
            nf = struct.unpack_from('<I', c, p)[0]
            faces = np.frombuffer(c, '<u4', nf * 3, p + 4).reshape(nf, 3).astype(np.int64)
        else:
            pos, nrm, uv, col, faces = _draco_core(c)
            nv = len(pos)
        res.update(pos=pos, nrm=nrm, uv=uv, col=col, faces=faces)
        if 'LODS' in chunks:
            lv, lc = chunks['LODS']
            lod_type, nhq, nlo = struct.unpack_from('<HBI', lc, 0)
            lods = np.frombuffer(lc, '<u4', nlo, 7)
            res['faces'] = _lod0(faces, lods)
        if 'SKINNING' in chunks:
            sv, sc = chunks['SKINNING']
            p = 0
            ns = struct.unpack_from('<I', sc, p)[0]; p += 4
            skin_raw = np.frombuffer(sc, np.uint8, ns * 8, p).reshape(ns, 8); p += ns * 8
            nb = struct.unpack_from('<I', sc, p)[0]; p += 4
            bones = _bones(sc, p, nb); p += nb * 60
            nsz = struct.unpack_from('<I', sc, p)[0]; p += 4
            _names(sc, p, nsz, bones); p += nsz
            nsub = struct.unpack_from('<I', sc, p)[0]; p += 4
            subs = _subsets(sc, p, nsub)
            res['bones'] = bones
            if ns == nv:
                res['skin_idx'], res['skin_w'] = _subset_remap(skin_raw, subs, nv)
        if 'FACS' in chunks:
            res['facs'] = _facs(chunks['FACS'][1])
        return res
    raise ValueError('unsupported mesh version ' + ver)


def _facs(c):
    """FACS chunk (dynamic heads): face bone names, control names and the bone pose of every control at full weight
    (translation px/py/pz in mesh units, rotation rx/ry/rz in degrees), columns = controls + 2- and 3-pose correctives"""
    _total, nbn, ncn, nqt, n2, n3 = struct.unpack_from('<IIIQII', c, 0)
    p = 28
    bones = [x.decode('utf-8', 'replace') for x in c[p:p + nbn].split(b'\0')[:-1]]
    p += nbn
    ctrls = [x.decode('utf-8', 'replace') for x in c[p:p + ncn].split(b'\0')[:-1]]
    p += ncn
    mats = []
    for _ in range(6):
        ver, rows, cols = struct.unpack_from('<HII', c, p)
        p += 10
        if ver == 1:
            v = np.frombuffer(c, '<f4', rows * cols, p).astype(np.float64)
            p += rows * cols * 4
        else:
            mn, mx = struct.unpack_from('<ff', c, p)
            p += 8
            q = np.frombuffer(c, '<u2', rows * cols, p).astype(np.float64)
            p += rows * cols * 2
            v = mn + (mx - mn) * q / 65535.0
        mats.append(v.reshape(rows, cols))
    two = np.frombuffer(c, '<u2', n2 // 2, p).reshape(-1, 2).astype(int)
    p += n2
    three = np.frombuffer(c, '<u2', n3 // 2, p).reshape(-1, 3).astype(int)
    return {'bones': bones, 'controls': ctrls, 'p': np.stack(mats[:3], -1), 'r': np.stack(mats[3:], -1),
            'two': two, 'three': three}


def _euler_xyz(r):
    rx, ry, rz = np.radians(r)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def facs_pose(mesh, weights):
    """deform a dynamic head with FACS control weights {control name: 0..1} (e.g. a smile) -> (pos, nrm)"""
    F = mesh.get('facs')
    if not F or mesh.get('skin_idx') is None or not weights:
        return mesh['pos'], mesh['nrm']
    nc = len(F['controls'])
    w = np.zeros(F['p'].shape[1])
    for i, n in enumerate(F['controls']):
        w[i] = float(weights.get(n, 0.0))
    for j, (a, b) in enumerate(F['two']):
        w[nc + j] = w[a] * w[b]
    for j, (a, b, c) in enumerate(F['three']):
        w[nc + len(F['two']) + j] = w[a] * w[b] * w[c]
    dpos = np.einsum('bck,c->bk', F['p'], w)
    drot = np.einsum('bck,c->bk', F['r'], w)
    bones = mesh['bones']
    names = [b['name'] for b in bones]
    rest = []
    for b in bones:
        M = np.eye(4)
        M[:3, :3] = np.asarray(b['R'], np.float64)
        M[:3, 3] = b['t']
        rest.append(M)
    delta = {}
    for k, bn in enumerate(F['bones']):
        if bn in names:
            D = np.eye(4)
            D[:3, :3] = _euler_xyz(drot[k])
            D[:3, 3] = dpos[k]
            delta[names.index(bn)] = D
    posed = [None] * len(bones)

    def world(i):
        if posed[i] is None:
            par = bones[i]['parent']
            local = (np.linalg.inv(rest[par]) @ rest[i]) if par >= 0 else rest[i]
            if i in delta:
                local = local @ delta[i]
            posed[i] = (world(par) @ local) if par >= 0 else local
        return posed[i]

    skin = np.stack([world(i) @ np.linalg.inv(rest[i]) for i in range(len(bones))])
    idx, wt = mesh['skin_idx'], mesh['skin_w']
    M = np.einsum('vk,vkij->vij', wt, skin[idx])
    tot = wt.sum(1)
    M[tot < 1e-6] = np.eye(4)
    hp = np.concatenate([mesh['pos'].astype(np.float64), np.ones((len(mesh['pos']), 1))], 1)
    pos = np.einsum('vij,vj->vi', M, hp)[:, :3]
    nrm = np.einsum('vij,vj->vi', M[:, :3, :3], mesh['nrm'].astype(np.float64))
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    return pos.astype(np.float32), nrm.astype(np.float32)


def _draco_core(c):
    """COREMESH version 2: Draco compressed geometry (needs the DracoPy package)."""
    import DracoPy
    p = c.find(b'DRACO')
    m = DracoPy.decode(c[p:])
    pos = np.asarray(m.points, np.float32).reshape(-1, 3)
    faces = np.asarray(m.faces, np.int64).reshape(-1, 3)
    n = len(pos)
    nrm = None
    if getattr(m, 'normals', None) is not None and len(m.normals):
        nrm = np.asarray(m.normals, np.float32).reshape(-1, 3)
    else:
        # Roblox stores the normals as a GENERIC attribute (3 x float32), not as Draco's NORMAL attribute
        for a in (m.attributes or []):
            if not isinstance(a, dict) or a.get('attribute_type') != 4 or a.get('num_components') != 3 or a.get('data_type') != 9:
                continue
            cand = np.asarray(a['data'], np.float32).reshape(-1, 3)
            if len(cand) == n and np.median(np.abs(np.linalg.norm(cand, axis=1) - 1)) < 0.05:
                nrm = cand
                break
    if nrm is None:
        nrm = vertex_normals(pos, faces)
    uv = np.asarray(m.tex_coord, np.float32).reshape(-1, 2) if getattr(m, 'tex_coord', None) is not None and len(m.tex_coord) else np.zeros((n, 2), np.float32)
    col = np.full((n, 4), 255, np.uint8)
    if getattr(m, 'colors', None) is not None and len(m.colors):
        cc = np.asarray(m.colors).reshape(n, -1)
        col[:, :cc.shape[1]] = cc[:, :4]
    return pos, nrm, uv, col, faces


def vertex_normals(pos, faces):
    """area-weighted smooth normals (vertices at the same place share them)"""
    pos = np.asarray(pos, np.float64)
    A, B, C = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    fn = np.cross(B - A, C - A)
    key = np.round(pos / 1e-5).astype(np.int64)
    _, weld = np.unique(key, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    acc = np.zeros((weld.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, weld[faces[:, k]], fn)
    vn = acc[weld]
    return (vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def load(path):
    return parse(open(path, 'rb').read())
