"""Minimal BFRES (Switch, version 10.2.2 as used by Tomodachi Life: Living the Dream) reader.
Reads models, skeletons, vertex buffers, shapes/meshes and material texture names, and can
decode vertex attributes into numpy arrays.  Writing is done by bfres_write.py (geometry swap)."""
import struct
import numpy as np

try:
    import zstandard
except ImportError:  # pragma: no cover
    zstandard = None


def load(path):
    data = open(path, 'rb').read()
    if path.endswith('.zs'):
        data = zstandard.ZstdDecompressor().decompress(data, max_output_size=1 << 28)
    return data


# attribute formats: (component count, numpy dtype, kind)
FMT = {
    0x1805: (3, '<f4', 'float'),
    0x1905: (4, '<f4', 'float'),
    0x1705: (2, '<f4', 'float'),
    0x1505: (4, '<f2', 'float'),
    0x1205: (2, '<f2', 'float'),
    0x1201: (2, '<u2', 'unorm'),
    0x1202: (2, '<i2', 'snorm'),
    0x1501: (4, '<u2', 'unorm'),
    0x0e02: (4, None, '1010102snorm'),
    0x0e01: (4, None, '1010102unorm'),
    0x0b01: (4, 'u1', 'unorm'),
    0x0b02: (4, 'i1', 'snorm'),
    0x0b03: (4, 'u1', 'uint'),
    0x0903: (2, 'u1', 'uint'),
    0x0901: (2, 'u1', 'unorm'),
    0x0203: (1, 'u1', 'uint'),
    0x0201: (1, 'u1', 'unorm'),
}


class Bfres:
    def __init__(self, data):
        self.d = bytearray(data)
        d = self.d
        assert d[:4] == b'FRES', d[:8]
        self.version = struct.unpack_from('<I', d, 8)[0]
        self.rlt_off = self.u32(0x18)
        self.file_size = self.u32(0x1C)
        self.name = self.str(self.u64(0x20))
        self.pool_off = self.u64(0xA8)
        bi = self.u64(0xB0)
        self.bufinfo_off = bi
        self.buf_size = self.u32(bi + 4)
        self.buf_data = self.u64(bi + 8)  # file offset of the GPU buffer data
        nmodels = self.u16(0xDC)
        marr = self.u64(0x28)
        self.models = [self.read_model(marr + i * 0x78) for i in range(nmodels)]

    # ---- primitive helpers
    def u8(self, o): return self.d[o]
    def u16(self, o): return struct.unpack_from('<H', self.d, o)[0]
    def s16(self, o): return struct.unpack_from('<h', self.d, o)[0]
    def u32(self, o): return struct.unpack_from('<I', self.d, o)[0]
    def u64(self, o): return struct.unpack_from('<Q', self.d, o)[0]
    def f32(self, o, n=1):
        v = struct.unpack_from('<%df' % n, self.d, o)
        return v if n > 1 else v[0]

    def str(self, o):
        if not o:
            return None
        n = self.u16(o)
        return bytes(self.d[o + 2:o + 2 + n]).decode('utf-8', 'replace')

    def dic(self, o):
        """ResDic -> list of names in node order (index 1..count)"""
        if not o:
            return []
        cnt = struct.unpack_from('<i', self.d, o + 4)[0]
        out = []
        for i in range(1, cnt + 1):
            n = o + 8 + i * 16
            out.append(self.str(self.u64(n + 8)))
        return out

    # ---- model
    def read_model(self, o):
        d = self
        m = {'off': o, 'name': d.str(d.u64(o + 8))}
        skl = d.u64(o + 0x18)
        vtx_arr = d.u64(o + 0x20)
        shp_arr = d.u64(o + 0x28)
        mat_arr = d.u64(o + 0x38)
        nvtx, nshp, nmat = d.u16(o + 0x68), d.u16(o + 0x6A), d.u16(o + 0x6C)
        m['skeleton'] = d.read_skeleton(skl)
        m['vbufs'] = [d.read_vbuf(vtx_arr + i * 0x58) for i in range(nvtx)]
        m['shapes'] = [d.read_shape(shp_arr + i * 0x60) for i in range(nshp)]
        m['materials'] = [d.read_material(mat_arr + i * 0xB0) for i in range(nmat)]
        return m

    def read_skeleton(self, o):
        d = self
        s = {'off': o, 'flags': d.u32(o + 4)}
        bone_arr = d.u64(o + 0x10)
        mtx_idx = d.u64(o + 0x18)
        inv = d.u64(o + 0x20)
        nb, nsmooth, nrigid = d.u16(o + 0x38), d.u16(o + 0x3A), d.u16(o + 0x3C)
        s['bone_arr'], s['mtx_idx_off'], s['inv_off'] = bone_arr, mtx_idx, inv
        s['n_smooth'], s['n_rigid'] = nsmooth, nrigid
        bones = []
        for i in range(nb):
            b = bone_arr + i * 0x58
            idx, parent, smooth, rigid, bill, nud = struct.unpack_from('<HHhhHH', d.d, b + 0x20)
            flags = d.u32(b + 0x2C)
            sc = d.f32(b + 0x30, 3)
            rot = d.f32(b + 0x3C, 4)
            tr = d.f32(b + 0x4C, 3)
            bones.append({'off': b, 'name': d.str(d.u64(b)), 'index': idx,
                          'parent': -1 if parent == 0xFFFF else parent,
                          'smooth': smooth, 'rigid': rigid, 'flags': flags,
                          'scale': sc, 'rot': rot, 'trans': tr})
        s['bones'] = bones
        nm = nsmooth + nrigid
        s['mtx_to_bone'] = list(struct.unpack_from('<%dH' % nm, d.d, mtx_idx)) if nm else []
        s['inv_mtx'] = [np.array(d.f32(inv + i * 48, 12)).reshape(3, 4) for i in range(nsmooth)] if inv else []
        return s

    def read_vbuf(self, o):
        d = self
        v = {'off': o}
        attr_arr = d.u64(o + 0x08)
        v['size_arr'] = d.u64(o + 0x30)
        v['stride_arr'] = d.u64(o + 0x38)
        v['buf_off'] = d.u32(o + 0x48)
        nattr, nbuf = d.u8(o + 0x4C), d.u8(o + 0x4D)
        v['count'] = d.u32(o + 0x50)
        v['skin_count'] = d.u8(o + 0x54)
        v['attrs'] = []
        for i in range(nattr):
            a = attr_arr + i * 16
            v['attrs'].append({'off': a, 'name': d.str(d.u64(a)), 'fmt': d.u32(a + 8),
                               'offset': d.u16(a + 12), 'buf': d.u16(a + 14)})
        v['buffers'] = []
        pos = v['buf_off']
        for i in range(nbuf):
            size = d.u32(v['size_arr'] + i * 16)
            stride = d.u32(v['stride_arr'] + i * 16)
            v['buffers'].append({'pool_off': pos, 'size': size, 'stride': stride})
            pos = (pos + size + 0xFF) & ~0xFF
        return v

    def read_shape(self, o):
        d = self
        s = {'off': o, 'name': d.str(d.u64(o + 8))}
        mesh_arr = d.u64(o + 0x18)
        skin_idx = d.u64(o + 0x20)
        s['bbox_off'] = d.u64(o + 0x38)
        s['radius_off'] = d.u64(o + 0x40)
        s['material'] = d.u16(o + 0x52)
        s['bone'] = d.u16(o + 0x54)
        s['vbuf'] = d.u16(o + 0x56)
        nskin, s['skin_count'], nmesh = d.u16(o + 0x58), d.u8(o + 0x5A), d.u8(o + 0x5B)
        s['skin_bones'] = list(struct.unpack_from('<%dH' % nskin, d.d, skin_idx)) if nskin else []
        s['meshes'] = []
        for i in range(nmesh):
            m = mesh_arr + i * 0x38
            sub = d.u64(m)
            nsub = d.u16(m + 0x34)
            s['meshes'].append({'off': m, 'pool_off': d.u32(m + 0x20), 'prim': d.u32(m + 0x24),
                                'ifmt': d.u32(m + 0x28), 'count': d.u32(m + 0x2C),
                                'first_vertex': d.u32(m + 0x30), 'size_info': d.u64(m + 0x18),
                                'submeshes': [struct.unpack_from('<II', d.d, sub + j * 8) for j in range(nsub)]})
        nb = sum(len(m['submeshes']) + 1 for m in s['meshes'])
        s['bboxes'] = [d.f32(s['bbox_off'] + i * 24, 6) for i in range(nb)] if s['bbox_off'] else []
        s['radius'] = [d.f32(s['radius_off'] + i * 16, 4) for i in range(max(1, nskin))] if s['radius_off'] else []
        return s

    def read_material(self, o):
        d = self
        m = {'off': o, 'name': d.str(d.u64(o + 8))}
        # texture name list + sampler dict (layout found empirically, see bfres notes)
        m['raw_ptrs'] = [d.u64(o + 0x10 + 8 * i) for i in range(20)]
        ntex = d.u8(o + 0xA3) if o + 0xA3 < len(d.d) else 0
        m['ntex_guess'] = ntex
        return m

    # ---- geometry access
    def attr(self, model, vbuf_i, name):
        v = model['vbufs'][vbuf_i]
        for a in v['attrs']:
            if a['name'] == name:
                return self.decode_attr(v, a)
        return None

    def decode_attr(self, v, a):
        b = v['buffers'][a['buf']]
        n = v['count']
        start = self.buf_data + b['pool_off'] + a['offset']
        comps, dt, kind = FMT[a['fmt']]
        stride = b['stride']
        raw = np.frombuffer(bytes(self.d[start:start + stride * (n - 1) + 16]), dtype='u1')
        rows = np.lib.stride_tricks.as_strided(raw, shape=(n, 16), strides=(stride, 1))
        if kind.startswith('1010102'):
            w = rows[:, :4].copy().view('<u4').reshape(n)
            out = np.zeros((n, 4), np.float32)
            for k in range(3):
                x = (w >> (10 * k)) & 0x3FF
                if kind == '1010102snorm':
                    x = np.where(x >= 512, x.astype(np.int32) - 1024, x)
                    out[:, k] = np.maximum(x / 511.0, -1.0)
                else:
                    out[:, k] = x / 1023.0
            wv = (w >> 30) & 3
            out[:, 3] = np.where(wv >= 2, wv.astype(np.int32) - 4, wv) if kind == '1010102snorm' else wv / 3.0
            return out
        size = np.dtype(dt).itemsize * comps
        arr = rows[:, :size].copy().view(dt).reshape(n, comps)
        if kind == 'unorm':
            return arr.astype(np.float32) / float(np.iinfo(dt).max)
        if kind == 'snorm':
            return np.maximum(arr.astype(np.float32) / float(np.iinfo(dt).max), -1.0)
        if kind == 'uint':
            return arr.astype(np.int32)
        return arr.astype(np.float32)

    def indices(self, mesh):
        start = self.buf_data + mesh['pool_off']
        dt = {0: 'u1', 1: '<u2', 2: '<u4'}[mesh['ifmt']]
        idx = np.frombuffer(bytes(self.d[start:start + mesh['count'] * np.dtype(dt).itemsize]), dtype=dt)
        return idx.astype(np.int64) + mesh['first_vertex']


def euler_to_mat(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_to_mat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def bone_world(skel):
    """rest-pose world matrices (4x4) of all bones"""
    W = []
    euler = bool(skel['flags'] & 0x1000)  # skeleton rotation mode: euler XYZ (else quaternion)
    for b in skel['bones']:
        sx, sy, sz = b['scale']
        rx, ry, rz, rw = b['rot']
        if euler:
            R = euler_to_mat(rx, ry, rz)
        else:
            R = quat_to_mat(rx, ry, rz, rw)
        L = np.eye(4)
        L[:3, :3] = R @ np.diag([sx, sy, sz])
        L[:3, 3] = b['trans']
        W.append(L if b['parent'] < 0 else W[b['parent']] @ L)
    return W
