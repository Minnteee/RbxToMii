"""Swap the geometry of a single-shape BFRES model (Tomodachi Life: Living the Dream, BFRES 10.2.2).
Keeps every structure of the template (skeleton, material, names) and rebuilds the GPU buffer pool:
all LOD meshes point at the new index buffer, vertex buffers keep the template's attribute formats.
replace_geometry(template_bytes, attrs={'_p0': (N,3), '_n0': (N,3|4), '_t0': (N,4), '_u0': (N,2), '_i0': (N,4), '_w0': (N,4)},
                 tris (M,3)) -> bytes"""
import struct
import numpy as np

import bfres


def _align(v, a):
    return (v + a - 1) // a * a


def _enc_1010102(v, w):
    v = np.clip(np.asarray(v, np.float64), -1, 1)
    q = np.round(v[:, :3] * 511).astype(np.int64) & 0x3FF
    ww = np.asarray(w, np.int64) & 3
    return (q[:, 0] | (q[:, 1] << 10) | (q[:, 2] << 20) | (ww << 30)).astype('<u4')


def encode_attr(fmt, data, n):
    data = np.asarray(data)
    if fmt == 0x1805:
        return np.ascontiguousarray(data[:, :3], '<f4').tobytes(), 12
    if fmt == 0x1905:
        return np.ascontiguousarray(data[:, :4], '<f4').tobytes(), 16
    if fmt == 0x0E02:
        w = data[:, 3] if data.shape[1] > 3 else np.ones(n)
        return _enc_1010102(data, np.where(w < 0, 3, 1)).tobytes(), 4
    if fmt == 0x1205:
        return np.ascontiguousarray(data[:, :2], '<f2').tobytes(), 4
    if fmt == 0x1505:
        return np.ascontiguousarray(data[:, :4], '<f2').tobytes(), 8
    if fmt == 0x1201:
        return np.ascontiguousarray(np.clip(np.round(data[:, :2] * 65535), 0, 65535), '<u2').tobytes(), 4
    if fmt == 0x0B03:
        return np.ascontiguousarray(data[:, :4], np.uint8).tobytes(), 4
    if fmt == 0x0B01:
        return np.ascontiguousarray(np.clip(np.round(data[:, :4] * 255), 0, 255), np.uint8).tobytes(), 4
    if fmt == 0x0903:
        return np.ascontiguousarray(data[:, :2], np.uint8).tobytes(), 2
    if fmt == 0x0901:
        return np.ascontiguousarray(np.clip(np.round(data[:, :2] * 255), 0, 255), np.uint8).tobytes(), 2
    if fmt == 0x0203:
        return np.ascontiguousarray(data[:, :1], np.uint8).tobytes(), 1
    raise ValueError('attribute format %x not supported' % fmt)


def replace_geometry(template, attrs, tris, bounds=None, skin_spheres=None):
    B = bfres.Bfres(template)
    d = bytearray(B.d)
    assert len(B.models) == 1 and len(B.models[0]['shapes']) == 1 and len(B.models[0]['vbufs']) == 1, 'single shape templates only'
    m = B.models[0]
    v = m['vbufs'][0]
    s = m['shapes'][0]
    n = len(attrs['_p0'])
    tris = np.asarray(tris, np.int64).reshape(-1)
    assert tris.max() < n
    ifmt = 1 if n < 65536 else 2
    ibytes = tris.astype('<u2' if ifmt == 1 else '<u4').tobytes()

    # ---- GPU pool: indices, then one buffer per vertex buffer slot (each 0x100 aligned)
    pool = bytearray()
    pool += ibytes
    vtx_off = _align(len(pool), 0x100)
    pool += bytes(vtx_off - len(pool))
    nbuf = len(v['buffers'])
    bufs = [None] * nbuf
    for a in v['attrs']:
        if a['name'] not in attrs:
            raise KeyError('missing attribute ' + a['name'])
        raw, size = encode_attr(a['fmt'], attrs[a['name']], n)
        stride = v['buffers'][a['buf']]['stride']
        assert a['offset'] == 0 and stride == size, ('interleaved buffers not supported', a)
        bufs[a['buf']] = raw
    sizes = []
    for i, raw in enumerate(bufs):
        if i:
            pool += bytes(_align(len(pool), 0x100) - len(pool))
        sizes.append(len(raw))
        pool += raw
    pool_size = _align(len(pool), 0x1000)
    pool += bytes(pool_size - len(pool))

    head = bytes(d[:B.buf_data])
    pool_struct = bytes(d[B.pool_off:B.pool_off + 0x120])
    rlt_old = B.rlt_off
    rlt_bytes = bytearray(d[rlt_old:B.file_size])
    new_pool_struct = B.buf_data + pool_size
    new_rlt = new_pool_struct + 0x120
    out = bytearray(head) + pool + pool_struct + rlt_bytes
    W = lambda fmt, off, *val: struct.pack_into(fmt, out, off, *val)

    # ---- header
    W('<I', 0x18, new_rlt)
    W('<I', 0x1C, len(out))
    W('<Q', 0xA8, new_pool_struct)
    W('<I', B.bufinfo_off + 4, pool_size)
    # ---- vertex buffer
    vo = v['off']
    W('<Q', vo + 0x18, new_pool_struct)
    W('<I', vo + 0x48, vtx_off)
    W('<I', vo + 0x50, n)
    for i, sz in enumerate(sizes):
        W('<I', v['size_arr'] + i * 16, sz)
    # ---- meshes (all LODs share the new index buffer)
    for me in s['meshes']:
        mo = me['off']
        W('<Q', mo + 0x08, new_pool_struct)
        W('<I', mo + 0x20, 0)
        W('<I', mo + 0x28, ifmt)
        W('<I', mo + 0x2C, len(tris))
        W('<I', mo + 0x30, 0)
        W('<I', me['size_info'], len(ibytes))
        sub = struct.unpack_from('<Q', out, mo)[0]
        nsub = struct.unpack_from('<H', out, mo + 0x34)[0]
        for k in range(nsub):
            W('<II', sub + 8 * k, 0, len(tris) if k == 0 else 0)
    # ---- bounds (rigid shapes: model space AABB + sphere; skinned: keep the template's)
    if bounds is not None and s['bbox_off']:
        # rigid shapes only: one AABB per mesh/submesh and ONE bounding sphere (skinned shapes have one
        # sphere per skin bone in bone space - those are kept from the template)
        assert not s['skin_bones'], 'bounds are only rewritten for rigid shapes'
        c, e = bounds
        nb = sum(len(me['submeshes']) + 1 for me in s['meshes'])
        for k in range(nb):
            W('<6f', s['bbox_off'] + 24 * k, *c, *e)
        W('<4f', s['radius_off'], *c, float(np.linalg.norm(e)))
    # skinned shapes: one bounding sphere per skin bone, in bone space (None = keep the template's)
    if skin_spheres is not None and s['skin_bones'] and s['radius_off']:
        assert len(skin_spheres) == len(s['skin_bones'])
        for k, sp in enumerate(skin_spheres):
            if sp is not None:
                W('<4f', s['radius_off'] + 16 * k, *[float(x) for x in sp])
    # ---- relocation table sections (positions inside the head are unchanged)
    r = new_rlt
    nsec = struct.unpack_from('<I', out, r + 8)[0]
    secs = [list(struct.unpack_from('<QIIII', out, r + 16 + 24 * i)) for i in range(nsec)]
    assert nsec == 6, nsec
    secs[1][1], secs[1][2] = B.buf_data, vtx_off
    secs[2][1], secs[2][2] = B.buf_data + vtx_off, pool_size - vtx_off
    secs[3][1], secs[3][2] = new_pool_struct, 0
    secs[4][1], secs[4][2] = new_pool_struct, 0x120
    secs[5][1], secs[5][2] = new_rlt, 0
    for i, sc in enumerate(secs):
        W('<QIIII', r + 16 + 24 * i, *sc)
    W('<I', r + 4, new_rlt)
    # _STR block header: size runs up to the relocation table
    so = struct.unpack_from('<H', out, 0x16)[0]
    if out[so:so + 4] == b'_STR':
        W('<I', so + 8, new_rlt - so)
    return bytes(out)


def _dict_ref_bit(name):
    nb = name.encode('utf-8')
    ref = 0
    while True:
        byte = nb[len(nb) - 1 - (ref >> 3)] if (ref >> 3) < len(nb) else 0
        if (byte >> (ref & 7)) & 1:
            return ref
        ref += 1


def rename_strings(data, mapping):
    """replace names in the string pool (e.g. the texture names a material uses): every new string is appended to
    the pool, every 64-bit pointer to the old string is redirected. Names stored in dictionaries are not touched."""
    B = bfres.Bfres(data)
    out = bytearray(data)
    rlt = B.rlt_off
    sec0_end = struct.unpack_from('<I', out, rlt + 16 + 12)[0]
    pos = _align(sec0_end, 4)
    for old, new in mapping.items():
        ob, nb = old.encode('utf-8'), new.encode('utf-8')
        pat = struct.pack('<H', len(ob)) + ob + b'\x00'
        start = out.find(pat)
        if start < 0:
            raise ValueError('Name %s is not in the template' % old)
        end = pos + 2 + len(nb) + 1
        assert end <= B.buf_data, 'no room for the new name'
        struct.pack_into('<H', out, pos, len(nb))
        out[pos + 2:pos + 2 + len(nb)] = nb
        out[pos + 2 + len(nb)] = 0
        old_ptr = struct.pack('<Q', start)
        hits = 0
        k = out.find(old_ptr, 0, B.buf_data)
        while k >= 0:
            if k % 8 == 0:
                struct.pack_into('<Q', out, k, pos)
                hits += 1
            k = out.find(old_ptr, k + 1, B.buf_data)
        if not hits:
            raise ValueError('kein Verweis auf %s gefunden' % old)
        pos = _align(end, 4)
    struct.pack_into('<I', out, rlt + 16 + 12, pos)
    return bytes(out)


def rename_model(data, new_name):
    """give the (single) model and the file a new name: the string is appended to the string pool
    (free space before the GPU buffers), pointers to the old name are redirected."""
    B = bfres.Bfres(data)
    out = bytearray(data)
    nb = new_name.encode('utf-8')
    rlt = B.rlt_off
    sec0_end = struct.unpack_from('<I', out, rlt + 16 + 12)[0]  # section 0 size
    pos = _align(sec0_end, 4)
    end = pos + 2 + len(nb) + 1
    assert end <= B.buf_data, 'no room for the new name'
    struct.pack_into('<H', out, pos, len(nb))
    out[pos + 2:pos + 2 + len(nb)] = nb
    out[pos + 2 + len(nb)] = 0
    struct.pack_into('<Q', out, 0x20, pos)
    struct.pack_into('<I', out, 0x10, pos + 2)
    mo = B.models[0]['off']
    struct.pack_into('<Q', out, mo + 0x08, pos)
    dic = struct.unpack_from('<Q', out, 0x30)[0]
    assert struct.unpack_from('<i', out, dic + 4)[0] == 1
    struct.pack_into('<I', out, dic + 8 + 16, _dict_ref_bit(new_name))
    struct.pack_into('<Q', out, dic + 8 + 16 + 8, pos)
    struct.pack_into('<I', out, rlt + 16 + 12, _align(end, 4))
    return bytes(out)
