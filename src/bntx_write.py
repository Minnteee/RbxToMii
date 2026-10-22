"""Write BNTX texture archives (Switch, block linear, full mip chains) in the layout the game's own files use.
build(name, rgba, kind)                         -> one texture, archive named like the texture
build_multi(file_name, [(name, rgba, kind, swz), ...]) -> several textures (e.g. the Tex/Lo/<project>.bntx files)
kinds: 'bc1_srgb' colour maps, 'bc3_srgb' colour maps with alpha, 'bc5_snorm_flat' flat normal map (all zero),
       'rgba8' small helper maps (R8G8B8A8 UNorm)"""
import struct
import numpy as np
from PIL import Image

import ugctex

KINDS = {  # format id, block w, block h, bytes per block, default channel swizzle
    'bc1_srgb': (0x1A06, 4, 4, 8, (2, 3, 4, 5)),
    'bc3_srgb': (0x1C06, 4, 4, 16, (2, 3, 4, 5)),
    'bc5_snorm_flat': (0x1E02, 4, 4, 16, (2, 3, 0, 1)),
    'rgba8': (0x0B01, 1, 1, 4, (2, 3, 4, 5)),
}


def _align(v, a):
    return (v + a - 1) // a * a


def _gob_blocks_y(hb):
    bh = 1
    while bh < (hb + 7) // 8:
        bh <<= 1
    return min(bh, 16)


def layout(w, h, bw, bh_px, bpp, nmips, bh_log2):
    out, off = [], 0
    for lvl in range(nmips):
        lw, lh = max(1, w >> lvl), max(1, h >> lvl)
        wb, hb = (lw + bw - 1) // bw, (lh + bh_px - 1) // bh_px
        g = 1 << bh_log2
        while g > 1 and hb <= (g // 2) * 8:
            g //= 2
        size = _align(wb * bpp, 64) * _align(hb, g * 8)
        out.append({'level': lvl, 'w': lw, 'h': lh, 'wb': wb, 'hb': hb, 'gob': g, 'off': off, 'size': size})
        off += size
    return out, off


def _encode_level(img, kind, wb, hb):
    if kind == 'bc5_snorm_flat':
        return bytes(wb * hb * 16)
    if kind == 'rgba8':
        return np.ascontiguousarray(img[:hb, :wb]).tobytes()
    pad = np.zeros((hb * 4, wb * 4, 4), np.uint8)
    pad[:img.shape[0], :img.shape[1]] = img
    if img.shape[1] < wb * 4:
        pad[:img.shape[0], img.shape[1]:] = img[:, -1:]
    if img.shape[0] < hb * 4:
        pad[img.shape[0]:] = pad[img.shape[0] - 1:img.shape[0]]
    if kind == 'bc1_srgb':
        return ugctex.encode_bc1(pad, alpha_cut=0)
    return ugctex.encode_bc3(pad)


def mip_chain(rgba, nmips):
    im = Image.fromarray(rgba, 'RGBA')
    out = [rgba]
    w, h = im.size
    for lvl in range(1, nmips):
        lw, lh = max(1, w >> lvl), max(1, h >> lvl)
        out.append(np.asarray(im.resize((lw, lh), Image.BOX)))
    return out


def encode_texture(rgba, kind, nmips=None):
    fmt, bw, bhp, bpp, _ = KINDS[kind]
    rgba = np.ascontiguousarray(rgba, np.uint8)
    h, w = rgba.shape[:2]
    if nmips is None:
        nmips = int(np.floor(np.log2(max(w, h)))) + 1  # down to 1 px, like the game's files (384 -> 9)
    hb0 = (h + bhp - 1) // bhp
    bh_log2 = int(np.log2(_gob_blocks_y(hb0)))
    levels, total = layout(w, h, bw, bhp, bpp, nmips, bh_log2)
    data = bytearray(total)
    for L, img in zip(levels, mip_chain(rgba, nmips)):
        lin = _encode_level(img, kind, L['wb'], L['hb'])
        sw = ugctex.swizzle(lin, L['wb'], L['hb'], bpp, L['gob'], size=L['size'])
        data[L['off']:L['off'] + L['size']] = sw[:L['size']]
    return {'fmt': fmt, 'w': w, 'h': h, 'nmips': nmips, 'bh_log2': bh_log2, 'levels': levels, 'data': bytes(data)}


def _bit(nb, i):
    j = i >> 3
    return (nb[len(nb) - 1 - j] >> (i & 7)) & 1 if j < len(nb) else 0


def dict_nodes(names):
    """Nintendo ResDic (patricia trie) for the given keys in order -> [[ref, left, right]] incl. root"""
    keys = [b''] + [n.encode('utf-8') for n in names]
    nodes = [[-1, 0, 0]]
    for idx in range(1, len(keys)):
        key = keys[idx]
        prev, cur = 0, nodes[0][1]
        while nodes[prev][0] < nodes[cur][0]:
            prev = cur
            cur = nodes[cur][2] if _bit(key, nodes[cur][0]) else nodes[cur][1]
        other = keys[cur]
        crit = 0
        while _bit(key, crit) == _bit(other, crit):
            crit += 1
        prev, cur = 0, nodes[0][1]
        while nodes[prev][0] < nodes[cur][0] and nodes[cur][0] < crit:
            prev = cur
            cur = nodes[cur][2] if _bit(key, nodes[cur][0]) else nodes[cur][1]
        new = [crit, cur, cur]
        if _bit(key, crit):
            new[2] = idx
        else:
            new[1] = idx
        nodes.append(new)
        if prev == 0:
            nodes[0][1] = idx
        elif _bit(key, nodes[prev][0]):
            nodes[prev][2] = idx
        else:
            nodes[prev][1] = idx
    return nodes


def build_multi(file_name, textures):
    """textures: list of (name, rgba, kind, swz or None); stored in the given order"""
    enc = []
    for name, rgba, kind, swz in textures:
        e = encode_texture(rgba, kind)
        e['name'], e['swz'] = name, swz or KINDS[kind][4]
        enc.append(e)
    n = len(enc)
    info_arr = 0x198
    STR = _align(info_arr + 8 * n, 8)
    names = [e['name'] for e in enc]
    strings = names if file_name in names else [file_name] + names
    pos = STR + 0x18
    str_off = {}
    for s_ in strings:
        str_off[s_] = pos
        pos += 2 + len(s_.encode('utf-8')) + 1
        pos = _align(pos, 2)
    dict_off = _align(pos, 8)
    brti_offs = []
    pos = dict_off + 8 + 16 * (n + 1)
    for e in enc:
        pos = _align(pos, 8)
        brti_offs.append(pos)
        pos += 0xA0 + 0x100 + 0x100 + 8 * e['nmips']
    sec0_end = pos
    data_start = _align(sec0_end + 0x10, 0x1000)
    brtd = data_start - 0x10
    doff = 0
    for e in enc:
        doff = _align(doff, 0x200)
        e['data_off'] = doff
        doff += len(e['data'])
    data_size = _align(doff, 0x1000)
    rlt = data_start + data_size
    ptr0 = [0x28, 0x38, 0x40] + [info_arr + 8 * i for i in range(n)] + [dict_off + 8 + 8 + 16 * i for i in range(n + 1)]
    for b in brti_offs:
        ptr0 += [b + 0x60, b + 0x68, b + 0x70, b + 0x80, b + 0x88]
    ptr1 = [0x30]
    for b, e in zip(brti_offs, enc):
        ptr1 += [b + 0x2A0 + 8 * i for i in range(e['nmips'])]

    def runs(ptrs):
        out = []
        for p in sorted(ptrs):
            if out and out[-1][0] + 8 * out[-1][2] == p and out[-1][2] < 255:
                out[-1][2] += 1
            else:
                out.append([p, 1, 1, 0])
        return out
    e0, e1 = runs(ptr0), runs(ptr1)
    fsize = rlt + 0x10 + 2 * 24 + 8 * (len(e0) + len(e1))
    f = bytearray(fsize)
    struct.pack_into('<4sI', f, 0, b'BNTX', 0)
    f[8:12] = bytes([1, 1, 4, 0])
    f[12:16] = bytes([0xFF, 0xFE, 0x0C, 0x40])
    struct.pack_into('<IHHII', f, 0x10, str_off[file_name] + 2, 0, STR, rlt, fsize)
    struct.pack_into('<4sIQQQQQII', f, 0x20, b'NX  ', n, info_arr, brtd, dict_off, 0x58, 0, 0, 0)
    for i, b in enumerate(brti_offs):
        struct.pack_into('<Q', f, info_arr + 8 * i, b)
    struct.pack_into('<4sIIIII', f, STR, b'_STR', brti_offs[0] - STR, brti_offs[0] - STR, 0, len(strings), 0)
    for s_, o in str_off.items():
        b = s_.encode('utf-8')
        struct.pack_into('<H', f, o, len(b))
        f[o + 2:o + 2 + len(b)] = b
    nodes = dict_nodes(names)
    struct.pack_into('<4si', f, dict_off, b'_DIC', n)
    for i, (ref, l, r) in enumerate(nodes):
        key = STR + 0x14 if i == 0 else str_off[names[i - 1]]
        struct.pack_into('<IHHQ', f, dict_off + 8 + 16 * i, ref & 0xFFFFFFFF, l, r, key)
    for i, (b, e) in enumerate(zip(brti_offs, enc)):
        nxt = brti_offs[i + 1] if i + 1 < n else brtd
        struct.pack_into('<4sIII', f, b, b'BRTI', nxt - b, nxt - b, 0)
        struct.pack_into('<BBHHHII', f, b + 0x10, 0x09, 2, 0, 0, e['nmips'], 1, e['fmt'])
        struct.pack_into('<IIIIIII', f, b + 0x20, 0x21, e['w'], e['h'], 1, 1, e['bh_log2'], 0x10007)
        struct.pack_into('<II', f, b + 0x50, len(e['data']), 0x200)
        f[b + 0x58:b + 0x5C] = bytes(e['swz'])
        f[b + 0x5C] = 1
        struct.pack_into('<QQQQQQQQ', f, b + 0x60, str_off[e['name']], 0x20, b + 0x2A0, 0, b + 0xA0, b + 0x1A0, 0, 0)
        for L in e['levels']:
            struct.pack_into('<Q', f, b + 0x2A0 + 8 * L['level'], data_start + e['data_off'] + L['off'])
        f[data_start + e['data_off']:data_start + e['data_off'] + len(e['data'])] = e['data']
    struct.pack_into('<4sIQ', f, brtd, b'BRTD', 0, 0x10 + data_size)
    struct.pack_into('<4sIII', f, rlt, b'_RLT', rlt, 2, 0)
    struct.pack_into('<QIIII', f, rlt + 0x10, 0, 0, sec0_end, 0, len(e0))
    struct.pack_into('<QIIII', f, rlt + 0x28, 0, brtd, 0x10 + data_size, len(e0), len(e1))
    o = rlt + 0x40
    for p, sc, oc, pc in e0 + e1:
        struct.pack_into('<IHBB', f, o, p, sc, oc, pc)
        o += 8
    return bytes(f)


def build(name, rgba, kind='bc1_srgb', nmips=None, swz=None):
    return build_multi(name, [(name, rgba, kind, swz)])
