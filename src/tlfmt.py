"""Format helpers for Tomodachi Life: Living the Dream (Nintendo EPD-style engine).
zstd (.zs), BYML v2-v7 (incl. hash dicts), SARC, MSBT."""
import struct, os, zstandard

_DCTX = zstandard.ZstdDecompressor()


def unzs(data: bytes) -> bytes:
    if data[:4] == b'\x28\xb5\x2f\xfd':
        return _DCTX.decompress(data, max_output_size=1 << 30)
    return data


def read(path):
    with open(path, 'rb') as f:
        d = f.read()
    return unzs(d) if path.endswith('.zs') else d


# ---------------- BYML ----------------
class Byml:
    def __init__(self, data: bytes):
        self.d = data
        m = data[:2]
        if m == b'YB':
            self.e = '<'
        elif m == b'BY':
            self.e = '>'
        else:
            raise ValueError('not byml: %r' % m)
        self.version = struct.unpack_from(self.e + 'H', data, 2)[0]
        self.keyoff, self.stroff, self.rootoff = struct.unpack_from(self.e + 'III', data, 4)
        self.keys = self._strtab(self.keyoff) if self.keyoff else []
        self.strs = self._strtab(self.stroff) if self.stroff else []

    def u24(self, o):
        return int.from_bytes(self.d[o:o + 3], 'little' if self.e == '<' else 'big')

    def u32(self, o):
        return struct.unpack_from(self.e + 'I', self.d, o)[0]

    def _strtab(self, off):
        if self.d[off] != 0xC2:
            raise ValueError('bad strtab type %x' % self.d[off])
        n = self.u24(off + 1)
        offs = [self.u32(off + 4 + 4 * i) for i in range(n + 1)]
        out = []
        for i in range(n):
            s = off + offs[i]
            e = self.d.index(b'\0', s)
            out.append(self.d[s:e].decode('utf-8', 'replace'))
        return out

    def root(self):
        if not self.rootoff:
            return None
        return self.node(self.d[self.rootoff], self.rootoff)

    def node(self, t, v):
        e = self.e
        if t == 0xA0:
            return self.strs[v]
        if t == 0xA1:
            n = self.u32(v)
            return bytes(self.d[v + 4:v + 4 + n])
        if t == 0xA2:
            n = self.u32(v)
            return bytes(self.d[v + 8:v + 8 + n])
        if t == 0xC0:
            return self.array(v)
        if t == 0xC1:
            return self.dict(v)
        if t in (0x20, 0x21):
            return self.hashdict(v, t)
        if t == 0xD0:
            return bool(v)
        if t == 0xD1:
            return struct.unpack(e + 'i', struct.pack(e + 'I', v))[0]
        if t == 0xD2:
            return struct.unpack(e + 'f', struct.pack(e + 'I', v))[0]
        if t == 0xD3:
            return v
        if t == 0xD4:
            return struct.unpack_from(e + 'q', self.d, v)[0]
        if t == 0xD5:
            return struct.unpack_from(e + 'Q', self.d, v)[0]
        if t == 0xD6:
            return struct.unpack_from(e + 'd', self.d, v)[0]
        if t == 0xFF:
            return None
        return {'__unk_type__': hex(t), 'v': v}

    def array(self, off):
        n = self.u24(off + 1)
        types = self.d[off + 4: off + 4 + n]
        vo = off + 4 + ((n + 3) & ~3)
        return [self.node(types[i], self.u32(vo + 4 * i)) for i in range(n)]

    def dict(self, off):
        n = self.u24(off + 1)
        r = {}
        for i in range(n):
            o = off + 4 + 8 * i
            k = self.u24(o)
            t = self.d[o + 3]
            v = self.u32(o + 4)
            r[self.keys[k]] = self.node(t, v)
        return r

    def hashdict(self, off, t):
        n = self.u24(off + 1)
        ks = 4 if t == 0x20 else 8
        es = ks + 4
        r = {}
        types = self.d[off + 4 + es * n: off + 4 + es * n + n]
        for i in range(n):
            o = off + 4 + es * i
            k = self.u32(o) if ks == 4 else struct.unpack_from(self.e + 'Q', self.d, o)[0]
            v = self.u32(o + ks)
            r[k] = self.node(types[i], v)
        return r


def byml(path_or_bytes):
    d = read(path_or_bytes) if isinstance(path_or_bytes, str) else unzs(path_or_bytes)
    return Byml(d).root()


# ---------------- SARC ----------------
def sarc(data: bytes):
    data = unzs(data)
    assert data[:4] == b'SARC', data[:4]
    bom = struct.unpack_from('>H', data, 6)[0]
    e = '<' if bom == 0xFFFE else '>'
    hdr, = struct.unpack_from(e + 'H', data, 4)
    dataoff, = struct.unpack_from(e + 'I', data, 0x0C)
    o = hdr
    assert data[o:o + 4] == b'SFAT'
    sfat_hs, nodes, hkey = struct.unpack_from(e + 'HHI', data, o + 4)
    entries = [struct.unpack_from(e + 'IIII', data, o + 0xC + 0x10 * i) for i in range(nodes)]
    o2 = o + 0xC + 0x10 * nodes
    assert data[o2:o2 + 4] == b'SFNT'
    names0 = o2 + 8
    out = {}
    for h, attr, s, en in entries:
        if attr & 0x01000000:
            no = names0 + (attr & 0xFFFF) * 4
            name = data[no:data.index(b'\0', no)].decode('utf-8')
        else:
            name = '%08x' % h
        out[name] = data[dataoff + s: dataoff + en]
    return out


def sarc_build(files: dict, endian='<', align=0x10, hash_key=0x65):
    """Build a SARC from {name: bytes}. Files are sorted by name hash (required)."""
    e = endian

    def nh(name):
        h = 0
        for c in name.encode('utf-8'):
            h = (h * hash_key + (c if c < 0x80 else c - 0x100)) & 0xFFFFFFFF
        return h

    names = sorted(files.keys(), key=nh)
    sfnt = bytearray()
    name_offs = {}
    for n in names:
        name_offs[n] = len(sfnt)
        sfnt += n.encode('utf-8') + b'\0'
        while len(sfnt) % 4:
            sfnt += b'\0'
    hdr_size = 0x14
    sfat_size = 0xC + 0x10 * len(names)
    sfnt_hdr = 8
    data_off = hdr_size + sfat_size + sfnt_hdr + len(sfnt)
    data_off = (data_off + align - 1) & ~(align - 1)
    blob = bytearray()
    ranges = []
    for n in names:
        while len(blob) % align:
            blob += b'\0'
        s = len(blob)
        blob += files[n]
        ranges.append((s, len(blob)))
    out = bytearray()
    total = data_off + len(blob)
    out += b'SARC' + struct.pack(e + 'HH', hdr_size, 0xFEFF) + struct.pack(e + 'IIHH', total, data_off, 0x0100, 0)
    out += b'SFAT' + struct.pack(e + 'HHI', 0xC, len(names), hash_key)
    for n, (s, en) in zip(names, ranges):
        out += struct.pack(e + 'IIII', nh(n), 0x01000000 | (name_offs[n] // 4), s, en)
    out += b'SFNT' + struct.pack(e + 'HH', 8, 0) + sfnt
    while len(out) < data_off:
        out += b'\0'
    out += blob
    return bytes(out)


# ---------------- MSBT ----------------
def msbt_sections(data: bytes):
    assert data[:8] == b'MsgStdBn', data[:8]
    e = '<' if data[8:10] == b'\xff\xfe' else '>'
    nsec, = struct.unpack_from(e + 'H', data, 0xE)
    o = 0x20
    secs = []
    for _ in range(nsec):
        mg = data[o:o + 4]
        sz, = struct.unpack_from(e + 'I', data, o + 4)
        secs.append((mg, o, sz))
        o = (o + 0x10 + sz + 15) & ~15
    return e, data[0xC], secs


def msbt(data: bytes):
    """returns ({label: (text_with_tags, raw_bytes)}, info)"""
    e, enc, secs = msbt_sections(data)
    sd = {mg: (o + 0x10, sz) for mg, o, sz in secs}
    labels = {}
    if b'LBL1' in sd:
        so, sz = sd[b'LBL1']
        ng, = struct.unpack_from(e + 'I', data, so)
        for g in range(ng):
            cnt, off = struct.unpack_from(e + 'II', data, so + 4 + 8 * g)
            p = so + off
            for _ in range(cnt):
                ln = data[p]
                name = data[p + 1:p + 1 + ln].decode('utf-8')
                idx, = struct.unpack_from(e + 'I', data, p + 1 + ln)
                labels[idx] = name
                p += 1 + ln + 4
    texts = []
    if b'TXT2' in sd:
        so, sz = sd[b'TXT2']
        n, = struct.unpack_from(e + 'I', data, so)
        offs = [struct.unpack_from(e + 'I', data, so + 4 + 4 * i)[0] for i in range(n)] + [sz]
        for i in range(n):
            texts.append(data[so + offs[i]: so + offs[i + 1]])
    out = {}
    for i, raw in enumerate(texts):
        out[labels.get(i, str(i))] = (decode_txt(raw, e, enc), raw)
    return out, {'endian': e, 'enc': enc, 'sections': [mg.decode() for mg, _, _ in secs]}


def decode_txt(raw, e='<', enc=1):
    if enc != 1:
        return raw.decode('utf-8', 'replace')
    s = []
    i = 0
    while i + 1 < len(raw):
        c, = struct.unpack_from(e + 'H', raw, i)
        i += 2
        if c == 0x0E:
            grp, tag, psz = struct.unpack_from(e + 'HHH', raw, i)
            i += 6
            params = raw[i:i + psz]
            i += psz
            s.append('[%d.%d:%s]' % (grp, tag, params.hex()))
        elif c == 0x0F:
            grp, tag = struct.unpack_from(e + 'HH', raw, i)
            i += 4
            s.append('[/%d.%d]' % (grp, tag))
        elif c == 0:
            break
        else:
            s.append(chr(c))
    return ''.join(s)


def encode_txt(s, e='<'):
    """inverse of decode_txt (UTF-16 only); tags written as [g.t:hex] / [/g.t]"""
    import re
    out = bytearray()
    pos = 0
    for m in re.finditer(r'\[(\d+)\.(\d+):([0-9a-f]*)\]|\[/(\d+)\.(\d+)\]', s):
        for ch in s[pos:m.start()]:
            out += struct.pack(e + 'H', ord(ch))
        if m.group(1) is not None:
            p = bytes.fromhex(m.group(3))
            out += struct.pack(e + 'HHHH', 0x0E, int(m.group(1)), int(m.group(2)), len(p)) + p
        else:
            out += struct.pack(e + 'HHH', 0x0F, int(m.group(4)), int(m.group(5)))
        pos = m.end()
    for ch in s[pos:]:
        out += struct.pack(e + 'H', ord(ch))
    out += b'\0\0'
    return bytes(out)


def msbt_replace_texts(data: bytes, new_raw: dict):
    """Rebuild TXT2 with replaced raw strings. new_raw: {index: raw_bytes_incl_terminator}."""
    e, enc, secs = msbt_sections(data)
    out = bytearray(data[:0x20])
    for mg, o, sz in secs:
        body = data[o + 0x10:o + 0x10 + sz]
        if mg == b'TXT2':
            n, = struct.unpack_from(e + 'I', body, 0)
            offs = [struct.unpack_from(e + 'I', body, 4 + 4 * i)[0] for i in range(n)] + [sz]
            strs = [body[offs[i]:offs[i + 1]] for i in range(n)]
            for i, r in new_raw.items():
                strs[i] = r
            nb = bytearray(struct.pack(e + 'I', n))
            cur = 4 + 4 * n
            for s_ in strs:
                nb += struct.pack(e + 'I', cur)
                cur += len(s_)
            for s_ in strs:
                nb += s_
            body = bytes(nb)
        out += mg + struct.pack(e + 'I', len(body)) + b'\0' * 8 + body
        while len(out) % 16:
            out += b'\xab'
    struct.pack_into(e + 'I', out, 0x12, len(out))
    return bytes(out)
