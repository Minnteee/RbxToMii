"""Typed BYML (v2-v7, little endian) parse + write, so tables can be edited without losing value types.
parse(bytes) -> node;  write(node, version) -> bytes
node kinds: str -> ('S', str); scalars -> ('b'|'i'|'f'|'u'|'l'|'L'|'d', value); list; dict (str keys);
            hash dicts -> ('H20'|'H21', {int: node}); binary -> ('bin', bytes)"""
import struct

T_STR, T_BIN, T_BIN2, T_ARR, T_DICT, T_STRTAB = 0xA0, 0xA1, 0xA2, 0xC0, 0xC1, 0xC2
SCALAR = {0xD0: 'b', 0xD1: 'i', 0xD2: 'f', 0xD3: 'u', 0xD4: 'l', 0xD5: 'L', 0xD6: 'd', 0xFF: 'null'}
SCALAR_T = {v: k for k, v in SCALAR.items()}


class Reader:
    def __init__(self, d):
        assert d[:2] == b'YB', d[:2]
        self.d = d
        self.version = struct.unpack_from('<H', d, 2)[0]
        ko, so, ro = struct.unpack_from('<III', d, 4)
        self.keys = self.strtab(ko) if ko else []
        self.strs = self.strtab(so) if so else []
        self.root_off = ro

    def u24(self, o):
        return int.from_bytes(self.d[o:o + 3], 'little')

    def strtab(self, off):
        assert self.d[off] == T_STRTAB
        n = self.u24(off + 1)
        offs = struct.unpack_from('<%dI' % (n + 1), self.d, off + 4)
        out = []
        for i in range(n):
            s = off + offs[i]
            out.append(self.d[s:self.d.index(b'\0', s)].decode('utf-8'))
        return out

    def node(self, t, v):
        d = self.d
        if t == T_STR:
            return ('S', self.strs[v])
        if t in SCALAR:
            k = SCALAR[t]
            if k == 'b':
                return ('b', bool(v))
            if k == 'i':
                return ('i', struct.unpack('<i', struct.pack('<I', v))[0])
            if k == 'f':
                return ('f', struct.unpack('<f', struct.pack('<I', v))[0])
            if k == 'u':
                return ('u', v)
            if k == 'l':
                return ('l', struct.unpack_from('<q', d, v)[0])
            if k == 'L':
                return ('L', struct.unpack_from('<Q', d, v)[0])
            if k == 'd':
                return ('d', struct.unpack_from('<d', d, v)[0])
            return ('null', None)
        if t == T_BIN:
            n = struct.unpack_from('<I', d, v)[0]
            return ('bin', bytes(d[v + 4:v + 4 + n]))
        if t == T_ARR:
            n = self.u24(v + 1)
            types = d[v + 4:v + 4 + n]
            vo = v + 4 + ((n + 3) & ~3)
            return [self.node(types[i], struct.unpack_from('<I', d, vo + 4 * i)[0]) for i in range(n)]
        if t == T_DICT:
            n = self.u24(v + 1)
            out = {}
            for i in range(n):
                o = v + 4 + 8 * i
                out[self.keys[self.u24(o)]] = self.node(d[o + 3], struct.unpack_from('<I', d, o + 4)[0])
            return out
        if t in (0x20, 0x21):
            n = self.u24(v + 1)
            ks = 4 if t == 0x20 else 8
            es = ks + 4
            types = d[v + 4 + es * n:v + 4 + es * n + n]
            out = {}
            for i in range(n):
                o = v + 4 + es * i
                k = struct.unpack_from('<I' if ks == 4 else '<Q', d, o)[0]
                out[k] = self.node(types[i], struct.unpack_from('<I', d, o + ks)[0])
            return ('H%02X' % t, out)
        raise ValueError('unknown node type %x' % t)

    def root(self):
        return self.node(self.d[self.root_off], self.root_off)


def parse(d):
    r = Reader(d)
    return r.root(), r.version


def _collect(node, keys, strs):
    if isinstance(node, tuple):
        if node[0] == 'S':
            strs.add(node[1])
        elif node[0] in ('H20', 'H21'):
            for v in node[1].values():
                _collect(v, keys, strs)
    elif isinstance(node, list):
        for v in node:
            _collect(v, keys, strs)
    elif isinstance(node, dict):
        for k, v in node.items():
            keys.add(k)
            _collect(v, keys, strs)


def _type_of(node):
    if isinstance(node, list):
        return T_ARR
    if isinstance(node, dict):
        return T_DICT
    k = node[0]
    if k == 'S':
        return T_STR
    if k == 'bin':
        return T_BIN
    if k in ('H20', 'H21'):
        return int(k[1:], 16)
    return SCALAR_T[k]


def _strtab_bytes(strings):
    n = len(strings)
    body = bytearray()
    offs = []
    base = 4 + 4 * (n + 1)
    for s in strings:
        offs.append(base + len(body))
        body += s.encode('utf-8') + b'\0'
    offs.append(base + len(body))
    out = bytearray(struct.pack('<I', T_STRTAB | (n << 8))) + struct.pack('<%dI' % (n + 1), *offs) + body
    while len(out) % 4:
        out += b'\0'
    return bytes(out)


def write(root, version=7):
    keys, strs = set(), set()
    _collect(root, keys, strs)
    keys = sorted(keys, key=lambda s: s.encode('utf-8'))
    strs = sorted(strs, key=lambda s: s.encode('utf-8'))
    kidx = {k: i for i, k in enumerate(keys)}
    sidx = {s: i for i, s in enumerate(strs)}
    out = bytearray(16)
    ko = len(out)
    out += _strtab_bytes(keys) if keys else b''
    so = len(out)
    out += _strtab_bytes(strs) if strs else b''

    def value(node):
        """inline value (u32) for a node, writing containers / 64-bit data as needed"""
        t = _type_of(node)
        if t == T_ARR or t == T_DICT or t in (0x20, 0x21):
            return container(node)
        k = node[0]
        if k == 'S':
            return sidx[node[1]]
        if k == 'b':
            return 1 if node[1] else 0
        if k == 'i':
            return struct.unpack('<I', struct.pack('<i', node[1]))[0]
        if k == 'f':
            return struct.unpack('<I', struct.pack('<f', node[1]))[0]
        if k == 'u':
            return node[1] & 0xFFFFFFFF
        if k == 'null':
            return 0
        if k in ('l', 'L', 'd'):
            while len(out) % 8:
                out.append(0)
            o = len(out)
            out.extend(struct.pack({'l': '<q', 'L': '<Q', 'd': '<d'}[k], node[1]))
            return o
        if k == 'bin':
            while len(out) % 4:
                out.append(0)
            o = len(out)
            out.extend(struct.pack('<I', len(node[1])) + node[1])
            while len(out) % 4:
                out.append(0)
            return o
        raise ValueError(node)

    seen = {}

    def freeze(node):
        if isinstance(node, list):
            return ('[', tuple(freeze(x) for x in node))
        if isinstance(node, dict):
            return ('{', tuple(sorted((k, freeze(v)) for k, v in node.items())))
        if isinstance(node, tuple) and node[0] in ('H20', 'H21'):
            return (node[0], tuple(sorted((k, freeze(v)) for k, v in node[1].items())))
        return node

    def container(node):
        key = freeze(node)
        if key in seen:
            return seen[key]
        while len(out) % 4:
            out.append(0)
        o = len(out)
        seen[key] = o
        if isinstance(node, list):
            n = len(node)
            out.extend(struct.pack('<I', T_ARR | (n << 8)))
            out.extend(bytes(_type_of(x) for x in node))
            while len(out) % 4:
                out.append(0)
            vo = len(out)
            out.extend(bytes(4 * n))
            for i, x in enumerate(node):
                struct.pack_into('<I', out, vo + 4 * i, value(x))
        elif isinstance(node, dict):
            items = sorted(node.items(), key=lambda kv: kidx[kv[0]])
            n = len(items)
            out.extend(struct.pack('<I', T_DICT | (n << 8)))
            eo = len(out)
            out.extend(bytes(8 * n))
            for i, (k, x) in enumerate(items):
                struct.pack_into('<I', out, eo + 8 * i, kidx[k] | (_type_of(x) << 24))
                struct.pack_into('<I', out, eo + 8 * i + 4, value(x))
        else:
            t = int(node[0][1:], 16)
            ks = 4 if t == 0x20 else 8
            items = sorted(node[1].items())
            n = len(items)
            out.extend(struct.pack('<I', t | (n << 8)))
            eo = len(out)
            out.extend(bytes((ks + 4) * n))
            out.extend(bytes(_type_of(x) for _, x in items))
            while len(out) % 4:
                out.append(0)
            for i, (h, x) in enumerate(items):
                struct.pack_into('<I' if ks == 4 else '<Q', out, eo + (ks + 4) * i, h)
                struct.pack_into('<I', out, eo + (ks + 4) * i + ks, value(x))
        return o

    ro = container(root)
    struct.pack_into('<2sHIII', out, 0, b'YB', version, ko if keys else 0, so if strs else 0, ro)
    return bytes(out)
