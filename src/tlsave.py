"""Tomodachi Life: Living the Dream save files (Mii.sav / Player.sav / Map.sav).
Layout (same family as TotK GameData saves):
  0x00 u32 magic 0x01020304, u32 version, u32 dataOffset, pad to 0x20
  0x20 (hash u32, value u32) pairs; (0, typeIndex) starts a type group
  values of small scalar types are inline, everything else is an offset into the file
  arrays: [u32 count][elements]; BoolArray elements are packed bits (u32 words)
"""
import struct, json, os

TYPES = ['Bool', 'BoolArray', 'Int', 'IntArray', 'Float', 'FloatArray', 'Enum', 'EnumArray',
         'Vector2', 'Vector2Array', 'Vector3', 'Vector3Array', 'String16', 'String16Array',
         'String32', 'String32Array', 'String64', 'String64Array', 'Binary', 'BinaryArray',
         'UInt', 'UIntArray', 'Int64', 'Int64Array', 'UInt64', 'UInt64Array',
         'WString16', 'WString16Array', 'WString32', 'WString32Array', 'WString64', 'WString64Array',
         'Struct']
INLINE = {'Bool', 'Int', 'Float', 'Enum', 'UInt', 'Struct'}
ELEM = {'IntArray': 4, 'FloatArray': 4, 'EnumArray': 4, 'UIntArray': 4, 'Int64Array': 8, 'UInt64Array': 8,
        'Vector2Array': 8, 'Vector3Array': 12, 'String16Array': 16, 'String32Array': 32, 'String64Array': 64,
        'WString16Array': 32, 'WString32Array': 64, 'WString64Array': 128}
SCALAR_SIZE = {'Int64': 8, 'UInt64': 8, 'Vector2': 8, 'Vector3': 12, 'String16': 16, 'String32': 32, 'String64': 64,
               'WString16': 32, 'WString32': 64, 'WString64': 128}


class Entry:
    __slots__ = ('type', 'hash', 'raw', 'pos', 'off', 'size')

    def __init__(self, t, h, raw, pos):
        self.type, self.hash, self.raw, self.pos = t, h, raw, pos
        self.off = None
        self.size = None


class Save:
    def __init__(self, path_or_bytes):
        d = open(path_or_bytes, 'rb').read() if isinstance(path_or_bytes, str) else path_or_bytes
        self.d = bytearray(d)
        self.magic, self.version, self.dataoff = struct.unpack_from('<III', d, 0)
        assert self.magic == 0x01020304
        o = 0x20
        t = None
        self.entries = []
        self.by_hash = {}
        while o < self.dataoff:
            h, v = struct.unpack_from('<II', d, o)
            if h == 0:
                t = TYPES[v] if v < len(TYPES) else 'T%d' % v
            else:
                e = Entry(t, h, v, o)
                self.entries.append(e)
                self.by_hash[h] = e
            o += 8
        offs = sorted(set(e.raw for e in self.entries if e.type not in INLINE))
        nxt = {a: b for a, b in zip(offs, offs[1:] + [len(d)])}
        for e in self.entries:
            if e.type not in INLINE:
                e.off = e.raw
                e.size = nxt[e.raw] - e.raw

    # ---- typed access ----
    def get(self, h):
        e = self.by_hash[h]
        return self.value(e)

    def value(self, e):
        d = self.d
        t = e.type
        if t == 'Bool':
            return bool(e.raw)
        if t in ('Int',):
            return struct.unpack('<i', struct.pack('<I', e.raw))[0]
        if t in ('UInt', 'Enum', 'Struct'):
            return e.raw
        if t == 'Float':
            return struct.unpack('<f', struct.pack('<I', e.raw))[0]
        o = e.off
        if t in ('Int64',):
            return struct.unpack_from('<q', d, o)[0]
        if t in ('UInt64',):
            return struct.unpack_from('<Q', d, o)[0]
        if t == 'Vector2':
            return struct.unpack_from('<2f', d, o)
        if t == 'Vector3':
            return struct.unpack_from('<3f', d, o)
        if t.startswith('WString') and not t.endswith('Array'):
            n = SCALAR_SIZE[t]
            return wstr(d[o:o + n])
        if t.startswith('String') and not t.endswith('Array'):
            n = SCALAR_SIZE[t]
            return d[o:o + n].split(b'\0')[0].decode('utf-8', 'replace')
        if t == 'Binary':
            n, = struct.unpack_from('<I', d, o)
            return bytes(d[o + 4:o + 4 + n])
        cnt, = struct.unpack_from('<I', d, o)
        if t == 'BoolArray':
            return [bool((d[o + 4 + (i >> 3)] >> (i & 7)) & 1) for i in range(cnt)]
        if t == 'BinaryArray':
            # [count][u32 elemSize?] -> derive element size from span
            span = e.size - 4
            es = span // cnt if cnt else 0
            return [bytes(d[o + 4 + i * es:o + 4 + (i + 1) * es]) for i in range(cnt)]
        es = ELEM.get(t)
        base = o + 4
        if t in ('IntArray',):
            return list(struct.unpack_from('<%di' % cnt, d, base))
        if t in ('UIntArray', 'EnumArray'):
            return list(struct.unpack_from('<%dI' % cnt, d, base))
        if t == 'FloatArray':
            return list(struct.unpack_from('<%df' % cnt, d, base))
        if t == 'Int64Array':
            return list(struct.unpack_from('<%dq' % cnt, d, base))
        if t == 'UInt64Array':
            return list(struct.unpack_from('<%dQ' % cnt, d, base))
        if t == 'Vector2Array':
            return [struct.unpack_from('<2f', d, base + 8 * i) for i in range(cnt)]
        if t == 'Vector3Array':
            return [struct.unpack_from('<3f', d, base + 12 * i) for i in range(cnt)]
        if t.startswith('WString'):
            return [wstr(d[base + es * i: base + es * (i + 1)]) for i in range(cnt)]
        if t.startswith('String'):
            return [d[base + es * i: base + es * (i + 1)].split(b'\0')[0].decode('utf-8', 'replace') for i in range(cnt)]
        return ('raw', bytes(d[o:o + e.size]))

    def elem_off(self, h, i):
        """file offset of array element i (non-bool arrays)"""
        e = self.by_hash[h]
        es = ELEM[e.type]
        return e.off + 4 + es * i

    def set_inline(self, h, raw_u32):
        e = self.by_hash[h]
        e.raw = raw_u32 & 0xFFFFFFFF
        struct.pack_into('<I', self.d, e.pos + 4, e.raw)

    def write_bytes(self, off, b):
        self.d[off:off + len(b)] = b

    def bytes(self):
        return bytes(self.d)


def wstr(b):
    s = bytes(b).decode('utf-16-le', 'replace')
    return s.split('\0')[0]


def enc_wstr(s, nbytes):
    b = s.encode('utf-16-le')[:nbytes - 2]
    return b + b'\0' * (nbytes - len(b))
