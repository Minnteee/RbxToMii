"""Resource size table (System/Resource/ResourceSizeTable.Product.100.Nin_NX_NVN.rsizetable.zs).
Every resource path (without .zs) has a memory size: CRC32(path) -> size, plus a name table for CRC collisions.
The game allocates this much for the resource, so changed/new files must be listed with a large enough size."""
import struct, zlib

# observed: size in table = decompressed size + overhead
OVERHEAD = {'.bntx': 4112, '.byml': 304, '.sarc': 8192, '.bfres': 61000}


class Restbl:
    def __init__(self, data):
        assert data[:6] == b'RESTBL'
        self.ver, self.strsz, ncrc, nname = struct.unpack_from('<IIII', data, 6)
        o = 22
        self.crc = {}
        for i in range(ncrc):
            h, s = struct.unpack_from('<II', data, o + 8 * i)
            self.crc[h] = s
        o += 8 * ncrc
        self.names = {}
        for i in range(nname):
            nm = data[o:o + self.strsz].split(b'\0')[0].decode()
            self.names[nm] = struct.unpack_from('<I', data, o + self.strsz)[0]
            o += self.strsz + 4

    def get(self, path):
        if path in self.names:
            return self.names[path]
        return self.crc.get(zlib.crc32(path.encode()))

    def set(self, path, size, known_paths=()):
        h = zlib.crc32(path.encode())
        if path in self.names:
            self.names[path] = size
            return
        clash = [p for p in known_paths if p != path and zlib.crc32(p.encode()) == h]
        if clash:
            self.names[path] = size
        else:
            self.crc[h] = size

    def write(self):
        out = bytearray(b'RESTBL')
        out += struct.pack('<IIII', self.ver, self.strsz, len(self.crc), len(self.names))
        for h in sorted(self.crc):
            out += struct.pack('<II', h, self.crc[h])
        for nm in sorted(self.names):
            b = nm.encode()
            out += b + bytes(self.strsz - len(b)) + struct.pack('<I', self.names[nm])
        return bytes(out)


def needed(path, decompressed_size, template_overhead=None):
    ext = '.' + path.rsplit('.', 1)[-1]
    ov = template_overhead if template_overhead is not None else OVERHEAD.get(ext, 8192)
    return (decompressed_size + ov + 0xFF) & ~0xFF
