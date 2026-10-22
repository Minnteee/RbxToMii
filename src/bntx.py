"""Minimal BNTX (Switch texture container) reader -> PIL images (first mip, first array layer)."""
import struct
from PIL import Image
from ugctex import deswizzle

# format id (upper byte) -> (block w, block h, bytes per block, pillow decoder spec)
FMT = {
    0x02: (1, 1, 1, ('raw', 'L')),          # R8
    0x07: (1, 1, 2, None),                   # R5G6B5 (unused here)
    0x09: (1, 1, 2, ('raw', 'LA')),          # R8G8
    0x0B: (1, 1, 4, ('raw', 'RGBA')),        # R8G8B8A8
    0x0C: (1, 1, 4, ('raw', 'BGRA')),        # B8G8R8A8
    0x1A: (4, 4, 8, ('bcn', 1)),             # BC1
    0x1B: (4, 4, 16, ('bcn', 2)),            # BC2
    0x1C: (4, 4, 16, ('bcn', 3)),            # BC3
    0x1D: (4, 4, 8, ('bcn', 4)),             # BC4
    0x1E: (4, 4, 16, ('bcn', 5)),            # BC5
    0x1F: (4, 4, 16, ('bcn', 6)),            # BC6H
    0x20: (4, 4, 16, ('bcn', 7)),            # BC7
}
# ASTC block sizes (format id -> block w, h); decoded with texture2ddecoder
ASTC = {0x2D: (4, 4), 0x2E: (5, 4), 0x2F: (5, 5), 0x30: (6, 5), 0x31: (6, 6), 0x32: (8, 5), 0x33: (8, 6),
        0x34: (8, 8), 0x35: (10, 5), 0x36: (10, 6), 0x37: (10, 8), 0x38: (10, 10), 0x39: (12, 10), 0x3A: (12, 12)}
for _k, (_w, _h) in ASTC.items():
    FMT[_k] = (_w, _h, 16, ('astc', (_w, _h)))


def read(data):
    assert data[:4] == b'BNTX', data[:4]
    nx = data.find(b'NX  ')
    count = struct.unpack_from('<I', data, nx + 4)[0]
    info_arr = struct.unpack_from('<Q', data, nx + 8)[0]
    out = []
    for i in range(count):
        bi = struct.unpack_from('<Q', data, info_arr + 8 * i)[0]
        assert data[bi:bi + 4] == b'BRTI'
        tile_mode = struct.unpack_from('<H', data, bi + 0x12)[0]
        mips = struct.unpack_from('<H', data, bi + 0x16)[0]
        fmt = struct.unpack_from('<I', data, bi + 0x1C)[0]
        w, h, depth, layers = struct.unpack_from('<IIII', data, bi + 0x24)
        layout = struct.unpack_from('<I', data, bi + 0x34)[0]
        img_size = struct.unpack_from('<I', data, bi + 0x50)[0]
        name_addr = struct.unpack_from('<Q', data, bi + 0x60)[0]
        ptrs = struct.unpack_from('<Q', data, bi + 0x70)[0]
        nlen = struct.unpack_from('<H', data, name_addr)[0]
        name = data[name_addr + 2:name_addr + 2 + nlen].decode('utf-8', 'replace')
        mip0 = struct.unpack_from('<Q', data, ptrs)[0]
        ftype = fmt >> 8
        bw, bh_, bpp, dec = FMT.get(ftype, (None, None, None, None))
        rec = {'name': name, 'w': w, 'h': h, 'format': hex(fmt), 'mips': mips, 'layers': layers,
               'tile': tile_mode, 'bh_log2': layout & 7, 'size': img_size}
        if dec is None:
            rec['image'] = None
            out.append(rec)
            continue
        wb, hb = (w + bw - 1) // bw, (h + bh_ - 1) // bh_
        raw = data[mip0:mip0 + img_size]
        if tile_mode == 0:  # block linear
            lin = deswizzle(raw, wb, hb, bpp, 1 << (layout & 7))
        else:
            lin = raw[:wb * hb * bpp]
        if dec[0] == 'astc':
            import texture2ddecoder
            bgra = texture2ddecoder.decode_astc(lin, wb * bw, hb * bh_, bw, bh_)
            im = Image.frombytes('RGBA', (wb * bw, hb * bh_), bgra, 'raw', 'BGRA').crop((0, 0, w, h))
        elif dec[0] == 'bcn':
            mode = {4: 'L', 5: 'RGB'}.get(dec[1], 'RGBA')
            try:
                im = Image.frombytes(mode, (wb * 4, hb * 4), lin, 'bcn', dec[1]).crop((0, 0, w, h)).convert('RGBA')
            except Exception:
                im = Image.new('RGBA', (w, h), (128, 128, 255, 255))
        else:
            im = Image.frombytes(dec[1] if dec[1] in ('L', 'LA', 'RGBA') else 'RGBA', (w, h), lin, 'raw', dec[1])
        rec['image'] = im
        out.append(rec)
    return out
