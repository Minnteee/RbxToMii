"""Tomodachi Life: Living the Dream UGC textures (Ugc*.ugctex / _Thumb.ugctex / .canvas, all zstd).
BC1 / BC3 block compression and Tegra X1 block-linear swizzle (numpy)."""
import numpy as np

# ---------------- Tegra X1 block linear ----------------

def _pow2_ceil(v):
    p = 1
    while p < v:
        p <<= 1
    return p


def block_height(h_blocks):
    bh = _pow2_ceil(max(1, (h_blocks + 7) // 8))
    return min(bh, 16)


def swizzle_offsets(w, h, bpp, bh=None):
    """Byte offset in the swizzled buffer for every element (row-major w*h)."""
    if bh is None:
        bh = block_height(h)
    gobs_w = (w * bpp + 63) // 64
    y = np.repeat(np.arange(h), w)
    x = np.tile(np.arange(w), h) * bpp
    gob = ((y // (8 * bh)) * 512 * bh * gobs_w + (x // 64) * 512 * bh + ((y % (8 * bh)) // 8) * 512)
    return gob + ((x % 64) // 32) * 256 + ((y % 8) // 2) * 64 + ((x % 32) // 16) * 32 + (y % 2) * 16 + (x % 16)


def swizzled_size(w, h, bpp, bh=None):
    if bh is None:
        bh = block_height(h)
    gobs_w = (w * bpp + 63) // 64
    rows = ((h + 8 * bh - 1) // (8 * bh)) * 8 * bh
    return gobs_w * 64 * rows


def deswizzle(data, w, h, bpp, bh=None):
    offs = swizzle_offsets(w, h, bpp, bh)
    src = np.frombuffer(data, np.uint8)
    idx = offs[:, None] + np.arange(bpp)[None, :]
    return src[idx].reshape(h, w * bpp).tobytes()


def swizzle(data, w, h, bpp, bh=None, size=None):
    offs = swizzle_offsets(w, h, bpp, bh)
    out = np.zeros(size or swizzled_size(w, h, bpp, bh), np.uint8)
    src = np.frombuffer(data, np.uint8).reshape(w * h, bpp)
    idx = offs[:, None] + np.arange(bpp)[None, :]
    out[idx] = src
    return out.tobytes()


# ---------------- BC1 / BC3 decode ----------------

def _565(c):
    r = ((c >> 11) & 31).astype(np.int32)
    g = ((c >> 5) & 63).astype(np.int32)
    b = (c & 31).astype(np.int32)
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1)


def _bc1_colors(blk):
    c0 = blk[:, 0].astype(np.uint32) | (blk[:, 1].astype(np.uint32) << 8)
    c1 = blk[:, 2].astype(np.uint32) | (blk[:, 3].astype(np.uint32) << 8)
    p0, p1 = _565(c0), _565(c1)
    four = (c0 > c1)[:, None]
    p2 = np.where(four, (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(four, (p0 + 2 * p1) // 3, 0)
    pal = np.stack([p0, p1, p2, p3], 1)  # n,4,3
    alpha = np.stack([np.full(len(blk), 255), np.full(len(blk), 255), np.full(len(blk), 255),
                      np.where(four[:, 0], 255, 0)], 1)
    idx = blk[:, 4:8].astype(np.uint32)
    bits = idx[:, 0] | (idx[:, 1] << 8) | (idx[:, 2] << 16) | (idx[:, 3] << 24)
    sel = (bits[:, None] >> (2 * np.arange(16))[None, :]) & 3  # n,16
    rgb = np.take_along_axis(pal, sel[:, :, None].repeat(3, 2), 1)
    a = np.take_along_axis(alpha, sel, 1)
    return rgb, a


def _blocks_to_image(px, wb, hb):
    # px: (wb*hb, 16, 4) -> (hb*4, wb*4, 4)
    return px.reshape(hb, wb, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(hb * 4, wb * 4, 4)


def decode_bc1(data, w, h):
    wb, hb = w // 4, h // 4
    blk = np.frombuffer(data, np.uint8)[:wb * hb * 8].reshape(-1, 8)
    rgb, a = _bc1_colors(blk)
    px = np.concatenate([rgb, a[:, :, None]], 2).astype(np.uint8)
    return _blocks_to_image(px, wb, hb)


def decode_bc3(data, w, h):
    wb, hb = w // 4, h // 4
    blk = np.frombuffer(data, np.uint8)[:wb * hb * 16].reshape(-1, 16)
    a0 = blk[:, 0].astype(np.int32)
    a1 = blk[:, 1].astype(np.int32)
    i = np.arange(8)
    six = (a0 > a1)[:, None]
    pal8 = np.where(six,
                    np.where(i == 0, a0[:, None], np.where(i == 1, a1[:, None],
                             ((8 - i) * a0[:, None] + (i - 1) * a1[:, None]) // 7)),
                    np.where(i == 0, a0[:, None], np.where(i == 1, a1[:, None],
                             np.where(i == 6, 0, np.where(i == 7, 255,
                                      ((6 - i) * a0[:, None] + (i - 1) * a1[:, None]) // 5)))))
    ab = np.zeros(len(blk), np.uint64)
    for k in range(6):
        ab |= blk[:, 2 + k].astype(np.uint64) << np.uint64(8 * k)
    asel = ((ab[:, None] >> (np.uint64(3) * np.arange(16, dtype=np.uint64))[None, :]) & np.uint64(7)).astype(np.int64)
    alpha = np.take_along_axis(pal8, asel, 1)
    cb = blk[:, 8:16].copy()
    # BC3 colour block is always 4-colour mode
    c0 = cb[:, 0].astype(np.uint32) | (cb[:, 1].astype(np.uint32) << 8)
    c1 = cb[:, 2].astype(np.uint32) | (cb[:, 3].astype(np.uint32) << 8)
    p0, p1 = _565(c0), _565(c1)
    pal = np.stack([p0, p1, (2 * p0 + p1) // 3, (p0 + 2 * p1) // 3], 1)
    idx = cb[:, 4:8].astype(np.uint32)
    bits = idx[:, 0] | (idx[:, 1] << 8) | (idx[:, 2] << 16) | (idx[:, 3] << 24)
    sel = (bits[:, None] >> (2 * np.arange(16))[None, :]) & 3
    rgb = np.take_along_axis(pal, sel[:, :, None].repeat(3, 2), 1)
    px = np.concatenate([rgb, alpha[:, :, None]], 2).astype(np.uint8)
    return _blocks_to_image(px, wb, hb)


# ---------------- BC1 / BC3 encode (range fit, good enough for UGC) ----------------

def _image_to_blocks(img):
    h, w = img.shape[:2]
    return img.reshape(h // 4, 4, w // 4, 4, img.shape[2]).transpose(0, 2, 1, 3, 4).reshape(-1, 16, img.shape[2])


def _to565(c):
    c = np.clip(np.rint(c), 0, 255).astype(np.uint32)
    return ((c[..., 0] >> 3) << 11) | ((c[..., 1] >> 2) << 5) | (c[..., 2] >> 3)


def _encode_color_blocks(rgb, transparent=None, four_only=False):
    """rgb: n,16,3 float. transparent: n,16 bool (BC1 punch-through). Returns n,8 uint8."""
    n = len(rgb)
    if transparent is None:
        transparent = np.zeros(rgb.shape[:2], bool)
    w = (~transparent).astype(np.float64)[:, :, None]
    cnt = np.maximum(w.sum(1), 1)
    mean = (rgb * w).sum(1) / cnt
    d = (rgb - mean[:, None, :]) * w
    cov = np.einsum('nki,nkj->nij', d, d)
    axis = np.ones((n, 3)) / np.sqrt(3)
    for _ in range(8):
        axis = np.einsum('nij,nj->ni', cov, axis)
        axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9)
    proj = np.einsum('nki,ni->nk', rgb - mean[:, None, :], axis)
    big = 1e9
    pmin = np.where(transparent, big, proj).min(1)
    pmax = np.where(transparent, -big, proj).max(1)
    pmin = np.where(pmin > pmax, 0, pmin)
    pmax = np.where(pmax < pmin, 0, pmax)
    e0 = mean + axis * pmax[:, None]
    e1 = mean + axis * pmin[:, None]
    c0 = _to565(e0)
    c1 = _to565(e1)
    has_t = transparent.any(1)
    if four_only:
        has_t[:] = False
    # 4-colour mode needs c0 > c1, 3-colour (+transparent) needs c0 <= c1
    swap = np.where(has_t, c0 > c1, c0 < c1)
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    eq = (c0 == c1) & ~has_t
    # equal endpoints in 4-colour mode: nudge so c0 > c1
    bump = eq & (c0 < 0xFFFF)
    c0 = np.where(bump, c0 + 1, c0)
    c1 = np.where(eq & ~bump, c1 - 1, c1)
    p0, p1 = _565(c0).astype(np.float64), _565(c1).astype(np.float64)
    four = (~has_t)[:, None, None]
    pal = np.stack([p0, p1,
                    np.where(four[:, 0], (2 * p0 + p1) / 3, (p0 + p1) / 2),
                    np.where(four[:, 0], (p0 + 2 * p1) / 3, 1e6)], 1)
    dist = ((rgb[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)
    sel = dist.argmin(2)
    sel = np.where(transparent, 3, sel)
    bits = (sel.astype(np.uint32) << (2 * np.arange(16, dtype=np.uint32))[None, :]).sum(1).astype(np.uint32)
    out = np.zeros((n, 8), np.uint8)
    out[:, 0] = c0 & 255
    out[:, 1] = c0 >> 8
    out[:, 2] = c1 & 255
    out[:, 3] = c1 >> 8
    for k in range(4):
        out[:, 4 + k] = (bits >> (8 * k)) & 255
    return out


def encode_bc1(img, alpha_cut=128):
    """img: h,w,4 uint8 -> BC1 bytes (linear block order). Pixels with alpha < alpha_cut become transparent."""
    blocks = _image_to_blocks(img.astype(np.float64))
    transparent = blocks[:, :, 3] < alpha_cut
    return _encode_color_blocks(blocks[:, :, :3], transparent).tobytes()


def encode_bc3(img):
    blocks = _image_to_blocks(img.astype(np.float64))
    a = blocks[:, :, 3]
    n = len(blocks)
    a0 = np.clip(np.rint(a.max(1)), 0, 255).astype(np.int64)
    a1 = np.clip(np.rint(a.min(1)), 0, 255).astype(np.int64)
    same = a0 == a1
    a0 = np.where(same & (a0 < 255), a0 + 1, a0)
    a1 = np.where(same & (a0 == 255) & (a1 == 255), 254, a1)
    i = np.arange(8)
    pal8 = np.where(i == 0, a0[:, None], np.where(i == 1, a1[:, None],
                    ((8 - i) * a0[:, None] + (i - 1) * a1[:, None]) / 7.0))
    asel = np.abs(a[:, :, None] - pal8[:, None, :]).argmin(2).astype(np.uint64)
    ab = (asel << (np.uint64(3) * np.arange(16, dtype=np.uint64))[None, :]).sum(1).astype(np.uint64)
    out = np.zeros((n, 16), np.uint8)
    out[:, 0] = a0
    out[:, 1] = a1
    for k in range(6):
        out[:, 2 + k] = ((ab >> np.uint64(8 * k)) & np.uint64(255)).astype(np.uint8)
    out[:, 8:] = _encode_color_blocks(blocks[:, :, :3], None, four_only=True)
    return out.tobytes()


# ---------------- high level ----------------

def read_tex(data, w, h, fmt, swz=True):
    bpp = 8 if fmt == 'bc1' else 16
    wb, hb = w // 4, h // 4
    lin = deswizzle(data, wb, hb, bpp) if swz else data
    return decode_bc1(lin, w, h) if fmt == 'bc1' else decode_bc3(lin, w, h)


def write_tex(img, fmt, swz=True, size=None):
    h, w = img.shape[:2]
    lin = encode_bc1(img) if fmt == 'bc1' else encode_bc3(img)
    if not swz:
        return lin
    bpp = 8 if fmt == 'bc1' else 16
    return swizzle(lin, w // 4, h // 4, bpp, size=size)
