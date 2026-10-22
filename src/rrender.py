"""Tiny software rasterizer for previews (orthographic, z-buffer, textured, lambert shaded).
render(pieces, yaw_deg, size) where each piece has .pos (N,3) .uv (N,2) .faces (M,3) and .tex (HxWx4 uint8 array) or .color"""
import numpy as np
from PIL import Image


def _rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def render(pieces, yaw=0.0, size=512, bg=(190, 190, 190), up=1, look=(0, 0, 1), bounds=None, pitch=0.0, alpha=False,
           light=(0.35, 0.6, -0.72), ambient=0.45, cull=False):
    """look: world direction the camera looks along (default +Z, i.e. camera stands at -Z looking at +Z).
    yaw rotates the model around Y before projecting."""
    R = _rot_y(yaw)
    if pitch:
        a = np.radians(pitch)
        Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        R = Rx @ R
    look = np.asarray(look, float)
    look /= np.linalg.norm(look)
    upv = np.array([0, 1.0, 0])
    right = np.cross(look, upv)
    right /= np.linalg.norm(right)
    upv = np.cross(right, look)
    allp = np.concatenate([(p.pos @ R.T) for p in pieces if len(p.pos)])
    px, py = allp @ right, allp @ upv
    if bounds is None:
        x0, x1, y0, y1 = px.min(), px.max(), py.min(), py.max()
    else:
        x0, x1, y0, y1 = bounds
    span = max(x1 - x0, y1 - y0) * 1.08
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    scale = size / span
    img = np.zeros((size, size, 3), np.float32)
    img[:] = np.array(bg, np.float32) / 255.0
    zb = np.full((size, size), np.inf, np.float32)
    light = np.array(light, float)
    light /= np.linalg.norm(light)
    for p in pieces:
        if not len(p.faces):
            continue
        P = p.pos @ R.T
        sx = (P @ right - cx) * scale + size / 2
        sy = size / 2 - (P @ upv - cy) * scale
        sz = P @ look
        tex = getattr(p, 'tex', None)
        if tex is not None:
            th, tw = tex.shape[:2]
            texf = tex.astype(np.float32) / 255.0
        col = np.array(getattr(p, 'color', (200, 200, 200))[:3], np.float32) / 255.0
        F = p.faces
        a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
        n = np.cross(b - a, c - a)
        nl = np.linalg.norm(n, axis=1)
        n = n / np.maximum(nl, 1e-12)[:, None]
        shade = ambient + (1 - ambient) * np.abs(n @ light)
        front = (n @ look) < 0      # CCW normal towards the camera (the game culls the other side)
        for fi in range(len(F)):
            if cull and not front[fi]:
                continue
            i0, i1, i2 = F[fi]
            xs = np.array([sx[i0], sx[i1], sx[i2]])
            ys = np.array([sy[i0], sy[i1], sy[i2]])
            bx0, bx1 = int(max(0, np.floor(xs.min()))), int(min(size - 1, np.ceil(xs.max())))
            by0, by1 = int(max(0, np.floor(ys.min()))), int(min(size - 1, np.ceil(ys.max())))
            if bx1 < bx0 or by1 < by0:
                continue
            gx, gy = np.meshgrid(np.arange(bx0, bx1 + 1) + 0.5, np.arange(by0, by1 + 1) + 0.5)
            d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(d) < 1e-12:
                continue
            w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
            w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
            w2 = 1 - w0 - w1
            m = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not m.any():
                continue
            z = w0 * sz[i0] + w1 * sz[i1] + w2 * sz[i2]
            sub = zb[by0:by1 + 1, bx0:bx1 + 1]
            m &= z < sub
            if not m.any():
                continue
            if tex is not None:
                u = w0 * p.uv[i0, 0] + w1 * p.uv[i1, 0] + w2 * p.uv[i2, 0]
                v = w0 * p.uv[i0, 1] + w1 * p.uv[i1, 1] + w2 * p.uv[i2, 1]
                tx = np.clip((np.mod(u, 1.0) * tw).astype(int), 0, tw - 1)
                ty = np.clip((np.mod(v, 1.0) * th).astype(int), 0, th - 1)
                texel = texf[ty, tx]
                if texel.shape[-1] == 4:
                    m &= texel[..., 3] > 0.35
                    rgb = texel[..., :3]
                else:
                    rgb = texel
            else:
                rgb = np.broadcast_to(col, m.shape + (3,))
            sub[m] = z[m]
            img[by0:by1 + 1, bx0:bx1 + 1][m] = rgb[m] * shade[fi]
    rgb = np.clip(img * 255, 0, 255).astype(np.uint8)
    if alpha:
        a = np.where(np.isfinite(zb), 255, 0).astype(np.uint8)
        return Image.fromarray(np.dstack([rgb, a]), 'RGBA')
    return Image.fromarray(rgb)


def sheet(images, labels=None):
    w = sum(i.size[0] for i in images)
    h = max(i.size[1] for i in images)
    out = Image.new('RGB', (w, h), (255, 255, 255))
    x = 0
    for i in images:
        out.paste(i, (x, 0))
        x += i.size[0]
    return out
