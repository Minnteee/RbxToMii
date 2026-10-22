"""Skeletal animations (FSKA) from a BFRES v10 file (Tomodachi Life: Living the Dream, Model/Animation*.bfres.zs).
Parses base values and curves and evaluates local bone transforms at a frame."""
import struct
import numpy as np


class Anims:
    def __init__(self, data):
        self.d = d = bytes(data)
        q = self.q
        n = struct.unpack_from('<i', d, q(0x60) + 4)[0]  # entries of the skeletal-animation dictionary
        arr = q(0x58)
        self.list = [self._anim(arr + i * 0x50) for i in range(n)]
        self.by_name = {a['name']: a for a in self.list}

    def q(self, o):
        return struct.unpack_from('<Q', self.d, o)[0]

    def s(self, o):
        if not o:
            return ''
        n = struct.unpack_from('<H', self.d, o)[0]
        return self.d[o + 2:o + 2 + n].decode('utf-8', 'replace')

    def _anim(self, o):
        d, q = self.d, self.q
        flags = struct.unpack_from('<I', d, o + 4)[0]
        frames, ncurve, baked, nbone, nud = struct.unpack_from('<iiIHH', d, o + 0x40)
        a = {'off': o, 'name': self.s(q(o + 8)), 'flags': flags, 'frames': frames, 'loop': bool(flags & 4),
             'euler': bool(flags & 0x1000), 'nbone': nbone, 'bone_arr': q(o + 0x28), 'bones': None}
        return a

    def bones(self, a):
        if a['bones'] is None:
            d, q = self.d, self.q
            out = []
            for i in range(a['nbone']):
                b = a['bone_arr'] + i * 0x38
                name = self.s(q(b))
                curves_off, base_off = q(b + 8), q(b + 0x10)
                flags = struct.unpack_from('<I', d, b + 0x28)[0]
                ncurve = d[b + 0x2E]
                base = {}
                p = base_off
                if flags & 0x8:
                    base['s'] = np.array(struct.unpack_from('<3f', d, p)); p += 12
                if flags & 0x10:
                    base['r'] = np.array(struct.unpack_from('<4f', d, p)); p += 16
                if flags & 0x20:
                    base['t'] = np.array(struct.unpack_from('<3f', d, p)); p += 12
                curves = [self._curve(curves_off + c * 0x30) for c in range(ncurve)]
                out.append({'name': name, 'flags': flags, 'base': base, 'curves': curves})
            a['bones'] = out
        return a['bones']

    def _curve(self, o):
        """v10 curves: frames (f32 / s10.5 / u8), keys = 8-byte header (tangent scale, tangent offset as f32)
        followed by n x (value, tangent) as unsigned u16/u8 (or f32); value = offset + scale * u"""
        d, q = self.d, self.q
        fr_off, key_off = q(o), q(o + 8)
        cflags, nkey, target = struct.unpack_from('<HHI', d, o + 0x10)
        start, end, _unk, scale, offset = struct.unpack_from('<5f', d, o + 0x18)  # v10 layout: scale @0x24, offset @0x28
        ftype, ktype, ctype = cflags & 3, (cflags >> 2) & 3, (cflags >> 4) & 7
        if ftype == 0:
            frames = np.array(struct.unpack_from('<%df' % nkey, d, fr_off))
        elif ftype == 1:
            frames = np.array(struct.unpack_from('<%dh' % nkey, d, fr_off)) / 32.0
        else:
            frames = np.array(struct.unpack_from('<%dB' % nkey, d, fr_off), float)
        if ktype == 0:
            tsc, toff = 1.0, 0.0
            raw = np.array(struct.unpack_from('<%df' % (2 * nkey), d, key_off), float).reshape(nkey, 2)
            vals, tans = raw[:, 0], raw[:, 1]
        else:
            tsc, toff = struct.unpack_from('<2f', d, key_off)
            fmt = {1: 'H', 2: 'B'}[ktype]
            raw = np.array(struct.unpack_from('<%d%s' % (2 * nkey, fmt), d, key_off + 8), float).reshape(nkey, 2)
            vals = offset + scale * raw[:, 0]
            tans = toff + tsc * raw[:, 1]
        return {'target': target, 'ctype': ctype, 'ktype': ktype, 'frames': frames, 'values': vals, 'tangents': tans,
                'start': start, 'end': end, 'flags': cflags}

    @staticmethod
    def eval_curve(c, f):
        """linear between key values (tangents are not needed for poses)"""
        fr, v = c['frames'], c['values']
        if len(fr) == 1 or f <= fr[0]:
            return float(v[0])
        if f >= fr[-1]:
            return float(v[-1])
        i = int(np.searchsorted(fr, f, side='right') - 1)
        span = fr[i + 1] - fr[i]
        t = (f - fr[i]) / span if span > 0 else 1.0
        return float(v[i] + (v[i + 1] - v[i]) * t)

    def local(self, a, f, rest=None, curves=True):
        """{bone name: (scale(3), rot(4: euler xyz + pad, or quat), trans(3))} at frame f.
        rest {bone name: (s, r, t)}: values for components the animation does not set (the model's own)"""
        out = {}
        for b in self.bones(a):
            fl = b['flags']
            rs = rest.get(b['name']) if rest else None
            s = np.array(rs[0], float) if rs is not None else np.ones(3)
            r = np.array(rs[1], float) if rs is not None else np.array([0, 0, 0, 1.0])
            t = np.array(rs[2], float) if rs is not None else np.zeros(3)
            if fl & (1 << 24) and fl & (1 << 25):   # scale one
                s = np.ones(3)
            if fl & (1 << 26):                      # rotate zero
                r = np.array([0, 0, 0, 0.0 if a['euler'] else 1.0])
            if fl & (1 << 27):                      # translate zero
                t = np.zeros(3)
            if 's' in b['base']:
                s = b['base']['s'].copy()
            if 'r' in b['base']:
                r = b['base']['r'].copy()
            if 't' in b['base']:
                t = b['base']['t'].copy()
            for c in (b['curves'] if curves else ()):
                v = self.eval_curve(c, f)
                tg = c['target']
                # target = offset in the g3d bone result: flags, scale (0x04), translate (0x10), pad, rotate (0x20)
                if 0x4 <= tg < 0x10:
                    s[(tg - 0x4) // 4] = v
                elif 0x10 <= tg < 0x1C:
                    t[(tg - 0x10) // 4] = v
                elif 0x20 <= tg < 0x30:
                    r[(tg - 0x20) // 4] = v
            out[b['name']] = (s, r, t)
        return out
