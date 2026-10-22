#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parser.py - a general reader for Roblox *binary* model / place files (.rbxl / .rbxm).

File layout
-----------
    "<roblox!" 89 FF 0D 0A 1A 0A | u16 version | i32 classCount | i32 instanceCount | 8 reserved bytes
    then chunks until END:
        4-byte name (META, SSTR, INST, PROP, PRNT, END, ...) | u32 compressedLen | u32 uncompressedLen
        | u32 reserved | payload
    compressedLen == 0  -> payload stored raw (uncompressedLen bytes)
    otherwise           -> LZ4 block, or a ZSTD frame when it starts with 28 B5 2F FD

Property values use the standard encodings: byte-interleaved big-endian arrays,
zig-zag integers, "rotated" floats (sign bit moved to bit 0), delta-accumulated
referents, the CFrame orientation-id table, etc.  Every PROP chunk is checked for
complete consumption, so an unknown/changed encoding is reported as a warning
instead of silently producing garbage.

Library use (load via importlib when running "python -I", which drops the script dir from sys.path):

    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("rbxl_parser", r"...\\parser.py")
    rbx = importlib.util.module_from_spec(spec); sys.modules["rbxl_parser"] = rbx
    spec.loader.exec_module(rbx)

    doc = rbx.read_file(r"C:\\path\\Place1.rbxl")
    ws = doc.find_service("Workspace")
    for inst in ws.get_descendants():
        print(inst.full_name(), inst.class_name, inst.props.get("Name"))

CLI:
    python -I parser.py FILE                 statistics (chunks, classes, warnings)
    python -I parser.py FILE --tree          print the instance tree
    python -I parser.py FILE --json OUT      dump every instance with every property to JSON
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import sys

import numpy as np

try:  # optional at import time, required for compressed chunks
    import lz4.block as _lz4block
except Exception:  # pragma: no cover
    _lz4block = None
try:
    import zstandard as _zstd
except Exception:  # pragma: no cover
    _zstd = None

__all__ = [
    "read_file", "read_bytes", "RobloxFile", "Instance", "RbxFormatError",
    "Color3", "Vector2", "Vector3", "CFrame", "EnumValue", "BrickColor", "Ref", "Content",
    "SharedString", "UDim", "UDim2", "Rect", "NumberRange", "NumberSequence", "ColorSequence",
    "Ray", "Faces", "Axes", "PhysicalProperties", "Font", "UniqueId", "Vector3int16",
    "decode_attributes", "decode_tags", "to_jsonable", "BRICKCOLORS", "TYPE_NAMES",
]

MAGIC = b"<roblox!"
SIGNATURE = b"\x89\xff\r\n\x1a\n"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


class RbxFormatError(Exception):
    pass


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _r(x, sig=7):
    """Round a float to `sig` significant digits for readable JSON (keeps ints/None)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return str(x)
    if x == 0.0:
        return 0.0
    return float(f"{x:.{sig}g}")


def _bytes_to_text(b: bytes):
    """Return str when the bytes are valid UTF-8, otherwise keep them as bytes."""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b


def _binary_summary(b: bytes, inline_limit=96):
    d = {"binary_length": len(b), "sha1": hashlib.sha1(b).hexdigest()}
    if len(b) <= inline_limit:
        d["base64"] = base64.b64encode(b).decode("ascii")
    else:
        d["head_hex"] = b[:32].hex()
    return d


# --------------------------------------------------------------------------------------
# value classes
# --------------------------------------------------------------------------------------
class Color3:
    """Color3 (floats 0..1, may exceed 1) or Color3uint8 (exact bytes)."""
    __slots__ = ("r", "g", "b", "uint8")

    def __init__(self, r, g, b, uint8=False):
        self.r, self.g, self.b, self.uint8 = r, g, b, uint8

    @property
    def rgb255(self):
        if self.uint8:
            return (int(self.r), int(self.g), int(self.b))
        return tuple(int(round(max(0.0, min(1.0, float(c))) * 255)) for c in (self.r, self.g, self.b))

    @property
    def floats(self):
        if self.uint8:
            return (self.r / 255.0, self.g / 255.0, self.b / 255.0)
        return (float(self.r), float(self.g), float(self.b))

    @property
    def hex(self):
        return "#%02X%02X%02X" % self.rgb255

    def to_json(self):
        d = {"rgb255": list(self.rgb255), "hex": self.hex}
        if not self.uint8:
            d["float"] = [_r(c) for c in (self.r, self.g, self.b)]
        return d

    def __repr__(self):
        return f"Color3({self.hex})"


class Vector2:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)

    def to_json(self):
        return [_r(self.x), _r(self.y)]

    def __repr__(self):
        return f"Vector2({self.x:g}, {self.y:g})"


class Vector3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def tuple(self):
        return (self.x, self.y, self.z)

    def to_json(self):
        return [_r(self.x), _r(self.y), _r(self.z)]

    def __repr__(self):
        return f"Vector3({self.x:g}, {self.y:g}, {self.z:g})"


class Vector3int16(Vector3):
    __slots__ = ()

    def to_json(self):
        return [int(self.x), int(self.y), int(self.z)]


class Vector2int16(Vector2):
    __slots__ = ()

    def to_json(self):
        return [int(self.x), int(self.y)]


_NORMALS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _special_rotation(rid):
    """Orientation id -> row-major 3x3 matrix.  id = 6*xNormalId + yNormalId + 1, where the
    right vector (column 0) is NormalId[x], the up vector (column 1) NormalId[y], column 2 = x cross y."""
    xi, yi = divmod(rid - 1, 6)
    if not (0 <= xi < 6 and 0 <= yi < 6) or xi % 3 == yi % 3:
        raise RbxFormatError(f"invalid CFrame orientation id 0x{rid:02x}")
    rx, ry = _NORMALS[xi], _NORMALS[yi]
    rz = _cross(rx, ry)
    return (rx[0], ry[0], rz[0], rx[1], ry[1], rz[1], rx[2], ry[2], rz[2])


class CFrame:
    """position (x, y, z) + row-major rotation R00 R01 R02 R10 R11 R12 R20 R21 R22."""
    __slots__ = ("pos", "rot", "orient_id")

    def __init__(self, pos, rot, orient_id=0):
        self.pos = tuple(float(v) for v in pos)
        self.rot = tuple(float(v) for v in rot)
        self.orient_id = orient_id

    def right_vector(self):
        r = self.rot
        return (r[0], r[3], r[6])

    def up_vector(self):
        r = self.rot
        return (r[1], r[4], r[7])

    def look_vector(self):
        r = self.rot
        return (-r[2], -r[5], -r[8])

    def point_to_world(self, p):
        r = self.rot
        x, y, z = p
        return (self.pos[0] + r[0] * x + r[1] * y + r[2] * z,
                self.pos[1] + r[3] * x + r[4] * y + r[5] * z,
                self.pos[2] + r[6] * x + r[7] * y + r[8] * z)

    def is_identity_rotation(self, eps=1e-5):
        ident = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        return all(abs(a - b) < eps for a, b in zip(self.rot, ident))

    def to_json(self):
        return {"position": [_r(v) for v in self.pos], "rotation": [_r(v, 6) for v in self.rot]}

    def __repr__(self):
        return "CFrame(pos=(%g, %g, %g))" % self.pos


class EnumValue(int):
    """An Enum token. The enum *type* is not stored in the file (it is implied by the property)."""

    def __repr__(self):
        return f"Enum({int(self)})"

    def to_json(self):
        return int(self)


class BrickColor:
    __slots__ = ("number",)

    def __init__(self, number):
        self.number = int(number)

    @property
    def name(self):
        e = BRICKCOLORS.get(self.number)
        return e[0] if e else None

    @property
    def rgb255(self):
        e = BRICKCOLORS.get(self.number)
        return e[1] if e else None

    def to_json(self):
        rgb = self.rgb255
        return {"number": self.number, "name": self.name,
                "rgb255": list(rgb) if rgb else None,
                "hex": ("#%02X%02X%02X" % rgb) if rgb else None}

    def __repr__(self):
        return f"BrickColor({self.number}, {self.name!r})"


class Ref:
    """Referent to another instance; `.instance` is filled in after the tree is built."""
    __slots__ = ("ref", "instance")

    def __init__(self, ref):
        self.ref = int(ref)
        self.instance = None

    def to_json(self):
        if self.ref < 0:
            return None
        if self.instance is None:
            return {"ref": self.ref, "path": None}
        return {"ref": self.ref, "path": self.instance.full_name(), "class": self.instance.class_name}

    def __repr__(self):
        return f"Ref({self.ref})"


class Content:
    """The 2024+ Content datatype (type 0x22): None, a URI, or an Object reference."""
    __slots__ = ("kind", "uri", "ref")

    def __init__(self, kind, uri=None, ref=None):
        self.kind, self.uri, self.ref = kind, uri, ref

    def to_json(self):
        if self.kind == "None":
            return {"content": "None"}
        if self.kind == "Uri":
            return {"content": "Uri", "uri": self.uri}
        return {"content": self.kind, "object": self.ref.to_json() if isinstance(self.ref, Ref) else self.ref}

    def __repr__(self):
        return f"Content({self.kind}, {self.uri or self.ref!r})"


class SharedString:
    __slots__ = ("index", "md5", "data")

    def __init__(self, index, md5, data):
        self.index, self.md5, self.data = index, md5, data

    def to_json(self):
        d = {"shared_string_index": self.index}
        if self.data is not None:
            d.update(_binary_summary(self.data))
        return d

    def __repr__(self):
        return f"SharedString(#{self.index}, {len(self.data) if self.data is not None else '?'} bytes)"


class UDim:
    __slots__ = ("scale", "offset")

    def __init__(self, scale, offset):
        self.scale, self.offset = float(scale), int(offset)

    def to_json(self):
        return {"scale": _r(self.scale), "offset": self.offset}


class UDim2:
    __slots__ = ("x", "y")

    def __init__(self, x: UDim, y: UDim):
        self.x, self.y = x, y

    def to_json(self):
        return {"x": self.x.to_json(), "y": self.y.to_json()}


class Rect:
    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = map(float, (x0, y0, x1, y1))

    def to_json(self):
        return {"min": [_r(self.x0), _r(self.y0)], "max": [_r(self.x1), _r(self.y1)]}


class NumberRange:
    __slots__ = ("min", "max")

    def __init__(self, lo, hi):
        self.min, self.max = float(lo), float(hi)

    def to_json(self):
        return {"min": _r(self.min), "max": _r(self.max)}


class NumberSequence:
    __slots__ = ("keypoints",)  # (time, value, envelope)

    def __init__(self, kps):
        self.keypoints = kps

    def to_json(self):
        return [{"time": _r(t), "value": _r(v), "envelope": _r(e)} for t, v, e in self.keypoints]


class ColorSequence:
    __slots__ = ("keypoints",)  # (time, Color3, envelope)

    def __init__(self, kps):
        self.keypoints = kps

    def to_json(self):
        return [{"time": _r(t), "color": c.to_json(), "envelope": _r(e)} for t, c, e in self.keypoints]


class Ray:
    __slots__ = ("origin", "direction")

    def __init__(self, o, d):
        self.origin, self.direction = o, d

    def to_json(self):
        return {"origin": [_r(v) for v in self.origin], "direction": [_r(v) for v in self.direction]}


class Faces:
    __slots__ = ("bits",)
    NAMES = ["Right", "Top", "Back", "Left", "Bottom", "Front"]

    def __init__(self, bits):
        self.bits = int(bits)

    def to_json(self):
        return [n for i, n in enumerate(self.NAMES) if self.bits & (1 << i)]


class Axes:
    __slots__ = ("bits",)
    NAMES = ["X", "Y", "Z"]

    def __init__(self, bits):
        self.bits = int(bits)

    def to_json(self):
        return [n for i, n in enumerate(self.NAMES) if self.bits & (1 << i)]


class PhysicalProperties:
    __slots__ = ("custom", "density", "friction", "elasticity", "friction_weight",
                 "elasticity_weight", "acoustic_absorption")

    def __init__(self, custom, density=None, friction=None, elasticity=None,
                 friction_weight=None, elasticity_weight=None, acoustic_absorption=None):
        self.custom = custom
        self.density, self.friction, self.elasticity = density, friction, elasticity
        self.friction_weight, self.elasticity_weight = friction_weight, elasticity_weight
        self.acoustic_absorption = acoustic_absorption

    def to_json(self):
        if not self.custom:
            return {"custom": False}
        d = {"custom": True, "density": _r(self.density), "friction": _r(self.friction),
             "elasticity": _r(self.elasticity), "friction_weight": _r(self.friction_weight),
             "elasticity_weight": _r(self.elasticity_weight)}
        if self.acoustic_absorption is not None:
            d["acoustic_absorption"] = _r(self.acoustic_absorption)
        return d


class Font:
    __slots__ = ("family", "weight", "style", "cached_face_id")

    def __init__(self, family, weight, style, cached):
        self.family, self.weight, self.style, self.cached_face_id = family, weight, style, cached

    def to_json(self):
        return {"family": self.family, "weight": self.weight, "style": self.style,
                "cached_face_id": self.cached_face_id}


class UniqueId:
    __slots__ = ("raw",)

    def __init__(self, raw: bytes):
        self.raw = raw

    def to_json(self):
        return self.raw.hex()


class OptionalCFrame:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def to_json(self):
        return None if self.value is None else self.value.to_json()


class SecurityCapabilities(int):
    def to_json(self):
        return int(self)


class RawValue:
    """Undecodable property payload (unknown type id)."""
    __slots__ = ("type_id", "data")

    def __init__(self, type_id, data):
        self.type_id, self.data = type_id, data

    def to_json(self):
        d = {"undecoded_type_id": self.type_id}
        d.update(_binary_summary(self.data))
        return d


TYPE_NAMES = {
    0x01: "String", 0x02: "Bool", 0x03: "Int32", 0x04: "Float", 0x05: "Double", 0x06: "UDim",
    0x07: "UDim2", 0x08: "Ray", 0x09: "Faces", 0x0A: "Axes", 0x0B: "BrickColor", 0x0C: "Color3",
    0x0D: "Vector2", 0x0E: "Vector3", 0x0F: "Vector2int16", 0x10: "CFrame", 0x11: "Quaternion",
    0x12: "Enum", 0x13: "Referent", 0x14: "Vector3int16", 0x15: "NumberSequence",
    0x16: "ColorSequence", 0x17: "NumberRange", 0x18: "Rect", 0x19: "PhysicalProperties",
    0x1A: "Color3uint8", 0x1B: "Int64", 0x1C: "SharedString", 0x1D: "ProtectedString",
    0x1E: "OptionalCFrame", 0x1F: "UniqueId", 0x20: "Font", 0x21: "SecurityCapabilities",
    0x22: "Content",
}


# --------------------------------------------------------------------------------------
# low level reader
# --------------------------------------------------------------------------------------
class _Reader:
    __slots__ = ("b", "p", "n")

    def __init__(self, b: bytes):
        self.b = b
        self.p = 0
        self.n = len(b)

    def remaining(self):
        return self.n - self.p

    def take(self, k):
        if k < 0 or self.p + k > self.n:
            raise RbxFormatError(f"read of {k} bytes at offset {self.p} runs past end ({self.n})")
        s = self.b[self.p:self.p + k]
        self.p += k
        return s

    def u8(self):
        return self.take(1)[0]

    def u16(self):
        return struct.unpack("<H", self.take(2))[0]

    def i16(self):
        return struct.unpack("<h", self.take(2))[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def i32(self):
        return struct.unpack("<i", self.take(4))[0]

    def f32(self):
        return struct.unpack("<f", self.take(4))[0]

    def f64(self):
        return struct.unpack("<d", self.take(8))[0]

    def string(self) -> bytes:
        return self.take(self.u32())

    # interleaved arrays ---------------------------------------------------------
    def interleaved(self, count, width):
        """Byte-interleaved array -> (count, width) uint8, each row big-endian."""
        if count == 0:
            return np.zeros((0, width), dtype=np.uint8)
        raw = self.take(count * width)
        a = np.frombuffer(raw, dtype=np.uint8).reshape(width, count)
        return np.ascontiguousarray(a.T)

    def arr_u32(self, count):
        a = self.interleaved(count, 4)
        return a.view(">u4").reshape(count).astype(np.uint32)

    def arr_u64(self, count):
        a = self.interleaved(count, 8)
        return a.view(">u8").reshape(count).astype(np.uint64)

    def arr_i32(self, count):
        u = self.arr_u32(count).astype(np.int64)
        return (u >> 1) ^ -(u & 1)

    def arr_i64(self, count):
        u = self.arr_u64(count)
        hi = (u >> np.uint64(1)).astype(np.int64)
        lo = (u & np.uint64(1)).astype(np.int64)
        return hi ^ -lo

    def arr_f32(self, count):
        u = self.arr_u32(count)
        bits = ((u >> np.uint32(1)) | (u << np.uint32(31))).astype(np.uint32)
        return bits.view(np.float32)

    def arr_ref(self, count):
        return np.cumsum(self.arr_i32(count)).astype(np.int64)


# --------------------------------------------------------------------------------------
# property decoding
# --------------------------------------------------------------------------------------
def _read_cframes(r: _Reader, n):
    rots, ids = [], []
    for _ in range(n):
        rid = r.u8()
        ids.append(rid)
        if rid == 0:
            rots.append(struct.unpack("<9f", r.take(36)))
        else:
            rots.append(_special_rotation(rid))
    xs, ys, zs = r.arr_f32(n), r.arr_f32(n), r.arr_f32(n)
    return [CFrame((xs[i], ys[i], zs[i]), rots[i], ids[i]) for i in range(n)]


def _decode_values(r: _Reader, t, n, doc):
    if t == 0x01:  # String
        return [_bytes_to_text(r.string()) for _ in range(n)]
    if t == 0x02:  # Bool
        return [b != 0 for b in r.take(n)]
    if t == 0x03:  # Int32
        return [int(v) for v in r.arr_i32(n)]
    if t == 0x04:  # Float
        return [float(v) for v in r.arr_f32(n)]
    if t == 0x05:  # Double
        return list(struct.unpack(f"<{n}d", r.take(8 * n)))
    if t == 0x06:  # UDim
        s, o = r.arr_f32(n), r.arr_i32(n)
        return [UDim(s[i], o[i]) for i in range(n)]
    if t == 0x07:  # UDim2
        sx, sy, ox, oy = r.arr_f32(n), r.arr_f32(n), r.arr_i32(n), r.arr_i32(n)
        return [UDim2(UDim(sx[i], ox[i]), UDim(sy[i], oy[i])) for i in range(n)]
    if t == 0x08:  # Ray
        out = []
        for _ in range(n):
            v = struct.unpack("<6f", r.take(24))
            out.append(Ray(v[:3], v[3:]))
        return out
    if t == 0x09:
        return [Faces(b) for b in r.take(n)]
    if t == 0x0A:
        return [Axes(b) for b in r.take(n)]
    if t == 0x0B:  # BrickColor
        return [BrickColor(v) for v in r.arr_u32(n)]
    if t == 0x0C:  # Color3 (floats)
        cr, cg, cb = r.arr_f32(n), r.arr_f32(n), r.arr_f32(n)
        return [Color3(float(cr[i]), float(cg[i]), float(cb[i])) for i in range(n)]
    if t == 0x0D:  # Vector2
        x, y = r.arr_f32(n), r.arr_f32(n)
        return [Vector2(x[i], y[i]) for i in range(n)]
    if t == 0x0E:  # Vector3
        x, y, z = r.arr_f32(n), r.arr_f32(n), r.arr_f32(n)
        return [Vector3(x[i], y[i], z[i]) for i in range(n)]
    if t == 0x0F:  # Vector2int16 (rare)
        out = []
        for _ in range(n):
            x, y = struct.unpack("<hh", r.take(4))
            out.append(Vector2int16(x, y))
        return out
    if t == 0x10:  # CFrame
        return _read_cframes(r, n)
    if t == 0x12:  # Enum
        return [EnumValue(int(v)) for v in r.arr_u32(n)]
    if t == 0x13:  # Referent
        return [Ref(v) for v in r.arr_ref(n)]
    if t == 0x14:  # Vector3int16
        out = []
        for _ in range(n):
            x, y, z = struct.unpack("<hhh", r.take(6))
            out.append(Vector3int16(x, y, z))
        return out
    if t == 0x15:  # NumberSequence
        out = []
        for _ in range(n):
            k = r.u32()
            kps = [struct.unpack("<3f", r.take(12)) for _ in range(k)]
            out.append(NumberSequence(kps))
        return out
    if t == 0x16:  # ColorSequence
        out = []
        for _ in range(n):
            k = r.u32()
            kps = []
            for _ in range(k):
                tt, cr, cg, cb, env = struct.unpack("<5f", r.take(20))
                kps.append((tt, Color3(cr, cg, cb), env))
            out.append(ColorSequence(kps))
        return out
    if t == 0x17:  # NumberRange
        return [NumberRange(*struct.unpack("<2f", r.take(8))) for _ in range(n)]
    if t == 0x18:  # Rect
        a, b, c, d = r.arr_f32(n), r.arr_f32(n), r.arr_f32(n), r.arr_f32(n)
        return [Rect(a[i], b[i], c[i], d[i]) for i in range(n)]
    if t == 0x19:  # PhysicalProperties
        out = []
        for _ in range(n):
            flag = r.u8()
            if flag & 1:
                vals = struct.unpack("<5f", r.take(20))
                aa = r.f32() if flag & 2 else None
                out.append(PhysicalProperties(True, *vals, acoustic_absorption=aa))
            else:
                out.append(PhysicalProperties(False))
        return out
    if t == 0x1A:  # Color3uint8: all R, then all G, then all B
        cr, cg, cb = r.take(n), r.take(n), r.take(n)
        return [Color3(cr[i], cg[i], cb[i], uint8=True) for i in range(n)]
    if t == 0x1B:  # Int64
        return [int(v) for v in r.arr_i64(n)]
    if t == 0x1C:  # SharedString (indices into SSTR)
        out = []
        for idx in r.arr_u32(n):
            idx = int(idx)
            if 0 <= idx < len(doc.shared_strings):
                md5, data = doc.shared_strings[idx]
                out.append(SharedString(idx, md5, data))
            else:
                out.append(SharedString(idx, None, None))
        return out
    if t == 0x1D:  # ProtectedString / bytecode
        return [_bytes_to_text(r.string()) for _ in range(n)]
    if t == 0x1E:  # OptionalCFrame: type byte 0x10, CFrame array, type byte 0x02, Bool array
        tag1 = r.u8()
        if tag1 != 0x10:
            raise RbxFormatError(f"OptionalCFrame: expected CFrame tag 0x10, got 0x{tag1:02x}")
        cfs = _read_cframes(r, n)
        tag2 = r.u8()
        if tag2 != 0x02:
            raise RbxFormatError(f"OptionalCFrame: expected Bool tag 0x02, got 0x{tag2:02x}")
        flags = r.take(n)
        return [OptionalCFrame(cfs[i] if flags[i] else None) for i in range(n)]
    if t == 0x1F:  # UniqueId (16-byte interleaved)
        a = r.interleaved(n, 16)
        return [UniqueId(bytes(a[i])) for i in range(n)]
    if t == 0x20:  # Font
        out = []
        for _ in range(n):
            fam = _bytes_to_text(r.string())
            weight = r.u16()
            style = r.u8()
            cached = _bytes_to_text(r.string())
            out.append(Font(fam, weight, style, cached))
        return out
    if t == 0x21:  # SecurityCapabilities (interleaved 64-bit)
        return [SecurityCapabilities(int(v)) for v in r.arr_u64(n)]
    if t == 0x22:  # Content
        kinds = r.arr_i32(n)
        uri_count = r.u32()
        uris = [_bytes_to_text(r.string()) for _ in range(uri_count)]
        obj_count = r.u32()
        objs = r.arr_ref(obj_count)
        ext_count = r.u32()
        ext = r.arr_ref(ext_count)
        out, ui, oi, ei = [], 0, 0, 0
        for k in kinds:
            k = int(k)
            if k == 0:
                out.append(Content("None"))
            elif k == 1:
                out.append(Content("Uri", uri=uris[ui] if ui < len(uris) else None))
                ui += 1
            elif k == 2:
                out.append(Content("Object", ref=Ref(objs[oi]) if oi < len(objs) else None))
                oi += 1
            else:
                out.append(Content(f"Kind{k}", ref=Ref(ext[ei]) if ei < len(ext) else None))
                ei += 1
        if ui != uri_count or oi != obj_count:
            doc.warnings.append(f"Content: used {ui}/{uri_count} uris, {oi}/{obj_count} objects")
        return out
    raise RbxFormatError(f"unknown property type id 0x{t:02x}")


# --------------------------------------------------------------------------------------
# attribute blobs / tags
# --------------------------------------------------------------------------------------
def _attr_value(r: _Reader, t):
    if t == 0x02:
        return _bytes_to_text(r.string())
    if t == 0x03:
        return r.u8() != 0
    if t == 0x04:
        return r.i32()
    if t == 0x05:
        return r.f32()
    if t == 0x06:
        return r.f64()
    if t == 0x09:
        return UDim(r.f32(), r.i32())
    if t == 0x0A:
        a, b, c, d = r.f32(), r.i32(), r.f32(), r.i32()
        return UDim2(UDim(a, b), UDim(c, d))
    if t == 0x0E:
        return BrickColor(r.u32())
    if t == 0x0F:
        return Color3(r.f32(), r.f32(), r.f32())
    if t == 0x10:
        return Vector2(r.f32(), r.f32())
    if t == 0x11:
        return Vector3(r.f32(), r.f32(), r.f32())
    if t == 0x14:
        pos = (r.f32(), r.f32(), r.f32())
        rid = r.u8()
        rot = struct.unpack("<9f", r.take(36)) if rid == 0 else _special_rotation(rid)
        return CFrame(pos, rot, rid)
    if t == 0x15:
        name = _bytes_to_text(r.string())
        return {"enum": name, "value": r.u32()}
    if t == 0x17:
        k = r.u32()
        kps = []
        for _ in range(k):
            env, tt, v = struct.unpack("<3f", r.take(12))
            kps.append((tt, v, env))
        return NumberSequence(kps)
    if t == 0x19:
        k = r.u32()
        kps = []
        for _ in range(k):
            env, tt, cr, cg, cb = struct.unpack("<5f", r.take(20))
            kps.append((tt, Color3(cr, cg, cb), env))
        return ColorSequence(kps)
    if t == 0x1B:
        return NumberRange(r.f32(), r.f32())
    if t == 0x1C:
        return Rect(r.f32(), r.f32(), r.f32(), r.f32())
    if t == 0x21:
        weight, style = r.u16(), r.u8()
        fam = _bytes_to_text(r.string())
        cached = _bytes_to_text(r.string())
        return Font(fam, weight, style, cached)
    raise RbxFormatError(f"unknown attribute type 0x{t:02x}")


def decode_attributes(blob):
    """Decode an AttributesSerialize blob -> dict (best effort; '_error' key on failure)."""
    if isinstance(blob, SharedString):
        blob = blob.data or b""
    if isinstance(blob, str):
        blob = blob.encode("utf-8")
    if not blob:
        return {}
    r = _Reader(blob)
    out = {}
    try:
        count = r.u32()
        for _ in range(count):
            key = r.string().decode("utf-8", "replace")
            t = r.u8()
            out[key] = _attr_value(r, t)
    except RbxFormatError as e:
        out["_error"] = str(e)
    return out


def decode_tags(blob):
    """Decode a Tags blob (NUL-separated names; String or SharedString typed) -> list."""
    if isinstance(blob, SharedString):
        blob = blob.data or b""
    if isinstance(blob, str):
        blob = blob.encode("utf-8")
    return [t.decode("utf-8", "replace") for t in blob.split(b"\0") if t]


# --------------------------------------------------------------------------------------
# instance tree
# --------------------------------------------------------------------------------------
class Instance:
    __slots__ = ("ref", "class_name", "props", "parent", "children", "is_service")

    def __init__(self, ref, class_name):
        self.ref = ref
        self.class_name = class_name
        self.props = {}
        self.parent = None
        self.children = []
        self.is_service = False

    @property
    def name(self):
        n = self.props.get("Name")
        if isinstance(n, bytes):
            n = n.decode("utf-8", "replace")
        return n if n is not None else self.class_name

    def full_name(self):
        parts, node, guard = [], self, 0
        while node is not None and guard < 10000:
            parts.append(node.name)
            node = node.parent
            guard += 1
        return ".".join(reversed(parts))

    def get_descendants(self):
        out, stack = [], list(reversed(self.children))
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(reversed(node.children))
        return out

    def find_first_child(self, name, recursive=False, ignore_case=False):
        pool = self.get_descendants() if recursive else self.children
        key = name.lower() if ignore_case else name
        for c in pool:
            n = c.name.lower() if ignore_case else c.name
            if n == key:
                return c
        return None

    def find_first_child_of_class(self, class_name, recursive=False):
        pool = self.get_descendants() if recursive else self.children
        for c in pool:
            if c.class_name == class_name:
                return c
        return None

    def children_of_class(self, *class_names):
        return [c for c in self.children if c.class_name in class_names]

    def find_first_ancestor_of_class(self, class_name):
        node = self.parent
        while node is not None:
            if node.class_name == class_name:
                return node
            node = node.parent
        return None

    def is_descendant_of(self, other):
        node = self.parent
        while node is not None:
            if node is other:
                return True
            node = node.parent
        return False

    def props_json(self, decode_blobs=True):
        d = {}
        for k in sorted(self.props):
            v = self.props[k]
            if decode_blobs and k == "AttributesSerialize" and v:
                d[k] = to_jsonable(decode_attributes(v))
            elif decode_blobs and k == "Tags" and v:
                d[k] = decode_tags(v)
            else:
                d[k] = to_jsonable(v)
        return d

    def to_json(self, recursive=True):
        d = {"ref": self.ref, "class": self.class_name, "name": self.name, "path": self.full_name(),
             "properties": self.props_json()}
        if recursive:
            d["children"] = [c.to_json(True) for c in self.children]
        return d

    def __repr__(self):
        return f"<{self.class_name} {self.full_name()!r}>"


def to_jsonable(v):
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (EnumValue, SecurityCapabilities)):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return _r(v)
    if isinstance(v, (bytes, bytearray)):
        return _binary_summary(bytes(v))
    if isinstance(v, np.generic):
        return to_jsonable(v.item())
    if hasattr(v, "to_json"):
        return v.to_json()
    if isinstance(v, dict):
        return {str(k): to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    if isinstance(v, Instance):
        return {"path": v.full_name(), "class": v.class_name}
    return repr(v)


class ClassInfo:
    __slots__ = ("class_id", "name", "referents", "is_service")

    def __init__(self, class_id, name, referents, is_service):
        self.class_id, self.name, self.referents, self.is_service = class_id, name, referents, is_service


class RobloxFile:
    def __init__(self):
        self.version = None
        self.class_count = 0
        self.instance_count = 0
        self.meta = {}
        self.shared_strings = []  # list of (md5 bytes, data bytes)
        self.classes = {}         # class_id -> ClassInfo
        self.instances = {}       # referent -> Instance
        self.roots = []
        self.chunks = []          # (name, compressed_len, uncompressed_len, compression)
        self.prop_types = {}      # (class_name, prop_name) -> type name
        self.warnings = []

    # ------------------------------------------------------------------ chunks
    def _chunk_meta(self, b):
        r = _Reader(b)
        for _ in range(r.u32()):
            k = r.string().decode("utf-8", "replace")
            v = r.string().decode("utf-8", "replace")
            self.meta[k] = v

    def _chunk_sstr(self, b):
        r = _Reader(b)
        version = r.u32()
        if version != 0:
            self.warnings.append(f"SSTR version {version}")
        for _ in range(r.u32()):
            md5 = r.take(16)
            data = r.string()
            self.shared_strings.append((md5, data))
        if r.remaining():
            self.warnings.append(f"SSTR: {r.remaining()} trailing bytes")

    def _chunk_inst(self, b):
        r = _Reader(b)
        class_id = r.u32()
        name = r.string().decode("utf-8", "replace")
        fmt = r.u8()
        count = r.u32()
        refs = [int(x) for x in r.arr_ref(count)]
        is_service = fmt == 1
        if is_service:
            r.take(count)  # one marker byte per instance
        if r.remaining():
            self.warnings.append(f"INST {name}: {r.remaining()} trailing bytes")
        self.classes[class_id] = ClassInfo(class_id, name, refs, is_service)
        for ref in refs:
            inst = Instance(ref, name)
            inst.is_service = is_service
            if ref in self.instances:
                self.warnings.append(f"duplicate referent {ref}")
            self.instances[ref] = inst

    def _chunk_prop(self, b):
        r = _Reader(b)
        class_id = r.u32()
        pname = r.string().decode("utf-8", "replace")
        t = r.u8()
        ci = self.classes.get(class_id)
        if ci is None:
            self.warnings.append(f"PROP {pname}: unknown class id {class_id}")
            return
        n = len(ci.referents)
        tname = TYPE_NAMES.get(t, f"0x{t:02x}")
        start = r.p
        try:
            values = _decode_values(r, t, n, self)
        except RbxFormatError as e:
            self.warnings.append(f"PROP {ci.name}.{pname} ({tname}): {e}; stored raw")
            raw = b[start:]
            values = [RawValue(t, raw)] * n
            r.p = r.n
        if r.remaining():
            self.warnings.append(f"PROP {ci.name}.{pname} ({tname}): {r.remaining()} trailing bytes")
        self.prop_types[(ci.name, pname)] = tname
        for ref, v in zip(ci.referents, values):
            self.instances[ref].props[pname] = v

    def _chunk_prnt(self, b):
        r = _Reader(b)
        version = r.u8()
        if version != 0:
            self.warnings.append(f"PRNT version {version}")
        count = r.u32()
        children = r.arr_ref(count)
        parents = r.arr_ref(count)
        for c, p in zip(children, parents):
            c, p = int(c), int(p)
            child = self.instances.get(c)
            if child is None:
                self.warnings.append(f"PRNT: unknown child {c}")
                continue
            if p >= 0:
                parent = self.instances.get(p)
                if parent is None:
                    self.warnings.append(f"PRNT: unknown parent {p}")
                    continue
                child.parent = parent
                parent.children.append(child)
        if r.remaining():
            self.warnings.append(f"PRNT: {r.remaining()} trailing bytes")

    # ------------------------------------------------------------------ post
    def _resolve(self):
        def fix(v):
            if isinstance(v, Ref):
                v.instance = self.instances.get(v.ref) if v.ref >= 0 else None
            elif isinstance(v, Content) and isinstance(v.ref, Ref):
                fix(v.ref)
        for inst in self.instances.values():
            for v in inst.props.values():
                fix(v)
        self.roots = [i for i in self.instances.values() if i.parent is None]

    # ------------------------------------------------------------------ queries
    def find_service(self, class_name):
        for r in self.roots:
            if r.class_name == class_name:
                return r
        return None

    def all_instances(self):
        out = []
        for r in self.roots:
            out.append(r)
            out.extend(r.get_descendants())
        return out

    def find_by_class(self, *class_names):
        return [i for i in self.instances.values() if i.class_name in class_names]

    def class_counts(self):
        return {ci.name: len(ci.referents) for ci in sorted(self.classes.values(), key=lambda c: c.name)}

    def to_json(self):
        return {"version": self.version, "class_count": self.class_count,
                "instance_count": self.instance_count, "meta": self.meta,
                "shared_strings": [{"index": i, "md5": m.hex(), **_binary_summary(d)}
                                   for i, (m, d) in enumerate(self.shared_strings)],
                "warnings": self.warnings,
                "roots": [r.to_json(True) for r in self.roots]}


def _decompress(raw, uncomp, name):
    if raw[:4] == ZSTD_MAGIC:
        if _zstd is None:
            raise RbxFormatError("zstandard module required")
        out = _zstd.ZstdDecompressor().decompress(raw, max_output_size=max(uncomp, 1))
        kind = "zstd"
    else:
        if _lz4block is None:
            raise RbxFormatError("lz4 module required")
        out = _lz4block.decompress(raw, uncompressed_size=uncomp)
        kind = "lz4"
    if len(out) != uncomp:
        raise RbxFormatError(f"chunk {name}: decompressed {len(out)} bytes, expected {uncomp}")
    return out, kind


def read_bytes(data: bytes) -> RobloxFile:
    if data[:8] != MAGIC:
        raise RbxFormatError("not a binary Roblox file (bad magic)")
    if data[8:14] != SIGNATURE:
        raise RbxFormatError("bad signature bytes after magic")
    doc = RobloxFile()
    doc.version, doc.class_count, doc.instance_count = struct.unpack_from("<Hii", data, 14)
    pos = 32
    handlers = {"META": doc._chunk_meta, "SSTR": doc._chunk_sstr, "INST": doc._chunk_inst,
                "PROP": doc._chunk_prop, "PRNT": doc._chunk_prnt}
    saw_end = False
    while pos + 16 <= len(data):
        name = data[pos:pos + 4].rstrip(b"\0").decode("ascii", "replace")
        comp, uncomp, _reserved = struct.unpack_from("<III", data, pos + 4)
        pos += 16
        size = comp if comp else uncomp
        raw = data[pos:pos + size]
        if len(raw) != size:
            raise RbxFormatError(f"chunk {name} truncated")
        pos += size
        if comp:
            payload, kind = _decompress(raw, uncomp, name)
        else:
            payload, kind = raw, "raw"
        doc.chunks.append((name, comp, uncomp, kind))
        if name == "END":
            saw_end = True
            break
        h = handlers.get(name)
        if h is None:
            doc.warnings.append(f"skipped unknown chunk {name!r} ({uncomp} bytes)")
            continue
        h(payload)
    if not saw_end:
        doc.warnings.append("no END chunk found")
    if pos != len(data):
        doc.warnings.append(f"{len(data) - pos} bytes after END chunk")
    if len(doc.classes) != doc.class_count:
        doc.warnings.append(f"header says {doc.class_count} classes, found {len(doc.classes)}")
    if len(doc.instances) != doc.instance_count:
        doc.warnings.append(f"header says {doc.instance_count} instances, found {len(doc.instances)}")
    doc._resolve()
    return doc


def read_file(path) -> RobloxFile:
    with open(path, "rb") as f:  # read-only
        return read_bytes(f.read())


# --------------------------------------------------------------------------------------
# BrickColor palette (number -> (name, (r, g, b)))
# --------------------------------------------------------------------------------------
BRICKCOLORS = {
    1: ("White", (242, 243, 243)), 2: ("Grey", (161, 165, 162)), 3: ("Light yellow", (249, 233, 153)),
    5: ("Brick yellow", (215, 197, 154)), 6: ("Light green (Mint)", (194, 218, 184)),
    9: ("Light reddish violet", (232, 186, 200)), 11: ("Pastel Blue", (128, 187, 219)),
    12: ("Light orange brown", (203, 132, 66)), 18: ("Nougat", (204, 142, 105)),
    21: ("Bright red", (196, 40, 28)), 22: ("Med. reddish violet", (196, 112, 160)),
    23: ("Bright blue", (13, 105, 172)), 24: ("Bright yellow", (245, 205, 48)),
    25: ("Earth orange", (98, 71, 50)), 26: ("Black", (27, 42, 53)), 27: ("Dark grey", (109, 110, 108)),
    28: ("Dark green", (40, 127, 71)), 29: ("Medium green", (161, 196, 140)),
    36: ("Lig. Yellowich orange", (243, 207, 155)), 37: ("Bright green", (75, 151, 75)),
    38: ("Dark orange", (160, 95, 53)), 39: ("Light bluish violet", (193, 202, 222)),
    40: ("Transparent", (236, 236, 236)), 41: ("Tr. Red", (205, 84, 75)),
    42: ("Tr. Lg blue", (193, 223, 240)), 43: ("Tr. Blue", (123, 182, 232)),
    44: ("Tr. Yellow", (247, 241, 141)), 45: ("Light blue", (180, 210, 228)),
    47: ("Tr. Flu. Reddish orange", (217, 133, 108)), 48: ("Tr. Green", (132, 182, 141)),
    49: ("Tr. Flu. Green", (248, 241, 132)), 50: ("Phosph. White", (236, 232, 222)),
    100: ("Light red", (238, 196, 182)), 101: ("Medium red", (218, 134, 122)),
    102: ("Medium blue", (110, 153, 202)), 103: ("Light grey", (199, 193, 183)),
    104: ("Bright violet", (107, 50, 124)), 105: ("Br. yellowish orange", (226, 155, 64)),
    106: ("Bright orange", (218, 133, 65)), 107: ("Bright bluish green", (0, 143, 156)),
    108: ("Earth yellow", (104, 92, 67)), 110: ("Bright bluish violet", (67, 84, 147)),
    111: ("Tr. Brown", (191, 183, 177)), 112: ("Medium bluish violet", (104, 116, 172)),
    113: ("Tr. Medi. reddish violet", (229, 173, 200)), 115: ("Med. yellowish green", (199, 210, 60)),
    116: ("Med. bluish green", (85, 165, 175)), 118: ("Light bluish green", (183, 215, 213)),
    119: ("Br. yellowish green", (164, 189, 71)), 120: ("Lig. yellowish green", (217, 228, 167)),
    121: ("Med. yellowish orange", (231, 172, 88)), 123: ("Br. reddish orange", (211, 111, 76)),
    124: ("Bright reddish violet", (146, 57, 120)), 125: ("Light orange", (234, 184, 146)),
    126: ("Tr. Bright bluish violet", (165, 165, 203)), 127: ("Gold", (220, 188, 129)),
    128: ("Dark nougat", (174, 122, 89)), 131: ("Silver", (156, 163, 168)),
    133: ("Neon orange", (213, 115, 61)), 134: ("Neon green", (216, 221, 86)),
    135: ("Sand blue", (116, 134, 157)), 136: ("Sand violet", (135, 124, 144)),
    137: ("Medium orange", (224, 152, 100)), 138: ("Sand yellow", (149, 138, 115)),
    140: ("Earth blue", (32, 58, 86)), 141: ("Earth green", (39, 70, 45)),
    143: ("Tr. Flu. Blue", (207, 226, 247)), 145: ("Sand blue metallic", (121, 136, 161)),
    146: ("Sand violet metallic", (149, 142, 163)), 147: ("Sand yellow metallic", (147, 135, 103)),
    148: ("Dark grey metallic", (87, 88, 87)), 149: ("Black metallic", (22, 29, 50)),
    150: ("Light grey metallic", (171, 173, 172)), 151: ("Sand green", (120, 144, 130)),
    153: ("Sand red", (149, 121, 119)), 154: ("Dark red", (123, 46, 47)),
    157: ("Tr. Flu. Yellow", (255, 246, 123)), 158: ("Tr. Flu. Red", (225, 164, 194)),
    168: ("Gun metallic", (117, 108, 98)), 176: ("Red flip/flop", (151, 105, 91)),
    178: ("Yellow flip/flop", (180, 132, 85)), 179: ("Silver flip/flop", (137, 135, 136)),
    180: ("Curry", (215, 169, 75)), 190: ("Fire Yellow", (249, 214, 46)),
    191: ("Flame yellowish orange", (232, 171, 45)), 192: ("Reddish brown", (105, 64, 40)),
    193: ("Flame reddish orange", (207, 96, 36)), 194: ("Medium stone grey", (163, 162, 165)),
    195: ("Royal blue", (70, 103, 164)), 196: ("Dark Royal blue", (35, 71, 139)),
    198: ("Bright reddish lilac", (142, 66, 133)), 199: ("Dark stone grey", (99, 95, 98)),
    200: ("Lemon metalic", (130, 138, 93)), 208: ("Light stone grey", (229, 228, 223)),
    209: ("Dark Curry", (176, 142, 68)), 210: ("Faded green", (112, 149, 120)),
    211: ("Turquoise", (121, 181, 181)), 212: ("Light Royal blue", (159, 195, 233)),
    213: ("Medium Royal blue", (108, 129, 183)), 216: ("Rust", (144, 76, 42)),
    217: ("Brown", (124, 92, 70)), 218: ("Reddish lilac", (150, 112, 159)),
    219: ("Lilac", (107, 98, 155)), 220: ("Light lilac", (167, 169, 206)),
    221: ("Bright purple", (205, 98, 152)), 222: ("Light purple", (228, 173, 200)),
    223: ("Light pink", (220, 144, 149)), 224: ("Light brick yellow", (240, 213, 160)),
    225: ("Warm yellowish orange", (235, 184, 127)), 226: ("Cool yellow", (253, 234, 141)),
    232: ("Dove blue", (125, 187, 221)), 268: ("Medium lilac", (52, 43, 117)),
    301: ("Slime green", (80, 109, 84)), 302: ("Smoky grey", (91, 93, 105)),
    303: ("Dark blue", (0, 16, 176)), 304: ("Parsley green", (44, 101, 29)),
    305: ("Steel blue", (82, 124, 174)), 306: ("Storm blue", (51, 88, 130)),
    307: ("Lapis", (16, 42, 220)), 308: ("Dark indigo", (61, 21, 133)),
    309: ("Sea green", (52, 142, 64)), 310: ("Shamrock", (91, 154, 76)),
    311: ("Fossil", (159, 161, 172)), 312: ("Mulberry", (89, 34, 89)),
    313: ("Forest green", (31, 128, 29)), 314: ("Cadet blue", (159, 173, 192)),
    315: ("Electric blue", (9, 137, 207)), 316: ("Eggplant", (123, 0, 123)),
    317: ("Moss", (124, 156, 107)), 318: ("Artichoke", (138, 171, 133)),
    319: ("Sage green", (185, 196, 177)), 320: ("Ghost grey", (202, 203, 209)),
    321: ("Lilac", (167, 94, 155)), 322: ("Plum", (123, 47, 123)),
    323: ("Olivine", (148, 190, 129)), 324: ("Laurel green", (168, 189, 153)),
    325: ("Quill grey", (223, 223, 222)), 327: ("Crimson", (151, 0, 0)),
    328: ("Mint", (177, 229, 166)), 329: ("Baby blue", (152, 194, 219)),
    330: ("Carnation pink", (255, 152, 220)), 331: ("Persimmon", (255, 89, 89)),
    332: ("Maroon", (117, 0, 0)), 333: ("Gold", (239, 184, 56)),
    334: ("Daisy orange", (248, 217, 109)), 335: ("Pearl", (231, 231, 236)),
    336: ("Fog", (199, 212, 228)), 337: ("Salmon", (255, 148, 148)),
    338: ("Terra Cotta", (190, 104, 98)), 339: ("Cocoa", (86, 36, 36)),
    340: ("Wheat", (241, 231, 199)), 341: ("Buttermilk", (254, 243, 187)),
    342: ("Mauve", (224, 178, 208)), 343: ("Sunrise", (212, 144, 189)),
    344: ("Tawny", (150, 85, 85)), 345: ("Rust", (143, 76, 42)),
    346: ("Cashmere", (211, 190, 150)), 347: ("Khaki", (226, 220, 188)),
    348: ("Lily white", (237, 234, 234)), 349: ("Seashell", (233, 218, 218)),
    350: ("Burgundy", (136, 62, 62)), 351: ("Cork", (188, 155, 93)),
    352: ("Burlap", (199, 172, 120)), 353: ("Beige", (202, 191, 163)),
    354: ("Oyster", (187, 179, 178)), 355: ("Pine Cone", (108, 88, 75)),
    356: ("Fawn brown", (160, 132, 79)), 357: ("Hurricane grey", (149, 137, 136)),
    358: ("Cloudy grey", (171, 168, 158)), 359: ("Linen", (175, 148, 131)),
    360: ("Copper", (150, 103, 102)), 361: ("Dirt brown", (86, 66, 54)),
    362: ("Bronze", (126, 104, 63)), 363: ("Flint", (105, 102, 92)),
    364: ("Dark taupe", (90, 76, 66)), 365: ("Burnt Sienna", (106, 57, 9)),
    1001: ("Institutional white", (248, 248, 248)), 1002: ("Mid gray", (205, 205, 205)),
    1003: ("Really black", (17, 17, 17)), 1004: ("Really red", (255, 0, 0)),
    1005: ("Deep orange", (255, 176, 0)), 1006: ("Alder", (180, 128, 255)),
    1007: ("Dusty Rose", (163, 75, 75)), 1008: ("Olive", (193, 190, 66)),
    1009: ("New Yeller", (255, 255, 0)), 1010: ("Really blue", (0, 0, 255)),
    1011: ("Navy blue", (0, 32, 96)), 1012: ("Deep blue", (33, 84, 185)),
    1013: ("Cyan", (4, 175, 236)), 1014: ("CGA brown", (170, 85, 0)),
    1015: ("Magenta", (170, 0, 170)), 1016: ("Pink", (255, 102, 204)),
    1017: ("Deep orange", (255, 175, 0)), 1018: ("Teal", (18, 238, 212)),
    1019: ("Toothpaste", (0, 255, 255)), 1020: ("Lime green", (0, 255, 0)),
    1021: ("Camo", (58, 125, 21)), 1022: ("Grime", (127, 142, 100)),
    1023: ("Lavender", (140, 91, 159)), 1024: ("Pastel light blue", (175, 221, 255)),
    1025: ("Pastel orange", (255, 201, 201)), 1026: ("Pastel violet", (177, 167, 255)),
    1027: ("Pastel blue-green", (159, 243, 233)), 1028: ("Pastel green", (204, 255, 204)),
    1029: ("Pastel yellow", (255, 255, 204)), 1030: ("Pastel brown", (255, 204, 153)),
    1031: ("Royal purple", (98, 37, 209)), 1032: ("Hot pink", (255, 0, 191)),
}


def nearest_brickcolor(rgb):
    """Closest palette entry to an (r, g, b) tuple -> (number, name, rgb, distance)."""
    best = None
    for num, (name, c) in BRICKCOLORS.items():
        d = sum((a - b) ** 2 for a, b in zip(rgb, c)) ** 0.5
        if best is None or d < best[3]:
            best = (num, name, c, d)
    return best


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _print_tree(inst, depth=0, max_depth=None, out=sys.stdout):
    out.write("  " * depth + f"{inst.name} [{inst.class_name}]\n")
    if max_depth is not None and depth >= max_depth:
        return
    for c in inst.children:
        _print_tree(c, depth + 1, max_depth, out)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Roblox binary (.rbxl/.rbxm) reader")
    ap.add_argument("file")
    ap.add_argument("--tree", action="store_true", help="print the instance tree")
    ap.add_argument("--depth", type=int, default=None, help="max depth for --tree")
    ap.add_argument("--json", metavar="OUT", help="dump all instances/properties to JSON")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    doc = read_file(a.file)
    print(f"version={doc.version} classes={doc.class_count} instances={doc.instance_count}")
    print("chunks:", ", ".join(f"{n}({k},{u}B)" for n, c, u, k in doc.chunks[:12]),
          f"... total {len(doc.chunks)}")
    print("meta:", doc.meta)
    print(f"shared strings: {len(doc.shared_strings)}")
    print("classes:", doc.class_counts())
    print(f"warnings ({len(doc.warnings)}):")
    for w in doc.warnings:
        print("  -", w)
    if a.tree:
        for r in doc.roots:
            _print_tree(r, 0, a.depth)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(doc.to_json(), f, indent=1, ensure_ascii=False)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
