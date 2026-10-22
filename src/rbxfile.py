"""Read Roblox model files the way the asset servers deliver them: binary (.rbxm/.rbxl), XML (.rbxmx), either one
possibly gzip-compressed. Gives the same Instance objects as roblox/parser.py (class_name, name, props, children)."""
import os, sys, gzip, zlib, importlib.util
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))


def _parser():
    if 'rbxl_parser' in sys.modules:
        return sys.modules['rbxl_parser']
    if getattr(sys, 'frozen', False):        # RbxToMii.exe: the parser is bundled as a module
        import parser as mod
        sys.modules['rbxl_parser'] = mod
        return mod
    for p in (os.path.join(HERE, 'parser.py'), os.path.join(HERE, '..', 'roblox', 'parser.py')):
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location('rbxl_parser', p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules['rbxl_parser'] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError('parser.py not found')


rbx = _parser()


def unpack(b):
    """gzip / zlib wrapped bodies -> raw bytes"""
    if b[:2] == b'\x1f\x8b':
        return gzip.decompress(b)
    if b[:2] in (b'\x78\x9c', b'\x78\xda', b'\x78\x01'):
        try:
            return zlib.decompress(b)
        except zlib.error:
            pass
    return b


class Doc:
    def __init__(self):
        self.instances = {}
        self.roots = []

    def all(self):
        out = []
        for r in self.roots:
            out.append(r)
            out.extend(r.get_descendants())
        return out


def _f(e, tag, default=0.0):
    c = e.find(tag)
    try:
        return float(c.text) if c is not None and c.text is not None else default
    except ValueError:
        return default


def _cframe(e):
    if e.find('CFrame') is not None:      # OptionalCoordinateFrame
        e = e.find('CFrame')
    pos = (_f(e, 'X'), _f(e, 'Y'), _f(e, 'Z'))
    rot = tuple(_f(e, k, 1.0 if k in ('R00', 'R11', 'R22') else 0.0)
                for k in ('R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22'))
    return rbx.CFrame(pos, rot)


def _content(e):
    url = e.find('url')
    text = (url.text if url is not None else e.text) or ''
    text = text.strip()
    return rbx.Content('uri', text) if text else None


def _value(e):
    t = e.tag
    txt = (e.text or '').strip()
    if t in ('string', 'ProtectedString'):
        return e.text or ''
    if t in ('bool',):
        return txt.lower() == 'true'
    if t in ('int', 'int64', 'token'):
        try:
            return int(txt)
        except ValueError:
            return 0
    if t in ('float', 'double'):
        try:
            return float(txt)
        except ValueError:
            return 0.0
    if t == 'Vector3':
        return rbx.Vector3(_f(e, 'X'), _f(e, 'Y'), _f(e, 'Z'))
    if t == 'Vector2':
        return rbx.Vector2(_f(e, 'X'), _f(e, 'Y'))
    if t in ('CoordinateFrame', 'OptionalCoordinateFrame'):
        return _cframe(e)
    if t in ('Content', 'ContentId'):
        return _content(e)
    if t == 'Color3':
        if e.find('R') is not None:
            return rbx.Color3(_f(e, 'R'), _f(e, 'G'), _f(e, 'B'))
        try:
            v = int(txt)
            return rbx.Color3((v >> 16) & 255, (v >> 8) & 255, v & 255, uint8=True)
        except ValueError:
            return None
    if t == 'Color3uint8':
        try:
            v = int(txt)
        except ValueError:
            return None
        return rbx.Color3((v >> 16) & 255, (v >> 8) & 255, v & 255, uint8=True)
    if t == 'BrickColor':
        try:
            return rbx.BrickColor(int(txt))
        except ValueError:
            return None
    return None   # binary blobs, refs, physics data ... not needed


def read_xml(b):
    root = ET.fromstring(b.decode('utf-8', 'replace'))
    doc = Doc()

    def item(el, parent):
        inst = rbx.Instance(el.get('referent') or str(len(doc.instances)), el.get('class') or '?')
        props = el.find('Properties')
        if props is not None:
            for p in props:
                n = p.get('name')
                if n is None:
                    continue
                v = _value(p)
                if v is not None:
                    inst.props[n] = v
        inst.parent = parent
        if parent is not None:
            parent.children.append(inst)
        else:
            doc.roots.append(inst)
        doc.instances[inst.ref] = inst
        for c in el.findall('Item'):
            item(c, inst)

    for el in root.findall('Item'):
        item(el, None)
    return doc


def load_bytes(b):
    b = unpack(b)
    if b.startswith(b'<roblox!'):
        d = rbx.read_bytes(b)
        doc = Doc()
        doc.instances = d.instances
        doc.roots = [i for i in d.instances.values() if i.parent is None]
        return doc
    if b.lstrip().startswith(b'<roblox'):
        return read_xml(b)
    raise ValueError('keine Roblox-Modelldatei')


def load(path):
    return load_bytes(open(path, 'rb').read())
