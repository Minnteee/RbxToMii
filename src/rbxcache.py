"""Read Roblox's local content cache (%LOCALAPPDATA%\\Roblox\\rbx-storage.db + rbx-storage\\xx\\<hash>).
Roblox Studio / the Roblox app keep every asset they loaded there - including the ones the public web API only
gives out after a login (dynamic heads, some accessories). The cache has no asset ids, so meshes are found again by
their exact bounding box (= the MeshPart's InitialSize) and images by comparing them with the public 420 px picture.
usage: python -I rbxcache.py index <out.json> [roblox dir]"""
import os, sys, json, struct, sqlite3, shutil, tempfile, io
import numpy as np

try:
    import zstandard
except ImportError:  # pragma: no cover
    zstandard = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rbxmesh

ZD = zstandard.ZstdDecompressor() if zstandard else None


def roblox_dir():
    return os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Roblox')


def _rows(root):
    """(hash hex, category, raw bytes) for every cache entry; the database is copied first (Roblox may be open)"""
    tmp = tempfile.mkdtemp(prefix='rbxcache_')
    try:
        for suf in ('', '-wal', '-shm'):
            src = os.path.join(root, 'rbx-storage.db' + suf)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp, 'rbx-storage.db' + suf))
        con = sqlite3.connect(os.path.join(tmp, 'rbx-storage.db'))
        rows = con.execute('select id, content, category from files').fetchall()
        con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for id_, content, cat in rows:
        h = id_.hex()
        if content is None:
            p = os.path.join(root, 'rbx-storage', h[:2], h)
            if not os.path.exists(p):
                continue
            try:
                content = open(p, 'rb').read()
            except OSError:
                continue
        yield h, cat, bytes(content)


def body(raw):
    """RBXH cache record -> (url, http status, header text, body bytes)"""
    if raw[:4] != b'RBXH':
        return None, None, b'', raw
    ulen = struct.unpack_from('<I', raw, 8)[0]
    url = raw[12:12 + ulen].decode('latin1')
    o = 12 + ulen + 1
    status, hlen, _hcrc, blen, _bcrc, _x = struct.unpack_from('<6I', raw, o)
    o += 24
    hdr = raw[o:o + hlen]
    b = raw[o + hlen:o + hlen + blen]
    if b[:4] == b'\x28\xb5\x2f\xfd' and ZD is not None:
        try:
            b = ZD.decompress(b, max_output_size=1 << 28)
        except Exception:
            pass
    return url, status, hdr, b


def kind(b):
    if b.startswith(b'version '):
        return 'mesh'
    if b.startswith(b'<roblox!'):
        return 'rbxm'
    if b.startswith(b'<roblox'):
        return 'rbxmx'
    if b.startswith(b'\xabKTX 20\xbb'):
        return 'ktx2'
    if b.startswith(b'\x89PNG'):
        return 'png'
    if b.startswith(b'\xff\xd8'):
        return 'jpg'
    return None


def decode_ktx2(b):
    """KTX2 (BC1/BC3/BC7/RGBA8, optional zstd supercompression) -> RGBA array of the largest level"""
    import texture2ddecoder
    vk, ts, w, h, d, layers, faces, levels, scs = struct.unpack_from('<9I', b, 12)
    off, length, ulen = struct.unpack_from('<3Q', b, 80)
    data = b[off:off + length]
    if scs == 2:
        data = ZD.decompress(data, max_output_size=max(ulen, 1) + 64)
    if vk in (131, 132, 133, 134):
        px = texture2ddecoder.decode_bc1(data, w, h)
    elif vk in (137, 138):
        px = texture2ddecoder.decode_bc3(data, w, h)
    elif vk in (145, 146):
        px = texture2ddecoder.decode_bc7(data, w, h)
    elif vk in (37, 43):   # R8G8B8A8 unorm / srgb
        return np.frombuffer(data, np.uint8)[:w * h * 4].reshape(h, w, 4).copy()
    else:
        raise ValueError('KTX2 format %d is not supported' % vk)
    a = np.frombuffer(px, np.uint8).reshape(h, w, 4)
    return a[..., [2, 1, 0, 3]].copy()      # BGRA -> RGBA


def ktx2_original_size(b):
    """Roblox stores the real image size as key/value data (textures are padded to powers of two)"""
    kvo, kvl = struct.unpack_from('<2I', b, 56)
    kv = b[kvo:kvo + kvl]
    out = {}
    p = 0
    while p + 4 <= len(kv):
        n = struct.unpack_from('<I', kv, p)[0]
        e = kv[p + 4:p + 4 + n]
        if b'\x00' in e:
            k, v = e.split(b'\x00', 1)
            out[k.decode('latin1')] = v.rstrip(b'\x00').decode('latin1')
        p += 4 + ((n + 3) & ~3)
    try:
        return int(out.get('RobloxOriginalWidth')), int(out.get('RobloxOriginalHeight'))
    except (TypeError, ValueError):
        return None


def image(b):
    """any cached picture -> RGBA array (KTX2 cropped to the original size)"""
    k = kind(b)
    if k == 'ktx2':
        a = decode_ktx2(b)
        o = ktx2_original_size(b)
        if o and (o[0] != a.shape[1] or o[1] != a.shape[0]):
            from PIL import Image
            a = np.asarray(Image.fromarray(a).resize(o, Image.LANCZOS))
        return a
    if k in ('png', 'jpg'):
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(b)).convert('RGBA'))
    return None


def index(root=None, log=None):
    """scan the whole cache once: meshes with bounding box size, images with size, models"""
    root = root or roblox_dir()
    out = {'meshes': [], 'images': [], 'models': []}
    for n, (h, cat, raw) in enumerate(_rows(root)):
        url, status, hdr, b = body(raw)
        k = kind(b)
        if k == 'mesh':
            try:
                m = rbxmesh.parse(b)
            except Exception:
                continue
            p = m['pos']
            if not len(p):
                continue
            lo, hi = p.min(0), p.max(0)
            out['meshes'].append({'h': h, 'size': [float(x) for x in (hi - lo)], 'n': int(len(p)),
                                  'ver': b[8:12].decode('latin1'), 'bones': [x['name'] for x in m['bones']][:8]})
        elif k in ('ktx2', 'png', 'jpg'):
            wh = None
            if k == 'ktx2':
                wh = ktx2_original_size(b) or struct.unpack_from('<2I', b, 20)
            out['images'].append({'h': h, 'k': k, 'wh': list(wh) if wh else None, 'bytes': len(b)})
        elif k in ('rbxm', 'rbxmx'):
            out['models'].append({'h': h, 'bytes': len(b)})
        if log and n % 5000 == 0:
            log('%d entries read' % n)
    out['root'] = root
    return out


def index_update(root=None, old=None, log=None, stop=None):
    """like index(), but keeps the entries of an older index and only reads what is new (much faster)"""
    root = root or roblox_dir()
    old = old or {'meshes': [], 'images': [], 'models': []}
    known = {e['h'] for k in ('meshes', 'images', 'models') for e in old.get(k, [])}
    known |= set(old.get('other', []))
    con = _db(root)
    ids = [r[0].hex() for r in con.execute('select id from files')]
    have = set(ids)
    out = {k: [e for e in old.get(k, []) if e['h'] in have] for k in ('meshes', 'images', 'models')}
    other = [h for h in old.get('other', []) if h in have]
    new = [h for h in ids if h not in known]
    for n, h in enumerate(new):
        if stop and stop():
            break
        try:
            b = read(h, root)
        except Exception:
            b = None
        k = kind(b) if b else None
        if k == 'mesh':
            try:
                m = rbxmesh.parse(b)
                p = m['pos']
                lo, hi = p.min(0), p.max(0)
                out['meshes'].append({'h': h, 'size': [float(x) for x in (hi - lo)], 'n': int(len(p)),
                                      'ver': b[8:12].decode('latin1'), 'bones': [x['name'] for x in m['bones']][:8]})
            except Exception:
                other.append(h)
        elif k in ('ktx2', 'png', 'jpg'):
            wh = None
            if k == 'ktx2':
                try:
                    wh = ktx2_original_size(b) or struct.unpack_from('<2I', b, 20)
                except Exception:
                    wh = None
            out['images'].append({'h': h, 'k': k, 'wh': list(wh) if wh else None, 'bytes': len(b)})
        elif k in ('rbxm', 'rbxmx'):
            out['models'].append({'h': h, 'bytes': len(b)})
        else:
            other.append(h)
        if log and n % 2000 == 0:
            log("Roblox's cache: %d of %d new entries read" % (n, len(new)))
    out['other'] = other
    out['root'] = root
    return out


def access_times(root=None):
    root = root or roblox_dir()
    return {r[0].hex(): r[1] for r in _db(root).execute('select id, atime from files')}


_DB = {}


def _db(root):
    """one private copy of the cache database per run (Roblox may have it open)"""
    if root not in _DB:
        import atexit
        tmp = tempfile.mkdtemp(prefix='rbxcache_')
        for suf in ('', '-wal', '-shm'):
            src = os.path.join(root, 'rbx-storage.db' + suf)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp, 'rbx-storage.db' + suf))
        con = sqlite3.connect(os.path.join(tmp, 'rbx-storage.db'))
        _DB[root] = (con, tmp)

        def _close(con=con, tmp=tmp):
            try:
                con.close()
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        atexit.register(_close)
    return _DB[root][0]


def read(h, root=None):
    """body bytes of one cache entry by hash"""
    root = root or roblox_dir()
    p = os.path.join(root, 'rbx-storage', h[:2], h)
    if os.path.exists(p):
        return body(open(p, 'rb').read())[3]
    row = _db(root).execute('select content from files where id = ?', (bytes.fromhex(h),)).fetchone()
    if row and row[0] is not None:
        return body(bytes(row[0]))[3]
    return None


def find_mesh(idx, size, tol=2e-3):
    """cached meshes whose bounding box equals a MeshPart's InitialSize"""
    s = np.asarray(size, float)
    hits = []
    for m in idx['meshes']:
        d = np.abs(np.asarray(m['size']) - s)
        if (d <= tol * np.maximum(s, 1e-3) + 1e-4).all():
            hits.append((float(d.sum()), m))
    hits.sort(key=lambda t: t[0])
    return [m for _, m in hits]


_THUMBS = {}


def small(h, root=None, size=64):
    """decoded picture of a cache entry shrunk to size x size (kept in memory for repeated comparisons)"""
    from PIL import Image
    key = (h, size)
    if key not in _THUMBS:
        try:
            a = image(read(h, root))
        except Exception:
            a = None
        _THUMBS[key] = None if a is None else (
            np.asarray(Image.fromarray(a).resize((size, size), Image.BOX), np.float32), a.shape[1] / a.shape[0])
    return _THUMBS[key]


def best_image(idx, ref_rgba, root=None, candidates=None, size=64):
    """cached picture most similar to a (small) reference picture of the same asset -> (error, hash)"""
    from PIL import Image
    ref = np.asarray(Image.fromarray(ref_rgba).convert('RGBA').resize((size, size), Image.BOX), np.float32)
    ar = ref_rgba.shape[1] / ref_rgba.shape[0]
    best = (1e18, None)
    for im in (candidates or idx['images']):
        wh = im.get('wh')
        if wh and abs(wh[0] / max(wh[1], 1) - ar) > 0.05:
            continue
        sm = small(im['h'], root, size)
        if sm is None:
            continue
        t, _ = sm
        al = (ref[..., 3:] + t[..., 3:]) / 510.0
        err = float((np.abs(ref[..., :3] - t[..., :3]) * al).mean() + np.abs(ref[..., 3] - t[..., 3]).mean() * 0.5)
        if err < best[0]:
            best = (err, im['h'])
    return best


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.argv[1] == 'index':
        res = index(sys.argv[3] if len(sys.argv) > 3 else None, log=print)
        json.dump(res, open(sys.argv[2], 'w', encoding='utf-8'))
        print('Meshes', len(res['meshes']), 'Bilder', len(res['images']), 'Modelle', len(res['models']))
