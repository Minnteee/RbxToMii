"""Recover GameData flag names: hash candidate strings (murmur3_32 / crc32 / fnv1a) and match GDL hashes."""
import sys, os, re, json, zlib, struct
sys.path.insert(0, '.')
import tlfmt


def murmur3_32(data: bytes, seed=0):
    c1, c2 = 0xcc9e2d51, 0x1b873593
    h = seed
    n = len(data)
    nb = n // 4
    for i in range(nb):
        k = struct.unpack_from('<I', data, i * 4)[0]
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF
    tail = data[nb * 4:]
    k = 0
    if len(tail) >= 3: k ^= tail[2] << 16
    if len(tail) >= 2: k ^= tail[1] << 8
    if len(tail) >= 1:
        k ^= tail[0]
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
    h ^= n
    h ^= h >> 16
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def fnv1a(data):
    h = 0x811c9dc5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def strings_from_bytes(b, minlen=3):
    for m in re.finditer(rb'[\x20-\x7e]{%d,}' % minlen, b):
        yield m.group().decode('ascii')


def walk_byml(x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(k, str):
                out.add(k)
            walk_byml(v, out)
    elif isinstance(x, list):
        for v in x:
            walk_byml(v, out)
    elif isinstance(x, str):
        out.add(x)


if __name__ == '__main__':
    gdl = json.load(open('../game/gdl_flat.json'))
    targets = {e['Hash'] for e in gdl}
    # also struct member hashes inside Struct DefaultValue
    for e in gdl:
        if e['type'] == 'Struct' and isinstance(e.get('DefaultValue'), dict):
            for k in e['DefaultValue'].keys():
                try:
                    targets.add(int(k))
                except Exception:
                    pass
    print('targets', len(targets))
    cands = set()
    for p in ['../game/exefs/update/main.bin']:
        cands.update(strings_from_bytes(open(p, 'rb').read(), 3))
    print('exe strings', len(cands))
    romfs = '../game/romfs'
    nfiles = 0
    for dp, dn, fns in os.walk(romfs):
        for fn in fns:
            path = os.path.join(dp, fn)
            try:
                d = tlfmt.read(path)
            except Exception:
                continue
            nfiles += 1
            if d[:2] in (b'YB', b'BY'):
                try:
                    walk_byml(tlfmt.Byml(d).root(), cands)
                except Exception:
                    cands.update(strings_from_bytes(d, 3))
            else:
                cands.update(strings_from_bytes(d, 3))
    print('files', nfiles, 'cands', len(cands))
    # split variants: tokens separated by non-identifier chars
    extra = set()
    for c in list(cands):
        for t in re.split(r'[^A-Za-z0-9_.]', c):
            if len(t) >= 3:
                extra.add(t)
    cands |= extra
    print('cands+tokens', len(cands))
    found = {}
    stats = {'murmur': 0, 'crc32': 0, 'fnv': 0}
    for c in cands:
        b = c.encode('utf-8')
        for nm, h in (('murmur', murmur3_32(b)), ('crc32', zlib.crc32(b) & 0xFFFFFFFF), ('fnv', fnv1a(b))):
            if h in targets and h not in found:
                found[h] = c
                stats[nm] += 1
    print(stats, 'found', len(found))
    json.dump({str(k): v for k, v in found.items()}, open('../game/hashnames.json', 'w'), indent=0, sort_keys=True)
    for h, n in list(found.items())[:60]:
        print('%08x %s' % (h, n))
