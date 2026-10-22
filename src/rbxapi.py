"""Roblox's public web APIs (no login): player lookup, avatar description, pictures, asset files.
Some files Roblox only gives out after a login (dynamic head meshes, some accessories / shirts) - those calls return
status 401 and the caller falls back to the local Roblox cache."""
import io, json, time, gzip, zlib, urllib.request, urllib.error

UA = 'Mozilla/5.0 (RbxToMii; private Tomodachi Life mod tool)'


class HttpError(Exception):
    def __init__(self, code, body=b''):
        super().__init__('HTTP %s' % code)
        self.code = code
        self.body = body


def _unpack(b):
    if b[:2] == b'\x1f\x8b':
        return gzip.decompress(b)
    return b


def request(url, data=None, headers=None, tries=5, timeout=60):
    body = json.dumps(data).encode() if data is not None else None
    hdr = {'User-Agent': UA, 'Accept': '*/*'}
    if body is not None:
        hdr['Content-Type'] = 'application/json'
    hdr.update(headers or {})
    err = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=hdr), timeout=timeout) as r:
                return _unpack(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                err = HttpError(e.code)
                time.sleep(1.5 + 2 * attempt)
                continue
            raise HttpError(e.code, e.read()[:300])
        except urllib.error.URLError as e:
            err = e
            time.sleep(1.5 + attempt)
    raise err


def getj(url, data=None, headers=None):
    return json.loads(request(url, data, headers).decode('utf-8'))


def user_lookup(name):
    """player name (or numeric user id) -> {'id', 'name', 'displayName'}"""
    s = str(name).strip()
    if s.isdigit():
        j = getj('https://users.roblox.com/v1/users/%s' % s)
        return {'id': j['id'], 'name': j['name'], 'displayName': j.get('displayName', j['name'])}
    j = getj('https://users.roblox.com/v1/usernames/users', {'usernames': [s], 'excludeBannedUsers': True})
    if not j.get('data'):
        raise LookupError('There is no Roblox player called "%s".' % s)
    d = j['data'][0]
    return {'id': d['id'], 'name': d['name'], 'displayName': d.get('displayName', d['name'])}


def avatar(uid):
    """avatar description: scales, body colours, asset list (v2 has layered-clothing order and adjustments)"""
    try:
        a = getj('https://avatar.roblox.com/v2/avatar/users/%d/avatar' % uid)
        a['_api'] = 'v2'
        return a
    except HttpError:
        a = getj('https://avatar.roblox.com/v1/users/%d/avatar' % uid)
        a['_api'] = 'v1'
        return a


def user_thumb(uid, kind='avatar', size='420x420'):
    """2D picture of the player (kind avatar / avatar-headshot) as PNG bytes"""
    for _ in range(25):
        j = getj('https://thumbnails.roblox.com/v1/users/%s?userIds=%d&size=%s&format=Png&isCircular=false' % (kind, uid, size))
        d = j['data'][0]
        if d['state'] == 'Completed':
            return request(d['imageUrl'])
        if d['state'] in ('Blocked', 'Error'):
            raise RuntimeError('Roblox liefert kein Bild (%s)' % d['state'])
        time.sleep(1.5)
    raise TimeoutError('The Roblox picture is not ready yet. Try again in a minute.')


def asset_thumb(aid, size='420x420'):
    """public picture of an image asset (the image itself, scaled) -> PNG bytes or None"""
    url = 'https://thumbnails.roblox.com/v1/assets?assetIds=%d&size=%s&format=Png&isCircular=false' % (aid, size)
    for _ in range(15):
        try:
            j = getj(url)
        except Exception:
            return None
        d = j['data'][0] if j.get('data') else {}
        if d.get('state') == 'Completed':
            try:
                return request(d['imageUrl'])
            except Exception:
                return None
        if d.get('state') in ('Blocked', 'Error', None):
            return None
        time.sleep(1.5)
    return None


def asset_thumbs(ids, size='150x150'):
    """public pictures of several assets at once -> {id: PNG bytes}"""
    out = {}
    ids = [int(i) for i in ids]
    for k in range(0, len(ids), 50):
        part = ids[k:k + 50]
        url = 'https://thumbnails.roblox.com/v1/assets?assetIds=%s&size=%s&format=Png&isCircular=false' % (','.join(map(str, part)), size)
        for _ in range(6):
            try:
                j = getj(url)
            except Exception:
                break
            pend = [d for d in j.get('data', []) if d.get('state') == 'Pending']
            for d in j.get('data', []):
                if d.get('state') == 'Completed' and d.get('imageUrl') and d['targetId'] not in out:
                    try:
                        out[d['targetId']] = request(d['imageUrl'])
                    except Exception:
                        pass
            if not pend:
                break
            time.sleep(1.5)
    return out


def asset(aid, fmt=None):
    """asset file bytes; fmt = Roblox-AssetFormat (avatar_meshpart_head / avatar_meshpart_accessory) for the
    MeshPart versions the Roblox client uses for avatars. Raises HttpError (401 = needs a login)."""
    hdr = {'Roblox-AssetFormat': fmt} if fmt else None
    return request('https://assetdelivery.roblox.com/v1/asset/?id=%d' % int(aid), headers=hdr)


def asset_name(aid):
    try:
        return getj('https://economy.roblox.com/v2/assets/%d/details' % int(aid)).get('Name')
    except Exception:
        return None
