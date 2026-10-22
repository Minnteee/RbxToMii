"""RbxToMii - put a Roblox character into Tomodachi Life: Living the Dream (Ryujinx).
Core without user interface: find Ryujinx and the save, load a Roblox player (public web APIs, no login) or an exported
Roblox Studio file, fill login-only parts from Roblox's local cache, build the game files, write them into the mod
folder, make the Roblox suit + head owned items and put them on a Mii of the island the player picks (Ryujinx closed).
Mod folder: mods/contents/<title>/RbxToMii, started from the game's clothing table and size table in templates/game
(7 spots; the wardrobe keeps the game's item names). Everything that gets replaced is zipped first (RbxToMii/backups).
CLI (tests): python -I rbxtomii.py env | fetch <player> | build <player> <spot> | install <player> <spot> [Mii] | miis"""
import os, sys, io, re, json, time, glob, shutil, zipfile, subprocess, struct

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from PIL import Image

import rbxapi, rbxfile, rbxassemble, rbxcache, rbxmesh, rbxchar, rbxmood, tlfmt, byml_write, restbl
import build_roblox_models as BRM

FROZEN = getattr(sys, 'frozen', False)
TOOL_DIR = os.path.dirname(sys.executable) if FROZEN else os.path.abspath(os.path.join(HERE, '..', 'rbxtomii'))
TEMPLATES = os.path.join(TOOL_DIR, 'templates')
DATA = os.path.join(TOOL_DIR, 'data')
TITLE_ID = '010051f0207b2000'
LANGS = ('EUde', 'EUen', 'USen')
RT_REL = 'System/Resource/ResourceSizeTable.Product.100.Nin_NX_NVN.rsizetable.zs'
CMP_REL = 'RSDB/ClothModelParam.Product.100.rstbl.byml.zs'
NAME_FILES = {'body': 'ReplaceMsg/ClothAll_Name.msbt', 'mask': 'ReplaceMsg/ClothHeadwear_Name.msbt'}
GRAMMAR = re.compile(r'^(\[201\.\d+:[0-9a-f]*\])')
MIIDB_REL = os.path.join('bis', 'system', 'save', '8000000000000030')
OWN_MOD = 'RbxToMii'                       # the mod folder RbxToMii writes
OWN_TABLE = 'rbxtomii_spots.json'          # its spot table (who is on which spot)
BASE = os.path.join(TEMPLATES, 'game')     # the game's clothing table + size table (its version: templates.json)
FLAGS = os.path.join(TEMPLATES, 'flags.json')   # names of the save values RbxToMii changes
OWN_README = """This mod folder was made by RbxToMii (https://github.com/Minnteee/RbxToMii).
It turns game costumes into Roblox characters: every spot turns one full-body suit and one full head mask
into the Roblox suit and head (for every Mii and in the shop).
To remove the Roblox characters: close Ryujinx and delete this folder (or disable it in Ryujinx: right-click the
game -> Manage Mods). Miis then wear the game's original items again.
backups\\ : what RbxToMii replaced (files and your save), one zip per change.
"""


def say(msg):
    print(msg, flush=True)


# ================================================================ where things are
def ryujinx_candidates():
    """Ryujinx data folders on this PC (normal install first, then portable installs)"""
    if os.environ.get('RBXTOMII_RYUJINX'):        # fixed folder (tests)
        return [os.environ['RBXTOMII_RYUJINX']]
    out = []
    a = os.path.join(os.environ.get('APPDATA', ''), 'Ryujinx')
    if os.path.isdir(a):
        out.append(a)
    home = os.path.expanduser('~')
    roots = [os.path.join(home, d) for d in ('Desktop', 'Downloads', 'Documents', 'OneDrive\\Desktop')] + [home, 'C:\\']
    for root in roots:
        try:
            subs = [os.path.join(root, d) for d in os.listdir(root)]
        except OSError:
            continue
        for d in subs:
            for exe in ('Ryujinx.exe', 'Switch.exe'):
                if os.path.exists(os.path.join(d, exe)) and os.path.isdir(os.path.join(d, 'portable')):
                    out.append(os.path.join(d, 'portable'))
    seen, res = set(), []
    for p in out:
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen:
            seen.add(k)
            res.append(p)
    return res


def mod_enabled(ry, folder):
    """False if Ryujinx lists this mod folder as disabled (games/<title>/mods.json, "Manage Mods"), else True"""
    try:
        mods = json.load(open(os.path.join(ry, 'games', TITLE_ID, 'mods.json'), encoding='utf-8-sig')).get('mods') or []
    except (OSError, ValueError, AttributeError):
        return True
    same_path = [m for m in mods if m.get('path') and
                 os.path.normcase(os.path.abspath(m['path'])) == os.path.normcase(os.path.abspath(folder))]
    same_name = [m for m in mods if (m.get('name') or '').lower() == os.path.basename(folder).lower()]
    for m in same_path or same_name:      # a moved / portable data folder keeps the names, not the paths
        return bool(m.get('enabled', True))
    return True


def own_dir(ry):
    return os.path.join(ry, 'mods', 'contents', TITLE_ID, OWN_MOD)


KNOWN_SAVE_SIZES = {1325148: '1.0.1', 1325368: '1.0.4'}      # Player.sav grows with new saved values


def save_player_size(save):
    try:
        return os.path.getsize(os.path.join(save, '0', 'Player.sav'))
    except OSError:
        return None


def update_selected(ry):
    """True if Ryujinx has an update selected for the game (RbxToMii's game files are from the version in templates.json)"""
    p = os.path.join(ry, 'games', TITLE_ID, 'updates.json')
    try:
        return bool(json.load(open(p, encoding='utf-8-sig')).get('selected'))
    except (OSError, ValueError, AttributeError):
        return False


def save_dir(ry):
    root = os.path.join(ry, 'bis', 'user', 'save')
    if not os.path.isdir(root):
        return None
    tid = int(TITLE_ID, 16)
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        try:
            b = open(os.path.join(d, 'ExtraData0'), 'rb').read(8)
        except OSError:
            continue
        if len(b) == 8 and struct.unpack('<Q', b)[0] == tid and os.path.exists(os.path.join(d, '0', 'Player.sav')):
            return d
    return None


def miidb_files(ry):
    return [p for p in (os.path.join(ry, MIIDB_REL, s, 'MiiDatabase.dat') for s in ('0', '1')) if os.path.exists(p)]


def roblox_default_rig():
    c = sorted(glob.glob(os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Roblox', 'Versions', '*', 'content', 'avatar',
                                      'characterR15DynamicHeadV2.rbxm')), key=os.path.getmtime)
    return c[-1] if c else None


def roblox_cache_present():
    return os.path.exists(os.path.join(rbxcache.roblox_dir(), 'rbx-storage.db'))


def environment(ry):
    """everything the tool needs, with a short explanation of what is missing (problems block "Add", notes don't).
    mode 'own': RbxToMii's own mod folder"""
    env = {'ryujinx': ry, 'mode': None, 'mod': None, 'table': None, 'backups': None, 'flags': None,
           'save': None, 'problems': [], 'notes': []}
    if not ry or not os.path.isdir(ry):
        env['problems'].append('Ryujinx folder not found. Choose it with "Change…" (usually %APPDATA%\\Ryujinx, '
                               'for the portable version the "portable" folder next to Ryujinx.exe).')
        return env
    own = own_dir(ry)
    env.update(mode='own', mod=own, table=os.path.join(own, OWN_TABLE), backups=os.path.join(own, 'backups'),
               flags=FLAGS)
    if not all(os.path.exists(os.path.join(BASE, rel.split('/')[-1])) for rel in (CMP_REL, RT_REL)):
        env['problems'].append('RbxToMii is incomplete (templates\\game is missing). Unpack the zip again.')
    if not os.path.exists(env['flags']):
        env['problems'].append('RbxToMii is incomplete (templates\\flags.json is missing). Unpack the zip again.')
    env['save'] = save_dir(ry)
    if not env['save']:
        env['problems'].append('No Tomodachi Life save found. Start the game once, make your island and let it save.')
    if not miidb_files(ry):
        env['problems'].append('The console Mii list was not found. Start the game once in Ryujinx and close it again.')
    V = templates()
    want = V.get('game_version', '1.0.4')
    size = save_player_size(env['save']) if env['save'] else None
    if not update_selected(ry):
        env['notes'].append('Ryujinx shows no update for Tomodachi Life. RbxToMii needs game version %s '
                            '(right-click the game → Manage Title Updates).' % want)
    elif size and V.get('player_sav_size') and size != V['player_sav_size']:
        have = KNOWN_SAVE_SIZES.get(size, 'another version')
        env['notes'].append('Your save was written by game version %s, but this RbxToMii is made for %s. Use game '
                            'version %s (Ryujinx: right-click the game → Manage Title Updates), play until the game '
                            'saves, then come back - otherwise the game may crash.' % (have, want, want))
    if env['mode'] == 'own' and os.path.isdir(env['mod']) and not mod_enabled(ry, env['mod']):
        env['notes'].append('The mod "RbxToMii" is turned off in Ryujinx. Turn it on (right-click the game → Manage '
                            'Mods), otherwise the Roblox characters do not show.')
    env['roblox'] = bool(roblox_default_rig())
    env['roblox_cache'] = roblox_cache_present()
    return env


def read_table(env):
    """rows of the spot table (who is on which spot)"""
    p = env.get('table')
    if p and os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))
    return []


def emulator_running(ignore=None):
    """paths of running Ryujinx processes (any name, also a renamed exe)"""
    ps = ("Get-Process | Where-Object { $_.ProcessName -match 'Ryujinx' -or $_.MainWindowTitle -match '^Ryujinx' } | "
          "ForEach-Object { if ($_.Path) { $_.Path } else { $_.ProcessName } }")
    try:
        out = subprocess.run(['powershell.exe', '-NoProfile', '-Command', ps], capture_output=True, text=True, timeout=60,
                             creationflags=0x08000000).stdout
    except Exception:
        return []
    res = [l.strip() for l in out.splitlines() if l.strip()]
    ign = ignore or os.environ.get('RBXTOMII_IGNORE')
    return [r for r in res if not (ign and ign.lower() in r.lower())]


def places(env):
    """the 7 Roblox spots with who is there now. 'tag' names the spot's model files (Spot1..Spot7)"""
    V = templates()
    by_body = {r.get('koerper_kleidung'): r for r in read_table(env) if r.get('koerper_kleidung')}
    rows = []
    for p in V['places']:
        r = by_body.get(p['body'])
        q = dict(p)
        q['tag'] = 'Spot%d' % p['nr']
        q['now'] = (r.get('mii_name') or r.get('roblox_name')) if r else None
        q['roblox'] = r.get('roblox_name') if r else None
        rows.append(q)
    return rows


# ================================================================ templates of the game
_V = {}


def templates():
    if 'v' not in _V:
        _V['v'] = json.load(open(os.path.join(TEMPLATES, 'templates.json'), encoding='utf-8'))
    return _V['v']


class ToolCtx:
    """what build_roblox_models needs from the game - read from RbxToMii's templates folder"""

    def __init__(self):
        V = templates()
        self.V = V
        self.mascot = tlfmt.read(os.path.join(TEMPLATES, 'ClothAllMascotLong.bfres.zs'))
        self.palette = V['palette']
        self.skin_names = V['skin_names']
        self.rest = {k: np.array(v) for k, v in V['rest'].items()}
        self.ref = {k: np.array(v) for k, v in V['ref'].items()}
        self.deltas = {ps: {b: np.array(M) for b, M in D.items()} for ps, D in V['deltas'].items()}
        self.mask_pivot_rest = np.array(V['mask_pivot_rest'])

    def ncol(self, item):
        return self.V['items'][item]['ncol']

    def mask_template(self, mask):
        src = self.V['items'][mask].get('template_from')
        if not src:
            return tlfmt.read(os.path.join(TEMPLATES, mask + '.bfres.zs'))
        import bfres_write
        d = tlfmt.read(os.path.join(TEMPLATES, src + '.bfres.zs'))
        d = bfres_write.rename_strings(d, {'%s_Body_%s' % (src, s): '%s_Body_%s' % (mask, s) for s in ('Alb.00', 'Mic', 'Nrm')})
        return bfres_write.rename_model(d, mask)

    def mic_body(self, item):
        return np.tile(np.array(self.V['items'][item]['mic'], np.uint8), (8, 8, 1))

    def mic_mask(self, mask):
        return np.tile(np.array(self.V['items'][mask]['mic'], np.uint8), (8, 8, 1))

    def has_emm(self, mask):
        return bool(self.V['items'][mask].get('emm'))


# ================================================================ Roblox side
class Store:
    """downloaded Roblox files (shared by all players): data/assets/<id>.bin, data/models/<id>_<format>.bin"""

    def __init__(self, root=DATA):
        self.root = root
        self.assets = os.path.join(root, 'assets')
        self.models = os.path.join(root, 'models')
        os.makedirs(self.assets, exist_ok=True)
        os.makedirs(self.models, exist_ok=True)
        self.status_path = os.path.join(self.assets, 'status.json')
        self.status = json.load(open(self.status_path, encoding='utf-8')) if os.path.exists(self.status_path) else {}

    def save(self):
        json.dump(self.status, open(self.status_path, 'w', encoding='utf-8'), indent=0)

    def model(self, aid, fmt):
        p = os.path.join(self.models, '%d_%s.bin' % (aid, fmt or 'raw'))
        b = None
        if os.path.exists(p) and time.time() - os.path.getmtime(p) < 3 * 86400:
            b = open(p, 'rb').read()
        else:
            try:
                b = rbxapi.asset(aid, fmt)
            except rbxapi.HttpError as e:
                b = b'HTTP%d' % e.code
            except Exception:
                b = None
            if b is not None:
                open(p, 'wb').write(b)
        if not b or b.startswith(b'HTTP'):
            return None
        try:
            return rbxfile.load_bytes(b)
        except Exception:
            return None

    def path(self, aid):
        return os.path.join(self.assets, '%d.bin' % aid)

    def have(self, aid):
        return os.path.exists(self.path(aid)) and str(self.status.get(str(aid), {}).get('status')) in ('200', 'cache', 'thumb')

    def download(self, aid):
        if self.have(aid) and self.status[str(aid)]['status'] == 200:
            return 200
        if str(self.status.get(str(aid), {}).get('status')) in ('401', '403', 'cache', 'thumb') and \
                time.time() - self.status[str(aid)].get('t', 0) < 86400:
            return 401             # asked Roblox today already: needs a login
        if os.path.exists(self.path(aid)) and self.status.get(str(aid), {}).get('status') in ('cache', 'thumb'):
            os.remove(self.path(aid))
        try:
            b = rbxapi.asset(aid)
            open(self.path(aid), 'wb').write(b)
            self.status[str(aid)] = {'status': 200, 'bytes': len(b)}
            return 200
        except rbxapi.HttpError as e:
            self.status[str(aid)] = {'status': e.code, 't': time.time()}
            return e.code
        except Exception as e:
            self.status[str(aid)] = {'status': -1, 'error': str(e)[:100]}
            return -1


def needed_assets(rig):
    """[(asset id, kind, what, part)] - kind: mesh / image / cage"""
    out = []
    for p in rig['parts']:
        if p.get('mesh'):
            out.append((p['mesh'], 'mesh', 'mesh', p))
        if p.get('tex'):
            out.append((p['tex'], 'image', 'tex', p))
        for k, v in (p.get('surface') or {}).items():
            if k == 'ColorMap' and v:
                out.append((v, 'image', 'tex', p))
        sm = p.get('special') or {}
        if sm.get('mesh'):
            out.append((sm['mesh'], 'mesh', 'special', p))
        if sm.get('tex'):
            out.append((sm['tex'], 'image', 'tex', p))
        for d in p.get('decals') or []:
            if d.get('tex'):
                out.append((d['tex'], 'image', 'decal', p))
        wl = p.get('wrap_layer') or {}
        for k in ('cage', 'reference'):
            if wl.get(k):
                out.append((wl[k], 'cage', k, p))
        wt = p.get('wrap_target') or {}
        if wt.get('cage'):
            out.append((wt['cage'], 'cage', 'target', p))
    for k in ('shirt', 'pants', 'tshirt', 'face'):
        if rig.get(k):
            out.append((rig[k], 'image', k, None))
    seen, res = set(), []
    for t in out:
        if t[0] not in seen:
            seen.add(t[0])
            res.append(t)
    return res


def cache_index(log=say, stop=None):
    """index of Roblox's local cache (kept in data/, only new entries are read)"""
    root = rbxcache.roblox_dir()
    if not roblox_cache_present():
        return None
    p = os.path.join(DATA, 'cache_index.json')
    old = json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None
    rbxcache._DB.pop(root, None)        # fresh copy of Roblox's database
    if old is None:
        log('Reading Roblox\'s cache on this PC for the first time (takes about a minute) …')
    idx = rbxcache.index_update(root, old, log=None, stop=stop)
    os.makedirs(DATA, exist_ok=True)
    json.dump(idx, open(p, 'w', encoding='utf-8'))
    return idx


def fill_from_cache(rig, store, todo, log=say):
    """login-only meshes / pictures from Roblox's local cache; pictures otherwise from Roblox's public preview"""
    root = rbxcache.roblox_dir()
    for aid, kind, what, p in todo:            # earlier matches are checked again
        if str(store.status.get(str(aid), {}).get('status')) in ('cache', 'thumb'):
            try:
                os.remove(store.path(aid))
            except OSError:
                pass
            store.status[str(aid)] = {'status': 401, 't': time.time()}
    idx = cache_index(log) if roblox_cache_present() else None
    found = {}
    anchors, anchors_login = [], []
    at = {}
    window = None
    ambiguous = {}     # mesh id -> (part, cached meshes of the same size, pick so far)
    if idx:
        at = rbxcache.access_times(root)
        # anchor times: when did Roblox last load this avatar's (public) parts? If most of them were loaded together,
        # Roblox showed this very avatar then - and its login-only parts were loaded in the same minutes
        todo_ids = {t[0] for t in todo}
        for p in rig['parts']:
            size = p.get('initial_size') or p.get('size')
            if p.get('mesh') and size and p['mesh'] not in todo_ids:
                hits = rbxcache.find_mesh(idx, size)
                if len(hits) == 1 and at.get(hits[0]['h']):
                    anchors.append(at[hits[0]['h']])
        span = 10 * 60 * 1000
        ts = sorted(anchors)
        best, j = [], 0
        for i in range(len(ts)):
            while ts[i] - ts[j] > span:
                j += 1
            if i - j + 1 > len(best):
                best = ts[j:i + 1]
        if len(best) >= 2 and len(best) >= 0.5 * len(ts):
            window = (best[0] - span, best[-1] + span)
        for aid, kind, what, p in todo:
            if kind != 'mesh' or p is None:
                continue
            size = p.get('initial_size') or p.get('size')
            hits = allhits = rbxcache.find_mesh(idx, size) if size else []
            if len(hits) > 1:     # several cached meshes of that size (e.g. heads): only one loaded with this avatar
                hits = [m for m in hits if window and window[0] <= at.get(m['h'], 0) <= window[1]]
            if not hits:
                if len(allhits) > 1:
                    ambiguous[aid] = (p, allhits, None)     # the texture may still tell which one it is
                continue
            hits.sort(key=lambda m: -at.get(m['h'], 0))
            pick = hits[0]
            if float(pick['ver']) < 6:
                same = [m for m in hits if m['n'] == pick['n'] and float(m['ver']) >= 6]
                if same:
                    pick = same[0]
            if len(hits) > 1:
                ambiguous[aid] = (p, hits, pick)
            data = rbxcache.read(pick['h'], root)
            open(store.path(aid), 'wb').write(data)
            store.status[str(aid)] = {'status': 'cache', 'cache': pick['h'], 'candidates': len(hits)}
            found[aid] = 'cache'
            if at.get(pick['h']):
                anchors_login.append(at[pick['h']])

    if idx and any(t[1] == 'cage' for t in todo):
        found.update(find_cages(rig, store, todo, idx, at, anchors + anchors_login, root, log))

    def candidates(anc, minutes, limit=600):
        if not anc:
            return []
        a = np.array(anc, float)
        win = minutes * 60 * 1000
        out = []
        for im in idx['images']:
            t = at.get(im['h'], 0)
            d = float(np.abs(a - t).min())
            if d <= win:
                out.append((d, im))
        out.sort(key=lambda x: x[0])
        return [im for _, im in out[:limit]]

    for aid, kind, what, p in todo:
        if kind != 'image' or aid in found:
            continue
        png = rbxapi.asset_thumb(aid)
        ref = np.asarray(Image.open(io.BytesIO(png)).convert('RGBA')) if png else None
        best = (1e18, None)
        if ref is not None and idx and (anchors_login or window):
            # first right around the moment Roblox loaded this avatar's login-only parts, then wider
            wide = anchors_login + ([t for t in anchors if window[0] <= t <= window[1]] if window else [])
            for anc, minutes in ((anchors_login, 3), (wide, 15)):
                cands = candidates(anc, minutes)
                if not cands:
                    continue
                best = min(best, rbxcache.best_image(idx, ref, root, candidates=cands), key=lambda t: t[0])
                if best[1] is not None and best[0] < 6.0:
                    break
        if ref is not None and idx:
            err, h = best
            if h is not None and err < 6.0:
                img = rbxcache.image(rbxcache.read(h, root))
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, 'PNG')
                open(store.path(aid), 'wb').write(buf.getvalue())
                store.status[str(aid)] = {'status': 'cache', 'cache': h, 'match_error': round(err, 2)}
                found[aid] = 'cache'
                continue
        if ref is not None:
            im = Image.fromarray(ref)
            if what in ('shirt', 'pants'):
                im = im.resize((585, 559), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, 'PNG')
            open(store.path(aid), 'wb').write(buf.getvalue())
            store.status[str(aid)] = {'status': 'thumb'}
            found[aid] = 'preview picture'
    # Many Roblox heads have exactly the same size, so the size alone often fits several cached meshes. The part's
    # texture decides: it was painted for one UV layout (eyes, mouth, ... on their own patches of the texture).
    for aid, (p, hits, pick) in ambiguous.items():
        tid = (p.get('surface') or {}).get('ColorMap') or p.get('tex')
        if not tid or not store.have(tid):
            continue
        try:
            tex = np.asarray(Image.open(store.path(tid)).convert('RGBA'))
        except Exception:
            continue
        scored = []
        for m in hits:
            try:
                mesh = rbxmesh.parse(rbxcache.read(m['h'], root))
            except Exception:
                continue
            scored.append((uv_misfit(mesh, tex), -float(float(m['ver']) >= 6), -at.get(m['h'], 0), m))
        if not scored:
            continue
        best = min(s[0] for s in scored)
        if pick is None and best > 0.002:
            continue            # no clean fit and no time hint: stays missing
        good = sorted([s for s in scored if s[0] <= best + 0.002], key=lambda s: s[1:3])
        m = good[0][3]
        if pick is not None and m['h'] == pick['h']:
            continue
        open(store.path(aid), 'wb').write(rbxcache.read(m['h'], root))
        store.status[str(aid)] = {'status': 'cache', 'cache': m['h'], 'candidates': len(hits), 'uv_misfit': round(best, 4)}
        found[aid] = 'cache'
    store.save()
    return found, idx


def _mesh_key(n, size):
    return int(n), tuple(round(float(x), 3) for x in size)


def find_cages(rig, store, todo, idx, at, anchors, root, log=say):
    """cages of layered clothing (WrapLayer: outer cage + inner 'reference' cage) and of body parts (WrapTarget, e.g. a
    dynamic head): Roblox gives most of them out only after a login. Roblox (also Roblox Studio) loads them in the same
    minute as the avatar's other meshes, so: cached meshes of that minute that are no known file of this avatar.
    Layered clothing cages all share one point layout with the body's cages (about 4900 points, as tall as a body);
    the inner cage is the smaller one of an item's pair."""
    if not anchors:
        return {}
    lo, hi = min(anchors) - 120000, max(anchors) + 120000
    known = {}
    for aid, kind, what, p in needed_assets(rig):
        if kind in ('mesh', 'cage') and store.have(aid):
            try:
                m = rbxmesh.load(store.path(aid))
                known[_mesh_key(len(m['pos']), m['pos'].max(0) - m['pos'].min(0))] = aid
            except Exception:
                pass
    cands = [m for m in idx['meshes'] if lo <= at.get(m['h'], 0) <= hi and _mesh_key(m['n'], m['size']) not in known]
    body_keys = set()                    # point layout of the body's cages (to recognise a cage)
    for p in rig['parts']:
        c = (p.get('wrap_target') or {}).get('cage')
        if c and store.have(c):
            try:
                body_keys |= set(int(k) for k in rbxchar.uv_keys(rbxmesh.load(store.path(c))['uv']))
            except Exception:
                pass

    def is_cage(m):
        try:
            mesh = rbxmesh.parse(rbxcache.read(m['h'], root))
        except Exception:
            return False
        k = rbxchar.uv_keys(mesh['uv'])
        return bool(body_keys) and float(np.isin(k, list(body_keys)).mean()) > 0.5

    full = [m for m in cands if m['n'] >= 3000 and m['size'][1] > 3.0]
    full = [m for m in full if is_cage(m)]
    vol = lambda m: float(np.prod(m['size']))
    found, used = {}, set()

    def take(aid, m, what):
        open(store.path(aid), 'wb').write(rbxcache.read(m['h'], root))
        store.status[str(aid)] = {'status': 'cache', 'cache': m['h'], 'cage_match': what}
        found[aid] = 'cache'
        used.add(m['h'])

    items = {}                           # layered item -> what is missing
    for aid, kind, what, p in todo:
        if kind == 'cage' and what in ('cage', 'reference') and p is not None:
            items.setdefault(p['path'], {'p': p})[what] = aid
    for path, it in sorted(items.items(), key=lambda kv: -vol({'size': kv[1]['p'].get('size') or [0, 0, 0]})):
        msize = np.array(it['p'].get('initial_size') or it['p'].get('size') or [0, 0, 0], float)
        free = [m for m in full if m['h'] not in used]
        if 'cage' in it and 'reference' in it:
            best = None
            for a in free:               # inner a, outer b: same layout, b encloses a and the item
                for b in free:
                    if a is b or a['n'] != b['n'] or vol(a) >= vol(b):
                        continue
                    sa, sb = np.array(a['size']), np.array(b['size'])
                    if (sa > sb + 1e-3).any():
                        continue
                    score = float(np.maximum(0, msize - sb).sum()) * 10 + float((sb - msize).clip(0).sum()) * 0.1
                    if best is None or score < best[0]:
                        best = (score, a, b)
            if best:
                take(it['reference'], best[1], 'inner cage of %s' % path)
                take(it['cage'], best[2], 'outer cage of %s' % path)
        elif free:                       # one of the two is there: the other one fits around / inside it
            wl = it['p'].get('wrap_layer') or {}
            want = 'reference' if 'reference' in it else 'cage'
            other = wl.get('cage' if want == 'reference' else 'reference')
            if other and store.have(other):
                om = rbxmesh.load(store.path(other))
                os_ = om['pos'].max(0) - om['pos'].min(0)
                if want == 'reference':  # inner cage: inside the outer one, the biggest such mesh
                    ok = [m for m in free if m['n'] == len(om['pos']) and (np.array(m['size']) <= os_ + 1e-3).all()]
                    ok.sort(key=lambda m: -vol(m))
                else:                    # outer cage: around the inner one, the smallest such mesh
                    ok = [m for m in free if m['n'] == len(om['pos']) and (np.array(m['size']) >= os_ - 1e-3).all()]
                    ok.sort(key=vol)
                if ok:
                    take(it[want], ok[0], '%s of %s' % (want, path))
    # most layered clothing is made on Roblox's standard inner cage: an inner cage still missing is taken from another
    # item of this avatar with the same point count
    for path, it in items.items():
        aid = it.get('reference')
        if aid and aid not in found:
            for p in rig['parts']:
                r = (p.get('wrap_layer') or {}).get('reference')
                if r and r != aid and store.have(r):
                    shutil.copy2(store.path(r), store.path(aid))
                    store.status[str(aid)] = {'status': 'cache', 'same_as': r, 'cage_match': 'standard inner cage'}
                    found[aid] = 'cache'
                    break
    # cages of body parts (dynamic heads): as big as the part
    for aid, kind, what, p in todo:
        if kind != 'cage' or what != 'target' or p is None or aid in found:
            continue
        ps = np.array(p.get('initial_size') or p.get('size') or [0, 0, 0], float)
        ok = [m for m in cands if m['h'] not in used and m['n'] < 3000 and
              (np.abs(np.array(m['size']) - ps) <= 0.12 * ps + 0.02).all()]
        ok.sort(key=lambda m: float(np.abs(np.array(m['size']) - ps).sum()))
        if ok:
            take(aid, ok[0], 'cage of %s' % p['path'])
    if found:
        log('Found %d clothing/head cages in Roblox\'s cache.' % len(found))
    return found


def uv_misfit(mesh, tex):
    """share of a mesh's triangles whose corners fall on both sides of a border in the texture (opaque / transparent,
    or feature / background colour) - about 0 for the mesh the texture was painted for"""
    a = tex[..., 3].astype(np.int32)
    if (a < 128).mean() > 0.02 and (a >= 128).mean() > 0.002:
        content = a >= 128
    else:
        rgb = tex[..., :3].astype(np.float32)
        bg = np.median(rgb.reshape(-1, 3), 0)
        content = np.linalg.norm(rgb - bg, axis=-1) > 40
    h, w = content.shape
    uv = np.mod(np.asarray(mesh['uv'], np.float64), 1.0)
    x = np.clip((uv[:, 0] * (w - 1)).round().astype(int), 0, w - 1)
    y = np.clip((uv[:, 1] * (h - 1)).round().astype(int), 0, h - 1)
    c = content[y, x][np.asarray(mesh['faces'])]
    return float((c.any(1) & ~c.all(1)).mean())


def mood_weights(rig, idx, log=say):
    m = rig.get('mood') or {'id': 0}
    if not m.get('id'):
        return templates()['smile'], 'Smile'
    name = rbxapi.asset_name(m['id']) or m.get('name')
    if idx:
        try:
            moods = rbxmood.cached_moods(idx, rbxcache.roblox_dir())
            w, name = rbxmood.mood_weights(m['id'], moods, {str(m['id']): name} if name else None)
            if w:
                return w, name
        except Exception:
            pass
    # mood not in Roblox's cache on this PC (its file needs a login). "Default Mood" keeps the head's own neutral face;
    # most other moods (and Roblox's "DefaultFallBackMood") are smiles - show Roblox's default smile
    if rbxmood.norm(name or '') == 'default':
        return {}, name
    return templates()['smile'], ('%s (shown as Smile)' % name) if name else 'Smile'


def _created(d):
    import datetime
    s = (d.get('Created') or '').replace('Z', '+00:00')
    try:
        return datetime.datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _norm_name(s):
    return re.sub(r'[^0-9a-z]+', '', (s or '').lower())


def _hist(png):
    """colour histogram of the coloured pixels of a picture (transparent / white / grey background left out)"""
    a = np.asarray(Image.open(io.BytesIO(png)).convert('RGBA').resize((96, 96))).reshape(-1, 4).astype(np.int32)
    rgb, al = a[:, :3], a[:, 3]
    sat = rgb.max(1) - rgb.min(1)
    keep = (al > 128) & ~((sat < 18) & (rgb.min(1) > 200))
    if keep.sum() < 50:
        keep = al > 128
    q = (rgb[keep] // 32)
    h = np.bincount(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], minlength=512).astype(np.float64)
    return h / max(h.sum(), 1)


def guess_template(aid, log=say, span=60, name=None):
    """classic shirt / pants / t-shirt whose item file needs a login: its picture is uploaded right before the item
    (slightly smaller id, created about half a second earlier). Among the pictures uploaded in that second the one
    with the same name wins, otherwise the one whose colours match Roblox's public picture of the item.
    name: the item's name from the avatar (used when the catalog has no details of the item, e.g. very old items)."""
    import difflib
    from concurrent.futures import ThreadPoolExecutor
    if aid > 2 ** 36:          # newer ids are not sequential
        return None
    try:
        item = rbxapi.getj('https://economy.roblox.com/v2/assets/%d/details' % aid)
    except Exception:
        item = None
    t0 = _created(item) if item else None
    if t0 is None:
        # no details of the item itself: a picture uploaded right before it under the same name
        n0 = _norm_name(re.sub(r'\.(jpe?g|png|bmp|gif|tga)$', '', name or '', flags=re.I))
        if not n0:
            return None
        for k in range(1, 4):
            try:
                d = rbxapi.getj('https://economy.roblox.com/v2/assets/%d/details' % (aid - k))
            except Exception:
                continue
            nk = _norm_name(re.sub(r'\.(jpe?g|png|bmp|gif|tga)$', '', d.get('Name') or '', flags=re.I))
            if d.get('AssetTypeId') == 1 and difflib.SequenceMatcher(None, n0, nk).ratio() >= 0.85:
                return aid - k
        return None

    def details(i):
        try:
            return i, rbxapi.getj('https://economy.roblox.com/v2/assets/%d/details' % i)
        except Exception:
            return i, None
    with ThreadPoolExecutor(6) as ex:
        found = dict(ex.map(details, [aid - k for k in range(1, span)]))
    cands = []
    for k in range(1, span):
        d = found.get(aid - k)
        t = _created(d) if d else None
        if t is None:
            continue
        if t0 - t > 30:
            break
        if d.get('AssetTypeId') == 1 and -0.05 <= t0 - t < 3:
            cands.append((aid - k, d.get('Name') or ''))
    if not cands:
        return None
    n0 = _norm_name(item.get('Name'))
    best = max(cands, key=lambda c: difflib.SequenceMatcher(None, n0, _norm_name(c[1])).ratio())
    if n0 and difflib.SequenceMatcher(None, n0, _norm_name(best[1])).ratio() >= 0.75:
        return best[0]
    thumbs = rbxapi.asset_thumbs([aid] + [c[0] for c in cands])
    if aid not in thumbs:
        return None
    h0 = _hist(thumbs[aid])
    scores = sorted(((float(np.minimum(h0, _hist(thumbs[c])).sum()), c) for c, _ in cands if c in thumbs), reverse=True)
    if scores and scores[0][0] >= 0.35 and (len(scores) == 1 or scores[0][0] >= 1.25 * scores[1][0]):
        return scores[0][1]
    return None


def recover_clothing(av, rig, report, log=say):
    """pictures of login-only classic clothing (see guess_template)"""
    key = {2: 'tshirt', 11: 'shirt', 12: 'pants'}
    types = {a['id']: a['assetType']['id'] for a in av.get('assets', []) if a.get('assetType')}
    keep = []
    for m in report['missing']:
        t = types.get(m[0])
        if t in key and not rig.get(key[t]):
            log('Looking up the picture of "%s" in the Roblox catalog …' % (m[2] or m[1]))
            tid = guess_template(m[0], log, name=m[2])
            if tid:
                rig[key[t]] = tid
                report['ok'].append(m)
                continue
        keep.append(m)
    report['missing'] = keep


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9_\-]', '_', s)[:40] or 'roblox'


def fetch_player(name, log=say):
    """Roblox player name -> character dict (rig, report, pictures) - public web APIs only"""
    log('Looking up %s …' % name)
    user = rbxapi.user_lookup(name)
    av = rbxapi.avatar(user['id'])
    key = user['name']
    pdir = os.path.join(DATA, 'players', safe_name(key))
    os.makedirs(pdir, exist_ok=True)
    for kind, fn in (('avatar', 'avatar.png'), ('avatar-headshot', 'headshot.png')):
        try:
            open(os.path.join(pdir, fn), 'wb').write(rbxapi.user_thumb(user['id'], kind))
        except Exception:
            pass
    json.dump(av, open(os.path.join(pdir, 'avatar.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    store = Store()
    rig, report = rbxassemble.assemble(av, store.model, None, log=log, default_parts=templates()['default_body'])
    recover_clothing(av, rig, report, log)
    return finish_character(key, user, rig, report, store, pdir, log)


def finish_character(key, user, rig, report, store, pdir, log=say):
    todo = []
    need = needed_assets(rig)
    log('Downloading %d Roblox files …' % len(need))
    for aid, kind, what, p in need:
        st = store.download(aid)
        if st != 200:              # login-only: matched again every time (Roblox's cache changes)
            todo.append((aid, kind, what, p))
    store.save()
    idx = None
    found = {}
    if todo:
        log('Looking for %d login-only files in Roblox\'s cache on this PC …' % len(todo))
        found, idx = fill_from_cache(rig, store, todo, log)
    still = [(aid, kind, what, p) for aid, kind, what, p in todo if aid not in found]
    if idx is None and (rig.get('mood') or {}).get('id') and roblox_cache_present():
        idx = cache_index(log)
    w, mood_name = mood_weights(rig, idx, log)
    mp = os.path.join(store.assets, 'moods.json')
    moods = json.load(open(mp, encoding='utf-8')) if os.path.exists(mp) else {}
    moods[key] = {'mood_id': (rig.get('mood') or {}).get('id', 0), 'name': mood_name, 'weights': w}
    json.dump(moods, open(mp, 'w', encoding='utf-8'), indent=1)
    rig['name'] = key
    json.dump(rig, open(os.path.join(pdir, 'rig.json'), 'w', encoding='utf-8'), indent=1)
    head = next((p for p in rig['parts'] if p['path'] == 'Head'), {})
    head_real = bool(head.get('mesh')) and store.have(head['mesh'])
    missing = [{'id': a, 'part': t, 'name': n} for a, t, n in report['missing']]
    body = [p for p in rig['parts'] if p['path'] in rbxassemble.BODY[1:]]
    if body and all((p.get('surface') or {}).get('ColorMap') for p in body):
        # a body with its own texture (SurfaceAppearance): Roblox draws no classic clothing on it, so it is not missing
        missing = [m for m in missing if m['part'] not in ('T-shirt', 'Shirt', 'Pants')]
    for aid, kind, what, p in still:
        if kind == 'mesh' and p is not None and p['path'] != 'Head':
            missing.append({'id': aid, 'part': p['path'].split('.')[0], 'name': ''})
    for p in rig['parts']:          # layered clothing without its cages: shown, but not fitted to the body
        wl = p.get('wrap_layer') or {}
        if wl and not all(wl.get(k) and store.have(wl[k]) for k in ('cage', 'reference')):
            missing.append({'id': wl.get('cage') or wl.get('reference'), 'part': 'Fit of layered clothing',
                            'name': p['path'].split('.')[0]})
    ch = {'key': key, 'user': user, 'rig': rig, 'dir': pdir, 'head_real': head_real, 'mood': mood_name if w is not None else None,
          'missing': missing, 'from_cache': {str(k): v for k, v in found.items()}, 'defaults': report['default'],
          'ok': len(report['ok'])}
    json.dump({k: v for k, v in ch.items() if k != 'rig'}, open(os.path.join(pdir, 'report.json'), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)
    return ch


def rig_from_file(path, log=say):
    """Roblox Studio export (.rbxm / .rbxmx / .rbxl): every model with a Humanoid -> [(name, rig)]"""
    doc = rbxfile.load(path)
    out = []
    for mdl in [i for i in doc.all() if i.class_name == 'Model' and any(c.class_name == 'Humanoid' for c in i.children)]:
        rig = {'parts': [], 'accessories': [], 'joints': [], 'source': 'file'}
        base = mdl.full_name()
        for c in mdl.get_descendants():
            if c.class_name in ('MeshPart', 'Part'):
                d = rbxassemble.part_info(c)
                d['path'] = c.full_name()[len(base) + 1:]
                d['cf'] = rbxassemble.cfd(c.props.get('CFrame'))
                d.pop('scale_type', None)
                if d['cf'] is None:
                    continue
                rig['parts'].append(d)
            elif c.class_name in ('Accessory', 'Hat'):
                h = c.find_first_child('Handle')
                rig['accessories'].append({'name': c.name, 'type': int(c.props.get('AccessoryType', 0) or 0),
                                           'handle': h.full_name()[len(base) + 1:] if h else None})
            elif c.class_name == 'Shirt':
                rig['shirt'] = rbxassemble.aid(c.props.get('ShirtTemplate'))
            elif c.class_name == 'Pants':
                rig['pants'] = rbxassemble.aid(c.props.get('PantsTemplate'))
            elif c.class_name == 'ShirtGraphic':
                rig['tshirt'] = rbxassemble.aid(c.props.get('Graphic'))
            elif c.class_name == 'BodyColors':
                rig['body_colors'] = {k: rbxassemble.color(v) for k, v in c.props.items() if 'Color' in k and rbxassemble.color(v)}
            elif c.class_name == 'HumanoidDescription':
                rig['mood'] = {'id': int(c.props.get('MoodAnimation') or 0), 'name': None}
                rig['scales'] = {k: c.props.get(k) for k in ('HeightScale', 'WidthScale', 'DepthScale', 'HeadScale',
                                                            'ProportionScale', 'BodyTypeScale')}
        names = {p['path'] for p in rig['parts']}
        if not all(b in names for b in rbxassemble.BODY):
            continue
        rig.setdefault('mood', {'id': 0, 'name': 'Smile'})
        out.append((mdl.name, rig))
    return out


def fetch_file(path, model_name=None, log=say):
    models = rig_from_file(path, log)
    if not models:
        raise ValueError('There is no R15 character (a model with a Humanoid) in this file.')
    if model_name:
        models = [m for m in models if m[0] == model_name] or models
    key, rig = models[0]
    log('Loading %s from the file …' % key)
    pdir = os.path.join(DATA, 'players', safe_name(key))
    os.makedirs(pdir, exist_ok=True)
    report = {'ok': [(None, 'part', p['path']) for p in rig['parts']], 'missing': [], 'default': []}
    user = {'id': None, 'name': key, 'displayName': key}
    try:      # picture for the preview, if the model is named like a player
        u = rbxapi.user_lookup(key)
        user = u
        for kind, fn in (('avatar', 'avatar.png'), ('avatar-headshot', 'headshot.png')):
            open(os.path.join(pdir, fn), 'wb').write(rbxapi.user_thumb(u['id'], kind))
    except Exception:
        pass
    return finish_character(key, user, rig, report, Store(), pdir, log)


# ================================================================ build
def build(ch, place, log=say):
    """character -> game files for one spot, plus preview pictures"""
    store = Store()
    rbxchar.NET3D = store.assets
    rbxchar.NET = ch['dir']
    rbxchar.DEFAULT_MOOD = templates()['smile']      # mood unknown on this PC: Roblox's default smile
    hs = os.path.join(ch['dir'], 'headshot.png')
    if os.path.exists(hs):
        shutil.copy2(hs, os.path.join(ch['dir'], '%s_avatar-headshot.png' % ch['key']))
    ctx = ToolCtx()
    log('Building the Tomodachi Life model …')
    files, stats, extra = BRM.make_character_files(ch['key'], {ch['key']: ch['rig']}, place['tag'], place['body'], place['mask'],
                                                   ctx, (2048, 2048), log=lambda s: None)
    prev = preview(extra, ctx)
    return {'files': files, 'stats': stats, 'preview': prev, 'icon_body': extra['icon_body'], 'icon_head': extra['icon_head'],
            'place': place, 'key': ch['key']}


def preview(extra, ctx, size=520):
    """front view (slightly turned) of the finished model in the Mii standing pose, like in the game"""
    import rrender
    A = extra['arrays']

    class P:
        pass
    b = P()
    b.pos, b.uv, b.faces, b.tex = A['Pref'], A['UV'], A['T'], extra['atlas_body']
    import miimap
    D = {b: ctx.ref[b] @ np.linalg.inv(ctx.rest[b]) for b in (miimap.MASK_PIVOT, 'Head')}
    h = P()
    h.pos = miimap.mask_to_world(A['P2'], ctx.rest, D, templates()['mask_scale'])
    h.uv, h.faces, h.tex = A['UV2'], A['T2'], extra['atlas_head']
    im = rrender.render([b, h], yaw=180 + 20, pitch=-5, size=size, alpha=True, cull=True)
    bg = Image.new('RGBA', im.size, (238, 241, 245, 255))
    bg.alpha_composite(im)
    return [bg.convert('RGB')]


# ================================================================ the island's Miis
# Mii.sav 'Mii.#05f5f62c' (hash 0x881ca27a): one 156-byte record per resident (size, CreateId, name, ..., height and
# build at +44/+45); the clothes of a Mii are Mii.MiiMisc.#3b098df0.<slot>.KeyHash (murmur3 of the cloth row name).
# RbxToMii puts the Roblox suit + head on a Mii the player picks (1.2.4; up to 1.2.3 it made console Mii templates,
# but a template only carries the face - a Mii made from it got the game's standard clothes).
MII_RECORDS = 0x881ca27a
CLOTH_SLOTS = ('Tops', 'Topslong', 'BottomsA', 'BottomsB', 'Shoes', 'Accessory', 'Headwear', 'All')


def mii_nickname(ch):
    """short name of the character for the wardrobe items: the display name if it has at most 10 characters, else
    the user name if that fits, else the display name cut to 10 characters"""
    names = [re.sub(r'[^A-Za-z0-9 _\-.]', '', n or '').strip() for n in (ch['user'].get('displayName'), ch['key'])]
    names = [n for n in names if n]
    for n in names:
        if len(n) <= 10:
            return n
    return names[0][:10] if names else 'Roblox'


def _mii_ed(savedir, flags):
    import ugcsave as U
    return U.Ed(os.path.join(savedir, 'Mii.sav'), U.load_flags(flags))


def residents(M):
    """[(index, name)] of the island's Miis (in the order of the save)"""
    e = M.sv.by_hash[MII_RECORDS]
    cnt = struct.unpack_from('<I', M.sv.d, e.off)[0]
    es = (e.size - 4) // cnt
    names = M.get('Mii.Name.Name')
    out = []
    for i in range(cnt):
        rec = bytes(M.sv.d[e.off + 4 + es * i:e.off + 4 + es * (i + 1)])
        if not any(rec[4:20]):           # empty place (no CreateId)
            continue
        out.append((i, names[i] if i < len(names) else ''))
    return out


def island_miis(env):
    """[{'index', 'name', 'roblox'}] of the island's Miis; roblox = the Roblox character whose suit the Mii wears ('')"""
    if not env.get('save') or not env.get('flags') or not os.path.exists(env['flags']):
        return []
    import ugcsave as U
    M = _mii_ed(os.path.join(env['save'], '0'), env['flags'])
    by_body = {U.mm(r['koerper_kleidung']): r for r in read_table(env) if r.get('koerper_kleidung')}
    out = []
    for i, name in residents(M):
        r = by_body.get(M.get('Mii.MiiMisc.#3b098df0.All.KeyHash', i))
        out.append({'index': i, 'name': name, 'roblox': (r.get('roblox_name') or '') if r else ''})
    return out


def dress_one(M, i, body, mask):
    """Roblox suit (All slot) + head (Headwear) on Mii i, everything else off; standard height / build"""
    import ugcsave as U
    pieces = {s: (0, 0, None) for s in ('Tops', 'Topslong', 'BottomsA', 'BottomsB', 'Shoes', 'Accessory')}
    pieces['All'] = (body, 0, None)
    pieces['Headwear'] = (mask, 0, None)
    U.dress_mii(M, i, pieces, facepaint=-1, front=False)
    for f in ('#ab277b66', '#b90d9b9a', '#ed14a1bc'):
        M.set('Mii.MiiMisc.#3b098df0.' + f, -1, i)
    # standard height / build: the Roblox body is made for it (the full-mask head does not scale with the body)
    e = M.sv.by_hash[MII_RECORDS]
    cnt = struct.unpack_from('<I', M.sv.d, e.off)[0]
    base = e.off + 4 + (e.size - 4) // cnt * i
    M.sv.d[base + 44] = 64
    M.sv.d[base + 45] = 64


# ================================================================ install
def _zs(b):
    import zstandard
    return zstandard.ZstdCompressor(level=19).compress(b)


def set_names(romfs, place, display):
    """wardrobe names of the spot's two items in all message archives -> {rel path: bytes}"""
    out = {}
    texts = {'de': ('Roblox-Anzug ' + display, 'Roblox-Kopf ' + display), 'en': ('Roblox suit: ' + display, 'Roblox head: ' + display)}
    for lang in LANGS:
        rel = 'Mals/%s.Product.100.sarc.zs' % lang
        p = os.path.join(romfs, rel)
        if not os.path.exists(p):
            continue
        files = tlfmt.sarc(open(p, 'rb').read())
        de = 'de' if lang == 'EUde' else 'en'
        for part, label, new in (('body', place['body'], texts[de][0]), ('mask', place['mask'], texts[de][1])):
            data = files[NAME_FILES[part]]
            msgs, info = tlfmt.msbt(data)
            if label not in msgs:
                continue
            tag = GRAMMAR.match(msgs[label][0])
            txt = (tag.group(1) if tag else '') + new
            index = list(msgs.keys()).index(label)
            files[NAME_FILES[part]] = tlfmt.msbt_replace_texts(data, {index: tlfmt.encode_txt(txt, info['endian'])})
        out[rel] = _zs(tlfmt.sarc_build(files))
    return out


def mark_owned(savedir, flags, cloth_indices):
    """the Roblox suit and head (colour 0) count as bought - they are in the wardrobe"""
    import ugcsave as U
    INV = 'Player.#3b098df0.#b94d3a0b.'
    P = U.Ed(os.path.join(savedir, 'Player.sav'), U.load_flags(flags))
    now = max(P.get(INV + '#d73851ab')) + 60
    for ci in cloth_indices:
        e = ci * 16
        if U.mm('Obtained') != P.get(INV + 'State', e):
            P.set(INV + 'State', 'Obtained', e)
            if not P.get(INV + '#3c82867e', e):
                P.set(INV + '#3c82867e', now, e)
            P.set(INV + '#d73851ab', now, e)
    P.save()


def install(res, ch, env, mii=None, log=say):
    """write the built files into the mod folder (spot of the character), set names and tables, make the suit + head
    owned and put them on the island Mii with index `mii` (None: files only) - after a backup of everything that
    changes. Ryujinx must be closed: the game keeps the island in memory and would write its old state back."""
    run = emulator_running()
    if run:
        raise RuntimeError('Ryujinx is still running. Close Ryujinx completely and press the button again.')
    mod, save = env['mod'], env['save']
    romfs = os.path.join(mod, 'romfs')
    place = res['place']
    V = templates()
    it = V['items']
    files = dict(res['files'])

    def current(rel):
        """the mod's file, or the game's own from templates/game (own mod folder, first time)"""
        p = os.path.join(romfs, *rel.split('/'))
        return p if os.path.exists(p) else os.path.join(BASE, rel.split('/')[-1])
    # table: the costume row shows the new model (body rows of new spots are redirected the first time)
    cmp_root, cmp_ver = byml_write.parse(tlfmt.read(current(CMP_REL)))
    rows = {r['__RowId'][1]: r for r in cmp_root}
    row = rows[it[place['body']]['model_hash']]
    cmp_changed = False
    if row['ModelUnit']['ModelName'][1] != 'ClothAllRoblox' + place['tag']:
        BRM.redirect_body_row(row, 'ClothAllRoblox' + place['tag'])
        cmp_changed = True
    for k, v in (it[place['mask']].get('row_fix') or {}).items():
        mrow = rows[it[place['mask']]['model_hash']]
        if mrow['ModelUnit'].get(k) != ('S', v):
            mrow['ModelUnit'][k] = ('S', v)
            cmp_changed = True
    if cmp_changed or not os.path.exists(os.path.join(romfs, *CMP_REL.split('/'))):
        files[CMP_REL] = _zs(byml_write.write(cmp_root, cmp_ver))
    tab = env['table']
    rows_t = read_table(env)
    # the same Roblox character again: keep its short name (wardrobe items)
    before = next((r for r in rows_t if r.get('kuerzel') == place['tag'] and r.get('roblox_name') == ch['key']), None)
    nick = (before.get('mii_name') or '')[:10] if before else ''
    nick = nick or mii_nickname(ch)
    files.update(set_names(romfs, place, nick))      # only text archives the mod has; else the game's names
    R = restbl.Restbl(tlfmt.read(current(RT_REL)))
    BRM.update_restbl(R, files)       # entries only grow (also the text archives, for long names)
    files[RT_REL] = _zs(R.write())
    mii_name = None
    if mii is not None:
        names = {m['index']: m['name'] for m in island_miis(env)}
        if mii not in names:
            raise RuntimeError('This Mii is not on the island any more. Choose the Mii again.')
        mii_name = names[mii]
    # backup: every file that gets replaced, the table and the save
    stamp = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(env['backups'], exist_ok=True)
    zp = os.path.join(env['backups'], 'RbxToMii_before_%s_%s.zip' % (safe_name(ch['key']), stamp))
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            p = os.path.join(romfs, *rel.split('/'))
            if os.path.exists(p):
                z.write(p, 'romfs/' + rel)
        if os.path.exists(tab):
            z.write(tab, os.path.relpath(tab, mod).replace(os.sep, '/'))
        for r, dirs, fs in os.walk(save):
            for f in fs:
                full = os.path.join(r, f)
                z.write(full, os.path.join('save', os.path.relpath(full, save)))
    # write everything
    for rel, data in files.items():
        p = os.path.join(romfs, *rel.split('/'))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p + '.new', 'wb').write(data)
        os.replace(p + '.new', p)
    if env.get('mode') == 'own' and not os.path.exists(os.path.join(mod, 'README.txt')):
        open(os.path.join(mod, 'README.txt'), 'w', encoding='utf-8').write(OWN_README)
    new_row = {'id': 'spot%d' % place['nr'], 'roblox_name': ch['key'], 'mii_name': (before or {}).get('mii_name') or nick,
               'kuerzel': place['tag'],
               'koerper_kleidung': place['body'], 'koerper_index': it[place['body']]['cloth_index'],
               'kopf_kleidung': place['mask'], 'kopf_index': it[place['mask']]['cloth_index'],
               'name_de_koerper': 'Roblox-Anzug ' + nick, 'name_de_kopf': 'Roblox-Kopf ' + nick,
               'name_en_koerper': 'Roblox suit: ' + nick, 'name_en_kopf': 'Roblox head: ' + nick,
               'nachgebaut': '-', 'kopf_quelle': 'RbxToMii %s' % time.strftime('%Y-%m-%d %H:%M'),
               'fehlt': '; '.join('%s %s' % (m['part'], m['name']) for m in ch.get('missing', [])) or '-'}
    if mii_name is not None:
        new_row['mii_name'] = mii_name
    bodies = [p['body'] for p in V['places']]
    rows_t = [r for r in rows_t if r.get('kuerzel') != place['tag'] and r.get('koerper_kleidung') != place['body']] + [new_row]
    rows_t.sort(key=lambda r: bodies.index(r.get('koerper_kleidung')) if r.get('koerper_kleidung') in bodies else 99)
    os.makedirs(os.path.dirname(tab), exist_ok=True)
    json.dump(rows_t, open(tab, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for sub in ('0', '1'):
        sd = os.path.join(save, sub)
        if os.path.exists(os.path.join(sd, 'Player.sav')):
            mark_owned(sd, env['flags'], [it[place['body']]['cloth_index'], it[place['mask']]['cloth_index']])
        if mii is not None and os.path.exists(os.path.join(sd, 'Mii.sav')):
            M = _mii_ed(sd, env['flags'])
            dress_one(M, mii, place['body'], place['mask'])
            M.save(os.path.join(sd, 'Mii.sav.new'))
            os.replace(os.path.join(sd, 'Mii.sav.new'), os.path.join(sd, 'Mii.sav'))
    if mii_name:
        log('%s now wears the Roblox look of %s.' % (mii_name, ch['key']))
    return {'backup': zp, 'files': len(files), 'nick': nick, 'mii': mii_name}


# ================================================================ CLI for tests
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    cmd = sys.argv[1]
    if cmd == 'env':
        for ry in ryujinx_candidates():
            e = environment(ry)
            print(json.dumps(e, ensure_ascii=False, indent=1))
            if e['mode']:
                print([(p['nr'], p['tag'], p['now']) for p in places(e)])
    elif cmd == 'fetch':
        ch = fetch_player(sys.argv[2])
        print(json.dumps({k: v for k, v in ch.items() if k != 'rig'}, ensure_ascii=False, indent=1)[:3000])
    elif cmd in ('build', 'install'):
        rig_dir = os.path.join(DATA, 'players', safe_name(sys.argv[2]))
        ch = json.load(open(os.path.join(rig_dir, 'report.json'), encoding='utf-8'))
        ch['rig'] = json.load(open(os.path.join(rig_dir, 'rig.json'), encoding='utf-8'))
        env = environment(ryujinx_candidates()[0])
        pl = [p for p in places(env) if p['nr'] == int(sys.argv[3])][0]
        res = build(ch, pl)
        res['preview'][0].save(os.path.join(rig_dir, 'preview.png'))
        print(res['stats'])
        if cmd == 'install':           # python -I rbxtomii.py install <player> <spot> [Mii index] (folder: RBXTOMII_RYUJINX)
            assert not env['problems'], env['problems']
            print(install(res, ch, env, int(sys.argv[4]) if len(sys.argv) > 4 else None))
    elif cmd == 'miis':
        env = environment(ryujinx_candidates()[0])
        for m in island_miis(env):
            print(m['index'], m['name'], ('<- Roblox ' + m['roblox']) if m['roblox'] else '')
