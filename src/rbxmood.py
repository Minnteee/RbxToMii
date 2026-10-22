"""Roblox moods: the face a dynamic head shows all the time (a one-keyframe KeyframeSequence of FACS values).
The mood asset itself needs a login to download, but Roblox keeps every mood it played in its local cache; the public
catalog API tells the mood's name, which is matched against the cached KeyframeSequence names.
Mood id 0 = Roblox's default mood "Smile"."""
import os, re, json, urllib.request, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))

# FACS control names (animations / FaceControls) -> short names used inside head meshes
FACS_ABBR = {
    'Corrugator': 'c_COR', 'ChinRaiser': 'c_CR', 'ChinRaiserUpperLip': 'c_CRUL', 'EyesLookDown': 'c_ELD',
    'EyesLookLeft': 'c_ELL', 'EyesLookRight': 'c_ELR', 'EyesLookUp': 'c_ELU', 'Funneler': 'c_FN', 'FlatPucker': 'c_FP',
    'JawDrop': 'c_JD', 'JawLeft': 'c_JL', 'JawRight': 'c_JR', 'LowerLipSuck': 'c_LLS', 'LipPresser': 'c_LP',
    'LipsTogether': 'c_LPT', 'MouthLeft': 'c_ML', 'MouthRight': 'c_MR', 'Pucker': 'c_PK', 'TongueDown': 'c_TD',
    'TongueOut': 'c_TO', 'TongueUp': 'c_TU', 'UpperLipSuck': 'c_ULS',
}
for side, s in (('Left', 'l'), ('Right', 'r')):
    for long, short in (('BrowLowerer', 'BL'), ('CheekPuff', 'CHP'), ('CheekRaiser', 'CHR'), ('Dimpler', 'DM'),
                        ('EyeClosed', 'EC'), ('EyeUpperLidRaiser', 'EULR'), ('InnerBrowRaiser', 'IBR'),
                        ('LipCornerDown', 'LCD'), ('LipCornerPuller', 'LCP'), ('LowerLipDepressor', 'LLD'),
                        ('LipStretcher', 'LS'), ('NoseWrinkler', 'NW'), ('OuterBrowRaiser', 'OBR'),
                        ('UpperLipRaiser', 'ULR')):
        FACS_ABBR[side + long] = '%s_%s' % (s, short)

DEFAULT_MOOD_NAME = 'Smile'


def _parser():
    import rbxfile
    return rbxfile.rbx


def norm(name):
    n = re.sub(r'[^a-z0-9]', '', (name or '').lower())
    for tail in ('mood2', 'mood'):
        if n.endswith(tail):
            n = n[:-len(tail)]
    if n.startswith('mood'):
        n = n[4:]
    return n


def cached_moods(idx, root=None):
    """{normalised name: {FACS name: value}} for every one-keyframe face pose in Roblox's cache"""
    import rbxcache
    rbx = _parser()
    out = {}
    for m in idx['models']:
        if m.get('bytes', 0) > 20000:     # moods are tiny one-keyframe animations
            continue
        b = rbxcache.read(m['h'], root)
        if not b or b'KeyframeSequence' not in b or b'NumberPose' not in b:
            continue
        try:
            doc = rbx.read_bytes(b)
        except Exception:
            continue
        ks = [i for i in doc.instances.values() if i.class_name == 'KeyframeSequence']
        kf = [i for i in doc.instances.values() if i.class_name == 'Keyframe']
        if not ks or len(kf) > 3:
            continue
        first = min(kf, key=lambda k: k.props.get('Time', 0.0)) if kf else None
        vals = {}
        for i in doc.instances.values():
            if i.class_name != 'NumberPose':
                continue
            if first is not None and not _below(i, first):
                continue
            vals[i.props.get('Name')] = float(i.props.get('Value', 0.0))
        if vals:
            out.setdefault(norm(ks[0].props.get('Name')), vals)
    return out


def _below(inst, anc):
    p = getattr(inst, 'parent', None)
    while p is not None:
        if p is anc:
            return True
        p = getattr(p, 'parent', None)
    return True   # parser without parent links: accept


def asset_name(aid):
    url = 'https://economy.roblox.com/v2/assets/%d/details' % aid
    try:
        j = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}), timeout=30).read())
        return j.get('Name')
    except Exception:
        return None


def mood_weights(mood_id, moods, name_cache=None):
    """FACS weights (short names) for a mood asset id; None if it is not in the cache (head stays neutral)"""
    if not mood_id:
        name = DEFAULT_MOOD_NAME
    else:
        name = (name_cache or {}).get(str(mood_id)) or asset_name(int(mood_id))
        if name_cache is not None and name:
            name_cache[str(mood_id)] = name
    vals = moods.get(norm(name)) if name else None
    if vals is None:
        return None, name
    return {FACS_ABBR[k]: v for k, v in vals.items() if k in FACS_ABBR}, name
