"""Put a Roblox R15 avatar together from its public avatar description, the way Roblox does it:
body parts (R15ArtistIntent) scaled by height/width/depth/head and the Rthro sliders (body type, proportions),
joined at their rig attachments, accessories hung on the matching body attachment and scaled like the part they sit
on, layered clothing with its cages, classic clothing, body colours.
Output: one character in the rigs.json format that rbxchar.py builds from (world CFrames, sizes, asset ids).

Scale rule (checked against Roblox's own result for 5 avatars): a part of scale type T (Classic, ProportionsNormal,
ProportionsSlender) gets  ((1-b) * 1 + b*(1-p) * N + b*p * S) / V_T  where N, S are the part's Classic->Normal and
Classic->Slender vectors, V_T the vector of its own type (Classic = 1), b = body type, p = proportions; times
(width, height, depth) for body parts, times head scale for the head and everything on it."""
import re, copy
import numpy as np

# Classic -> ProportionsNormal / ProportionsSlender per R15 part (Roblox engine values, read back from Roblox's
# own scaling of avatars: exact to 3-4 decimals)
_ARM_N, _ARM_S = (1.129, 1.342, 1.132), (1.0036, 1.2079, 1.0062)
_HAND_N, _HAND_S = (1.0, 1.0, 1.0), (0.9476, 1.174, 1.0942)
_UPLEG_N, _UPLEG_S = (1.023, 1.506, 1.023), (0.9761, 1.4009, 0.9093)
_LOLEG_N, _LOLEG_S = (1.023, 1.506, 1.023), (0.9761, 1.3005, 0.9093)
_FOOT_N, _FOOT_S = (1.079, 1.267, 1.129), (1.0296, 1.1333, 1.0036)
SCALE = {
    'UpperTorso': ((1.033, 1.309, 1.14), (0.9053, 1.2042, 1.0133)),
    'LowerTorso': ((1.033, 1.309, 1.14), (0.9857, 1.0046, 1.0133)),
    'LeftUpperArm': (_ARM_N, _ARM_S), 'LeftLowerArm': (_ARM_N, _ARM_S), 'LeftHand': (_HAND_N, _HAND_S),
    'RightUpperArm': (_ARM_N, _ARM_S), 'RightLowerArm': (_ARM_N, _ARM_S), 'RightHand': (_HAND_N, _HAND_S),
    'LeftUpperLeg': (_UPLEG_N, _UPLEG_S), 'LeftLowerLeg': (_LOLEG_N, _LOLEG_S), 'LeftFoot': (_FOOT_N, _FOOT_S),
    'RightUpperLeg': (_UPLEG_N, _UPLEG_S), 'RightLowerLeg': (_LOLEG_N, _LOLEG_S), 'RightFoot': (_FOOT_N, _FOOT_S),
    'Head': ((0.625, 0.625, 0.625), (0.5947, 0.625, 0.5947)),
}
# rigid accessories on the body (not on the head)
ACC_N = (0.9862, 1.2285, 1.0823)
ACC_S = tuple(a * s / n for a, s, n in zip(ACC_N, SCALE['UpperTorso'][1], SCALE['UpperTorso'][0]))

BODY = ['Head', 'UpperTorso', 'LowerTorso', 'LeftUpperArm', 'LeftLowerArm', 'LeftHand', 'RightUpperArm',
        'RightLowerArm', 'RightHand', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot', 'RightUpperLeg', 'RightLowerLeg',
        'RightFoot']
# (child, parent, joint attachment)
TREE = [('UpperTorso', 'LowerTorso', 'WaistRigAttachment'), ('Head', 'UpperTorso', 'NeckRigAttachment'),
        ('LeftUpperArm', 'UpperTorso', 'LeftShoulderRigAttachment'), ('LeftLowerArm', 'LeftUpperArm', 'LeftElbowRigAttachment'),
        ('LeftHand', 'LeftLowerArm', 'LeftWristRigAttachment'), ('RightUpperArm', 'UpperTorso', 'RightShoulderRigAttachment'),
        ('RightLowerArm', 'RightUpperArm', 'RightElbowRigAttachment'), ('RightHand', 'RightLowerArm', 'RightWristRigAttachment'),
        ('LeftUpperLeg', 'LowerTorso', 'LeftHipRigAttachment'), ('LeftLowerLeg', 'LeftUpperLeg', 'LeftKneeRigAttachment'),
        ('LeftFoot', 'LeftLowerLeg', 'LeftAnkleRigAttachment'), ('RightUpperLeg', 'LowerTorso', 'RightHipRigAttachment'),
        ('RightLowerLeg', 'RightUpperLeg', 'RightKneeRigAttachment'), ('RightFoot', 'RightLowerLeg', 'RightAnkleRigAttachment')]
BODY_ASSET = {27: ['UpperTorso', 'LowerTorso'], 28: ['RightUpperArm', 'RightLowerArm', 'RightHand'],
              29: ['LeftUpperArm', 'LeftLowerArm', 'LeftHand'], 30: ['LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot'],
              31: ['RightUpperLeg', 'RightLowerLeg', 'RightFoot'], 79: ['Head'], 17: ['Head']}
ACCESSORY_ASSET = {8, 41, 42, 43, 44, 45, 46, 47, 64, 65, 66, 67, 68, 69, 70, 71, 72, 76, 77}
NAMES = {2: 'T-shirt', 8: 'Hat', 11: 'Shirt', 12: 'Pants', 17: 'Head', 18: 'Face', 27: 'Torso', 28: 'Right arm',
         29: 'Left arm', 30: 'Left leg', 31: 'Right leg', 41: 'Hair', 42: 'Face accessory', 43: 'Neck accessory',
         44: 'Shoulder accessory', 45: 'Front accessory', 46: 'Back accessory', 47: 'Waist accessory',
         64: 'T-shirt (3D)', 65: 'Shirt (3D)', 66: 'Pants (3D)', 67: 'Jacket (3D)', 68: 'Sweater (3D)', 69: 'Shorts (3D)',
         70: 'Left shoe (3D)', 71: 'Right shoe (3D)', 72: 'Dress / skirt (3D)', 76: 'Eyebrows', 77: 'Eyelashes',
         78: 'Mood', 79: 'Head'}
# body attachment -> part, when a part has no attachment of that name (old bodies)
ATTACH_PART = {'HatAttachment': 'Head', 'HairAttachment': 'Head', 'FaceFrontAttachment': 'Head',
               'FaceCenterAttachment': 'Head', 'NeckAttachment': 'UpperTorso', 'BodyFrontAttachment': 'UpperTorso',
               'BodyBackAttachment': 'UpperTorso', 'LeftCollarAttachment': 'UpperTorso',
               'RightCollarAttachment': 'UpperTorso', 'WaistFrontAttachment': 'LowerTorso',
               'WaistBackAttachment': 'LowerTorso', 'WaistCenterAttachment': 'LowerTorso',
               'LeftShoulderAttachment': 'LeftUpperArm', 'RightShoulderAttachment': 'RightUpperArm',
               'LeftGripAttachment': 'LeftHand', 'RightGripAttachment': 'RightHand',
               'LeftFootAttachment': 'LeftFoot', 'RightFootAttachment': 'RightFoot'}
COLOR_KEYS = {'headColor3': 'HeadColor3', 'torsoColor3': 'TorsoColor3', 'leftArmColor3': 'LeftArmColor3',
              'rightArmColor3': 'RightArmColor3', 'leftLegColor3': 'LeftLegColor3', 'rightLegColor3': 'RightLegColor3'}
COLOR_ID_KEYS = {'headColorId': 'HeadColor3', 'torsoColorId': 'TorsoColor3', 'leftArmColorId': 'LeftArmColor3',
                 'rightArmColorId': 'RightArmColor3', 'leftLegColorId': 'LeftLegColor3', 'rightLegColorId': 'RightLegColor3'}
ID_RE = re.compile(r'(?:id=|rbxassetid://|/asset/\?id=|^)(\d{3,})', re.I)


# ---------------------------------------------------------------- small helpers
def aid(v):
    if v is None:
        return None
    if hasattr(v, 'uri'):
        v = v.uri
    if isinstance(v, bytes):
        v = v.decode('utf-8', 'replace')
    if not isinstance(v, str):
        return None
    m = ID_RE.search(v.strip())
    return int(m.group(1)) if m else None


def v3(x):
    return None if x is None else [float(x.x), float(x.y), float(x.z)]


def cfd(c):
    return None if c is None else {'p': list(c.pos), 'r': list(c.rot)}


def mat(cf):
    M = np.eye(4)
    M[:3, :3] = np.array(cf['r'], float).reshape(3, 3)
    M[:3, 3] = cf['p']
    return M


def cf_of(M):
    return {'p': [float(x) for x in M[:3, 3]], 'r': [float(x) for x in M[:3, :3].reshape(-1)]}


def color(c):
    if c is None:
        return None
    if hasattr(c, 'rgb255'):
        r = c.rgb255
        r = r() if callable(r) else r
        return list(r) if r else None
    return None


def part_info(p):
    """MeshPart / Part (+ SpecialMesh) -> dict in the rigs.json part format (positions still local)"""
    pp = p.props
    d = {'name': p.name, 'class': p.class_name, 'size': v3(pp.get('size') or pp.get('Size')),
         'color': color(pp.get('Color3uint8') or pp.get('Color') or pp.get('BrickColor')),
         'transparency': float(pp.get('Transparency', 0.0) or 0.0)}
    if p.class_name == 'MeshPart':
        d['mesh'] = aid(pp.get('MeshId')) or aid(pp.get('MeshContent'))
        d['tex'] = aid(pp.get('TextureID')) or aid(pp.get('TextureContent'))
        d['initial_size'] = v3(pp.get('InitialSize')) or d['size']
        d['double_sided'] = bool(pp.get('DoubleSided'))
    att, st = {}, None
    for c in p.children:
        cp = c.props
        if c.class_name == 'Attachment':
            att[c.name] = cfd(cp.get('CFrame'))
        elif c.class_name == 'Vector3Value' and c.name.endswith('Attachment'):   # legacy heads
            val = cp.get('Value')
            att[c.name] = {'p': v3(val), 'r': [1, 0, 0, 0, 1, 0, 0, 0, 1]}
        elif c.class_name == 'StringValue' and c.name == 'AvatarPartScaleType':
            st = cp.get('Value') or None
        elif c.class_name == 'SpecialMesh':
            d['special'] = {'type': int(cp.get('MeshType', 5) or 5), 'mesh': aid(cp.get('MeshId')),
                            'tex': aid(cp.get('TextureId')), 'scale': v3(cp.get('Scale')) or [1, 1, 1],
                            'offset': v3(cp.get('Offset')) or [0, 0, 0], 'vcolor': v3(cp.get('VertexColor'))}
        elif c.class_name == 'SurfaceAppearance':
            d['surface'] = {k: aid(cp.get(k)) for k in ('ColorMap', 'NormalMap', 'MetalnessMap', 'RoughnessMap') if cp.get(k)}
            d['surface']['alpha_mode'] = int(cp.get('AlphaMode', 0) or 0)
            d['surface']['color'] = color(cp.get('Color'))
        elif c.class_name == 'WrapLayer':
            d['wrap_layer'] = {'cage': aid(cp.get('CageMeshId')) or aid(cp.get('CageMeshContent')),
                               'reference': aid(cp.get('ReferenceMeshId')) or aid(cp.get('ReferenceMeshContent')),
                               'order': cp.get('Order'), 'puffiness': cp.get('Puffiness'),
                               'bind_offset': cfd(cp.get('BindOffset')), 'cage_origin': cfd(cp.get('CageOrigin')),
                               'reference_origin': cfd(cp.get('ReferenceOrigin')), 'import_origin': cfd(cp.get('ImportOrigin'))}
        elif c.class_name == 'WrapTarget':
            d['wrap_target'] = {'cage': aid(cp.get('CageMeshId')) or aid(cp.get('CageMeshContent')),
                                'cage_origin': cfd(cp.get('CageOrigin')), 'import_origin': cfd(cp.get('ImportOrigin'))}
        elif c.class_name in ('Decal', 'Texture'):
            d.setdefault('decals', []).append({'tex': aid(cp.get('Texture')) or aid(cp.get('TextureContent')),
                                               'face': int(cp.get('Face', 5) or 5), 'color': color(cp.get('Color3')),
                                               'transparency': float(cp.get('Transparency', 0.0) or 0.0)})
    d['attachments'] = att
    d['scale_type'] = st or 'Classic'
    return d


def type_vec(n, s, t):
    return np.ones(3) if t == 'Classic' else np.array(n if t == 'ProportionsNormal' else s, float)


def conv(n, s, t, b, p):
    """scale of something of scale type t on an avatar with body type b and proportions p"""
    target = (1 - b) * np.ones(3) + b * (1 - p) * np.array(n, float) + b * p * np.array(s, float)
    return target / type_vec(n, s, t)


def scale_part(d, s):
    s = np.asarray(s, float)
    d['size'] = [float(x) for x in np.array(d['size'], float) * s]
    for an, cf in d['attachments'].items():
        cf['p'] = [float(x) for x in np.array(cf['p'], float) * s]
    if 'special' in d:
        d['special']['scale'] = [float(x) for x in np.array(d['special']['scale'], float) * s]
        d['special']['offset'] = [float(x) for x in np.array(d['special']['offset'], float) * s]


def r15_parts(doc):
    """the R15 body parts of a body-part asset (Roblox uses R15ArtistIntent when there is one)"""
    roots = {r.name: r for r in doc.roots}
    for fn in ('R15ArtistIntent', 'R15Fixed', 'R15'):
        if fn in roots:
            return [c for c in roots[fn].children if c.class_name in ('MeshPart', 'Part')]
    out = []
    for i in doc.all():
        if i.class_name in ('MeshPart', 'Part') and i.name in BODY:
            out.append(i)
    return out


def euler_xyz(rx, ry, rz):
    a, b, c = np.radians([rx, ry, rz])
    X = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Y = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Z = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return X @ Y @ Z


def body_colors(av):
    out = {}
    for k, nk in COLOR_KEYS.items():
        h = (av.get('bodyColor3s') or {}).get(k)
        if h:
            h = h.lstrip('#')
            out[nk] = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    if not out and av.get('bodyColors'):
        import rbxfile
        for k, nk in COLOR_ID_KEYS.items():
            cid = av['bodyColors'].get(k)
            if cid is not None:
                c = rbxfile.rbx.BrickColor(int(cid))
                rgb = c.rgb255() if callable(c.rgb255) else c.rgb255
                if rgb:
                    out[nk] = list(rgb)
    return out


def scales_of(av):
    s = av.get('scales') or {}
    return {'HeightScale': float(s.get('height', 1)), 'WidthScale': float(s.get('width', 1)),
            'DepthScale': float(s.get('depth', s.get('width', 1))), 'HeadScale': float(s.get('head', 1)),
            'ProportionScale': float(s.get('proportion', 0)), 'BodyTypeScale': float(s.get('bodyType', 0))}


# ---------------------------------------------------------------- the avatar
def assemble(av, get_model, default_doc=None, log=print, default_parts=None):
    """av: avatar description (avatar API); get_model(asset_id, fmt) -> rbxfile.Doc or None (no access);
    default_doc: Roblox's default R15 character (content/avatar/characterR15DynamicHeadV2.rbxm).
    -> (rig dict, report dict)"""
    sc = scales_of(av)
    b, p = sc['BodyTypeScale'], sc['ProportionScale']
    body_s = np.array([sc['WidthScale'], sc['HeightScale'], sc['DepthScale']])
    head_s = sc['HeadScale']
    report = {'ok': [], 'missing': [], 'default': []}
    assets = [a for a in av.get('assets', []) if a.get('assetType')]
    meta = {a['id']: a.get('meta') or {} for a in assets}
    body = {}
    # 1) body parts and head
    for a in assets:
        t = a['assetType']['id']
        if t not in BODY_ASSET:
            continue
        doc = get_model(a['id'], 'avatar_meshpart_head' if t in (17, 79) else None)
        if doc is None:
            report['missing'].append((a['id'], NAMES.get(t, str(t)), a.get('name')))
            continue
        got = []
        for mp in r15_parts(doc):
            if mp.name in BODY_ASSET[t]:
                body[mp.name] = part_info(mp)
                got.append(mp.name)
        if t in (17, 79) and 'Head' not in got:
            heads = [i for i in doc.all() if i.class_name == 'MeshPart' and i.name == 'Head']
            if heads:
                body['Head'] = part_info(heads[0])
                got.append('Head')
        (report['ok'] if got else report['missing']).append((a['id'], NAMES.get(t, str(t)), a.get('name')))
    # Roblox's default body parts / head for every slot without an asset
    if default_parts is None:
        default_parts = {i.name: part_info(i) for i in default_doc.all() if i.class_name in ('MeshPart', 'Part') and i.name in BODY}
    for pn in BODY:
        if pn not in body:
            body[pn] = copy.deepcopy(default_parts[pn])
            body[pn]['scale_type'] = 'Classic'
            report['default'].append(pn)
    # 2) scale (sizes and attachment positions)
    pscale = {}
    for pn, d in body.items():
        n, s = SCALE[pn]
        f = conv(n, s, d['scale_type'], b, p) * (head_s if pn == 'Head' else body_s)
        scale_part(d, f)
        pscale[pn] = f
    # 3) join at the rig attachments (LowerTorso at the origin, Roblox axes: character looks along -Z)
    world = {'LowerTorso': np.eye(4)}
    for child, parent, an in TREE:
        pa = body[parent]['attachments'].get(an)
        ca = body[child]['attachments'].get(an)
        if pa is None or ca is None:
            raise ValueError('Gelenk %s fehlt an %s/%s' % (an, parent, child))
        world[child] = world[parent] @ mat(pa) @ np.linalg.inv(mat(ca))
    rig = {'parts': [], 'accessories': [], 'joints': [], 'scales': sc, 'body_colors': body_colors(av),
           'source': 'roblox-api', 'avatar_type': av.get('playerAvatarType')}
    for pn in BODY:
        d = body[pn]
        d['path'] = pn
        d['cf'] = cf_of(world[pn])
        d.pop('scale_type', None)
        rig['parts'].append(d)
    body_att = {}
    for pn in BODY:
        for an in body[pn]['attachments']:
            body_att.setdefault(an, pn)
    # 4) accessories (rigid and layered)
    used = set()
    for a in assets:
        t = a['assetType']['id']
        if t not in ACCESSORY_ASSET:
            continue
        doc = get_model(a['id'], 'avatar_meshpart_accessory')
        if doc is None:
            report['missing'].append((a['id'], NAMES.get(t, str(t)), a.get('name')))
            continue
        accs = [i for i in doc.all() if i.class_name in ('Accessory', 'Hat')]
        if not accs:
            report['missing'].append((a['id'], NAMES.get(t, str(t)), a.get('name')))
            continue
        acc = accs[0]
        h = acc.find_first_child('Handle')
        if h is None:
            report['missing'].append((a['id'], NAMES.get(t, str(t)), a.get('name')))
            continue
        hd = part_info(h)
        att_names = [an for an in hd['attachments'] if an in body_att or an in ATTACH_PART]
        layered = 'wrap_layer' in hd
        if not att_names and not layered:
            report['missing'].append((a['id'], NAMES.get(t, str(t)), a.get('name')))
            continue
        an = att_names[0] if att_names else None
        target = body_att.get(an) or ATTACH_PART.get(an) or 'UpperTorso'
        mt = meta.get(a['id']) or {}
        if layered:
            f = np.ones(3)
            if mt.get('order') is not None:
                hd['wrap_layer']['order'] = int(mt['order'])
            if mt.get('puffiness') is not None:
                hd['wrap_layer']['puffiness'] = float(mt['puffiness'])
        elif target == 'Head':
            n, s = SCALE['Head']
            f = conv(n, s, hd['scale_type'], b, p) * head_s
        else:
            f = conv(ACC_N, ACC_S, hd['scale_type'], b, p) * body_s
        # avatar-editor adjustments of rigid accessories (position / rotation / scale)
        adj = np.eye(4)
        if not layered and mt:
            if mt.get('scale'):
                f = f * np.array([mt['scale'].get('X', 1), mt['scale'].get('Y', 1), mt['scale'].get('Z', 1)], float)
            if mt.get('rotation'):
                adj[:3, :3] = euler_xyz(mt['rotation'].get('X', 0), mt['rotation'].get('Y', 0), mt['rotation'].get('Z', 0))
            if mt.get('position'):
                adj[:3, 3] = [mt['position'].get('X', 0), mt['position'].get('Y', 0), mt['position'].get('Z', 0)]
        scale_part(hd, f)
        if an is not None and an in body[target]['attachments']:
            M = world[target] @ mat(body[target]['attachments'][an]) @ adj @ np.linalg.inv(mat(hd['attachments'][an]))
        elif an is not None:
            M = world[target] @ adj @ np.linalg.inv(mat(hd['attachments'][an]))
        else:
            M = world[target]
        name = acc.name
        k = 2
        while name in used:
            name = '%s (%d)' % (acc.name, k)
            k += 1
        used.add(name)
        hd['path'] = name + '.Handle'
        hd['cf'] = cf_of(M)
        hd.pop('scale_type', None)
        rig['parts'].append(hd)
        rig['accessories'].append({'name': name, 'type': int(acc.props.get('AccessoryType', 0) or 0),
                                   'handle': hd['path'], 'asset': a['id'], 'asset_type': t})
        report['ok'].append((a['id'], NAMES.get(t, str(t)), a.get('name')))
    # 5) classic clothing
    for a in assets:
        t = a['assetType']['id']
        if t not in (2, 11, 12, 18):
            continue
        doc = get_model(a['id'], None)
        if doc is None:
            report['missing'].append((a['id'], NAMES.get(t, str(t)), a.get('name')))
            continue
        for i in doc.all():
            if i.class_name == 'Shirt':
                rig['shirt'] = aid(i.props.get('ShirtTemplate'))
                rig['shirt_asset'] = a['id']
            elif i.class_name == 'Pants':
                rig['pants'] = aid(i.props.get('PantsTemplate'))
                rig['pants_asset'] = a['id']
            elif i.class_name == 'ShirtGraphic':
                rig['tshirt'] = aid(i.props.get('Graphic'))
            elif i.class_name == 'Decal' and t == 18:
                rig['face'] = aid(i.props.get('Texture'))
        report['ok'].append((a['id'], NAMES.get(t, str(t)), a.get('name')))
    # 6) mood (face expression of dynamic heads)
    moods = [a for a in assets if a['assetType']['id'] == 78]
    rig['mood'] = {'id': moods[0]['id'], 'name': moods[0].get('name')} if moods else {'id': 0, 'name': 'Smile'}
    return rig, report
