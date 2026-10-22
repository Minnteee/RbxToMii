"""Retarget an assembled Roblox character (rbxchar pieces, Roblox world space, default pose) onto the Mii rig:
 * body model: every piece except head attachments, in the Mii T-pose model space, skinned to the Mii skeleton
 * head model: head + head attachments, rigid, in the full-mask (Head bone) space
Each Roblox part P gets an affine map  x -> Jm + R (S (F (x - Jr)))  (F turns Roblox -Z-facing into Mii +Z-facing)."""
import numpy as np

import bfres

F = np.diag([-1.0, 1.0, -1.0])
LIMBS = {  # part: (proximal rig joint, distal rig joint or None, mii proximal bone, mii distal bone, mii skin bone)
    'LeftUpperArm': ('LeftShoulderRigAttachment', 'LeftElbowRigAttachment', 'Arm_1_L', 'Arm_2_L', 'Arm_1_L'),
    'LeftLowerArm': ('LeftElbowRigAttachment', 'LeftWristRigAttachment', 'Arm_2_L', 'Wrist_L', 'Arm_2_L'),
    'LeftHand': ('LeftWristRigAttachment', None, 'Wrist_L', None, 'Wrist_L'),
    'RightUpperArm': ('RightShoulderRigAttachment', 'RightElbowRigAttachment', 'Arm_1_R', 'Arm_2_R', 'Arm_1_R'),
    'RightLowerArm': ('RightElbowRigAttachment', 'RightWristRigAttachment', 'Arm_2_R', 'Wrist_R', 'Arm_2_R'),
    'RightHand': ('RightWristRigAttachment', None, 'Wrist_R', None, 'Wrist_R'),
    'LeftUpperLeg': ('LeftHipRigAttachment', 'LeftKneeRigAttachment', 'Leg_1_L', 'Leg_2_L', 'Leg_1_L'),
    'LeftLowerLeg': ('LeftKneeRigAttachment', 'LeftAnkleRigAttachment', 'Leg_2_L', 'Ankle_L', 'Leg_2_L'),
    'LeftFoot': ('LeftAnkleRigAttachment', None, 'Ankle_L', None, 'Ankle_L'),
    'RightUpperLeg': ('RightHipRigAttachment', 'RightKneeRigAttachment', 'Leg_1_R', 'Leg_2_R', 'Leg_1_R'),
    'RightLowerLeg': ('RightKneeRigAttachment', 'RightAnkleRigAttachment', 'Leg_2_R', 'Ankle_R', 'Leg_2_R'),
    'RightFoot': ('RightAnkleRigAttachment', None, 'Ankle_R', None, 'Ankle_R'),
}
R15_ALIAS = {'Root': 'LowerTorso', 'HumanoidRootNode': 'LowerTorso', 'DynamicHead': 'Head'}
# torso skin weights by Mii height (from the game's own costume): Waist -> Spine_1 -> Spine_2 -> Neck
TORSO_Y = np.array([0.55, 0.62, 0.70, 0.78, 0.85, 0.92, 1.08, 1.16])
TORSO_W = {'Waist': [1.0, 0.85, 0.50, 0.25, 0.05, 0.0, 0.0, 0.0],
           'Spine_1': [0.0, 0.15, 0.50, 0.60, 0.45, 0.05, 0.0, 0.0],
           'Spine_2': [0.0, 0.0, 0.0, 0.15, 0.50, 0.95, 0.9, 0.2],
           'Neck': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8]}


def rot_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 degrees: any perpendicular axis
        p = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 0, 1.0])
        v = np.cross(a, p)
        v /= np.linalg.norm(v)
        return 2 * np.outer(v, v) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


class PartMap:
    """x (Roblox world) -> Jm + R @ S @ F @ (x - Jr), S = k*I + (s_ax - k) a a^T  (+ optional per-axis torso scale)"""

    def __init__(self, Jr, Jm, R, S, bone):
        self.Jr, self.Jm, self.R, self.S, self.bone = np.asarray(Jr, float), np.asarray(Jm, float), R, S, bone
        self.L = R @ S @ F

    def __call__(self, x):
        return (np.asarray(x) - self.Jr) @ self.L.T + self.Jm

    def normal(self, n):
        Ni = np.linalg.inv(self.L).T
        m = np.asarray(n) @ Ni.T
        return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def mii_rig(template_bfres_bytes):
    B = bfres.Bfres(template_bfres_bytes)
    sk = B.models[0]['skeleton']
    W = bfres.bone_world(sk)
    pos = {b['name']: W[i][:3, 3] for i, b in enumerate(sk['bones'])}
    palette = [sk['bones'][b]['name'] for b in sk['mtx_to_bone']]
    return pos, palette


# full masks are drawn ~1.45x larger than body units (measured in game on Miis with height/build 64/64)
MASK_SCALE = 1.45
# where the game puts a full mask (calibrated in game for 1.2.4 with marker models): the mask's origin sits on the
# NECK bone - 0.09 below the Head bone, which builds up to 1.2.3 assumed (every Roblox head was 0.09 too low) - and
# the mask turns with the head; the scale MASK_SCALE was confirmed exactly
MASK_PIVOT = 'Neck'


def mask_frame(rest, D):
    """origin (3,) and rotation (3x3) of the full-mask space in a pose; D = {bone: posed world @ inv(rest world)}.
    mask point v -> world: origin + (MASK_SCALE * v) @ R.T ; world w -> mask: ((w - origin) @ R) / MASK_SCALE"""
    Dp = np.asarray(D[MASK_PIVOT])
    origin = Dp[:3, :3] @ np.asarray(rest[MASK_PIVOT])[:3, 3] + Dp[:3, 3]
    return origin, np.asarray(D['Head'])[:3, :3]


def mask_to_world(v, rest, D, scale=None):
    o, R = mask_frame(rest, D)
    return o + ((scale or MASK_SCALE) * np.asarray(v)) @ R.T


def build_maps(pieces, joints, mii, rig, head_boost=1.05, leg_inset=0.025):
    """part maps for every R15 part + head map (into the Head-bone / full-mask space)"""
    parts = {p['path']: p for p in rig['parts']}
    J = {name: np.mean(list(v.values()), axis=0) for name, v in joints.items()}
    maps = {}
    # ---- torso: hips centre -> neck, width from the shoulder joints
    hip_c = (J['LeftHipRigAttachment'] + J['RightHipRigAttachment']) / 2
    neck = J['NeckRigAttachment']
    sh_half = abs(J['LeftShoulderRigAttachment'][0] - J['RightShoulderRigAttachment'][0]) / 2
    mii_sh_half = abs(mii['Arm_1_L'][0])
    sx = mii_sh_half / sh_half                      # also the global thickness scale (studs -> Mii units)
    k = sx
    mii_hip_y = mii['Leg_1_L'][1]
    s_ax_torso = (mii['Neck'][1] - mii_hip_y) / (neck[1] - hip_c[1])
    S_t = np.diag([sx, s_ax_torso, k])
    torso = PartMap(hip_c, np.array([0, mii_hip_y, 0.0]), np.eye(3), S_t, None)
    maps['UpperTorso'] = maps['LowerTorso'] = torso
    # ---- limbs
    for pn, (jp, jd, bp, bd, skin) in LIMBS.items():
        if pn not in parts:
            continue
        Jr = J[jp]
        Jm = mii[bp].copy()
        if 'Leg' in pn or 'Foot' in pn:
            # Roblox legs sit flush under the torso; keep the pivot (skinning) at the Mii hip but move the geometry in
            Jm[0] -= np.sign(Jm[0]) * leg_inset
        if jd is not None:
            Dr, Dm = J[jd], mii[bd].copy()
            if 'Leg' in pn:
                Dm[0] -= np.sign(Dm[0]) * leg_inset
            a_r = F @ (Dr - Jr)
            a_m = Dm - Jm
            s_ax = np.linalg.norm(a_m) / max(np.linalg.norm(a_r), 1e-6)
        else:
            # hands / feet: continue the previous segment
            prev = {'LeftHand': 'LeftLowerArm', 'RightHand': 'RightLowerArm', 'LeftFoot': 'LeftLowerLeg', 'RightFoot': 'RightLowerLeg'}[pn]
            pm = maps[prev]
            a_r = F @ np.array([0, -1.0, 0])
            a_m = pm.R @ a_r
            s_ax = None
        R = rot_between(a_r, a_m) if 'Arm' in pn or 'Hand' in pn else np.eye(3)
        a = a_r / np.linalg.norm(a_r)
        if pn.endswith('Foot'):
            # vertical: ankle -> sole lands on the floor; horizontal: thickness scale
            p = parts[pn]
            sole = p['cf']['p'][1] - p['size'][1] / 2
            sy = mii[bp][1] / max(Jr[1] - sole, 1e-6)
            S = np.diag([k, sy, k])
        else:
            if s_ax is None:
                s_ax = maps[{'LeftHand': 'LeftLowerArm', 'RightHand': 'RightLowerArm'}[pn]].s_ax
            S = k * np.eye(3) + (s_ax - k) * np.outer(a, a)
        pmap = PartMap(Jr, Jm, R, S, skin)
        pmap.s_ax = s_ax if s_ax is not None else k
        maps[pn] = pmap
    # ---- head: rigid, into the Head bone frame (full mask space); neck joint sits at the Mii neck
    hk = k * head_boost / MASK_SCALE
    neck_in_head = (mii['Neck'] - mii['Head']) / MASK_SCALE
    maps['Head'] = PartMap(J['NeckRigAttachment'], neck_in_head, np.eye(3), hk * np.eye(3), 'Head')
    maps['_k'] = k
    return maps


def torso_weights(y):
    return {b: float(np.interp(y, TORSO_Y, w)) for b, w in TORSO_W.items()}


def bone_weights_for(part, mii_pos_y, limb_bone=None):
    if part in ('UpperTorso', 'LowerTorso'):
        if part == 'LowerTorso':
            return {'Waist': 1.0}
        return torso_weights(mii_pos_y)
    if part in LIMBS:
        return {(limb_bone or {}).get(part, LIMBS[part][4]): 1.0}
    return {'Spine_2': 1.0}


# elbows further than this from the Mii elbow (standing pose, Mii units): the whole arm stays straight on the
# upper-arm bone (classic Roblox arm) - bending it around the far-away Mii elbow would tear the forearm off
RIGID_ARM_LIMIT = 0.15


def retarget(pieces, joints, mii, rig, **kw):
    """-> (body list of (piece, pos, nrm, weights list[dict]), head list of (piece, pos, nrm))"""
    maps = build_maps(pieces, joints, mii, rig, **kw)
    body, head = [], []
    for p in pieces:
        if not len(p.faces):
            continue
        if p.attach == 'Head' or p.kind == 'head':
            m = maps['Head']
            head.append((p, m(p.pos), m.normal(p.nrm)))
            continue
        if p.skin is not None:
            names, idx, w = p.skin
            pos = np.zeros_like(p.pos)
            nrm = np.zeros_like(p.pos)
            W = [dict() for _ in range(len(p.pos))]
            parts_used = {}
            for j, nm in enumerate(names):
                nm = R15_ALIAS.get(nm, nm)
                if nm == 'Head':
                    nm = 'UpperTorso'
                parts_used[j] = nm if nm in maps else 'UpperTorso'
            for slot in range(4):
                wj = w[:, slot]
                for j in np.unique(idx[:, slot]):
                    sel = (idx[:, slot] == j) & (wj > 0)
                    if not sel.any():
                        continue
                    m = maps[parts_used.get(int(j), 'UpperTorso')]
                    pos[sel] += wj[sel, None] * m(p.pos[sel])
                    nrm[sel] += wj[sel, None] * m.normal(p.nrm[sel])
            tot = w.sum(1)
            bad = tot < 1e-6
            if bad.any():
                pos[bad] = maps['UpperTorso'](p.pos[bad])
                nrm[bad] = maps['UpperTorso'].normal(p.nrm[bad])
                tot = np.where(bad, 1.0, tot)
            pos /= tot[:, None]
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            for v in range(len(p.pos)):
                acc = {}
                for slot in range(4):
                    if w[v, slot] <= 0:
                        continue
                    part = parts_used.get(int(idx[v, slot]), 'UpperTorso')
                    for b, bw in bone_weights_for(part, pos[v, 1]).items():
                        if bw > 0:
                            acc[b] = acc.get(b, 0) + bw * w[v, slot]
                W[v] = acc or {'Spine_2': 1.0}
            body.append((p, pos, nrm, W))
            continue
        part = p.attach if p.attach in maps else 'UpperTorso'
        m = maps[part]
        pos = m(p.pos)
        nrm = m.normal(p.nrm)
        if p.kind == 'body':
            W = [bone_weights_for(part, y) for y in pos[:, 1]]
        else:
            W = [{'Waist': 1.0} if part == 'LowerTorso' else bone_weights_for(part, 0.95 if part == 'UpperTorso' else y)
                 for y in pos[:, 1]]
        body.append((p, pos, nrm, W))
    return body, head, maps


# ================================================================ exact proportions
# The game moves costume vertices with the Mii body's joints (tested: the costume's own skeleton is ignored),
# so joints cannot be moved.  The Roblox character therefore keeps its exact shape with ONE scale factor;
# every part is bound to the Mii skeleton in the pose a Mii stands in most of the time, where it then looks exactly
# like Roblox.
# 1.2.3: that pose is the game's idle 'Wait' (also the start of every talk): compared with 'StandStraightH64W64'
# (bind pose up to 1.2.2) the Mii turns its arms by 23 degrees, its wrists by 35 and tilts the head 10 degrees down -
# bound in StandStraight, hands came off the sleeves and the head looked sunk into the shirt.
REF_ANIM = 'Wait'
REF_FRAME = 0.0
JOINT_FIT = (('LeftHipRigAttachment', 'Leg_1_L', 3.0), ('RightHipRigAttachment', 'Leg_1_R', 3.0),
             ('LeftKneeRigAttachment', 'Leg_2_L', 2.0), ('RightKneeRigAttachment', 'Leg_2_R', 2.0),
             ('LeftAnkleRigAttachment', 'Ankle_L', 1.0), ('RightAnkleRigAttachment', 'Ankle_R', 1.0),
             ('NeckRigAttachment', 'Neck', 2.0),
             ('LeftShoulderRigAttachment', 'Arm_1_L', 1.0), ('RightShoulderRigAttachment', 'Arm_1_R', 1.0))


def reference_pose(template_bfres_bytes, anim_bytes, anim=REF_ANIM, frame=REF_FRAME):
    """world matrices {bone: 4x4} of the Mii skeleton at rest and in the bind pose (a 64/64 Mii, like the game)"""
    import fska, miipose
    sk = bfres.Bfres(template_bfres_bytes).models[0]['skeleton']
    A = fska.Anims(anim_bytes)
    rest = miipose.world(sk)
    loc, euler = miipose.blend_hw(A, anim, frame, 64, 64, miipose.rest_local(sk))
    ref = miipose.world(sk, loc, euler)
    names = [b['name'] for b in sk['bones']]
    return {n: rest[i] for i, n in enumerate(names)}, {n: ref[i] for i, n in enumerate(names)}


# ---------------------------------------------------------------- limb weights (1.2.3)
# Roblox shoulders / elbows / wrists sit far from the Mii's (a Mii's arm pivot is up to 0.15 inside a blocky Roblox
# torso): every arm bend of the Mii tore Roblox arms and hands apart. Now a Roblox arm is one piece hanging from the
# Roblox shoulder: at the shoulder it moves exactly with the torso, further down it follows the Mii's upper arm a
# little (ARM_SWING at the hand) - no gap anywhere, the arms still swing when walking. Elbow and wrist bends are left
# out (wrong animation, but nothing comes apart).
# Legs fit the Mii well (gaps under 0.02 in everyday poses): hip, knee and ankle stay jointed, with a short blend
# zone at each joint so blocky parts and trousers bend together.
ARM_SWING = 0.5
ARM_RAMP = 0.6          # the swing share is reached this far down the arm (0 shoulder, 1 wrist)
# 1.2.5: arms jointed like the legs (the Mii's eating / waving / holding moves reach the Roblox hands), with wide
# blend zones at shoulder, elbow and wrist that stretch instead of tearing where Roblox and Mii joints differ
# RbxToMii 1.1: arms rigid per Roblox part (upper arm, forearm, hand), like Roblox's own R15 rig.
# The wide blend zones bent the blocks themselves in strong poses (clapping, hands on hips, eating): the part of an arm
# block above the Roblox shoulder stayed on the torso and the elbow zone folded both blocks. Rigid parts only overlap
# a little at the joints, which Roblox characters do too.
ARM_MODE = 'rigid'      # 'jointed' = 1.2.5 blend zones, 'swing' = 1.2.3/1.2.4 one-piece arm
ARM_BLEND = (0.14, 0.18, 0.10)   # half widths at shoulder, elbow, wrist (share of the shoulder-wrist length)
LEG_BLEND = 0.08        # half width of the blend zone at hip / knee / ankle (share of the hip-ankle length)
ARM_PARTS = {'LeftUpperArm': 'L', 'LeftLowerArm': 'L', 'LeftHand': 'L',
             'RightUpperArm': 'R', 'RightLowerArm': 'R', 'RightHand': 'R'}
LEG_PARTS = {'LeftUpperLeg': 'L', 'LeftLowerLeg': 'L', 'LeftFoot': 'L',
             'RightUpperLeg': 'R', 'RightLowerLeg': 'R', 'RightFoot': 'R'}


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class LimbWeights:
    """Mii bone weights for points of Roblox limbs (positions in the Mii bind pose space)"""

    def __init__(self, Jm):
        """Jm: Roblox rig joints already mapped into the Mii bind space {attachment name: xyz}"""
        self.arm = {}
        self.leg = {}
        for s, S in (('L', 'Left'), ('R', 'Right')):
            sh, wr = Jm.get(S + 'ShoulderRigAttachment'), Jm.get(S + 'WristRigAttachment')
            el = Jm.get(S + 'ElbowRigAttachment')
            if sh is not None and wr is not None:
                a = wr - sh
                L = max(float(np.linalg.norm(a)), 1e-6)
                te = float((el - sh) @ (a / L)) / L if el is not None else 0.5
                self.arm[s] = (sh, a / L, L, torso_weights(float(sh[1])), te)
            hip, kn, an = (Jm.get(S + k + 'RigAttachment') for k in ('Hip', 'Knee', 'Ankle'))
            if hip is not None and an is not None:
                a = an - hip
                L = max(float(np.linalg.norm(a)), 1e-6)
                a = a / L
                tk = float((kn - hip) @ a) / L if kn is not None else 0.5
                self.leg[s] = (hip, a, L, tk)

    def arm_weights(self, side, pos):
        sh, a, L, base, te = self.arm[side]
        t = ((np.asarray(pos) - sh) @ a) / L
        out = []
        if ARM_MODE == 'swing':
            w = ARM_SWING * _smooth(t / ARM_RAMP)
            bone = 'Arm_1_' + side
            for wi in np.atleast_1d(w):
                d = {b: (1 - wi) * v for b, v in base.items() if v > 0}
                d[bone] = d.get(bone, 0.0) + float(wi)
                out.append(d)
            return out
        hs, he, hw = ARM_BLEND
        w1 = _smooth((t + hs) / (2 * hs))            # torso -> upper arm (around the Roblox shoulder)
        w2 = _smooth((t - te + he) / (2 * he))       # upper arm -> forearm (around the Roblox elbow)
        w3 = _smooth((t - 1 + hw) / (2 * hw))        # forearm -> hand (around the Roblox wrist)
        for a1, a2, a3 in zip(np.atleast_1d(w1), np.atleast_1d(w2), np.atleast_1d(w3)):
            d = {b: (1 - a1) * v for b, v in base.items() if v > 0}
            for b, v in (('Arm_1_' + side, a1 * (1 - a2)), ('Arm_2_' + side, a1 * a2 * (1 - a3)), ('Wrist_' + side, a1 * a2 * a3)):
                if v > 1e-4:
                    d[b] = d.get(b, 0.0) + float(v)
            out.append(d)
        return out

    def leg_weights(self, side, pos):
        hip, a, L, tk = self.leg[side]
        t = ((np.asarray(pos) - hip) @ a) / L
        h = LEG_BLEND
        w1 = _smooth((t + h) / (2 * h))          # waist -> thigh
        w2 = _smooth((t - tk + h) / (2 * h))     # thigh -> shin
        w3 = _smooth((t - 1 + h) / (2 * h))      # shin -> foot
        out = []
        for a1, a2, a3 in zip(np.atleast_1d(w1), np.atleast_1d(w2), np.atleast_1d(w3)):
            d = {'Waist': 1 - a1, 'Leg_1_' + side: a1 * (1 - a2), 'Leg_2_' + side: a1 * a2 * (1 - a3),
                 'Ankle_' + side: a1 * a2 * a3}
            out.append({b: float(v) for b, v in d.items() if v > 1e-4})
        return out

    def for_part(self, part, pos):
        """weights for points pos (N,3) of R15 part `part` - None if the part is not an arm / leg"""
        if part in ARM_PARTS and ARM_PARTS[part] in self.arm:
            if ARM_MODE == 'rigid':
                side = ARM_PARTS[part]
                bone = {'UpperArm': 'Arm_1_', 'LowerArm': 'Arm_2_', 'Hand': 'Wrist_'}[part[len('Left'):] if part.startswith('Left') else part[len('Right'):]]
                return [{bone + side: 1.0} for _ in range(len(np.atleast_2d(pos)))]
            return self.arm_weights(ARM_PARTS[part], pos)
        if part in LEG_PARTS and LEG_PARTS[part] in self.leg:
            return self.leg_weights(LEG_PARTS[part], pos)
        return None


def fit_uniform(pieces, J, ref):
    """one scale s (studs -> Mii units) + placement: soles on the floor, hips centred like the Mii's"""
    feet = [p for p in pieces if p.kind == 'body' and p.attach in ('LeftFoot', 'RightFoot', 'LeftLowerLeg', 'RightLowerLeg')]
    sole = min(p.pos[:, 1].min() for p in feet) if feet else min(p.pos[:, 1].min() for p in pieces if len(p.pos))
    num = den = 0.0
    for jn, bone, w in JOINT_FIT:
        if jn in J:
            h = J[jn][1] - sole
            num += w * h * ref[bone][1, 3]
            den += w * h * h
    s = num / den
    hip_r = (J['LeftHipRigAttachment'] + J['RightHipRigAttachment']) / 2
    hip_m = (ref['Leg_1_L'][:3, 3] + ref['Leg_1_R'][:3, 3]) / 2
    c = np.array([hip_r[0], sole, hip_r[2]])
    o = np.array([hip_m[0], 0.0, hip_m[2]])
    return s, c, o


JOINT_INFO = (('Shoulder', 'LeftShoulderRigAttachment', 'Arm_1_L'), ('Elbow', 'LeftElbowRigAttachment', 'Arm_2_L'),
              ('Wrist', 'LeftWristRigAttachment', 'Wrist_L'), ('Hip', 'LeftHipRigAttachment', 'Leg_1_L'),
              ('Knee', 'LeftKneeRigAttachment', 'Leg_2_L'), ('Ankle', 'LeftAnkleRigAttachment', 'Ankle_L'),
              ('Neck', 'NeckRigAttachment', 'Head'))


def retarget_uniform(pieces, joints, rig, rest, ref):
    """-> body list (piece, pos in the bind pose, nrm, weights), head list (piece, mask pos, mask nrm), info.
    Positions are the Roblox ones times one scale factor; nothing is stretched."""
    J = {name: np.mean(list(v.values()), axis=0) for name, v in joints.items() if len(v)}
    s, c, o = fit_uniform(pieces, J, ref)
    G = lambda x: (np.asarray(x) - c) @ (s * F).T + o
    limbs = LimbWeights({k: G(v) for k, v in J.items()})
    joint_off = {lab: round(float(np.linalg.norm(G(J[jn]) - ref[bone][:3, 3])), 3)
                 for lab, jn, bone in JOINT_INFO if jn in J}

    def part_weights(part, pts):
        lw = limbs.for_part(part, pts)
        if lw is not None:
            return lw
        if part == 'LowerTorso':
            return [{'Waist': 1.0} for _ in range(len(pts))]
        return [{b: w for b, w in torso_weights(float(y)).items() if w > 0} for y in pts[:, 1]]

    # full mask space (mask_frame): origin on the Neck bone, turned with the head, MASK_SCALE larger - in the bind pose
    m_origin, m_rot = mask_frame(rest, {b: ref[b] @ np.linalg.inv(rest[b]) for b in (MASK_PIVOT, 'Head')})
    body, head = [], []
    for p in pieces:
        if not len(p.faces):
            continue
        pos = G(p.pos)
        nrm = np.asarray(p.nrm) @ F.T
        if p.attach == 'Head' or p.kind == 'head':
            hp = ((pos - m_origin) @ m_rot) / MASK_SCALE
            hn = nrm @ m_rot
            head.append((p, hp, hn / np.maximum(np.linalg.norm(hn, axis=1, keepdims=True), 1e-12)))
            continue
        if p.skin is not None:
            # layered clothing: Roblox's own smooth skin weights, every Roblox part replaced by its Mii weights
            names, idx, w = p.skin
            used = {}
            for j, nm in enumerate(names):
                nm = R15_ALIAS.get(nm, nm)
                if nm == 'Head':
                    nm = 'UpperTorso'
                used[j] = nm if (nm in LIMBS or nm in ('UpperTorso', 'LowerTorso')) else 'UpperTorso'
            Wl = [dict() for _ in range(len(pos))]
            live = w > 0
            for j in np.unique(idx[live]):
                share = np.where(live & (idx == j), w, 0.0).sum(1)
                sel = np.nonzero(share > 0)[0]
                for v, d in zip(sel, part_weights(used.get(int(j), 'UpperTorso'), pos[sel])):
                    acc = Wl[v]
                    for b, bw in d.items():
                        acc[b] = acc.get(b, 0.0) + bw * share[v]
            body.append((p, pos, nrm, [d or {'Spine_2': 1.0} for d in Wl]))
            continue
        part = p.attach if (p.attach in LIMBS or p.attach in ('UpperTorso', 'LowerTorso')) else 'UpperTorso'
        if p.kind == 'body':
            Wl = part_weights(part, pos)
        elif part == 'UpperTorso':   # rigid accessories keep their shape: the weights of one point of their part
            Wl = [{b: w for b, w in torso_weights(0.95).items() if w > 0}] * len(pos)
        else:
            Wl = part_weights(part, pos.mean(0, keepdims=True)) * len(pos)
        body.append((p, pos, nrm, Wl))
    return body, head, {'s': float(s), 'c': c.tolist(), 'o': o.tolist(), 'joint_offset': joint_off,
                        'arm_mode': ARM_MODE, 'bind_pose': REF_ANIM}


def unskin(pos, nrm, I, Wt, palette, rest, ref):
    """standing-pose positions -> rest-pose (bind) positions for the weights the game will use"""
    D = np.stack([ref[b] @ np.linalg.inv(rest[b]) for b in palette])      # rest -> standing per palette entry
    M = np.einsum('vk,vkij->vij', Wt, D[I])                                 # blended per vertex
    Mi = np.linalg.inv(M)
    hp = np.concatenate([pos, np.ones((len(pos), 1))], 1)
    out = np.einsum('vij,vj->vi', Mi, hp)[:, :3]
    n = np.einsum('vji,vj->vi', M[:, :3, :3], nrm)                          # inverse-transpose of Mi = M^T
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return out, n


def skin_spheres(pos_rest, I, Wt, palette, rest, skin_bones_names, margin=0.02):
    """bounding sphere per skin bone in bone space (culling) -> list of (cx, cy, cz, r) or None (keep template)"""
    out = []
    for b in skin_bones_names:
        k = palette.index(b) if b in palette else -1
        sel = ((I == k) & (Wt > 0)).any(1) if k >= 0 else np.zeros(len(pos_rest), bool)
        if not sel.any():
            out.append(None)
            continue
        inv = np.linalg.inv(rest[b])
        pb = pos_rest[sel] @ inv[:3, :3].T + inv[:3, 3]
        lo, hi = pb.min(0), pb.max(0)
        cc = (lo + hi) / 2
        out.append((*cc, float(np.linalg.norm(pb - cc, axis=1).max()) + margin))
    return out
