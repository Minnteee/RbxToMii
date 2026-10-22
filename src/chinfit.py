"""Keep a Roblox head's chin in front of the chest in the game's everyday poses.
The full mask hangs rigidly on the Mii's neck (origin on the Neck bone, turned with the head - miimap.mask_frame).
In several standing / talking poses the Mii bends its upper spine and head a little, so a long Roblox jaw (e.g. an
open-mouth head) slides behind the shirt and disappears. The head is moved forward just as far as needed (often 0) -
never up/down or sideways."""
import numpy as np

MASK_SCALE = 1.45   # same as miimap.MASK_SCALE (full masks are drawn this much larger)

# everyday poses: standing habits, talking, walking (not sleeping / falling / looking down on purpose)
POSES = ('Wait', 'StandStraight', 'StandAffected', 'StandCute', 'StandHandsFront', 'StandLeanForward', 'StandOneFoot',
         'StandProudly', 'StandShy', 'Talk', 'TalkWaitBend', 'TalkWaitOneHand', 'Walk@0', 'Walk@8', 'Walk@16', 'Walk@24')
TORSO_BONES = ('Waist', 'Spine_1', 'Spine_2', 'Neck')


def pose_deltas(skel, anims):
    """{pose: {bone: 4x4 posed world @ inverse rest world}}, rest position of the mask pivot (Neck bone)"""
    import miipose
    names = [b['name'] for b in skel['bones']]
    rest = miipose.world(skel)
    out = {}
    for ps in POSES:
        an, fr = (ps.split('@') + ['0'])[:2]
        try:
            loc, euler = miipose.blend_hw(anims, an, float(fr), 64, 64, miipose.rest_local(skel))
        except KeyError:
            continue
        pose = miipose.world(skel, loc, euler)
        out[ps] = {n: pose[i] @ np.linalg.inv(rest[i]) for i, n in enumerate(names)}
    return out, rest[names.index('Neck')][:3, 3].copy()


def front_depth(tris, q, chunk=256):
    """tris: T x 3 x 3 (x, y, z) -> for each query point the largest z of a triangle covering its (x, y); -inf if none"""
    out = np.full(len(q), -np.inf)
    if not len(tris) or not len(q):
        return out
    lo = tris[:, :, :2].min(1)
    hi = tris[:, :, :2].max(1)
    qlo, qhi = q[:, :2].min(0), q[:, :2].max(0)
    keep = (hi[:, 0] >= qlo[0]) & (lo[:, 0] <= qhi[0]) & (hi[:, 1] >= qlo[1]) & (lo[:, 1] <= qhi[1])
    tris, lo, hi = tris[keep], lo[keep], hi[keep]
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(d) > 1e-12
    a, b, c, d, lo, hi = a[ok], b[ok], c[ok], d[ok], lo[ok], hi[ok]
    if not len(a):
        return out
    for s in range(0, len(q), chunk):
        x = q[s:s + chunk, 0][:, None]
        y = q[s:s + chunk, 1][:, None]
        inb = (x >= lo[None, :, 0]) & (x <= hi[None, :, 0]) & (y >= lo[None, :, 1]) & (y <= hi[None, :, 1])
        w0 = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / d
        w1 = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / d
        w2 = 1 - w0 - w1
        hit = inb & (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        z = w0 * a[:, 2] + w1 * b[:, 2] + w2 * c[:, 2]
        out[s:s + chunk] = np.where(hit, z, -np.inf).max(1)
    return out


def chin_points(P2, N2, head_v, share=0.35):
    """points that face forwards in the lower part of the face - head mesh or face accessory (masks, beards...);
    the height range comes from the head mesh itself (hats / long hair do not stretch it)"""
    ref = head_v if head_v.any() else np.ones(len(P2), bool)
    y = P2[ref, 1]
    lim = y.min() + share * (y.max() - y.min())
    return (P2[:, 1] <= lim) & (P2[:, 2] > 0) & (N2[:, 2] > 0.5)


def _skin(P, I, Wt, palette, D):
    hp = np.concatenate([P, np.ones((len(P), 1))], 1)
    out = np.zeros((len(P), 3))
    for k in range(I.shape[1]):
        for j in np.unique(I[:, k]):
            sel = (I[:, k] == j) & (Wt[:, k] > 0)
            if sel.any():
                out[sel] += Wt[sel, k:k + 1] * (hp[sel] @ D[palette[j]].T)[:, :3]
    return out


def forward_shift(P, I, Wt, palette, faces, occluder_v, P2, N2, head_v, deltas, pivot_rest, clearance=0.004, limit=0.06,
                  log=None):
    """P, I, Wt: body in bind pose (Mii units) with palette indices/weights; faces: body triangles;
    occluder_v: body vertices that may hide the chin (bare body / clothing, not rigid accessories);
    P2, N2: head in mask space (miimap.mask_frame, divided by MASK_SCALE); head_v: head-mesh vertices;
    deltas, pivot_rest: from pose_deltas().
    -> forward shift in Mii units (apply as P2[:, 2] += shift / MASK_SCALE), report dict"""
    sel = chin_points(P2, N2, head_v)
    if sel.sum() < 8:
        return 0.0, {'points': int(sel.sum())}
    dom = np.array([palette[i] for i in I[np.arange(len(I)), Wt.argmax(1)]])
    occ = occluder_v & np.isin(dom, TORSO_BONES)
    tf = faces[occ[faces].all(1)]
    used = np.unique(tf)
    remap = np.full(len(P), -1)
    remap[used] = np.arange(len(used))
    hw = MASK_SCALE * P2[sel]
    need, per_pose, base = [], {}, None
    for ps in POSES:
        if ps not in deltas:
            continue
        D = deltas[ps]
        bp = _skin(P[used], I[used], Wt[used], palette, D)
        Dn, Dh = np.asarray(D['Neck']), np.asarray(D['Head'])
        q = (Dn[:3, :3] @ pivot_rest + Dn[:3, 3]) + hw @ Dh[:3, :3].T
        m = q[:, 2] - front_depth(bp[remap[tf]], q)
        if base is None:      # Wait = the pose the character was fitted in (Roblox look)
            base = m > 0
            if base.sum() < 8:
                return 0.0, {'points': int(sel.sum()), 'visible': int(base.sum())}
        lack = np.clip(clearance - m[base], 0, None)
        v = float(np.percentile(lack, 90))     # at most 10 % of the chin may hide behind the chest
        per_pose[ps] = round(v, 4)
        need.append(v)
    shift = float(min(max(need) if need else 0.0, limit))
    if log:
        hit = ', '.join('%s %.3f' % kv for kv in per_pose.items() if kv[1] > 0)
        log('   chin clearance: head %.3f forward%s' % (shift, (' (%s)' % hit) if hit else ''))
    return shift, {'points': int(sel.sum()), 'visible': int(base.sum()) if base is not None else 0,
                   'per_pose': per_pose, 'limit': limit}
