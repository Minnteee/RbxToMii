"""Pose the Mii skeleton (from a cloth template) with the game's skeletal animations (fska.Anims)."""
import numpy as np
import bfres, fska


def local_matrix(s, r, t, euler):
    if euler:
        R = bfres.euler_to_mat(r[0], r[1], r[2])
    else:
        R = bfres.quat_to_mat(*r)
    L = np.eye(4)
    L[:3, :3] = R @ np.diag(s)
    L[:3, 3] = t
    return L


def world(skel, anim_local=None, euler_anim=True):
    """world matrices of every bone of skel; anim_local {name: (s, r, t)} overrides the rest values"""
    euler_skel = bool(skel['flags'] & 0x1000)
    W = []
    for b in skel['bones']:
        if anim_local is not None and b['name'] in anim_local:
            s, r, t = anim_local[b['name']]
            L = local_matrix(s, r, t, euler_anim)
        else:
            L = local_matrix(b['scale'], b['rot'], b['trans'], euler_skel)
        W.append(L if b['parent'] < 0 else W[b['parent']] @ L)
    return W


def rest_local(skel):
    return {b['name']: (np.array(b['scale']), np.array(b['rot']), np.array(b['trans'])) for b in skel['bones']}


def blend_hw(A, base, frame, h=64, w=64, rest=None, curves=True):
    """bilinear blend of the H0W0 / H0W127 / H127W0 / H127W127 variants (local values) like the game"""
    names = {k: base + k for k in ('H0W0', 'H0W127', 'H127W0', 'H127W127')}
    if not all(n in A.by_name for n in names.values()):
        a = A.by_name[base] if base in A.by_name else A.by_name[base + 'H64W64']
        return A.local(a, frame % max(a['frames'], 1), rest, curves), a['euler']
    loc = {k: A.local(A.by_name[n], frame % max(A.by_name[n]['frames'], 1), rest, curves) for k, n in names.items()}
    fh, fw = h / 127.0, w / 127.0
    wts = {'H0W0': (1 - fh) * (1 - fw), 'H0W127': (1 - fh) * fw, 'H127W0': fh * (1 - fw), 'H127W127': fh * fw}
    out = {}
    for bn in loc['H0W0']:
        s = sum(wts[k] * loc[k][bn][0] for k in wts)
        r = sum(wts[k] * loc[k][bn][1] for k in wts)
        t = sum(wts[k] * loc[k][bn][2] for k in wts)
        out[bn] = (s, r, t)
    return out, A.by_name[names['H0W0']]['euler']
