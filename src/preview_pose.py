"""Render the built game files the way the game poses them: the body costume skinned to the Mii skeleton in a pose
from the game's animations, the full mask where the game puts it (miimap.mask_frame: Neck origin, head turn).
usage: python -I preview_pose.py <romfs> <body model> <mask> <out.jpg> [Anim[@frame]:yaw ...]"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bfres, bntx, tlfmt, rrender, fska, miipose, miimap

WORK = os.path.abspath(os.path.join(HERE, '..'))
GAME = os.path.join(WORK, 'game', 'romfs')


class P:
    pass


def load_model(romfs, name, texname):
    B = bfres.Bfres(tlfmt.read(os.path.join(romfs, 'Model', name + '.bfres.zs')))
    m = B.models[0]
    p = P()
    p.pos = B.attr(m, 0, '_p0').astype(float)
    p.uv = B.attr(m, 0, '_u0').astype(float)
    p.faces = B.indices(m['shapes'][0]['meshes'][0]).reshape(-1, 3)
    sk = m['skeleton']
    p.skinned = bool(m['shapes'][0]['skin_bones'])
    if p.skinned:
        p.idx = B.attr(m, 0, '_i0').astype(int)
        p.w = B.attr(m, 0, '_w0').astype(float)
        p.palette = [sk['bones'][b]['name'] for b in sk['mtx_to_bone']]
    t = bntx.read(tlfmt.read(os.path.join(romfs, 'Tex', texname + '.bntx.zs')))[0]
    p.tex = np.asarray(t['image'].convert('RGBA'))
    return p


def posed(body, head, sk, A, anim, frame=0):
    names = [b['name'] for b in sk['bones']]
    rest = miipose.world(sk)
    if anim == 'rest':
        pose = rest
    else:
        loc, euler = miipose.blend_hw(A, anim, frame, 64, 64, miipose.rest_local(sk))
        pose = miipose.world(sk, loc, euler)
    D = {n: pose[i] @ np.linalg.inv(rest[i]) for i, n in enumerate(names)}
    b = P()
    b.__dict__.update(body.__dict__)
    hp = np.concatenate([body.pos, np.ones((len(body.pos), 1))], 1)
    out = np.zeros((len(hp), 3))
    for k in range(4):
        for j in np.unique(body.idx[:, k]):
            sel = (body.idx[:, k] == j) & (body.w[:, k] > 0)
            out[sel] += body.w[sel, k:k + 1] * (hp[sel] @ D[body.palette[j]].T)[:, :3]
    b.pos = out
    h = P()
    h.__dict__.update(head.__dict__)
    h.pos = miimap.mask_to_world(head.pos, {n: rest[i] for i, n in enumerate(names)}, D)
    return b, h


def main():
    romfs, body_name, mask, out = sys.argv[1:5]
    poses = sys.argv[5:] or ['StandStraightH64W64:180', 'StandStraightH64W64:90', 'SitChair:90', 'PoseHandUpR:180']
    body = load_model(romfs, body_name, body_name + 'Tex_Body_Alb.00')
    head = load_model(romfs, mask, mask + '_Body_Alb.00')
    sk = bfres.Bfres(tlfmt.read(os.path.join(romfs, 'Model', body_name + '.bfres.zs'))).models[0]['skeleton']
    A = fska.Anims(tlfmt.read(os.path.join(GAME, 'Model', 'AnimationProduct.bfres.zs')))
    ims = []
    for ps in poses:
        an, yaw = ps.split(':')
        an, fr = (an.split('@') + ['0'])[:2]
        b, h = posed(body, head, sk, A, an, float(fr))
        ims.append(rrender.render([b, h], yaw=float(yaw), size=420))
    rrender.sheet(ims, [p.replace('H64W64', '') for p in poses]).save(out, quality=90)


if __name__ == '__main__':
    main()
