"""Render the generated game files (body costume + full mask) the way the game would place them:
body in the Mii T-pose (or with arms lowered via the skin weights), head on the Head bone.
usage: python -I preview_game_model.py <romfs> <body model name> <mask name> <out.jpg> [arms_down_deg]"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bfres, bntx, tlfmt, rrender


class P:
    pass


def load_model(romfs, name, texname):
    B = bfres.Bfres(tlfmt.read(os.path.join(romfs, 'Model', name + '.bfres.zs')))
    m = B.models[0]
    s = m['shapes'][0]
    p = P()
    p.pos = B.attr(m, 0, '_p0').astype(float)
    p.uv = B.attr(m, 0, '_u0').astype(float)
    p.faces = B.indices(s['meshes'][0]).reshape(-1, 3)
    p.idx = B.attr(m, 0, '_i0')
    p.w = B.attr(m, 0, '_w0')
    p.skel = m['skeleton']
    t = bntx.read(tlfmt.read(os.path.join(romfs, 'Tex', texname + '.bntx.zs')))[0]
    p.tex = np.asarray(t['image'].convert('RGBA'))
    return p


def pose_arms_down(p, deg):
    """rotate the arm bones about Z at the shoulders (simple FK), skin with the stored weights"""
    sk = p.skel
    W = bfres.bone_world(sk)
    names = [b['name'] for b in sk['bones']]
    world = [w.copy() for w in W]
    a = np.radians(deg)
    for side, sgn in (('L', -1), ('R', 1)):
        sh = names.index('Arm_1_' + side)
        piv = W[sh][:3, 3]
        Rz = np.array([[np.cos(sgn * a), -np.sin(sgn * a), 0], [np.sin(sgn * a), np.cos(sgn * a), 0], [0, 0, 1]])
        M = np.eye(4)
        M[:3, :3] = Rz
        M[:3, 3] = piv - Rz @ piv
        for i, n in enumerate(names):
            # every bone below the shoulder (arm chain) follows
            j = i
            while j >= 0 and j != sh:
                j = sk['bones'][j]['parent']
            if j == sh:
                world[i] = M @ W[i]
    skin = [world[b] @ np.linalg.inv(W[b]) for b in sk['mtx_to_bone']]
    out = np.zeros_like(p.pos)
    hp = np.concatenate([p.pos, np.ones((len(p.pos), 1))], 1)
    for s in range(4):
        for j in np.unique(p.idx[:, s]):
            sel = p.idx[:, s] == j
            out[sel] += p.w[sel, s:s + 1] * (hp[sel] @ skin[j].T)[:, :3]
    return out


def main():
    romfs, body_name, mask, out = sys.argv[1:5]
    deg = float(sys.argv[5]) if len(sys.argv) > 5 else 75
    body = load_model(romfs, body_name, body_name + 'Tex_Body_Alb.00')
    head = load_model(romfs, mask, mask + '_Body_Alb.00')
    head.pos = head.pos * 1.45 + np.array([0, 1.156, 0])  # full masks are drawn ~1.45x larger (miimap.MASK_SCALE)
    down = P()
    down.__dict__.update(body.__dict__)
    down.pos = pose_arms_down(body, deg)
    ims = [rrender.render([body, head], yaw=180, size=420), rrender.render([down, head], yaw=180, size=420),
           rrender.render([down, head], yaw=135, size=420), rrender.render([down, head], yaw=0, size=420),
           rrender.render([down, head], yaw=270, size=420)]
    rrender.sheet(ims).save(out, quality=90)


if __name__ == '__main__':
    main()
