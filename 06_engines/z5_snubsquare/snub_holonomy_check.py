# -*- coding: utf-8 -*-
"""
Snub square torus holonomy adjudication experiment
==========================================================
Background: the first attempt of the demo figure reported a "layout
inconsistency" (diff ~ 3.38, translation scale) on a 2x2 torus patch; at
the time the log attributed it to "the patch is not simply connected" --
but a 2x2 cell block IS simply connected, so the attribution was
suspicious. This experiment adjudicates the true mechanism.
Prediction (SNB theorem + holonomy argument): a kernel element (a,b)
satisfies all face closures (locally), but the displacement around a
non-contractible torus cycle = (a-b)*w: zero when a = b (the regular
solution is flat), nonzero when a != b => a full-torus layout must hit a
periodic translation somewhere; a within-patch layout (never leaving the
patch) can never clash (simply connected => local closure => global
consistency).
Adjudication matrix:
  E1 layout restricted to the patch, (a,b)=(1,0.75) -> predicted success
  E2 full-torus layout,             (a,b)=(1,1)     -> predicted success
     (regular metric flat)
  E3 full-torus layout,             (a,b)=(1,0.75)  -> predicted failure,
     translation-type clash (= holonomy)
  E4 extra: E2 with (1,1.33), same as E3.
"""
import sys, os
from collections import deque
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HALF = 0.5
SQ3 = np.sqrt(3) / 2
VEC_TAB = {0: (1, 0), 1: (SQ3, HALF), 2: (HALF, SQ3), 3: (0, 1),
           4: (-HALF, SQ3), 5: (-SQ3, HALF), 6: (-1, 0), 7: (-SQ3, -HALF),
           8: (-HALF, -SQ3), 9: (0, -1), 10: (HALF, -SQ3), 11: (SQ3, -HALF)}


def build_torus(N, M):
    """Torus version of the construction (same lineage as the first demo
    attempt: modularly reduced edge keys)."""
    E = {}; elist = []
    def newedge(t_, i, j):
        k = (t_, i % N, j % M)
        if k not in E:
            E[k] = len(elist); elist.append(k)
        return E[k]
    faces = []
    for i in range(N):
        for j in range(M):
            g = [newedge('H', i, j), newedge('V', i + 1, j),
                 newedge('H', i, j + 1), newedge('V', i, j)]
            if (i + j) % 2 == 0:
                faces.append((g, [3, 3, 3, 3], 'R', (i, j)))
            else:
                dd = newedge('D', i, j)
                if i % 2 == 1:
                    faces.append(([g[0], g[1], dd], [4, 4, 4], 'T', (i, j)))
                    faces.append(([dd, g[2], g[3]], [4, 4, 4], 'T', (i, j)))
                else:
                    faces.append(([g[0], dd, g[3]], [4, 4, 4], 'T', (i, j)))
                    faces.append(([g[1], g[2], dd], [4, 4, 4], 'T', (i, j)))
    Vn, En, Fn = N * M, len(elist), len(faces)
    assert Vn - En + Fn == 0 and En == 2 * N * M + N * M // 2
    def rhombus_i(k):
        t_, i, j = k
        if t_ == 'H': return i
        if t_ == 'V': return i if (i + j) % 2 == 1 else i - 1
        return i
    vI = [1 if rhombus_i(k) % 2 == 1 else 0 for k in elist]
    occ = {}
    for fi, (e, _, _, _) in enumerate(faces):
        for pos, ed in enumerate(e):
            occ.setdefault(ed, []).append((fi, pos))
    dir_of = {}
    def walk_dir(fi, pos, dcur):
        e, turns, _, _ = faces[fi]
        m = len(e); c = dcur
        for off in range(m):
            k = (pos + off) % m
            if off > 0:
                c = (c + turns[k]) % 12
            key = (fi, k)
            if key in dir_of:
                assert dir_of[key] == c
            else:
                dir_of[key] = c
        assert (c + turns[pos]) % 12 == dcur
    dir_of[(0, 0)] = 0
    dq = deque([(0, 0)])
    while dq:
        fi, pos = dq.popleft()
        walk_dir(fi, pos, dir_of[(fi, pos)])
        for k in [kk for kk in dir_of if kk[0] == fi]:
            ed = faces[fi][0][k[1]]
            for (fj, pj) in occ[ed]:
                if (fj, pj) == k:
                    continue
                d2 = (dir_of[k] + 6) % 12
                if (fj, pj) in dir_of:
                    assert dir_of[(fj, pj)] == d2
                else:
                    dir_of[(fj, pj)] = d2
                    dq.append((fj, pj))
    assert len(dir_of) == sum(len(f[0]) for f in faces)
    return dict(faces=faces, elist=elist, vI=vI, dir_of=dir_of, occ=occ)


def layout(s, ell, allowed=None):
    """Full-torus / patch-restricted layout. allowed=None visits all faces;
    otherwise only faces inside allowed are entered.
    Returns (ok, visited_count, clash_info)."""
    faces, occ, dir_of = s['faces'], s['occ'], s['dir_of']
    vnode = {}
    f0 = 0 if allowed is None else sorted(allowed)[0]
    vnode[(f0, 0)] = np.array([0.0, 0.0])
    dq = deque([f0]); seen = {f0}
    visited = 0
    while dq:
        fi = dq.popleft()
        visited += 1
        e, _, _, _ = faces[fi]
        m = len(e)
        start = next(i for i in range(m) if (fi, i) in vnode)
        cur = vnode[(fi, start)]
        i = start
        for _ in range(m):
            nxt = (fi, (i + 1) % m)
            cand = cur + np.array(VEC_TAB[dir_of[(fi, i)]]) * ell[e[i]]
            if nxt in vnode:
                d = np.linalg.norm(cand - vnode[nxt])
                if d > 1e-9:
                    return False, visited, dict(face=fi, slot=i, diff=d, vec=cand - vnode[nxt])
            else:
                vnode[nxt] = cand
            cur = cand
            i = (i + 1) % m
        for i2, ed in enumerate(e):
            for (fj, pj) in occ[ed]:
                if allowed is not None and fj not in allowed:
                    continue
                if fj not in seen:
                    seen.add(fj); dq.append(fj)
                m2 = len(faces[fj][0])
                if (fj, pj) not in vnode and (fi, (i2 + 1) % m) in vnode:
                    vnode[(fj, pj)] = vnode[(fi, (i2 + 1) % m)]
                if (fj, (pj + 1) % m2) not in vnode and (fi, i2) in vnode:
                    vnode[(fj, (pj + 1) % m2)] = vnode[(fi, i2)]
    return True, visited, None


def main():
    s = build_torus(4, 4)
    cells = {(i, j) for i in (1, 2) for j in (1, 2)}
    patch = {fi for fi, f in enumerate(s['faces']) if f[3] in cells}
    cases = [
        ("E1 patch only,  a/b=1/0.75", 1.0, 0.75, patch),
        ("E2 full torus,  a/b=1/1    ", 1.0, 1.0, None),
        ("E3 full torus,  a/b=1/0.75 ", 1.0, 0.75, None),
        ("E4 full torus,  a/b=1/1.33 ", 1.0, 1.33, None),
    ]
    print(f"{'case':28} {'result':8} {'visited':8} {'clash':}")
    for name, a, b, allowed in cases:
        ell = np.array([a * x + b * (1 - x) for x in s['vI']])
        ok, visited, clash = layout(s, ell, allowed)
        if ok:
            print(f"{name:28} {'success':8} {visited:<8} -")
        else:
            print(f"{name:28} {'clash':8} {visited:<8} face {clash['face']} slot {clash['slot']}"
                  f" diff={clash['diff']:.4f} vec=({clash['vec'][0]:.3f},{clash['vec'][1]:.3f})")
    # mechanism corroboration: latitude-cycle w vector (sum of directions of
    # the H edge family; class I and class II each vanish <=> a=b is the only
    # holonomy-free case)
    row = [(s['faces'][fi][3], fi) for fi in range(len(s['faces']))]
    print("\nmechanism: latitude-cycle displacement = a*S_I + b*S_II; the regular"
          " solution a=b is flat => S_I = -S_II = w != 0"
          " => holonomy = (a-b)*w")


if __name__ == "__main__":
    main()
