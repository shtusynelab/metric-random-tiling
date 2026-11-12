# -*- coding: utf-8 -*-
"""Diagnosis: root cause of the DFS finding zero solutions for the k=2 tile set.
1) whether class binding (triangle equal-value / square opposite-edge pairs)
   contradicts the real solution space (a wrong binding would cut off true
   solutions);
2) whether the class-constraint system has a solution over {0,1}
   (LP relaxation + small-scale integer check);
3) cross-check: take a rational point of the real solution space and see
   whether classes are naturally equal-valued."""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "z4_3464"))

from rth_zint_demo import PlaneRTH
from RTH_numerical_verification import rref_and_param

t = PlaneRTH(9, 9)
E = t.E
T, piv, free, dim = rref_and_param(t.xrows, t.yrows, E)
print(f"E={E} dim(R)={dim}", flush=True)

parent = list(range(E))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


for (keys, pts, ftype), segs in zip(t.faces, t.face_edges):
    ids = [s[0] for s in segs]
    if ftype == 3:
        for b in ids[1:]:
            r0, rb = find(ids[0]), find(b)
            if r0 != rb:
                parent[rb] = r0
    elif ftype == 4:
        for a, b in ((ids[0], ids[2]), (ids[1], ids[3])):
            r0, rb = find(a), find(b)
            if r0 != rb:
                parent[rb] = r0

classes = {}
cls_of = [0] * E
for e in range(E):
    r = find(e)
    classes.setdefault(r, len(classes))
    cls_of[e] = classes[r]
NC = len(classes)
print(f"classes={NC}", flush=True)

# --- 1) class binding vs real solution space: row vector (e1 - e2) must lie
#     in the constraint row space ---
# rref_and_param gives Tm (x[piv] = -Tm f): kernel basis = images of unit
# vectors f_j.
# Kernel basis vector b_j: b_j[free_j]=1, b_j[piv]=-Tm[:,j].
Tm = np.array([[int(v) for v in row] for row in T], dtype=np.int64)
basis = np.zeros((dim, E), dtype=np.int64)
for j, fc in enumerate(free):
    basis[j, fc] = 1
    for i, pc in enumerate(piv):
        basis[j, pc] = -Tm[i, j]
# Equal-value check: for each class take two member edges; the difference
# vector on the kernel basis must be identically 0.
bad = 0
by_cls = {}
for e in range(E):
    by_cls.setdefault(cls_of[e], []).append(e)
for c, members in by_cls.items():
    e0 = members[0]
    for e1 in members[1:]:
        diff = basis[:, e0] - basis[:, e1]
        if diff.any():
            bad += 1
            if bad <= 5:
                print(f"  class {c}: edges {e0}/{e1} unequal in the real "
                      f"solution space (diff vector has "
                      f"{np.count_nonzero(diff)} nonzeros)", flush=True)
print(f"class binding vs real solution space: "
      f"{'consistent' if bad == 0 else f'{bad} unequal pairs!'}",
      flush=True)

# --- 2) LP relaxation: class variables x in [0,1], constraint rows
#     sum(coef*x) = 0 ---
rows = []
for ri in range(t.F):
    for idx in (0, 1):
        dr, di = {}, {}
        for eid, sgn in t.face_edges[ri]:
            qd = t.edge_cs[eid][idx]
            p2 = int(qd.a * 2) * sgn
            q2 = int(qd.b * 2) * sgn
            c = cls_of[eid]
            if p2:
                dr[c] = dr.get(c, 0) + p2
            if q2:
                di[c] = di.get(c, 0) + q2
        for d in (dr, di):
            d = {k: v for k, v in d.items() if v != 0}
            if len(d) >= 2:
                rows.append(d)
print(f"rows={len(rows)}", flush=True)

try:
    from scipy.optimize import linprog
    A = np.zeros((len(rows), NC))
    for i, d in enumerate(rows):
        for c, cf in d.items():
            A[i, c] = cf
    tri_cls = by_cls_key = None
    tri_set = set()
    for (keys, pts, ftype), segs in zip(t.faces, t.face_edges):
        if ftype == 3:
            tri_set.add(cls_of[segs[0][0]])
    assert len(tri_set) == 1
    tri_cls = tri_set.pop()
    b_eq = np.zeros(len(rows))
    # fix triangle class = 1: b_eq -= A[:, tri_cls]
    b_eq = b_eq - A[:, tri_cls]
    bounds = [(0, 1)] * NC
    bounds[tri_cls] = (1, 1)
    r2 = linprog(np.zeros(NC), A_eq=A, b_eq=b_eq, bounds=bounds,
                 method="highs")
    print(f"LP relaxation (tri=1): "
          f"{'feasible' if r2.status == 0 else 'infeasible ' + r2.message}",
          flush=True)
    if r2.status == 0:
        x = r2.x
        frac = np.sum((x > 1e-9) & (x < 1 - 1e-9))
        print(f"  LP solution: {int(np.sum(x > 0.5))} classes ~1, "
              f"{frac} fractional classes", flush=True)
except ImportError:
    print("scipy unavailable; skipping LP", flush=True)
