# -*- coding: utf-8 -*-
"""Fig 1b flat-band cell asset (k=2 binary version): RTH {0,1} solution
sampling (replaces the earlier three-color k=3 version).

The delivered nine-point area-law counting uses the binary alphabet
k=2 (Omega(L=2, k=2) = 128 = 2^7), so the asset switches to k=2
solutions; the display alphabet is {1,2} (the {0,1} data shifted).

Route: PlaneRTH(9,9) construction; triangle locking (theorem: all
triangle edges carry one global value) via explicit union-find + locked
value 1; square opposite-edge equality via union-find; hexagon closure
= the two projection conditions of the 60-deg hexagon (QD integerized
and split into rational / sqrt3 rows), class-variable DFS + row
interval pruning + single-DOF propagation. A seed scan picks the
solution with the most balanced counts of the two edge lengths inside
the window (strongest random-tiling look); the rendering convention
matches the three-color version (grey edges + hexagon-area viridis +
PIL enhancement).

Output: G4_rth_binary.png next to this script (300 dpi).
"""
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                           # release repo root
sys.path.insert(0, str(REPO / "06_engines" / "z4_3464"))    # rth_zint_demo + RTH_numerical_verification

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon

from rth_zint_demo import PlaneRTH, build_embed, embed
from RTH_numerical_verification import QD

W = L = 9
t = PlaneRTH(W, L)
E = t.E
print(f"PlaneRTH({W},{L}): V={t.V} F={t.F} E={E}", flush=True)

# ---------- union-find: triangle equal-value + square opposite-edge equality ----------
parent = list(range(E))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def union(a, b):
    ra, rb = find(a), find(b)
    assert ra != rb or True
    if ra != rb:
        parent[rb] = ra


n_tri = n_sq = 0
for (keys, pts, ftype), segs in zip(t.faces, t.face_edges):
    if ftype == 3:
        e0, e1, e2 = (s[0] for s in segs)
        union(e0, e1)
        union(e1, e2)
        n_tri += 1
    elif ftype == 4:
        ids = [s[0] for s in segs]
        # face edges in pts cyclic order: opposite edges = [0] vs [2], [1] vs [3]
        union(ids[0], ids[2])
        union(ids[1], ids[3])
        n_sq += 1
print(f"union: {n_tri} triangles, {n_sq} squares", flush=True)

classes = {}
for e in range(E):
    r = find(e)
    if r not in classes:
        classes[r] = len(classes)
cls_of = np.array([classes[find(e)] for e in range(E)], dtype=np.int64)
NC = len(classes)

# Triangle lock: all triangle edges in one class? Verify the triangle
# class is unique and lock its value to 1 (k=2 branch t=1; the t=0
# branch is the global reflection x->1-x, no visual difference after
# the display shift)
tri_cls = set()
for (keys, pts, ftype), segs in zip(t.faces, t.face_edges):
    if ftype == 3:
        for s in segs:
            tri_cls.add(int(cls_of[s[0]]))
assert len(tri_cls) == 1, f"triangle lock should yield a single equal-value class, got {tri_cls}"
tri_cls = tri_cls.pop()

# ---------- hexagon (and remaining face) closure rows: QD split into rational / sqrt3 integer rows ----------
# row coefficients: coeff[cls] = sum sgn*2*(p or q); once fully determined, sum coeff*val == 0
rows = []  # each row: dict cls -> int coef
for ri in range(t.F):
    for comp in ("x", "y"):
        idx = 0 if comp == "x" else 1
        dr, di = {}, {}
        for eid, sgn in t.face_edges[ri]:
            qd = t.edge_cs[eid][idx]   # (cosQD, sinQD) tuple, indexed by row component
            p2 = int(qd.a * 2) * sgn   # rational part x2
            q2 = int(qd.b * 2) * sgn   # sqrt3 part x2
            c = cls_of[eid]
            if p2:
                dr[c] = dr.get(c, 0) + p2
            if q2:
                di[c] = di.get(c, 0) + q2
        for d in (dr, di):
            d = {k: v for k, v in d.items() if v != 0}
            if len(d) >= 2:
                rows.append(d)
            elif len(d) == 1:
                (c, v), = d.items()
                assert v == 0, f"single-class nonzero row: cls {c} coef {v} (face {ri} {comp})"
print(f"closure rows: {len(rows)} (hex + rational/sqrt3 split)", flush=True)

# class -> row indices for rows
cls_rows = [[] for _ in range(NC)]
for ri_, d in enumerate(rows):
    for c in d:
        cls_rows[c].append(ri_)

# ---------- DFS: class variables {0,1}, triangle class locked to 1; row interval pruning + single-DOF propagation ----------


def solve(seed, max_backtrack=200000):
    rng = random.Random(seed)
    val = [None] * NC
    val[tri_cls] = 1
    row_sum = [0] * len(rows)      # partial sums over determined entries
    row_abs = [sum(abs(v) for v in d.values()) for d in rows]  # sum of |coef|
    # row_sum init: the triangle class is already determined, its
    # coefficient must be booked, otherwise every row containing it is
    # constrained wrongly
    for ri_, d in enumerate(rows):
        if tri_cls in d:
            row_sum[ri_] += d[tri_cls] * 1

    def row_feasible(ri_):
        d = rows[ri_]
        fixed = row_sum[ri_]
        lo = hi = fixed
        for c, cf in d.items():
            if val[c] is None:
                lo -= abs(cf)
                hi += abs(cf)
        return lo <= 0 <= hi

    # trigger-driven propagation: queued rows check feasibility + single-DOF
    # determination; all assignments recorded in trail
    def propagate(queue, trail):
        while queue:
            ri_ = queue.pop()
            d = rows[ri_]
            if not row_feasible(ri_):
                return False
            un = [(c, cf) for c, cf in d.items() if val[c] is None]
            if len(un) == 1:
                c, cf = un[0]
                need = -row_sum[ri_]
                if need % cf != 0:
                    return False
                v = need // cf
                if v not in (0, 1):
                    return False
                if not assign(c, v, queue, trail):
                    return False
        return True

    def assign(c, v, queue, trail):
        val[c] = v
        trail.append((c, v))
        for ri_ in cls_rows[c]:
            row_sum[ri_] += rows[ri_][c] * v
            queue.append(ri_)
        return True

    free_cls = [c for c in range(NC) if c != tri_cls]
    order0 = free_cls[:]
    rng.shuffle(order0)
    # order by constraint degree (RLV greedy) + random jitter:
    # most-constrained classes first
    order0.sort(key=lambda c: (-len(cls_rows[c]), rng.random()))

    sys.setrecursionlimit(10000)
    backtracks = [0]

    def dfs(unassigned):
        live = [c for c in unassigned if val[c] is None]
        if not live:
            return all(row_sum[ri_] == 0 for ri_ in range(len(rows)))
        # dynamic MRV: among unassigned classes pick the most constrained
        # (fewest undetermined classes among its rows)
        best_c, best_key = None, None
        for c in live:
            key = min(len([cc for cc in rows[ri_] if val[cc] is None])
                      for ri_ in cls_rows[c])
            if best_key is None or key < best_key:
                best_c, best_key = c, key
                if key <= 2:
                    break
        c = best_c
        rest = [x for x in unassigned if x != c]
        vals = [0, 1]
        rng.shuffle(vals)
        for v in vals:
            if backtracks[0] > max_backtrack:
                return False
            trail = []
            queue = []
            ok = assign(c, v, queue, trail) and propagate(queue, trail)
            if ok and dfs(rest):
                return True
            backtracks[0] += 1
            # undo all trail assignments in reverse (including propagated
            # ones, preventing state leakage)
            for c2, v2 in reversed(trail):
                for ri_ in cls_rows[c2]:
                    row_sum[ri_] -= rows[ri_][c2] * v2
                val[c2] = None
        return False

    if dfs(free_cls):
        return val, backtracks[0]
    return None, backtracks[0]


# ---------- seed scan + geometric verification + balanced in-window solution choice ----------
eid2end, adj, start = build_embed(t)


def verify_solution(cls_val):
    """Exact per-face QD closure + unique triangle lock value."""
    s = np.zeros(E, dtype=np.int64)
    for e in range(E):
        s[e] = cls_val[cls_of[e]]
    assert set(np.unique(s)) <= {0, 1}, "solution alphabet is {0,1}"
    tri_vals = set()
    for (keys, pts, ftype), segs in zip(t.faces, t.face_edges):
        tx = QD(0, 0, 3)
        ty = QD(0, 0, 3)
        for eid, sgn in segs:
            Lv = QD(int(s[eid]), 0, 3)
            cx, sy = t.edge_cs[eid]
            if sgn > 0:
                tx = tx + cx * Lv
                ty = ty + sy * Lv
            else:
                tx = tx - cx * Lv
                ty = ty - sy * Lv
        assert tx.iszero() and ty.iszero(), f"closure failed ftype={ftype}"
        if ftype == 3:
            for eid, _ in segs:
                tri_vals.add(int(s[eid]))
    assert len(tri_vals) == 1, f"triangle lock violated: {tri_vals}"
    return s


def window_stats(s):
    pos = embed(t, adj, start, s)
    allx = [p[0] for p in pos.values()]
    ally = [p[1] for p in pos.values()]
    cx = (min(allx) + max(allx)) / 2
    cy = (min(ally) + max(ally)) / 2
    span = max(max(allx) - min(allx), max(ally) - min(ally))
    # inscribed-square approximate window (for solution choice only; need
    # not match the final render's largest_window)
    half = 0.20 * span
    cnt = {1: 0, 2: 0}
    nhx = 0
    for (keys, pts, ftype), segs in zip(t.faces, t.face_edges):
        if ftype != 6:
            continue
        P = [pos[k] for k in keys if k in pos]
        if len(P) != 6:
            continue
        if not all(cx - half <= p[0] <= cx + half and
                   cy - half <= p[1] <= cy + half for p in P):
            continue
        nhx += 1
        for eid, _ in segs:
            cnt[int(s[eid]) + 1] += 1
    bal = min(cnt.values())
    return bal, nhx, cnt


best = None
n_solved = 0
for seed in range(20261000, 20261120):
    val, bt = solve(seed)
    if val is None:
        print(f"seed {seed}: no solution (bt={bt})", flush=True)
        continue
    n_solved += 1
    s = verify_solution(val)
    bal, nhx, cnt = window_stats(s)
    if best is None or (bal, nhx) > best[0]:
        best = ((bal, nhx), seed, s.copy(), cnt, bt)
    if n_solved >= 40:
        break
assert best is not None, "seed scan found no solution"
(score, seed, s, cnt, bt) = best
print(f"best seed={seed} score={score} window edges 1/2 = "
      f"{cnt.get(1, 0)}/{cnt.get(2, 0)} bt={bt} (solved {n_solved})",
      flush=True)

# Display convention: data {0,1} shifted by +1 -> {1,2} (the alphabet is
# shift-invariant; zero-length edges cannot be drawn)
s = s + 1
assert set(np.unique(s)) == {1, 2}, f"display alphabet should be {{1,2}}: {np.unique(s)}"

# ---------- render (same convention as the three-color version: grey edges + hexagon-area viridis) ----------
pos = embed(t, adj, start, s)
allx0 = [p[0] for p in pos.values()]
ally0 = [p[1] for p in pos.values()]
span_x0 = max(allx0) - min(allx0)
span_y0 = max(ally0) - min(ally0)
half_w0 = 0.34 * span_x0
half_h0 = half_w0 / 0.62
cx0, cy0 = (min(allx0) + max(allx0)) / 2, (min(ally0) + max(ally0)) / 2


def largest_window(face_polys, cx, cy, aspect=0.535, lo=0.30, hi=None):
    from matplotlib.path import Path as MplPath
    paths = [MplPath(np.asarray(f, float)) for f in face_polys if len(f) >= 3]
    px = np.concatenate([np.asarray(f, float)[:, 0] for f in face_polys])
    py = np.concatenate([np.asarray(f, float)[:, 1] for f in face_polys])
    if hi is None:
        hi = 0.5 * max(px.max() - px.min(), py.max() - py.min()) / aspect

    def covered(half_w):
        half_h = half_w / aspect
        xs = np.linspace(cx - half_w, cx + half_w, 36)
        ys = np.linspace(cy - half_h, cy + half_h, 36)
        X, Y = np.meshgrid(xs, ys)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        ok = np.zeros(len(pts), bool)
        for pa in paths:
            ok |= pa.contains_points(pts, radius=-1e-9)
            if ok.all():
                return True
        return bool(ok.all())

    if not covered(lo):
        return lo
    a, b = lo, hi
    for _ in range(16):
        mid = 0.5 * (a + b)
        if covered(mid):
            a = mid
        else:
            b = mid
    return a


all_faces_pts = [([pos[k] for k in keys]) for keys, pts, ftype in t.faces
                 if all(k in pos for k in keys)]
half_w0 = largest_window(all_faces_pts, cx0, cy0, aspect=0.535)
half_h0 = half_w0 / 0.535
win_box = (cx0 - half_w0, cx0 + half_w0, cy0 - half_h0, cy0 + half_h0)


def touch_win(pts):
    return any(win_box[0] <= p[0] <= win_box[1] and
               win_box[2] <= p[1] <= win_box[3] for p in pts)


def poly_area(pts):
    x, y = zip(*pts)
    n = len(x)
    return 0.5 * abs(sum(x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i]
                         for i in range(n)))


hex_faces = []
for keys, pts, ftype in t.faces:
    if ftype != 6:
        continue
    P = [pos[k] for k in keys if k in pos]
    if len(P) == 6 and touch_win(P):
        hex_faces.append((P, poly_area(P)))
am = sum(a for _, a in hex_faces) / len(hex_faces)
cmap = plt.get_cmap("viridis")
a_lo = min(a for _, a in hex_faces) / am
a_hi = max(a for _, a in hex_faces) / am
print(f"hexagons in window: {len(hex_faces)}, area/mean in "
      f"[{a_lo:.2f}, {a_hi:.2f}]", flush=True)

fig = plt.figure(figsize=(2.6, 4.0))
ax = fig.add_axes([0.02, 0.01, 0.96, 0.98])
ax.set_aspect("equal")
ax.axis("off")
for keys, pts, ftype in t.faces:
    if any(k not in pos for k in keys):
        continue
    P = [pos[k] for k in keys]
    if not touch_win(P):
        continue
    if ftype == 6:
        a = poly_area(P) / am
        rgb = cmap((a - a_lo) / (a_hi - a_lo + 1e-12))
        ax.add_patch(MplPolygon(P, closed=True, facecolor=rgb,
                                edgecolor="none", zorder=1))
    else:
        ax.add_patch(MplPolygon(P, closed=True, facecolor="#e9e9e9",
                                edgecolor="none", zorder=1))
segs = []
for eid, (a, b, ux, uy) in eid2end.items():
    if a not in pos or b not in pos:
        continue
    segs.append([pos[a], pos[b]])
ax.add_collection(LineCollection(segs, colors="#3a3a3a", linewidths=1.5,
                                 capstyle="round", zorder=3))
ax.set_xlim(win_box[0], win_box[1])
ax.set_ylim(win_box[2], win_box[3])
out = HERE / "G4_rth_binary.png"
fig.savefig(out, dpi=300)

from PIL import Image, ImageEnhance
img = Image.open(out).convert("RGB")
arr = np.asarray(img, int)
bg = arr[2, 2].tolist()
diff = np.abs(arr - np.array(bg, int)).sum(axis=2)
ys, xs = np.where(diff > 18)
if len(xs):
    pad = 6
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
           min(img.size[0], xs.max() + pad),
           min(img.size[1], ys.max() + pad))
    img = img.crop(box)
img = ImageEnhance.Color(img).enhance(1.45)
img = ImageEnhance.Contrast(img).enhance(1.08)
img.save(out)
print("saved G4_rth_binary.png", flush=True)
