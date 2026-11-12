# -*- coding: utf-8 -*-
"""Fig 1b flat-band cell asset: RTH {4,5,6} solution resampling
(three-color balanced version).

Reuses the PlaneRTH constructor and integer-kernel sampling from
rth_zint_demo; a seed scan counts the three edge-color classes and picks
the solution maximizing the smallest color count (strongest random-tiling
look); renders a full-bleed portrait single panel.
Output: G4_rth_colorful.png next to this script (300 dpi).
"""
import math
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

from rth_zint_demo import (PlaneRTH, sample_int_solution, build_embed,
                           embed, EDGE_COL, FACE_FILL)
from RTH_numerical_verification import rref_and_param

W = L = 9
t = PlaneRTH(W, L)
E = t.E
T, piv, free, dim = rref_and_param(t.xrows, t.yrows, E)
Tm = np.array([[int(v) for v in row] for row in T], dtype=np.int64)

# Triangle edges share one color = the triangle locking theorem (all
# triangle edges locked to a single value, which varies with the
# solution; locked-value frequency 4/5/6 = 7%/85%/9%). Seed scan: among
# locking solutions pick the most balanced three-color counts (largest
# minimum color count). Base patch enlarged 5x5 -> 7x7 so the crop
# window grows with the patch and still covers fully without white
# corners.
tri_eids = set()
for (keys, pts, ftype), segs in zip(t.faces, t.face_edges):
    if ftype == 3:
        for eid, _sgn in segs:
            tri_eids.add(eid)
best = None
for seed in range(20260000, 20260800):
    s = sample_int_solution(Tm, piv, free, E, seed)
    lock_vals = set(int(s[e]) for e in tri_eids)
    if len(lock_vals) != 1:
        continue
    cnt = np.bincount(s, minlength=7)[4:7]
    bal = int(cnt.min())
    if best is None or bal > best[0]:
        best = (bal, seed, s.copy(), cnt.copy())
assert best is not None, "no triangle-locked solution in seed scan"
bal, seed, s, cnt = best
lock_vals = set(int(s[e]) for e in tri_eids)
assert len(lock_vals) == 1, f"triangle lock violated: {lock_vals}"
print(f"seed={seed} lock={lock_vals.pop()} counts 4/5/6 = "
      f"{cnt.tolist()} (min-color {bal}, scanned 800)", flush=True)

eid2end, adj, start = build_embed(t)
pos = embed(t, adj, start, s)

# Face coloring: hexagons by area with viridis (the face type with the
# largest area variation: a hexagon has 6 independent 4/5/6 edges, so
# its relative area variation exceeds the triangle's [locked value] and
# the square's [opposite-edge equality]); triangles/squares light grey;
# edges three-colored. Color = hexagon area/mean within the crop window.
win_cx, win_cy = None, None
allx0 = [p[0] for p in pos.values()]
ally0 = [p[1] for p in pos.values()]
span_x0 = max(allx0) - min(allx0)
span_y0 = max(ally0) - min(ally0)
half_w0 = 0.34 * span_x0       # deep crop (full-bleed regular rectangle)
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


all_faces_pts = [([pos[k] for k in keys if k in pos]) for keys, pts, ftype
                 in t.faces]
all_faces_pts = [f for f in all_faces_pts if len(f) == ftype + 0]
all_faces_pts = [([pos[k] for k in keys]) for keys, pts, ftype in t.faces
                 if all(k in pos for k in keys)]
half_w0 = largest_window(all_faces_pts, cx0, cy0, aspect=0.535)
half_h0 = half_w0 / 0.535
win_box = (cx0 - half_w0, cx0 + half_w0, cy0 - half_h0, cy0 + half_h0)


def in_win(pts):
    return all(win_box[0] <= p[0] <= win_box[1] and
               win_box[2] <= p[1] <= win_box[3] for p in pts)


def touch_win(pts):
    """Partial window overlap also counts: edge hexagons are colored by
    their own area; clipping is left to the axes xlim/ylim, and colors
    use the full-face area = the tiling solution's physical quantity,
    independent of the crop window."""
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
segs, cols = [], []
for eid, (a, b, ux, uy) in eid2end.items():
    if a not in pos or b not in pos:
        continue
    segs.append([pos[a], pos[b]])
ax.add_collection(LineCollection(segs, colors="#3a3a3a", linewidths=1.5,
                                 capstyle="round", zorder=3))
ax.set_xlim(win_box[0], win_box[1])
ax.set_ylim(win_box[2], win_box[3])
out = HERE / "G4_rth_colorful.png"
fig.savefig(out, dpi=300)
from PIL import Image, ImageEnhance
import numpy as np
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
print("saved G4_rth_colorful.png", flush=True)
