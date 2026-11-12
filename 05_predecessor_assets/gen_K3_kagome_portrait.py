# -*- coding: utf-8 -*-
"""Fig 1b kagome-cell portrait asset: reuses the construction logic of
kagome_real_demo.py (engine 06_engines/z4_kagome), single panel
(seed 20260927) + deep-crop window (large cut => the rectangular window
lies entirely inside the network, full-bleed without white corners).
Output: G3_kagome_portrait.png next to this script (300 dpi).
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                           # release repo root
sys.path.insert(0, str(REPO / "06_engines" / "z4_kagome"))   # kagome_real_demo
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import kagome_real_demo as M

W = L = 8
hex_vids, hex_pos, rows, n_vars = M.K.build_patch(W, L)

v2base, v2real = {}, {}
for hi in range(len(hex_vids)):
    for k, v in enumerate(hex_vids[hi]):
        a, b = hex_pos[hi][k]
        v2base[v] = (a, b)
        v2real[v] = (a + b * 0.5, b * math.sqrt(3) / 2)
Uset = set(M.K.U)
nbrs = {v: [] for v in range(n_vars)}
for a in range(n_vars):
    for b in range(a + 1, n_vars):
        d = (v2base[b][0] - v2base[a][0], v2base[b][1] - v2base[a][1])
        if d in Uset:
            nbrs[a].append(b)
            nbrs[b].append(a)
interior = [v for v in range(n_vars) if len(nbrs[v]) == 3]

mid_ref = {}
for a in interior:
    for b in nbrs[a]:
        key = (min(a, b), max(a, b))
        if key not in mid_ref:
            mid_ref[key] = ((v2real[a][0] + v2real[b][0]) / 2,
                            (v2real[a][1] + v2real[b][1]) / 2)
bonds = sorted(mid_ref)
bid = {bd: i for i, bd in enumerate(bonds)}
kedges = []
for hv in interior:
    bs = sorted(bid[(min(hv, b), max(hv, b))] for b in nbrs[hv])
    for i, j in ((0, 1), (1, 2), (0, 2)):
        mi, mj = mid_ref[bonds[bs[i]]], mid_ref[bonds[bs[j]]]
        dx, dy = mj[0] - mi[0], mj[1] - mi[1]
        Lref = math.hypot(dx, dy)
        kedges.append((hv, bs[i], bs[j], dx / Lref, dy / Lref, Lref))
adj = {i: [] for i in range(len(bonds))}
for hv, i, j, ux, uy, Lref in kedges:
    adj[i].append((j, hv, ux, uy))
    adj[j].append((i, hv, -ux, -uy))
s0 = kedges[0][5]
mid_ref = {bd: (x / s0, y / s0) for bd, (x, y) in mid_ref.items()}


def embed(t):
    start = next(i for i in range(len(bonds)) if adj[i])
    pos = {start: (0.0, 0.0)}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt, hv, ux, uy in adj[cur]:
            if nxt in pos:
                continue
            px, py = pos[cur]
            pos[nxt] = (px + t[hv] * ux, py + t[hv] * uy)
            stack.append(nxt)
    return pos


t, _ = M.sample_t_field(W, L, 20260927)
pos = embed(t)
segs, cols = [], []
cmap = plt.get_cmap("viridis")
norm = Normalize(vmin=0.5, vmax=1.5)
for hv, i, j, ux, uy, Lref in kedges:
    if i not in pos or j not in pos:
        continue
    segs.append([pos[i], pos[j]])
    cols.append(cmap(norm(t[hv] / t.mean())))

fig = plt.figure(figsize=(2.35, 3.55))
ax = fig.add_axes([0.02, 0.01, 0.96, 0.98])
ax.set_aspect("equal")
ax.axis("off")
ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.5,
                                 capstyle="round", zorder=3))
xs_ = [p[0] for p in pos.values()]
ys_ = [p[1] for p in pos.values()]
cx, cy = (min(xs_) + max(xs_)) / 2, (min(ys_) + max(ys_)) / 2
# tall portrait window (aspect 0.62) deep crop: window fully inside the
# patch => full-bleed
half_w = 5.6
half_h = half_w / 0.62
ax.set_xlim(cx - half_w, cx + half_w)
ax.set_ylim(cy - half_h, cy + half_h)
fig.savefig(HERE / "G3_kagome_portrait.png", dpi=300)
print("saved G3_kagome_portrait.png", flush=True)
