# -*- coding: utf-8 -*-
"""Flip certificate figure v3: honeycomb {1,2,3} flower-pair flip double
completion -- real-metric rendering.
Edge length = edge value (colour = length, one encoding); equiangular 120
deg; BFS-layout tiling; per-face closure + revisit-consistency asserted.
Mode: central face (3,5) hexagon v=-1; each ring neighbour = shared edge -1
+ two radial edges +1.
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths_common as PC  # noqa: E402

SLOT_JSON = os.path.join(str(PC.DATA_ROOT), 'lower_bounds', 'mode_slots.json')
OUT_DIR = os.path.join(str(PC.FIGURES), 'assets_certificates')

S3 = math.sqrt(3.0)
OFF = {0: (0, -1), 1: (1, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 1), 5: (-1, 0)}
DIR60 = [(math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k)))
         for k in range(6)]

slots = json.load(open(SLOT_JSON, encoding='utf-8'))
v = {}
for key, val in slots.items():
    i, j, k = [int(x) for x in key.strip('()').split(',')]
    v[(i, j, k)] = val

P = set()
for (i, j, k) in v:
    P.add((i, j))
    P.add((i + OFF[k][0], j + OFF[k][1]))
CENTRAL = (3, 5)
assert CENTRAL in P

# adjacency of P (torus L arbitrarily large; the flower has no local wrap)
def neighbor_face(face, k):
    return (face[0] + OFF[k][0], face[1] + OFF[k][1])


def values_of(face, sign):
    """completed length values of a face's six edges:
    P faces = 2 + sign*v, sea faces = 2"""
    if face in P:
        return [2 + sign * v.get((face[0], face[1], k), 0) for k in range(6)]
    return [2] * 6


def place_from_anchor(ell, k_anchor, anchor_xy, theta0):
    """Neighbour-face gluing: edge k'=(k+3)%6 of the new face is the shared
    edge traversed in reverse, with its start at anchor_xy; base angle
    theta0 given. Vertices follow from the generic walk: vertex_{k'} is
    known; step along e = k', k'+1, ..., k'-1 in order."""
    kp = (k_anchor + 3) % 6
    theta0 = math.radians(math.degrees(theta0) + 180.0)  # reversed base angle
    verts = [None] * 6
    pos = anchor_xy
    verts[kp] = pos
    for m in range(6):
        e = (kp + m) % 6
        d = DIR60[e]
        nxt = (pos[0] + ell[e] * d[0], pos[1] + ell[e] * d[1])
        if m < 5:
            verts[(e + 1) % 6] = nxt
        pos = nxt
    return verts, theta0


def layout(sign):
    """BFS layout: returns placed = {face: (verts6, theta0_deg)};
    closure and revisit consistency asserted"""
    placed = {}
    # central face: vertex_0 = origin, theta0 = 0
    ell0 = values_of(CENTRAL, sign)
    vs = [(0.0, 0.0)]
    for k in range(6):
        d = DIR60[k]
        vs.append((vs[-1][0] + ell0[k] * d[0], vs[-1][1] + ell0[k] * d[1]))
    assert math.hypot(vs[6][0] - vs[0][0], vs[6][1] - vs[0][1]) < 1e-9, 'central not closed'
    placed[CENTRAL] = (vs[:6], 0.0)
    queue = [CENTRAL]
    while queue:
        face = queue.pop(0)
        fverts, ftheta = placed[face]
        ell = values_of(face, sign)
        for k in range(6):
            nb = neighbor_face(face, k)
            if nb in placed:
                continue
            nverts, _ = place_from_anchor(values_of(nb, sign), k,
                                          fverts[(k + 1) % 6], math.radians(ftheta))
            # consistency: edge k+3 of the neighbour must be the same
            # segment as edge k of this face, reversed
            a = nverts[(k + 3) % 6]
            b = nverts[(k + 4) % 6]
            assert (math.hypot(a[0] - fverts[(k + 1) % 6][0],
                               a[1] - fverts[(k + 1) % 6][1]) < 1e-9
                    and math.hypot(b[0] - fverts[k][0],
                                   b[1] - fverts[k][1]) < 1e-9), ('glue', face, k, nb)
            closed = math.hypot(nverts[5][0] + ell[5] * DIR60[5][0] - nverts[0][0],
                                nverts[5][1] + ell[5] * DIR60[5][1] - nverts[0][1])
            placed[nb] = (nverts, 0.0)
            queue.append(nb)
        if len(placed) > 60:
            break
    return placed


COL = {1: '#1f77b4', 2: '#9a9a9a', 3: '#d62728'}
CTX = '#c6c6c6'

fig, axes = plt.subplots(1, 2, figsize=(10.8, 6.0))
for ax, sign, tag in ((axes[0], +1, 'completion = base 2 + v'),
                      (axes[1], -1, 'completion = base 2 - v')):
    placed = layout(sign)
    xs, ys = [], []
    for face, (fverts, _) in placed.items():
        fill = '#efefef' if face in P else '#ffffff'
        ax.add_patch(MplPolygon(fverts, closed=True, facecolor=fill,
                                edgecolor=CTX, lw=0.9, zorder=1))
        for (x, y) in fverts:
            xs.append(x)
            ys.append(y)
    labeled = set()
    for face in sorted(placed):
        fverts, _ = placed[face]
        ell = values_of(face, sign)
        for k in range(6):
            val = ell[k]
            a, b = fverts[k], fverts[(k + 1) % 6]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=COL[val], lw=3.2,
                    solid_capstyle='butt', zorder=3)
            key = (round((a[0] + b[0]) / 2, 6), round((a[1] + b[1]) / 2, 6))
            if key not in labeled and val != 2:
                labeled.add(key)
                mx, my = key
                ax.text(mx, my, str(val), color=COL[val], fontsize=9,
                        fontweight='bold', ha='center', va='center', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.12', fc='white',
                                  ec='none', alpha=0.85))
        # minimal patch envelope: black outline on P boundary edges
        # (seat values agree across it, differ inside)
        for (i, j) in sorted(P):
            fv = placed[(i, j)][0]
            for k in range(6):
                nb = (i + OFF[k][0], j + OFF[k][1])
                if nb not in P:
                    a, b = fv[k], fv[(k + 1) % 6]
                    ax.plot([a[0], b[0]], [a[1], b[1]], color='#000000',
                            lw=3.2, solid_capstyle='butt', zorder=4)
    ax.text(0.02, 0.98, tag, transform=ax.transAxes, fontsize=10.5,
            va='top', ha='left', family='monospace')
    ax.set_aspect('equal')
    ax.axis('off')
    m = 1.0
    ax.set_xlim(min(xs) - m, max(xs) + m)
    ax.set_ylim(min(ys) - m, max(ys) + m)
handles = [Line2D([0], [0], color=COL[x], lw=3) for x in (1, 2, 3)]
axes[1].legend(handles, ['edge value 1', 'edge value 2', 'edge value 3'],
               loc='lower left', fontsize=8.5, frameon=False)
fig.tight_layout()
out = os.path.join(OUT_DIR, 'flip_certificate.png')
fig.savefig(out, dpi=200)
print('saved:', out, ' placed faces per panel:', len(placed))
