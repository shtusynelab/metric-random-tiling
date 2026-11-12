# -*- coding: utf-8 -*-
"""L2 main-text Fig. 3 -- honeycomb geometric limit (L2-local file).

Layout: canvas 10.82x6.30 in (same frame as Fig 1; after the typesetting
scale-down to 183 mm the font sizes return to the original 1:1 design
values), two rows x four columns; all font sizes/line widths = original
design pt x K, applied uniformly by scale_fig after rendering (panel-
internal original pt values untouched):
  col 1: a counting construction (vertical: patch on top, tube|ring below,
  L-shaped arrow flow) / b tile library;
  cols 2-3: c sample tilings of the three alphabets (spanning two columns,
  heavy black edges = Fig 1/2 language);
  col 2: d resonant/non-resonant counting, three stacked strips;
  col 3: e lambda(Ly)/Ly width scaling.
Letter order: a,b in the left column; c = patterns; d = counts; e = lambda.
Parent = the L1 canonical fig2_honeycomb_metric_entropy.py (frozen); this
file changes only layout coordinates/canvas/font base; panel drawing logic
and data sources are unchanged. Output:
L2_metric_random_tiling/figs/fig3_honeycomb_metric_entropy.{png,pdf}

Manuscript-facing alphabets are uniformly 1-based: display and text always
use {1,2}/{1,2,3}/{1,2,4}; the data alphabets {0,1}/{0,1,2}/{0,1,3} are
translation-equivalent (both closure equations are invariant under
L -> L+1 and Omega is identical digit-for-digit; affine invariance);
0-based labels are internal to the counting scripts and no longer appear
on the manuscript face.
"""
import math
import queue
import random
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent            # L2_metric_random_tiling/scripts
sys.path.insert(0, str(HERE))                     # l1_style/gen_S9_torus_construction
import paths_common as PC  # noqa: E402
ROOT = PC.HC                                      # honeycomb engine dir
FIGS = PC.FIGURES                                 # delivered figure dir (this figure's output)
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))                     # pseudocrystal_lib

import pseudocrystal_lib as pl
from l1_style import plt, panel_label, savefig, DOUBLE_COL
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import gen_S9_torus_construction as s9   # piece-tube-ring construction drawing
                                         # code (reused for top-row panel a)

EDGE_GRAY = '#3a3a3a'   # uniform edge gray (matches the Fig 1 gray skeleton;
                        # length is carried by geometry)
ALPHA_C = {'binary': '#2166ac', 'arithmetic': '#c0392b',
           'nonarith': '#e6811e'}
ALPHA_LAB = {'binary': '{1, 2} (binary)', 'arithmetic': '{1, 2, 3} (arith.)',
             'nonarith': '{1, 2, 4} (non-arith.)'}
ALPH_VALS = {'binary': (1, 2), 'arithmetic': (1, 2, 3),
             'nonarith': (1, 2, 4)}
# width colors/markers (panel c width-scaling version; Ly7/9 are the
# transpose three-point series)
W_COL = {'Ly2': '#762a83', 'Ly3': '#2166ac', 'Ly5': '#5aae61',
         'Ly7': '#e08214', 'Ly9': '#8c510a'}
W_LAB = {'Ly2': '$L_y=2$', 'Ly3': '$L_y=3$', 'Ly5': '$L_y=5$',
         'Ly7': '$L_y=7$', 'Ly9': '$L_y=9$'}
W_MK = {'Ly2': 'o', 'Ly3': 's', 'Ly5': '^', 'Ly7': 'D', 'Ly9': 'v'}
LY = {'Ly2': 2, 'Ly3': 3, 'Ly5': 5, 'Ly7': 7, 'Ly9': 9}
A_TAG = {'binary': '$\\{1,2\\}$ (binary)',
         'arithmetic': '$\\{1,2,3\\}$ (arith.)',
         'nonarith': '$\\{1,2,4\\}$ (non-arith.)'}
CMAP = plt.get_cmap('viridis')

# ---------------- new-layout constants (inches) ----------------
WIN, HIN = 10.82, 6.30        # canvas = same frame as Fig 1
K = WIN / 7.205               # font/linewidth compensation: scaled uniformly
                              # at render time; after scale-down to 183 mm the
                              # sizes return to the 6-8 pt of the 1:1 design
ML, MR = 0.50, 0.50           # left/right margins
# four columns: col 1 a/b stacked; c (three tilings stacked), d (counting
# strips) and e (lambda/Ly) all span two rows (stretched vertically -> d's
# data points stay readable)
COLW = (2.68, 1.96, 2.23, 1.60)     # d-e gap widened (keeps the lambda/Ly
                                    # label off d's right edge)
GUTS = (0.40, 0.40, 0.55)
RBOT, RGAP, RTOP = 0.55, 0.25, 0.10   # bottom margin / row gap / top margin
ROWH1 = ROWH2 = 2.70
CX = (ML,
      ML + COLW[0] + GUTS[0],
      ML + COLW[0] + COLW[1] + GUTS[0] + GUTS[1],
      ML + COLW[0] + COLW[1] + COLW[2] + sum(GUTS))
GUT = GUTS[0]                  # legacy alias
RY1, RY2 = RBOT + ROWH2 + RGAP, RBOT    # bottom y of row 1 (top) / row 2
TALL0, TALLH = RBOT, HIN - RTOP - RBOT  # bottom y / height of two-row panels


def fx(x_in):
    return x_in / WIN


def fy(y_in):
    return y_in / HIN


def fit_axes(ax, x_ref_in, y_ref_in, max_w_in, max_h_in, anchor='top',
             halign='center'):
    """Fit an already-drawn equal-aspect 2D axes box to its data aspect
    ratio (prevents horizontal/vertical clipping), horizontally per halign
    ('center' = centered on x_ref / 'left' = left edge at x_ref), top (or
    bottom) edge at y_ref_in, size capped at max_w x max_h."""
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ar = (xl[1] - xl[0]) / (yl[1] - yl[0])           # data width/height
    w = max_w_in
    h = w / ar
    if h > max_h_in:
        h = max_h_in
        w = h * ar
    x0 = x_ref_in if halign == 'left' else x_ref_in - w / 2
    y0 = y_ref_in - h if anchor == 'top' else y_ref_in
    ax.set_position([fx(x0), fy(y0), fx(w), fy(h)])
    return w, h


def scale_fig(fig, k):
    """Globally scale all text/linewidths/markers by k after rendering
    (panel-internal original pt values unchanged)."""
    def sq(t):
        t.set_fontsize(t.get_fontsize() * k)
    for ax in fig.axes:
        for t in ax.texts:
            sq(t)
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            sq(t)
        for lb in (ax.xaxis.label, ax.yaxis.label):
            sq(lb)
        for ln in ax.lines:
            ln.set_linewidth(ln.get_linewidth() * k)
            ln.set_markersize(ln.get_markersize() * k)
            ln.set_markeredgewidth(ln.get_markeredgewidth() * k)
        for col in ax.collections:
            lws = col.get_linewidths()
            if len(lws):
                col.set_linewidth(lws[0] * k)
        for pt in ax.patches:
            if hasattr(pt, 'set_linewidth'):
                pt.set_linewidth(pt.get_linewidth() * k)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                sq(t)
            for h in lg.legend_handles:
                if hasattr(h, 'set_linewidth'):
                    h.set_linewidth(h.get_linewidth() * k)
                if hasattr(h, 'set_markersize'):
                    h.set_markersize(h.get_markersize() * k)
    for t in fig.texts:
        sq(t)


def polygon_area(pts):
    x, y = zip(*pts)
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i]
                         for i in range(-1, len(x) - 1)))


def hex_pts(colors, s):
    """Equiangular hexagon vertices (centered): edge directions step by
    60 deg, edge length = value x s. The closure assert is the geometric
    closure test (reused by the sampler as closure test)."""
    weights = [c + 1 for c in colors]
    a0 = np.deg2rad(90)
    pts = [(0.0, 0.0)]
    for k, w in enumerate(weights):
        a = a0 + np.deg2rad(60) * k
        pts.append((pts[-1][0] + s * w * np.cos(a),
                    pts[-1][1] + s * w * np.sin(a)))
    assert abs(pts[-1][0]) < 1e-8 and abs(pts[-1][1]) < 1e-8, \
        f'tile not closed: {weights}'
    gx = sum(p[0] for p in pts[:-1]) / 6
    gy = sum(p[1] for p in pts[:-1]) / 6
    return [(px - gx, py - gy) for px, py in pts[:-1]]


def _closes(lengths):
    try:
        hex_pts(tuple(v - 1 for v in lengths), 0.1)
        return True
    except AssertionError:
        return False


def sample_metric_patch(n_rings, alphabet, seed0=7004, tries=200,
                        criterion='balanced'):
    """Random sampling under pure metric-closure constraints: per-hole
    backtracking DFS over a finite honeycomb block (n_rings rings), each
    hole's 6 edges assigned from the alphabet with geometric closure.
    criterion='balanced' (default): the seed scan picks the sample with the
    most balanced alphabet-value frequencies. criterion='separated': under
    a near-balance constraint (both short/long domains sizable), maximize
    the same-area-class adjacency share -- realizes the "layered short/long
    domains" caption."""
    valid = [L for L in product(alphabet, repeat=6) if _closes(L)]
    assert valid, f'no closed hexagon in {alphabet}'
    cells = [(i, j) for i in range(-n_rings + 1, n_rings)
             for j in range(-n_rings + 1, n_rings)
             if (abs(i) + abs(j) + abs(i + j)) // 2 <= n_rings - 1]
    cellset = set(cells)
    order = [cells[0]]
    seen = {cells[0]}
    bq = queue.Queue()
    bq.put(cells[0])
    while not bq.empty():
        c = bq.get()
        for d in pl.DIRECTIONS:
            nb = (c[0] + d[0], c[1] + d[1])
            if nb in cellset and nb not in seen:
                seen.add(nb)
                order.append(nb)
                bq.put(nb)
    assert len(order) == len(cells)
    slot_edge = {}
    for c in cells:
        for k, d in enumerate(pl.DIRECTIONS):
            nb = (c[0] + d[0], c[1] + d[1])
            if nb in cellset:
                key = tuple(sorted([(c, k), (nb, (k + 3) % 6)]))
            else:
                key = ((c, k),)
            slot_edge[(c, k)] = key

    best = None
    target = 1.0 / len(alphabet) + 0.03
    for seed in range(seed0, seed0 + tries):
        rng = random.Random(seed)
        assign = {}
        steps = [0]

        def rec(idx):
            if idx == len(order):
                return True
            steps[0] += 1
            if steps[0] > 200000:
                raise RuntimeError('backtracking cap')
            c = order[idx]
            known = {}
            for k in range(6):
                e = slot_edge[(c, k)]
                if e in assign:
                    known[k] = assign[e]
            cands = [t for t in valid
                     if all(t[k] == v for k, v in known.items())]
            rng.shuffle(cands)
            for t in cands:
                new = {slot_edge[(c, k)]: t[k] for k in range(6)
                       if slot_edge[(c, k)] not in assign}
                assign.update(new)
                if rec(idx + 1):
                    return True
                for e in new:
                    del assign[e]
            return False

        try:
            ok = rec(0)
        except RuntimeError:
            continue
        if not ok or set(assign.values()) < set(alphabet):
            continue
        from collections import Counter
        cnt = Counter(assign.values())
        score = max(cnt.values()) / len(assign)
        if criterion == 'separated':
            if score > 0.55:
                continue                 # near-balance: both domains sizable
            per = {c: sum(assign[slot_edge[(c, k)]] for k in range(6))
                   for c in order}
            m = sum(per.values()) / len(per)
            cls = {c: per[c] >= m for c in per}
            same = tot = 0
            for c in order:
                for k, d in enumerate(pl.DIRECTIONS):
                    nb = (c[0] + d[0], c[1] + d[1])
                    if nb in cellset and nb > c:
                        tot += 1
                        if cls[c] == cls[nb]:
                            same += 1
            rank = same / max(tot, 1)    # phase separation: same-class share
        else:
            rank = -score                # balanced: the more balanced the better
        if best is None or rank > best[0]:
            sol = {}
            for c in order:
                t = tuple(assign[slot_edge[(c, k)]] for k in range(6))
                sol[c] = valid.index(t)
            best = (rank, sol, SimpleNamespace(weights=valid), seed, cnt)
            if criterion == 'balanced' and score <= target:
                break
    if best is None:
        raise RuntimeError(f'no sample for {alphabet}')
    return best[1], best[2], best[3]


def _hole_polygons(sol, lib):
    """Per-hole vertex coordinates (BFS reconstruction, same as
    gen_G_batch2.G5b; coordinate keys rounded to 4)."""
    data = {c: lib.weights[idx] for c, idx in sol.items()}
    pos = {(0, 0): (0.0, 0.0)}
    hole_verts = {}
    bfs = queue.Queue()
    bfs.put((0, 0))
    seen = {(0, 0)}
    while not bfs.empty():
        curr = bfs.get()
        if curr not in data:
            continue
        wx, wy = pos[curr]
        w = data[curr]
        verts = [(wx, wy)]
        vx, vy = float(wx), float(wy)
        for i in range(6):
            vx += float(w[i]) * math.cos(math.radians(i * 60))
            vy += float(w[i]) * math.sin(math.radians(i * 60))
            verts.append((vx, vy))
        hole_verts[curr] = verts
        for i in range(6):
            d = pl.DIRECTIONS[i]
            nb = (curr[0] + d[0], curr[1] + d[1])
            if nb in data and nb not in seen:
                n_w = data[nb]
                k = (i + 3) % 6
                ax_ = ay_ = 0.0
                for step in range(k + 1):
                    if step == k:
                        ax_n, ay_n = ax_, ay_
                    ax_ += float(n_w[step]) * math.cos(math.radians(step * 60))
                    ay_ += float(n_w[step]) * math.sin(math.radians(step * 60))
                dest = verts[i + 1]
                pos[nb] = (dest[0] - ax_n, dest[1] - ay_n)
                seen.add(nb)
                bfs.put(nb)
    return hole_verts


# ------------------------------------------------------------------ panel a
def draw_tile(ax, cx, cy, colors, s, norm, lw=1.0, val=None):
    """Tile: face color = norm(val) (absolute area by default; pass val for
    area/mean)."""
    pts = hex_pts(colors, s)
    if val is None:
        val = polygon_area(pts)
    ax.add_patch(plt.Polygon([(px + cx, py + cy) for px, py in pts],
                             closed=True, facecolor=CMAP(norm(val)),
                             edgecolor='none', zorder=1))
    for k in range(6):
        a, b = pts[k], pts[(k + 1) % 6]
        ax.plot([a[0] + cx, b[0] + cx], [a[1] + cy, b[1] + cy],
                color=EDGE_GRAY, lw=lw, solid_capstyle='round', zorder=2)
    return max(math.hypot(px, py) for px, py in pts)   # circumradius


def tile_orbits(alphabet):
    """Dihedral-class representatives of all closable hexagons over the
    alphabet (sorted by increasing perimeter)."""
    def closable(L):
        return (L[1] + L[2] == L[4] + L[5] and
                2 * L[0] + L[1] + L[5] == 2 * L[3] + L[2] + L[4])

    def d6_canon(L):
        rots = [tuple(L[k:] + L[:k]) for k in range(6)]
        m = (L[0],) + tuple(reversed(L[1:]))
        rots += [tuple(m[k:] + m[:k]) for k in range(6)]
        return min(rots)

    orbits = {}
    for L in product(alphabet, repeat=6):
        if closable(L):
            orbits.setdefault(d6_canon(L), []).append(L)
    reps = sorted(orbits, key=lambda t: (sum(t), t))     # increasing perimeter
    print(f'[QA] tile library {alphabet}: {len(reps)} dihedral orbits '
          f'({sum(len(v) for v in orbits.values())} closable hexagons)',
          flush=True)
    return reps


def panel_a(fig, x0, y0, w, h, clim):
    """Tile library, two groups stacked ({1,2,3}: 15 classes with the {1,2}
    subset marked by a dashed box; {1,2,4}: 13 classes). x0/y0/w/h =
    figure fraction coordinates (passed in by main)."""
    h_in = 2.45                                        # actual group-box height
                                                       # (bottom = colorbar bottom)
    s = 0.20
    dx = 2.40                                          # column pitch narrowed:
                                                       # tiles must not touch the box
    row_h = 1.75
    libs = {nm: tile_orbits(ALPH_VALS[nm])
            for nm in ('arithmetic', 'nonarith')}
    H_dat, W_dat = 11.74, 12.30
    # axis window tight-cropped to the group box (0.08..12.22 / 0.10..11.70),
    # rotated labels hang outside the axis; box width on the column-width
    # scale (left edge on the column baseline x0, aligned with the left edge
    # of the a patch)
    w_t = h_in * ((W_dat - 0.16) / (11.70 - 0.10)) / WIN
    ax = fig.add_axes([x0, y0 + (h * HIN - h_in) / 2 / HIN, w_t, h_in / HIN])
    # coloring convention unified with the c colorbar: per-library
    # area/library-mean, color range = clim (the same ruler as the c
    # colorbar); the former absolute-area convention left the two groups
    # inconsistent with c
    areas_lib = {nm: [polygon_area(hex_pts(tuple(v - 1 for v in t), s))
                      for t in g] for nm, g in libs.items()}
    rel_of = {}
    for nm, g in libs.items():
        m = sum(areas_lib[nm]) / len(areas_lib[nm])
        rel_of[nm] = {t: a / m for t, a in zip(g, areas_lib[nm])}
    norm = Normalize(*clim)
    radii = {nm: [max(math.hypot(px, py) for px, py in
                      hex_pts(tuple(v - 1 for v in t), s)) for t in g]
             for nm, g in libs.items()}
    xs_c = [1.30 + i * dx for i in range(5)]
    for nm, g in libs.items():
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                ddx = abs(xs_c[i % 5] - xs_c[j % 5])
                ddy = abs((i // 5) - (j // 5)) * row_h
                assert ddx >= radii[nm][i] + radii[nm][j] or \
                    ddy >= radii[nm][i] + radii[nm][j], f'overlap {nm} {i},{j}'
    y = H_dat
    # group 1: {1,2,3} 15 classes; the first 5 classes (all values <= 2) form
    # the {1,2} library, marked by a dashed box on a single row
    g = libs['arithmetic']
    relmap = rel_of['arithmetic']
    subset = [t for t in g if max(t) <= 2]
    rest = [t for t in g if max(t) > 2]
    assert len(subset) == 5 and len(subset) + len(rest) == len(g)
    ax.text(-0.55, 8.75, '$\\{1,2,3\\}$', fontsize=7.5,
            ha='center', va='center', rotation=90,
            color=ALPHA_C['arithmetic'])
    y -= 0.70
    y_sub = y - 0.85
    for c, t in enumerate(subset):
        draw_tile(ax, xs_c[c], y_sub, tuple(v - 1 for v in t), s, norm,
                  val=relmap[t])
    # subset box pad 0.55; after moving the {1,2} label above the box top it
    # still had to clear the red box top, so the whole group moved up 0.70
    # and the sub-box half-height went 1.00 -> 0.85
    ax.add_patch(Rectangle((xs_c[0] - 0.55, y_sub - 0.85),
                           (xs_c[4] - xs_c[0]) + 1.10, 1.70,
                           fill=False, ls=(0, (4, 2)), lw=0.6,
                           edgecolor=ALPHA_C['binary'], zorder=1))
    # {1,2} label rides on the box top edge with a white background
    ax.text(xs_c[0] - 0.25, y_sub + 0.85, '$\\{1,2\\}$', fontsize=6.5,
            ha='left', va='center', color=ALPHA_C['binary'], zorder=4,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    y -= 1.70
    for i, t in enumerate(rest):
        r, c = divmod(i, 5)
        draw_tile(ax, xs_c[c], y - row_h / 2 - r * row_h,
                  tuple(v - 1 for v in t), s, norm, val=relmap[t])
    y -= 2 * row_h + 0.12
    # group 2: {1,2,4} 13 classes
    g = libs['nonarith']
    relmap = rel_of['nonarith']
    ax.text(-0.55, 2.78, '$\\{1,2,4\\}$', fontsize=7.5,
            ha='center', va='center', rotation=90,
            color=ALPHA_C['nonarith'])
    y -= 0.35
    for i, t in enumerate(g):
        r, c = divmod(i, 5)
        draw_tile(ax, xs_c[c], y - row_h / 2 - r * row_h,
                  tuple(v - 1 for v in t), s, norm, val=relmap[t])
    y -= 3 * row_h + 0.12
    assert y >= 0, 'vertical overflow in tile-library stack'
    # two group outer boxes (box lines inset 0.15 units: the original box
    # edges sat exactly on the axis boundary and lost half their stroke,
    # giving uneven edge weights -- top/bottom clipped, inner sides full)
    fx0, fx1 = 0.23, W_dat - 0.23
    ax.add_patch(Rectangle((fx0, 5.72), fx1 - fx0,
                           11.55 - 5.72, fill=False,
                           ls=(0, (5, 3)), lw=1.1,
                           edgecolor=ALPHA_C['arithmetic'], zorder=1))
    ax.add_patch(Rectangle((fx0, 0.25), fx1 - fx0, 5.42 - 0.25,
                           fill=False, ls=(0, (5, 3)), lw=1.1,
                           edgecolor=ALPHA_C['nonarith'], zorder=1))
    ax.set_xlim(0.08, W_dat - 0.08)
    ax.set_ylim(0.10, 11.70)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # 'b' letter pinned to the tile-library content top (axes top != content
    # top, letterbox; same pinning as a/c)
    fig.text(x0 - 0.008, y0 + h - 0.002, 'b', fontsize=8, fontweight='bold',
             ha='right', va='top', color='#1a1a1a')


def panel_b(fig, x0, y0, w, h):
    """Sample tilings of the three alphabets, stacked vertically (spanning
    two rows). {1,2}/{1,2,3}/{1,2,4}; n_rings=4, i.e. 37 holes, backtracking
    sampling under pure metric-closure constraints; shared horizontal
    area/mean colorbar. Block heights set by column width / max aspect
    ratio (equal heights, centered widths); edge language = Fig 1/2 heavy
    black lines. x0/y0/w/h = figure fraction coordinates (column 2, spans
    two rows)."""
    PAD = 0.15
    samples = []
    lo_list, hi_list = [], []
    ROT = {'binary': 90.0, 'arithmetic': 330.0, 'nonarith': 30.0}
    for name in ('binary', 'arithmetic', 'nonarith'):
        sol, lib, seed = sample_metric_patch(
            4, ALPH_VALS[name], tries=2000 if name == 'binary' else 200,
            criterion='separated' if name == 'binary' else 'balanced')
        hole_verts = _hole_polygons(sol, lib)
        th = math.radians(ROT[name])               # specified orientation
        rot = np.array([[math.cos(th), -math.sin(th)],
                        [math.sin(th), math.cos(th)]])
        polys = [np.array([rot @ np.array(v) for v in verts[1:]])
                 for verts in hole_verts.values()]
        arts = np.array([polygon_area(p) for p in polys])
        rel = arts / arts.mean()
        allp = np.concatenate(polys, axis=0)
        _cx = (allp[:, 0].min() + allp[:, 0].max()) / 2
        _cy = (allp[:, 1].min() + allp[:, 1].max()) / 2
        _hx = np.ptp(allp[:, 0]) / 2 + PAD
        _hy = np.ptp(allp[:, 1]) / 2 + PAD
        samples.append((name, polys, rel, _cx, _cy, _hx, _hy))
        lo_list.append(rel.min())
        hi_list.append(rel.max())
        from collections import Counter
        cnt = Counter(v for c in sol for v in lib.weights[sol[c]])
        _per = {c: sum(lib.weights[sol[c]]) for c in sol}
        _m = sum(_per.values()) / len(_per)
        _cls = {c: _per[c] >= _m for c in _per}
        _same = _tot = 0
        for c in sol:
            for d in pl.DIRECTIONS:
                nb = (c[0] + d[0], c[1] + d[1])
                if nb in sol and nb > c:
                    _tot += 1
                    if _cls[c] == _cls[nb]:
                        _same += 1
        print(f'[QA] sample {name}: seed={seed}, holes={len(polys)}, '
              f'edge values {dict(sorted(cnt.items()))}, '
              f'area/mean {rel.min():.2f}-{rel.max():.2f}, '
              f'class-assortativity {_same / max(_tot, 1):.3f}', flush=True)
    clim = (min(lo_list), max(hi_list))
    # vertical stack: block height = column width / max(aspect ratio) ->
    # equal heights, centered widths; the whole band = 3x(block + label row)
    # + horizontal colorbar block, centered in [y0, y0+h]
    ar_max = max(s[5] / s[6] for s in samples)
    lab_h = 0.21 / HIN
    bar_y = fy(RY2 + 0.42)      # colorbar bottom edge = d/e box bottom line
    y_top = y0 + h              # tops of the three blocks = top line of a/c
    h_net = min((w * WIN) / ar_max,
                ((y_top - bar_y) * HIN - 0.21) / 3 - 0.21) / HIN
    for i, (name, polys, rel, cx, cy, hx, hy) in enumerate(samples):
        b_top = y_top - i * (h_net + lab_h)
        ax = fig.add_axes([x0, b_top - h_net, w, h_net])
        pc = PolyCollection(polys, array=rel, cmap=CMAP,
                            edgecolors='black', linewidths=0.8)
        pc.set_clim(*clim)
        ax.add_collection(pc)
        ax.set_xlim(cx - hx, cx + hx)
        ax.set_ylim(cy - hy, cy + hy)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        lab = (ALPHA_LAB[name].split(' (')[0])[1:-1]
        fig.text(x0 + w / 2, b_top - h_net - 0.5 * lab_h, f'$\\{{{lab}\\}}$',
                 fontsize=6.8, ha='center', va='center', color='#333333')
    fig.text(x0 - 0.008, y_top, 'c', fontsize=8, fontweight='bold',
             ha='right', va='top', color='#1a1a1a')
    # horizontal colorbar: bar spans the full column width (horizontally
    # aligned with the three patterns), bottom edge pinned to the d/e box
    # bottom line; "tile area / mean" centered below the bar
    cax = fig.add_axes([x0, bar_y, w, 0.11 / HIN])
    cb = fig.colorbar(ScalarMappable(Normalize(*clim), CMAP), cax=cax,
                      orientation='horizontal')
    cb.set_ticks([round(clim[0], 1), round(clim[1], 1)])
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.ax.tick_params(labelsize=5.2, width=0.5, length=1.2, pad=1.0)
    fig.text(x0 + w / 2, bar_y - 0.06 / HIN, 'tile area / mean',
             fontsize=6.0, ha='center', va='top', color='#333333')
    return clim          # color range shared with the tile-library panel
                         # (same ruler)


# ------------------------------------------------------------------ panel d
# TM Perron lambda(Ly) full table (after spectral correction; the arithmetic
# W=8 point is the 3.12.12 interlocking check)
TM_LAM = {
    'binary': {2: 0.89588, 3: 0.96657, 4: 1.24245, 5: 1.32452,
               6: 1.59197, 7: 1.73615, 8: 1.99572, 9: 2.17424},
    'nonarith': {2: 1.24245, 3: 1.38629, 4: 1.50444, 5: 1.58305,
                 6: 1.78945, 7: 1.89771},
    'arithmetic': {2: 1.60944, 3: 2.01215, 4: 2.50211, 5: 3.01488,
                   6: 3.54934, 7: 4.09316, 8: 4.643},
}
# mixed global model (s,c,A',q; binary has an extra parity pair of A)
MIXPAR = {'binary': (0.2094, 0.268, 0.320, 0.112, 0.572),
          'nonarith': (0.2240, 0.2209, 0.8696, 0.5743),
          'arithmetic': (0.5436, 0.289, 1.607, 0.270)}
# convention correction: the non-arithmetic class is a free fit; the
# binary/non-arithmetic s values are not resolved by the width series
# (fit band 0.17-0.27, difference smaller than the band), labels unified
# to ~0.2
S_END = {'binary': '', 'nonarith': '',
         'arithmetic': '0.544'}   # band replaces the two '0.2' labels


def panel_counts(fig, x0, y0, w, h):
    """Panel d (left column of the former combined c/d piece): resonant-size
    counts lnOmega vs Lx, long chains, three tightly stacked strips.
    x0/y0/w/h = figure fraction coordinates (the lambda panel was split off
    as panel_lambda).
    Resonant sizes (Lx = 0 mod Ly, filled) sit on the asymptotic dashed
    lines lnOmega = lambda(Ly)*Lx (TM Perron) -- the statistics converge to
    the spectral asymptote; for the arithmetic alphabet both branches merge
    onto the asymptote after a crossover of ~10-20 columns.
    The non-resonant branch (formerly open symbols + dotted line
    lnOmega = ln k*(Lx+Ly+1)) is no longer drawn -- its record and selection
    rule live in SI S6.3 + SI Fig. S5, while its bitwise exactness is still
    asserted by this script's QA block (data-integrity check, decoupled
    from the plot).
    Slot geometry/font sizes 1:1 with the standalone canonical
    fig2cd_symmetric_forms.py (tick 7/A_TAG 8/legend 7, ms 5.0, ~10 points
    per width after subsampling). Data: shape-class CSVs (Lx<=20) +
    TSDHC_long_trace{,_arith5}.csv (long chains Lx<=150; headerless,
    BOM-contaminated rows filtered by row[2].isdigit() and stripped of
    \ufeff)."""
    import csv
    lnk_of = {'binary': np.log(2), 'nonarith': np.log(3),
              'arithmetic': np.log(3)}

    # ---- data: shape-class CSVs (Lx<=20) + long-chain CSVs (Lx=21..150) ----
    long = {}
    for lf in (ROOT / 'TSDHC_long_trace.csv',
               ROOT / 'TSDHC_long_trace_arith5.csv'):
        with open(lf, encoding='utf-8-sig') as f:
            for row in csv.reader(f):
                if len(row) < 7 or not row[2].isdigit():
                    continue
                key = (row[0].lstrip('\ufeff'), int(row[2]))
                long.setdefault(key, ([], []))
                long[key][0].append(int(row[3]))
                long[key][1].append(float(row[5]))
    df3 = pd.read_csv(ROOT / PC.fname("P0_PM_torus_trace_shape_classes_Ly2Ly3.csv"))
    df5 = pd.read_csv(ROOT / PC.fname("P0_PM_torus_trace_shape_classes_Ly5.csv"))

    # ---- QA: non-resonant perimeter law bitwise exact (binary k=2 /
    #      nonarith k=3) ----
    for nm, k in (("binary", 2), ("nonarith", 3)):
        lnk_ = np.log(k)
        for ly, dd in ((2, df3), (3, df3), (5, df5)):
            for _, r in dd[(dd.alphabet == nm) & (dd.Ly == ly)].iterrows():
                if int(r.Lx) % ly:
                    assert abs(float(r.lnOmega)
                               - lnk_ * (int(r.Lx) + ly + 1)) < 1e-6, \
                        f"perimeter law broken: {nm} {int(r.Lx)}x{ly}"
            for xx, yy in zip(*long.get((nm, ly), ([], []))):
                if xx % ly:
                    assert abs(yy - lnk_ * (xx + ly + 1)) < 1e-6, \
                        f"perimeter law broken: {nm} {xx}x{ly}"
    print("[QA] closed form bitwise exact on non-resonant sizes (data "
          "check, decoupled from the plot; SI S6.3/S5) "
          "(binary k=2, nonarith k=3, Ly2+Ly3+Ly5, short+long)", flush=True)

    def series_split(gname, ly, target=10):
        """Subsample the resonant/non-resonant classes each at equal spacing
        (~target points per class; large-Ly resonant points are scarce, so
        the ratios are tuned separately to keep filled/open visually
        balanced; the QA uses the full data and is unaffected)."""
        d = df3[(df3.alphabet == gname) & (df3.Ly == ly)].sort_values('Lx')
        lx = d.Lx.values.astype(float)
        yy = d.lnOmega.values.astype(float)
        if ly == 5:
            d5 = df5[(df5.alphabet == gname)].sort_values('Lx')
            lx = np.concatenate([lx, d5.Lx.values.astype(float)])
            yy = np.concatenate([yy, d5.lnOmega.values.astype(float)])
        if (gname, ly) in long:
            l2, y2 = long[(gname, ly)]
            o = np.argsort(l2)
            lx = np.concatenate([lx, np.array(l2)[o]])
            yy = np.concatenate([yy, np.array(y2)[o]])
        o = np.argsort(lx)
        lx, yy = lx[o], yy[o]
        res = (lx % ly) == 0
        keep = []
        for m in (res, ~res):
            idx = np.where(m)[0]
            if idx.size:
                keep.append(idx[::max(1, len(idx) // target)])
        sel = np.sort(np.concatenate(keep))
        return lx[sel], yy[sel]

    # ---- layout: box lines on the column baselines, tick text/ylabel hang
    #      outside the boxes ----
    lx0, lxw = x0, w
    ay0, ayh = y0, h
    gh = ayh / 3
    ytop = ay0 + ayh
    spans = {'arithmetic': (0.0, 350.0), 'nonarith': (0.0, 225.0),
             'binary': (0.0, 215.0)}
    ytk = {'arithmetic': [0, 100, 200, 300], 'nonarith': [0, 100, 200],
           'binary': [0, 100, 200]}
    xtk = [0, 25, 50, 75, 100, 125, 150]
    xmax = 152
    axes_map = {}
    for gname in ("arithmetic", "nonarith", "binary"):
        ax = fig.add_axes([lx0, ytop - gh, lxw, gh])
        axes_map[gname] = ax
        ytop -= gh
        lnk = lnk_of[gname]
        for ly in (2, 3, 5):
            lx, yy = series_split(gname, ly)
            c, mk = W_COL[f'Ly{ly}'], W_MK[f'Ly{ly}']
            res = (lx % ly) == 0
            lam = TM_LAM[gname][ly]
            xr = np.array([0, xmax])
            ax.plot(xr, lam * xr, ls=(0, (5, 2)), color=c, lw=1.5,
                    zorder=2, alpha=0.85)
            # only resonant (entropy-carrying) sizes drawn; the non-resonant
            # branch is not plotted (its exact closed form and selection rule
            # are recorded in SI S6.3 + SI Fig. S5)
            ax.plot(lx[res], yy[res], mk, ms=5.0, mfc=c, mec=c, mew=0.0,
                    zorder=3, label=W_LAB[f'Ly{ly}']
                    if gname == "arithmetic" else None)
        lo, hi = spans[gname]
        ax.set_xlim(0, xmax)
        ax.set_ylim(lo, hi)
        ax.set_xticks(xtk)
        ax.set_yticks(ytk[gname])
        # "0" tick labels hidden -- adjacent strips collide 0/200
        # back-to-back at the shared boundary
        ax.set_yticklabels([''] + [f'{v}' for v in ytk[gname][1:]])
        ax.tick_params(labelsize=7, length=2.0, width=0.8, pad=1.8)
        if gname == "nonarith":
            # middle strip: x ticks drawn on the upper side only (upward)
            ax.tick_params(labelbottom=False, bottom=False, top=True,
                           direction="out", length=1.8)
        elif gname == "binary":
            ax.tick_params(bottom=True, labelbottom=True, top=True,
                           direction="out", length=1.8)
        else:
            ax.tick_params(labelbottom=False, bottom=False, top=False)
        ax.text(0.975, 0.08, A_TAG[gname], transform=ax.transAxes,
                fontsize=8, ha="right", va="bottom", color="#333333")
    axes_map["arithmetic"].legend(fontsize=6, handlelength=1.0, ncol=1,
                                  loc="upper left", borderaxespad=0.25,
                                  labelspacing=0.22, handletextpad=0.4,
                                  frameon=True, facecolor='white',
                                  edgecolor='none', framealpha=1.0)
    # fill/open protocol line removed (the non-resonant branch is not drawn;
    # its record lives in SI)
    axes_map["binary"].set_xlabel(r"$L_x$", fontsize=7.5, labelpad=1.2)
    fig.text(x0 - 0.049, ay0 + 0.44 * ayh, r"$\ln\Omega$",
             fontsize=7.5, rotation=90, ha="center", va="center")
    # panel letter: d (three counting strips; pattern c and lambda e live in
    # panel_b / panel_lambda)
    fig.text(x0 - 0.008, y0 + h - 0.002, "d", fontsize=8,
             fontweight="bold", ha="right", va="top", color="#1a1a1a")

def panel_lambda(fig, x0, y0, w, h):
    """Panel e: lambda/Ly vs Ly for all TM width points + the mixed global
    model s+c/Ly+A'*q^Ly (binary even/odd branches) + endpoint s labels
    (0.544; the other two classes unresolved, ~0.2).
    At the right-edge infinity tick position, direct planar open-boundary
    measurements are marked (open diamonds; binary 0.231 / arithmetic
    0.560; error bars = fit-window spread) -- cylinder extrapolation and
    planar direct measurement read side by side. x0/y0/w/h = figure
    fraction coordinates."""
    ax_d = fig.add_axes([x0, y0, w, h])
    # reviewer round: the two sub-law classes resolve only to a common
    # fit band; draw it (0.17-0.27) instead of hedging in the labels.
    ax_d.axhspan(0.17, 0.27, color="#9aa0a6", alpha=0.20, lw=0.0, zorder=0)
    ax_d.text(2.15, 0.219, "fit band 0.17" + chr(8211) + "0.27", fontsize=6.2,
              color="#555555", ha="left", va="center", zorder=6)
# the panel text "mixed global fit" was removed (stated in the caption;
# frees the lower right)
    xr = np.linspace(1.6, 10.4, 250)
    for gname in ("binary", "nonarith", "arithmetic"):
        c = ALPHA_C[gname]
        wss = sorted(TM_LAM[gname])
        ax_d.plot(wss, [TM_LAM[gname][w_] / w_ for w_ in wss], 'o',
                  ms=5.5, mfc=c, mec=c, mew=0.0, zorder=3,
                  label=("$\\{1,2\\}$" if gname == "binary" else
                         "$\\{1,2,3\\}$" if gname == "arithmetic" else
                         "$\\{1,2,4\\}$"))
        if gname == "binary":
            s, cc, Ae, Ao, q = MIXPAR[gname]
            for A, lsty in ((Ae, "-"), (Ao, (0, (3, 1.5)))):
                ax_d.plot(xr, s + cc / xr + A * q ** xr, linestyle=lsty,
                          color=c, lw=1.0, zorder=2, alpha=0.75)
            yend = s + cc / 10.4 + Ao * q ** 10.4
            if S_END[gname]:
                ax_d.text(10.35, yend - 0.022, S_END[gname], color=c,
                          fontsize=6.5, ha='right', va='top')
        else:
            s, cc, A, q = MIXPAR[gname]
            ax_d.plot(xr, s + cc / xr + A * q ** xr, '-', color=c,
                      lw=1.2, zorder=2, alpha=0.85)
            yend = s + cc / 10.4 + A * q ** 10.4
            if S_END[gname]:
                ax_d.text(10.35, yend + 0.018, S_END[gname], color=c,
                          fontsize=6.5, ha='right', va='bottom')
    # direct planar open-boundary measurement (at the infinity tick, open
    # diamonds)
    for gname, (aval, aerr) in (('binary', (0.231, 0.0005)),
                                ('arithmetic', (0.560, 0.004))):
        c = ALPHA_C[gname]
        ax_d.errorbar([10.9], [aval], yerr=[aerr], fmt='D', ms=5.0,
                      mfc='none', mec=c, mew=1.0, ecolor=c, elinewidth=0.8,
                      capsize=1.6, capthick=0.8, zorder=5)
    ax_d.set_xlim(1.6, 11.5)
    ax_d.set_ylim(0.1, 0.9)
    ax_d.set_xticks([2, 4, 6, 8, 10, 10.9])
    ax_d.set_xticklabels(['2', '4', '6', '8', '10', r'$\infty$'])
    ax_d.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax_d.set_xlabel("$L_y$", fontsize=7.5, labelpad=1.2)
    ax_d.set_ylabel(r"$\lambda/L_y$", fontsize=7.5, labelpad=3.5)
    ax_d.tick_params(labelsize=7, length=2.0, width=0.8, pad=1.8)
    from matplotlib.lines import Line2D
    _h, _l = ax_d.get_legend_handles_labels()
    _h = _h + [Line2D([], [], marker='D', mfc='none', mec='#555555',
                      mew=1.0, ms=5.0, ls='none')]
    _l = _l + ['planar disk']
    ax_d.legend(_h, _l, fontsize=7, handlelength=1.0, ncol=1,
                loc="upper right", borderaxespad=0.3, labelspacing=0.24,
                handletextpad=0.4)
    fig.text(x0 - 0.008, y0 + h - 0.002, "e", fontsize=8,
             fontweight="bold", ha="right", va="top", color="#1a1a1a")


def main():
    fig = plt.figure(figsize=(WIN, HIN))
    import matplotlib.patches as mpatches
    # ---- a counting construction (vertical: patch on top / tube|ring
    #      below; L-shaped arrow flow) ----
    ax_p = fig.add_axes([fx(CX[0]), fy(RY1 + 1.50), fx(COLW[0]), fy(1.20)])
    ax_t = fig.add_axes([fx(CX[0]), fy(RY1), fx(1.35), fy(1.20)],
                        projection='3d', computed_zorder=False)
    ax_r = fig.add_axes([fx(CX[0] + COLW[0] - 1.10), fy(RY1), fx(1.10),
                         fy(1.20)],
                        projection='3d', computed_zorder=False)
    s9.panel_a(ax_p)
    pw, ph = fit_axes(ax_p, CX[0] + 0.25, RY1 + ROWH1, COLW[0] - 0.25, 1.25,
                      anchor='top', halign='left')   # shifted right 0.25
                                                     # (visual weight left)
    p_bot = RY1 + ROWH1 - ph
    s9.panel_b(ax_t)
    s9.panel_c(ax_r)
    fig.text(fx(CX[0]) - 0.008, fy(RY1 + ROWH1), 'a', fontsize=8,
             fontweight='bold', ha='right', va='top', color='#1a1a1a')
    # patch->tube vertical arrow + two glue label lines (label block
    # vertically aligned with the arrow: +/-0.075 in about the arrow
    # midpoint)
    ya0, ya1 = p_bot - 0.08, RY1 + 1.23
    fig.add_artist(mpatches.FancyArrowPatch(
        (fx(CX[0] + 0.46), fy(ya0)), (fx(CX[0] + 0.46), fy(ya1)),
        transform=fig.transFigure, arrowstyle='-|>', mutation_scale=9,
        lw=1.1, color='#555555'))
    ymid = 0.5 * (ya0 + ya1)
    fig.text(fx(CX[0] + 0.66), fy(ymid + 0.075), "roll up:", fontsize=6.8,
             ha='left', va='center', color='#333333')
    fig.text(fx(CX[0] + 0.66), fy(ymid - 0.075),
             "glue $A$" + chr(8211) + "$A'", fontsize=6.8,
             ha='left', va='center', color='#333333')
    # tube->ring horizontal arrow + two glue label lines (centered in the
    # gap band right above the arrow)
    fig.add_artist(mpatches.FancyArrowPatch(
        (fx(CX[0] + 1.30), fy(RY1 + 0.62)), (fx(CX[0] + 1.50), fy(RY1 + 0.62)),
        transform=fig.transFigure, arrowstyle='-|>', mutation_scale=9,
        lw=1.1, color='#555555'))
    fig.text(fx(CX[0] + 1.40), fy(RY1 + 0.48), "close:", fontsize=6.8,
             ha='center', va='center', color='#333333')
    fig.text(fx(CX[0] + 1.40), fy(RY1 + 0.33),
             "glue $B$" + chr(8211) + "$B'", fontsize=6.8,
             ha='center', va='center', color='#333333')
    # ---- c stacked (column 2, spans two rows; drawn first to set the
    #      area/mean color range) / b tile library (column 1, row 2) ----
    clim_am = panel_b(fig, fx(CX[1]), fy(TALL0), fx(COLW[1]), fy(TALLH))
    panel_a(fig, fx(CX[0]), fy(RY2 + 0.42), fx(COLW[0]), fy(2.45), clim_am)
    # ---- d counting strips (column 3, spans two rows; x-label band at the
    #      bottom) / e lambda(Ly) (column 4, spans two rows) ----
    panel_counts(fig, fx(CX[2]), fy(TALL0 + 0.42), fx(COLW[2]),
                 fy(TALLH - 0.42))
    panel_lambda(fig, fx(CX[3]), fy(TALL0 + 0.42), fx(COLW[3]),
                 fy(TALLH - 0.42))
    scale_fig(fig, K)
    savefig(fig, str(FIGS / "fig3_honeycomb_metric_entropy"))
    # QA spot checks: 3x3 anchors (trace-method CSV) + cross-validation
    # against the Fig 1c extension CSV + exact non-resonant straight line
    # (semantic self-check of the zigzag structure)
    df = pd.read_csv(ROOT / PC.fname("P0_PM_torus_trace_shape_classes_Ly2Ly3.csv"))
    a33 = df[(df.Lx == 3) & (df.Ly == 3)][['alphabet', 'Omega']]
    print('[QA] 3x3 anchors:', a33.to_dict('records'))
    t = pd.read_csv(ROOT / PC.fname("P0_PM_torus_trace_extension_Ly3.csv"))
    m = t.merge(df[df.Ly == 3], on=['alphabet', 'Lx', 'Ly'],
                suffixes=('_fig1c', '_new'))
    print(f'[QA] overlap with fig1c trace CSV: {len(m)} rows, '
          f'all equal: {(m.Omega_fig1c == m.Omega_new).all()}')
    d = df[(df.alphabet == 'binary') & (df.Ly == 3) & (df.Lx % 3 != 0)]
    print(f'[QA] binary off-resonance == 2^(Lx+4): '
          f'{all(int(r.Omega) == 2 ** (r.Lx + 4) for _, r in d.iterrows())}')
    d = df[(df.alphabet == 'nonarith') & (df.Ly == 3) & (df.Lx % 3 != 0)]
    print(f'[QA] nonarith off-resonance == 3^(Lx+4): '
          f'{all(int(r.Omega) == 3 ** (r.Lx + 4) for _, r in d.iterrows())}')
    d = df[(df.alphabet == 'arithmetic') & (df.Lx == 20) & (df.Ly == 3)]
    print(f'[QA] arith 20x3: {int(d.Omega.iloc[0])} '
          f'(lnOm/F={float(d.lnOmega_per_F.iloc[0]):.4f})')


if __name__ == "__main__":
    main()
