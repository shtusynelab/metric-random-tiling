# -*- coding: utf-8 -*-
"""P0_PM G asset generation (Fig 1 assets).

G1  lozenge tiling, three orientations as three face colors (classic
    random tiling; reuses the LozengeTiling class from gen_tilings.py
    collected into this directory: random perfect matching + flip MC,
    same seeds/parameters);
G2  square-grid three-value edge-color disorder pattern (analytic
    construction: the z4_square solution structure = horizontal edges
    constant per column, vertical edges constant per row; one sample of
    Omega=k^(Lx+Ly));
G5  honeycomb three-value edge-color disorder pattern (regenerated with
    the pseudocrystal_lib quaternary alphabet; de-MOF styling: three-color
    edge network, no vertex spheres, no chemistry labels).

Outputs: G1_lozenge_color.png / G2_square_edges.png /
     G5_honeycomb_edges.png written next to this script (300 dpi, English,
     no panel titles).
"""
import math
import pickle
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                           # release repo root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
import numpy as np

sys.path.insert(0, str(HERE))                # gen_tilings.py lives here
sys.path.insert(0, str(REPO / "06_engines" / "z3_hexagonal_P0"))

EDGE_COLORS = {4: "#2166ac", 5: "#4d9a4d", 6: "#c0392b"}   # blue/green/red = 4/5/6
TRI_COLORS = ["#66c2a5", "#8da0cb", "#fc8d62"]             # three lozenge orientations


def gen_G1(radius=12.0, Nu=48, Nv=48, sweeps=300):
    """Lozenge three-color faces (circular window).

    Note: flips conserve orientation counts, so orientation uniformity can
    only come from the initial matching: a seed scan picks the random
    perfect matching with the flattest orientation counts, and the MC run
    only decorrelates configurations.
    """
    from gen_tilings import LozengeTiling
    best = None
    for seed in range(1, 61):
        t = LozengeTiling(Nu, Nv, seed=seed)
        t.random_perfect_matching()
        st = t.orientation_stats()
        score = max(st) - min(st)
        if best is None or score < best[0]:
            best = (score, seed, st)
    score, seed, st = best
    print(f"  G1 best seed={seed} init stats={st} (spread={score})", flush=True)
    t = LozengeTiling(Nu, Nv, seed=seed)
    t.random_perfect_matching()
    t.run_mc(sweeps)
    print(f"  G1 after MC:   {t.orientation_stats()}", flush=True)
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.5, math.sqrt(3) / 2])

    def T(i, j):
        return i * e1 + j * e2

    center = np.array([Nu / 2 + 0.25, Nv / 2 * math.sqrt(3) / 2 + 0.25])
    R2 = radius * radius
    polys, colors = [], []
    dmap = {(0, 0): 0, (Nu - 1, 0): 1, (0, Nv - 1): 2}
    for (i, j, k), m in t.match.items():
        if k != 0:
            continue
        di, dj = (m[0] - i) % Nu, (m[1] - j) % Nv
        ori = dmap[(di, dj)]
        if ori == 0:
            ring = [T(i, j), T(i + 1, j), T(i + 1, j + 1), T(i, j + 1)]
        elif ori == 1:
            ring = [T(i, j), T(i + 1, j), T(i, j + 1), T(i - 1, j + 1)]
        else:
            ring = [T(i, j), T(i, j + 1), T(i + 1, j), T(i + 1, j - 1)]
        pts = [p - center for p in ring]
        if all(float(p @ p) <= R2 for p in pts):
            polys.append(pts)
            colors.append(TRI_COLORS[ori])
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    pc = PolyCollection(polys, facecolors=colors, edgecolors="white",
                        linewidths=0.5)
    ax.add_collection(pc)
    ax.set_aspect("equal")
    lim = radius * 0.98
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(HERE / "G1_lozenge_color.png", dpi=300)
    fig.savefig(HERE / "G1_lozenge_color.pdf")
    plt.close(fig)
    print(f"  G1 saved: {len(polys)} lozenges", flush=True)


def gen_G2(m=8, n=12, seed=7):
    """Square-grid continuous-alphabet version: the row/column-constant
    solution structure is kept, edge lengths take continuous values
    (independent continuous samples per row/column), colored by
    edge-length/mean with viridis; cell width = horizontal edge length,
    cell height = vertical edge length (color = length = area in one
    encoding)."""
    rng = random.Random(seed)
    col_vals = [rng.uniform(0.65, 1.35) for _ in range(m)]     # column -> horizontal edge
    row_vals = [rng.uniform(0.65, 1.35) for _ in range(n)]     # row -> vertical edge
    xs = [0.0]
    for i in range(m):
        xs.append(xs[-1] + col_vals[i])
    ys = [0.0]
    for j in range(n):
        ys.append(ys[-1] + row_vals[j])
    segs, cols = [], []
    cmap = plt.get_cmap("viridis")

    def vcol(v):
        return cmap((v - 0.65) / 0.70)

    for j in range(n + 1):
        for i in range(m):
            segs.append([(xs[i], ys[j]), (xs[i + 1], ys[j])])
            cols.append(vcol(col_vals[i]))
    for i in range(m + 1):
        for j in range(n):
            segs.append([(xs[i], ys[j]), (xs[i], ys[j + 1])])
            cols.append(vcol(row_vals[j]))
    fig, ax = plt.subplots(figsize=(3.2, 3.2 * (ys[-1] / xs[-1])))
    lc = LineCollection(segs, colors=cols, linewidths=3.8,
                        capstyle="round")
    ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, xs[-1] + 0.35)
    ax.set_ylim(-0.35, ys[-1] + 0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(HERE / "G2_square_edges.png", dpi=300)
    fig.savefig(HERE / "G2_square_edges.pdf")
    plt.close(fig)
    print(f"  G2 saved: {m}x{n} continuous viridis (portrait)", flush=True)


def gen_G5(n_rings=4, seed=7004):
    """Honeycomb three-value edge colors (quaternary alphabet; de-MOF
    styling: no vertex spheres, no chemistry labels)."""
    import pseudocrystal_lib as pl
    lib = pl.build_library(("222", "223", "233", "333"))
    sol, _ = pl.solve_tessellation(n_rings, lib, periodic=False,
                                   rng=random.Random(seed))
    g = pl.extract_geometry(sol, lib)
    pts = g["metal_nodes"]
    key2idx = {k: i for i, k in enumerate(g["node_keys"])}
    segs, cols = [], []
    for e, w in zip(g["edges"], g["edge_weights"]):
        a, b = pts[key2idx[e[0]]], pts[key2idx[e[1]]]
        segs.append([(a[0], a[1]), (b[0], b[1])])
        cols.append(EDGE_COLORS[w])
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    lc = LineCollection(segs, colors=cols, linewidths=3.4, capstyle="round")
    ax.add_collection(lc)
    ax.set_aspect("equal")
    xs, ys = pts[:, 0], pts[:, 1]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + 1.5
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(HERE / "G5_honeycomb_edges.png", dpi=300)
    fig.savefig(HERE / "G5_honeycomb_edges.pdf")
    plt.close(fig)
    # motif spot check (stdout record): are all three edge colors present
    from collections import Counter
    wcount = Counter(int(w) for w in g["edge_weights"])
    print(f"  G5 saved: {len(segs)} edges, weight counts={dict(wcount)}",
          flush=True)


if __name__ == "__main__":
    print(f"= P0_PM G assets {matplotlib.get_backend()} =", flush=True)
    gen_G1()
    gen_G2()
    gen_G5()
    print("G assets done", flush=True)
