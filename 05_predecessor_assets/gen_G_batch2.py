# -*- coding: utf-8 -*-
"""Fig 1a asset upgrade (G assets, batch 2):
G5b  large honeycomb metric disorder (n_rings=7, 127 holes, matching the
     scale of the lozenge circular window next to it); faces colored by
     hole area (viridis; area = side-length data), edges thin dark grey;
     area is set by the equiangular-hexagon side lengths: color = length =
     area in a single encoding.
Five dictionary thumbnails (thumbs/): periodic / quasicrystal / random
     tiling / amorphous rendered independently at small scale (fewer
     details, readable when shrunk) + metric (honeycomb n=3 solution,
     faces colored).
Outputs: G5b_honeycomb_faces.png/.pdf and thumbs/*.png written next to
     this script.
"""
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                              # release repo root
sys.path.insert(0, str(HERE))                   # gen_tilings.py lives here
sys.path.insert(0, str(REPO / "06_engines" / "z3_hexagonal_P0"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

import pseudocrystal_lib as pl
from gen_tilings import (periodic_honeycomb, penrose_edges, random_lozenge,
                         voronoi_network)


def _hole_polygons(sol, lib):
    """Rebuild per-hole vertex coordinates with the same BFS walk as
    pseudocrystal_lib.extract_geometry (the lib does not return hole_verts,
    so the walk logic is replicated here; lib node keys are coordinate
    tuples rounded to 4 decimals)."""
    import queue as _q
    data = {c: lib.weights[idx] for c, idx in sol.items()}
    pos = {(0, 0): (0.0, 0.0)}
    hole_verts = {}
    bfs = _q.Queue()
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
                ax = ay = 0.0
                for step in range(k + 1):
                    if step == k:
                        ax_n, ay_n = ax, ay
                    ax += float(n_w[step]) * math.cos(math.radians(step * 60))
                    ay += float(n_w[step]) * math.sin(math.radians(step * 60))
                dest = verts[i + 1]
                pos[nb] = (dest[0] - ax_n, dest[1] - ay_n)
                seen.add(nb)
                bfs.put(nb)
    return hole_verts


def polygon_area(pts):
    x, y = zip(*pts)
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i]
                         for i in range(-1, len(x) - 1)))


def gen_G5b(n_rings=7, seed=7004):
    lib = pl.build_library(('222', '223', '233', '333'))
    sol, _ = pl.solve_tessellation(n_rings, lib, periodic=False,
                                   rng=random.Random(seed))
    hole_verts = _hole_polygons(sol, lib)
    polys = [v[1:] for v in hole_verts.values()]      # drop duplicated first point
    areas = np.array([polygon_area(p) for p in polys])
    anorm = areas / areas.mean()    # normalize by mean area (colorbar scale)
    print(f"  G5b: {len(polys)} holes, area/mean {anorm.min():.3f}-"
          f"{anorm.max():.3f} (mean {areas.mean():.3f})", flush=True)
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    pc = PolyCollection(polys, array=anorm, cmap=cmap,
                        edgecolors='white', linewidths=0.7)
    pc.set_clim(anorm.min(), anorm.max())
    ax.add_collection(pc)
    ax.set_aspect('equal')
    allp = np.concatenate(polys, axis=0)
    cx, cy = (allp[:, 0].min() + allp[:, 0].max()) / 2, \
             (allp[:, 1].min() + allp[:, 1].max()) / 2
    half = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) / 2 + 0.5
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(HERE / "G5b_honeycomb_faces.png", dpi=300)
    fig.savefig(HERE / "G5b_honeycomb_faces.pdf")
    plt.close(fig)
    print("  G5b saved", flush=True)


def _render(ax, segs, R):
    for x1, y1, x2, y2 in segs:
        ax.plot([x1, x2], [y1, y2], color='#3a3a3a', lw=1.1)
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def gen_thumbs():
    out = HERE / "thumbs"
    out.mkdir(exist_ok=True)
    R = 7.0
    # 1 periodic honeycomb (small patch)
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    _render(ax, periodic_honeycomb(radius=R), R)
    fig.savefig(out / "periodic.png", dpi=300)
    plt.close(fig)
    # 2 Penrose (two fewer substitution generations)
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    _render(ax, penrose_edges(radius=R, n_gen=5), R)
    fig.savefig(out / "quasicrystal.png", dpi=300)
    plt.close(fig)
    # 3 random lozenge (small window)
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    segs = random_lozenge(radius=R, Nu=26, Nv=26, seed=1, sweeps=60)
    _render(ax, segs, R)
    fig.savefig(out / "random_tiling.png", dpi=300)
    plt.close(fig)
    # 4 Voronoi (sparse)
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    _render(ax, voronoi_network(radius=R, seed=23), R)
    fig.savefig(out / "amorphous.png", dpi=300)
    plt.close(fig)
    # 5 metric disorder: honeycomb n=3 solution (colored faces + grey edges)
    lib = pl.build_library(('222', '223', '233', '333'))
    sol, _ = pl.solve_tessellation(3, lib, periodic=False,
                                   rng=random.Random(7004))
    hole_verts = _hole_polygons(sol, lib)
    polys = [v[1:] for v in hole_verts.values()]
    areas = np.array([polygon_area(p) for p in polys])
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    pc = PolyCollection(polys, array=areas, cmap=cmap,
                        edgecolors='white', linewidths=1.0)
    pc.set_clim(areas.min(), areas.max())
    ax.add_collection(pc)
    ax.set_aspect('equal')
    allp = np.concatenate(polys, axis=0)
    cx, cy = (allp[:, 0].min() + allp[:, 0].max()) / 2, \
             (allp[:, 1].min() + allp[:, 1].max()) / 2
    half = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) / 2 + 0.4
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out / "metric.png", dpi=300)
    plt.close(fig)
    print("  thumbs saved (5)", flush=True)


if __name__ == "__main__":
    print("= G assets batch 2 =", flush=True)
    gen_G5b()
    gen_thumbs()
    print("done", flush=True)
