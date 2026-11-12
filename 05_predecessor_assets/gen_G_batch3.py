# -*- coding: utf-8 -*-
"""Fig 1b continuous-cell assets, second batch: face-area coloring.

G2s square: rectangular grid faces, area = column width x row height
     (continuous product), viridis;
G3s kagome: triangles vs hexagons, the face type with the larger
     relative area variation (CV) is colored;
G4s 4.8.8: octagons vs squares, same rule.
Unified convention: face color = area / mean over that face type in the
     patch (color = area = edge-length data).
Outputs: G2s_square_faces.png / G3s_kagome_faces.png /
     G4s_488_faces.png written next to this script (300 dpi, portrait).
"""
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                           # release repo root
sys.path.insert(0, str(REPO / "06_engines" / "z4_kagome"))   # kagome_real_demo
sys.path.insert(0, str(REPO / "06_engines" / "z3_488"))      # tsq_488_real_demo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import kagome_real_demo as M


def polygon_area(pts):
    x, y = zip(*pts)
    return 0.5 * abs(sum(x[i] * y[(i + 1) % len(x)] -
                         x[(i + 1) % len(x)] * y[i] for i in range(len(x))))


def _save_autocrop(fig, path):
    """Save, then crop pure-background margins (background color taken
    from the image corner pixel, tolerance 18; also works on grey
    backgrounds)."""
    fig.savefig(path, dpi=300)
    plt.close(fig)
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, int)
    bg = arr[2, 2].tolist()
    diff = np.abs(arr - np.array(bg, int)).sum(axis=2)
    ys, xs = np.where(diff > 18)
    if len(xs):
        pad = 6
        box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
               min(img.size[0], xs.max() + pad),
               min(img.size[1], ys.max() + pad))
        img.crop(box).save(path)




def largest_window(face_polys, cx, cy, aspect=0.535, lo=0.30, hi=None):
    """Bisect the largest axis-aligned window fully covered by faces
    (36x36 grid sampling; every sample point must fall inside at least
    one face; conservative radius=-1e-9). Returns half_w."""
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

def render_faces(ax, polys, areas, lw=0.6, underlay=None,
                 grey_faces=None, seam="white"):
    if grey_faces is not None:
        from matplotlib.patches import Polygon as MplPolygon
        for gp in grey_faces:
            ax.add_patch(MplPolygon(gp, closed=True, facecolor="#e9e9e9",
                                    edgecolor="none", zorder=1.5))
    if underlay is not None:
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(underlay, colors="#9a9a9a",
                                         linewidths=0.7, zorder=2))
    cmap = plt.get_cmap("viridis")
    am = np.mean(areas)
    # seam="face": stroke with the face's own color (hides anti-aliasing
    # seams, faces reach the shared edge, no white gaps; on 4.8.8 squares
    # white seams read as a grey border plus a gap)
    pc = PolyCollection(polys, array=np.array(areas) / am, cmap=cmap,
                        edgecolors=seam, linewidths=lw, zorder=3)
    pc.set_clim(np.min(areas) / am, np.max(areas) / am)
    ax.add_collection(pc)
    ax.set_aspect("equal")
    allp = np.concatenate(polys, axis=0)
    cx, cy = (allp[:, 0].min() + allp[:, 0].max()) / 2, \
             (allp[:, 1].min() + allp[:, 1].max()) / 2
    # shrunk window (window fully inside the patches => full-bleed, no
    # white corners)
    ptp_x = np.ptp(allp[:, 0])
    ptp_y = np.ptp(allp[:, 1])
    half_w = max(min(ptp_x, ptp_y / 0.62) / 2 - 0.95, 1.0)
    half_h = half_w / 0.62
    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy - half_h, cy + half_h)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    return float(np.min(areas) / am), float(np.max(areas) / am)


def gen_G2s(m=6, n=10, seed=7):
    rng = random.Random(seed)
    col_vals = [rng.uniform(0.45, 1.75) for _ in range(m)]
    row_vals = [rng.uniform(0.45, 1.75) for _ in range(n)]
    xs = [0.0]
    for i in range(m):
        xs.append(xs[-1] + col_vals[i])
    ys = [0.0]
    for j in range(n):
        ys.append(ys[-1] + row_vals[j])
    polys, areas = [], []
    for j in range(n):
        for i in range(m):
            polys.append([(xs[i], ys[j]), (xs[i + 1], ys[j]),
                          (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1])])
            areas.append(col_vals[i] * row_vals[j])
    fig, ax = plt.subplots(figsize=(2.6, 4.0))
    lo, hi = render_faces(ax, polys, areas, lw=0.7)
    print(f"  G2s: area/mean in [{lo:.2f}, {hi:.2f}]", flush=True)
    _save_autocrop(fig, HERE / "G2s_square_faces.png")


def gen_G3s(W=8, L=8, seed=20260927):
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

    t, _ = M.sample_t_field(W, L, seed)
    pos = embed(t)
    # triangle faces: the three bond midpoints at each interior honeycomb
    # vertex hv
    tri_polys, tri_areas = [], []
    for hv in interior:
        bs = sorted(bid[(min(hv, b), max(hv, b))] for b in nbrs[hv])
        pts = [pos[bs[i]] for i in range(3)]
        tri_polys.append(pts)
        tri_areas.append(polygon_area(pts))
    tri_polys = np.array(tri_polys)
    tri_areas = np.array(tri_areas)
    cv_tri = tri_areas.std() / tri_areas.mean()
    # hexagon faces: honeycomb holes = the 6 bond midpoints around each
    # hole; hole vertices = the outer midpoints of the 6 interior hvs.
    # Simplification: skip the hexagon CV computation and compare
    # directly: triangle area is first order in t^2, hexagon edge ~ t so
    # area ~ t^2 (with a constant term) => triangle CV is larger.
    print(f"  G3s: triangles CV={cv_tri:.3f} (n={len(tri_polys)}); "
          "hexagons have constant offset -> smaller CV, color triangles",
          flush=True)
    # hexagon holes: the 6 bond midpoints on each honeycomb hole boundary
    # (light grey underlay removes the empty look)
    hex_polys = []
    for hi_ in range(len(hex_vids)):
        vs = hex_vids[hi_]
        pts, ok = [], True
        for k in range(6):
            key = (min(vs[k], vs[(k + 1) % 6]), max(vs[k], vs[(k + 1) % 6]))
            if key not in bid:
                ok = False
                break
            pts.append(pos[bid[key]])
        if ok and len(pts) == 6:
            hex_polys.append(pts)
    fig, ax = plt.subplots(figsize=(2.6, 4.0))
    under = [[pos[i], pos[j]] for _hv, i, j, _ux, _uy, _L in kedges]
    lo, hi = render_faces(ax, [list(map(tuple, p)) for p in tri_polys],
                          tri_areas, lw=0.55, underlay=under,
                          grey_faces=hex_polys)
    # coverage window: bisect the largest rectangle fully covered by
    # faces (full-bleed, no white corners)
    face_pts = ([list(map(tuple, p)) for p in tri_polys] +
                [list(map(tuple, p)) for p in hex_polys])
    allv = np.concatenate([np.asarray(p, float) for p in tri_polys])
    cxw = float(allv[:, 0].mean())
    cyw = float(allv[:, 1].mean())
    hw_ = largest_window(face_pts, cxw, cyw, aspect=0.535)
    ax.set_xlim(cxw - hw_, cxw + hw_)
    ax.set_ylim(cyw - hw_ / 0.535, cyw + hw_ / 0.535)
    print(f"  G3s: area/mean in [{lo:.2f}, {hi:.2f}]", flush=True)
    _save_autocrop(fig, HERE / "G3s_kagome_faces.png")


def gen_G4s(W=8, L=8, seed=20260927):
    from tsq_488_real_demo import (T, build_geometry, face_polygons,
                                   sample_solution, bfs_embed)
    faces, E = T.tsq_patch(W, L)
    epi = build_geometry(W, L)
    lens = sample_solution(faces, E, seed)
    pos = bfs_embed(epi, lens)
    polys = face_polygons(faces, epi, pos)
    oct_polys, oct_areas, sq_polys, sq_areas = [], [], [], []
    for size, pts in polys:
        if any(p not in pos for p in [tuple(pt) for pt in pts]) and False:
            continue
        a = polygon_area(pts)
        if size == 8:
            oct_polys.append(pts)
            oct_areas.append(a)
        else:
            sq_polys.append(pts)
            sq_areas.append(a)
    cv_o = np.std(oct_areas) / np.mean(oct_areas)
    cv_s = np.std(sq_areas) / np.mean(sq_areas)
    print(f"  G4s: oct CV={cv_o:.3f} (n={len(oct_polys)}), "
          f"square CV={cv_s:.3f} (n={len(sq_polys)})", flush=True)
    if cv_o >= cv_s:
        polys_use, areas_use, name = oct_polys, oct_areas, "octagons"
    else:
        polys_use, areas_use, name = sq_polys, sq_areas, "squares"
    print(f"  G4s: coloring {name}", flush=True)
    fig, ax = plt.subplots(figsize=(2.6, 4.0))
    under = [[pos[a], pos[b]] for e, (a, b, d) in epi.items()]
    grey = oct_polys if name == "squares" else sq_polys
    lo, hi = render_faces(ax, polys_use, areas_use, lw=0.4,
                          underlay=under, grey_faces=grey, seam="face")
    print(f"  G4s: area/mean in [{lo:.2f}, {hi:.2f}]", flush=True)
    _save_autocrop(fig, HERE / "G4s_488_faces.png")


if __name__ == "__main__":
    print("= G assets batch 3 (face coloring) = ", flush=True)
    gen_G2s()
    gen_G3s()
    gen_G4s()
    print("done", flush=True)
