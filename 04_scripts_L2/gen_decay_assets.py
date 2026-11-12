# -*- coding: utf-8 -*-
"""gen_decay_assets.py - Fig 2 four-tile pattern inset assets (Fig 1
area-coloring language).
Language (consistent with Fig 1 c-e): the target face type is colored by
face area / panel mean on viridis; non-target face types are filled
#eeeeee; all edges are thick black lines (linewidth calibrated to the inset
display scale: asset 2.6-4.0 in -> inset ~1.4 in).
Solution data sources and parameters carry over from the earlier
face-colored version and the current Fig 2 edge-colored version
(gen_G_pieces / gen_K3); the pattern geometry is unchanged, only the
rendering language:
  sq_faces    square: row/column constant solutions (continuous alphabet
              0.65-1.35, seed 7), all faces = target class;
  kagome_faces kagome: hexagon holes colored normalized by their own area,
              triangles grey;
  q488_faces  4.8.8: octagons colored normalized by their own area, squares
              grey;
  elt_faces   3.3.3.4.4: free square-row implementation (triangle rows
              width-locked), squares colored, triangles grey; constructor
              z4_3464/RTH_numerical_verification.build_33344 (32/32
              regression); torus patch sampling uses the same method and
              seed (20261003) as the earlier metric-demo assets.
Output: <FIGURES>/assets_decay/{sq,kagome,q488,elt}_faces.png (300 dpi)"""
import math
import random
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

OUT = PC.FIGURES / "assets_decay"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PC.engine("z4_kagome")))
sys.path.insert(0, str(PC.engine("z3_488")))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
import importlib  # noqa: E402
import kagome_real_demo as M  # noqa: E402

LW = 0.9            # black edge linewidth (design pt; ~0.4-0.6 pt after scaling into the inset, matching the Fig 1 visual weight)
GREY = "#eeeeee"


def polygon_area(pts):
    x, y = zip(*pts)
    return 0.5 * abs(sum(x[i] * y[(i + 1) % len(x)] -
                         x[(i + 1) % len(x)] * y[i] for i in range(len(x))))


def save_autocrop(fig, path):
    """Save and crop away the plain background margin (from the L1
    gen_G_batch3 helper)."""
    fig.savefig(path, dpi=300)
    plt.close(fig)
    from PIL import Image
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, int)
    bg = arr[2, 2].tolist()
    diff = np.abs(arr - np.array(bg, int)).sum(axis=2)
    ys, xs = np.where(diff > 18)
    if len(xs):
        pad = 6
        img.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                  min(img.size[0], xs.max() + pad),
                  min(img.size[1], ys.max() + pad))).save(path)


def hex_envelope(centroid, hole_pts, face_polys):
    """Grow the central hole into an envelope hexagon: bisect for the largest
    k such that the 6 vertices + edge sample points all lie inside the patch
    faces (point-in-polygon test, matplotlib.path.Path)."""
    from matplotlib.path import Path as MplPath
    paths = [MplPath(np.asarray(f, float)) for f in face_polys if len(f) >= 3]
    cx, cy = centroid
    v0 = np.asarray(hole_pts, float) - [cx, cy]

    def inside(k):
        pts = []
        for j in range(6):                      # sample along polygon edges (vertex order, not angular order)
            for f_ in np.linspace(0, 1, 13, endpoint=False):
                p = v0[j] * (1 - f_) + v0[(j + 1) % 6] * f_
                pts.append((cx + k * p[0], cy + k * p[1]))
        pts = np.array(pts)
        ok = np.zeros(len(pts), bool)
        for pa in paths:
            ok |= pa.contains_points(pts, radius=-1e-9)
        return bool(ok.all())

    a, b = 0.5, 12.0
    for _ in range(28):
        mid = 0.5 * (a + b)
        if inside(mid):
            a = mid
        else:
            b = mid
    return a * v0 + [cx, cy]


def save_autocrop_after_mask(path):
    """Crop tightly to non-background content after masking (variant of the
    L1 save_autocrop)."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, int)
    bg = arr[2, 2].tolist()
    diff = np.abs(arr - np.array(bg, int)).sum(axis=2)
    ys, xs = np.where(diff > 18)
    if len(xs):
        pad = 6
        img.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                  min(img.size[0], xs.max() + pad),
                  min(img.size[1], ys.max() + pad))).save(path)


def render_fig1(ax, polys, areas, grey=None, win=None, aspect=0.62):
    """Fig 1 language render: grey faces (first) -> target faces on viridis
    (area/mean) + all-black thick edges.
    win=(cx, cy, half_w) overrides the window; None = automatic from the
    face union (0.95 shrink margin)."""
    if grey:
        for gp in grey:
            ax.add_patch(MplPolygon(gp, closed=True, facecolor=GREY,
                                    edgecolor="black", lw=LW, zorder=2))
    am = float(np.mean(areas))
    pc = PolyCollection(polys, array=np.array(areas) / am,
                        cmap=plt.get_cmap("viridis"),
                        edgecolors="black", linewidths=LW, zorder=3)
    vmin, vmax = float(np.min(areas)) / am, float(np.max(areas)) / am
    if vmax - vmin < 1e-9:          # identical areas (rigid): value 1.0 maps to the colour-bar midpoint
        vmin, vmax = 0.5, 1.5
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_aspect("equal")
    ax.axis("off")
    allp = np.concatenate([np.asarray(p, float) for p in polys], axis=0)
    if win is None:
        cx, cy = (allp[:, 0].min() + allp[:, 0].max()) / 2, \
                 (allp[:, 1].min() + allp[:, 1].max()) / 2
        ptp_x, ptp_y = np.ptp(allp[:, 0]), np.ptp(allp[:, 1])
        half_w = max(min(ptp_x, ptp_y / aspect) / 2 - 0.95, 1.0)
    else:
        cx, cy, half_w = win
    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy - half_w / aspect, cy + half_w / aspect)
    return float(np.min(areas)) / am, float(np.max(areas)) / am


def _fig():
    fig, ax = plt.subplots(figsize=(2.6, 4.0))
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


# ---------------- a square: row/column constants (continuous alphabet) ----------------
def gen_sq(m=8, n=12, seed=7):
    rng = random.Random(seed)
    col_vals = [rng.uniform(0.65, 1.35) for _ in range(m)]
    row_vals = [rng.uniform(0.65, 1.35) for _ in range(n)]
    xs = [0.0]
    for v in col_vals:
        xs.append(xs[-1] + v)
    ys = [0.0]
    for v in row_vals:
        ys.append(ys[-1] + v)
    polys, areas = [], []
    for j in range(n):
        for i in range(m):
            polys.append([(xs[i], ys[j]), (xs[i + 1], ys[j]),
                          (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1])])
            areas.append(col_vals[i] * row_vals[j])
    fig, ax = _fig()
    cx = (xs[0] + xs[-1]) / 2
    cy = (ys[0] + ys[-1]) / 2
    lo, hi = render_fig1(ax, polys, areas, win=(cx, cy, 0.62 * (ys[-1] - ys[0]) / 2 * 0.98))
    print(f"  sq_faces: area/mean [{lo:.2f},{hi:.2f}]")
    save_autocrop(fig, OUT / "sq_faces.png")


# ---------------- b kagome: triangle variables (same lineage as the face-colored version) ----------------
def gen_kagome(W=20, L=20, seed=20260927, rings=3):
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
            mid_ref.setdefault(key, ((v2real[a][0] + v2real[b][0]) / 2,
                                     (v2real[a][1] + v2real[b][1]) / 2))
    bonds = sorted(mid_ref)
    bid = {bd: i for i, bd in enumerate(bonds)}
    kedges = []
    for hv in interior:
        bs = sorted(bid[(min(hv, b), max(hv, b))] for b in nbrs[hv])
        for i_, j_ in ((0, 1), (1, 2), (0, 2)):
            mi, mj = mid_ref[bonds[bs[i_]]], mid_ref[bonds[bs[j_]]]
            dx, dy = mj[0] - mi[0], mj[1] - mi[1]
            Lref = math.hypot(dx, dy)
            kedges.append((hv, bs[i_], bs[j_], dx / Lref, dy / Lref, Lref))
    adj = {i_: [] for i_ in range(len(bonds))}
    for hv, i_, j_, ux, uy, Lref in kedges:
        adj[i_].append((j_, hv, ux, uy))
        adj[j_].append((i_, hv, -ux, -uy))
    s0 = kedges[0][5]
    mid_ref = {bd: (x / s0, y / s0) for bd, (x, y) in mid_ref.items()}

    def embed(t):
        start = next(i_ for i_ in range(len(bonds)) if adj[i_])
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
    tri_polys, tri_areas = [], []
    for hv in interior:
        bs = sorted(bid[(min(hv, b), max(hv, b))] for b in nbrs[hv])
        pts = [pos[bs[i_]] for i_ in range(3)]
        tri_polys.append(pts)
        tri_areas.append(polygon_area(pts))
    hex_polys, hex_fidx = [], []
    for fi_, vs in enumerate(hex_vids):
        pts, ok = [], True
        for k in range(6):
            key = (min(vs[k], vs[(k + 1) % 6]), max(vs[k], vs[(k + 1) % 6]))
            if key not in bid:
                ok = False
                break
            pts.append(pos[bid[key]])
        if ok and len(pts) == 6:
            hex_polys.append(pts)
            hex_fidx.append(fi_)

    # ---- tile-aligned hexagon envelope ----
    # metric disorder pushes the actual geometry off the ideal lattice (by up
    # to 2x), so nearest-neighbour tests by distance/angle are unreliable;
    # use a **topological** criterion instead: hole adjacency built from
    # shared honeycomb vertices (metric-free), BFS rings from the central
    # hole; a triangle whose three neighbouring holes are all selected =>
    # its boundary is made of tile edges, so no tile gets truncated.
    v2f = {}
    for fi_, vs in enumerate(hex_vids):
        for v in vs:
            v2f.setdefault(v, []).append(fi_)
    adjf = {fi_: set() for fi_ in range(len(hex_vids))}
    for v, fl in v2f.items():
        for i_ in range(len(fl)):
            for j_ in range(i_ + 1, len(fl)):
                adjf[fl[i_]].add(fl[j_])
                adjf[fl[j_]].add(fl[i_])
    allv = np.concatenate([np.asarray(p, float) for p in tri_polys])
    cxw, cyw = float(allv[:, 0].mean()), float(allv[:, 1].mean())
    cent = np.array([np.asarray(hex_polys[k], float).mean(axis=0)
                     for k in range(len(hex_polys))])
    k0 = int(np.argmin(np.hypot(cent[:, 0] - cxw, cent[:, 1] - cyw)))
    f0 = hex_fidx[k0]
    dist = {f0: 0}
    dq = deque([f0])
    while dq:
        u = dq.popleft()
        for w in adjf[u]:
            if w not in dist:
                dist[w] = dist[u] + 1
                dq.append(w)
    sel_f = {fi_ for fi_, dd_ in dist.items() if dd_ <= rings}
    tri_sel = []
    for i_, hv in enumerate(interior):
        fl = v2f.get(hv, [])
        # outermost-ring "stitching triangles" (>= 2 neighbouring holes
        # selected) are also included and outlined
        if len(fl) == 3 and sum(f in sel_f for f in fl) >= 2:
            tri_sel.append(tri_polys[i_])
    hex_sel = [hex_polys[k] for k in range(len(hex_polys))
               if hex_fidx[k] in sel_f]
    allenv = np.concatenate([np.asarray(p, float)
                             for p in (tri_sel + hex_sel)])
    ex0, ex1 = allenv[:, 0].min(), allenv[:, 0].max()
    ey0, ey1 = allenv[:, 1].min(), allenv[:, 1].max()
    hex_sel_a = [polygon_area(p) for p in hex_sel]
    hex_s = [list(map(tuple, p)) for p in hex_sel]
    tri_s = [list(map(tuple, p)) for p in tri_sel]

    fig, ax = _fig()
    ax.add_collection(PolyCollection(tri_s, facecolors=GREY,
                                     edgecolors="black", linewidths=LW,
                                     zorder=2))
    am = float(np.mean(hex_sel_a))
    pch = PolyCollection(hex_s, array=np.array(hex_sel_a) / am,
                         cmap=plt.get_cmap("viridis"),
                         edgecolors="black", linewidths=LW, zorder=3)
    vmin, vmax = float(np.min(hex_sel_a)) / am, float(np.max(hex_sel_a)) / am
    if vmax - vmin < 1e-9:
        vmin, vmax = 0.5, 1.5
    pch.set_clim(vmin, vmax)
    ax.add_collection(pch)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(ex0 - 0.03, ex1 + 0.03)
    ax.set_ylim(ey0 - 0.03, ey1 + 0.03)
    save_autocrop(fig, OUT / "kagome_faces.png")
    lo, hi = vmin, vmax
    print(f"  kagome_faces: aligned (topological) hex envelope rings={rings} "
          f"({ex1-ex0:.1f}x{ey1-ey0:.1f}), faces {len(tri_s)}+{len(hex_s)}, "
          f"hex area/mean [{lo:.2f},{hi:.2f}]")


# ---------------- c 4.8.8: octagons vs squares by CV ----------------
def gen_q488(W=8, L=8, seed=20260927):
    from tsq_488_real_demo import (T, build_geometry, face_polygons,
                                   sample_solution, bfs_embed)
    faces, E = T.tsq_patch(W, L)
    epi = build_geometry(W, L)
    lens = sample_solution(faces, E, seed)
    pos = bfs_embed(epi, lens)
    polys = face_polygons(faces, epi, pos)
    oct_polys, oct_areas, sq_polys, sq_areas = [], [], [], []
    for size, pts in polys:
        a = polygon_area(pts)
        (oct_polys if size == 8 else sq_polys).append(pts)
        (oct_areas if size == 8 else sq_areas).append(a)
    # fixed coloring: octagons colored (normalized by their own areas), squares grey
    use_p, use_a, grey = oct_polys, oct_areas, sq_polys
    fig, ax = _fig()
    allv = np.concatenate([np.asarray(p, float) for p in use_p])
    cxw, cyw = float(allv[:, 0].mean()), float(allv[:, 1].mean())
    lo, hi = render_fig1(ax, [list(map(tuple, p)) for p in use_p], use_a,
                         grey=[list(map(tuple, p)) for p in grey],
                         win=(cxw, cyw, 13.0))
    print(f"  q488_faces: area/mean [{lo:.2f},{hi:.2f}]")
    save_autocrop(fig, OUT / "q488_faces.png")


# ---------------- d 3.3.3.4.4: subextensive 2L+1 implementation ----------------
# same method and seed as the earlier metric-demo assets;
# constructor z4_3464/RTH_numerical_verification.build_33344 (32/32 regression)
from collections import deque  # noqa: E402


def _patch_from_torus(t, fx=0.62, fy=0.62, u0=0.20, v0=0.20):
    sel = []
    for fi, (keys, pts, ftype) in enumerate(t.faces):
        uv = t._Minv @ np.mean(pts, axis=0)
        if u0 <= uv[0] <= u0 + fx and v0 <= uv[1] <= v0 + fy:
            sel.append(fi)
    assert len(sel) >= 8, f"patch too small: {len(sel)}"
    ecount = {}
    for fi in sel:
        for eid, sgn in t.face_edges[fi]:
            ecount[eid] = ecount.get(eid, 0) + 1
    assert max(ecount.values()) <= 2
    remap = {e: i for i, e in enumerate(sorted(ecount))}
    V_, E_ = set(), set()
    for fi in sel:
        V_.update(t.faces[fi][0])
        E_.update(t.face_edges[fi][k][0] for k in range(len(t.face_edges[fi])))
    assert len(V_) - len(E_) + len(sel) == 1, (
        f"patch not simply connected (chi={len(V_)-len(E_)+len(sel)}):"
        " the window wraps the torus - shrink fx/fy or shift u0/v0")
    return sel, remap


def _float_rows(t, sel, remap):
    rows = []
    for fi in sel:
        rx = np.zeros(len(remap)); ry = np.zeros(len(remap))
        for eid, sgn in t.face_edges[fi]:
            cx_qd, sy_qd = t.edge_cs[eid]
            rx[remap[eid]] += sgn * (float(cx_qd.a) + np.sqrt(t.d) * float(cx_qd.b))
            ry[remap[eid]] += sgn * (float(sy_qd.a) + np.sqrt(t.d) * float(sy_qd.b))
        rows.append(rx); rows.append(ry)
    return np.array(rows)


def _sample_lengths(rows, rng, amp=0.5):
    u_, s_, vh = np.linalg.svd(rows)
    rank = int(np.sum(s_ > s_.max() * 1e-9))
    basis = vh[rank:].T
    ones = np.ones(rows.shape[1])
    assert np.max(np.abs(rows @ ones)) < 1e-8
    d = basis @ rng.standard_normal(rows.shape[1] - rank)
    d = d / (np.max(np.abs(d)) + 1e-300)
    ell = ones + amp * d
    return ell, rows.shape[1] - rank


def _edge_vec(t, eid):
    cx_qd, sy_qd = t.edge_cs[eid]
    return np.array([float(cx_qd.a) + np.sqrt(t.d) * float(cx_qd.b),
                     float(sy_qd.a) + np.sqrt(t.d) * float(sy_qd.b)])


def _layout(t, sel, remap, ell, tol=1e-6):
    place = {}
    f0 = sel[0]
    place[t.faces[f0][0][0]] = np.asarray(t.faces[f0][1][0], float)
    face_of_edge = {}
    for fi in sel:
        for eid, sgn in t.face_edges[fi]:
            face_of_edge.setdefault(eid, []).append(fi)
    dq = deque([f0]); seen = {f0}
    while dq:
        fi = dq.popleft()
        keys, pts, _ = t.faces[fi]
        segs = t.face_edges[fi]
        m = len(keys)
        start = next(i for i in range(m) if keys[i] in place)
        cur = np.array(place[keys[start]], float)
        i = start
        for _ in range(m):
            eid, sgn = segs[i]
            nxt = keys[(i + 1) % m]
            cand = cur + _edge_vec(t, eid) * sgn * ell[remap[eid]]
            if nxt in place:
                assert np.linalg.norm(cand - place[nxt]) < tol
            else:
                place[nxt] = cand
            cur = cand
            i = (i + 1) % m
        for eid, sgn in segs:
            for fj in face_of_edge[eid]:
                if fj not in seen:
                    seen.add(fj); dq.append(fj)
    polys, ftypes = [], []
    for fi in sel:
        keys = t.faces[fi][0]
        assert all(k in place for k in keys)
        polys.append([tuple(place[k]) for k in keys])
        ftypes.append(t.faces[fi][2])
    return polys, ftypes


def gen_elt(L=12, M=4, fx=0.90, fy=0.90, u0=0.075, v0=0.05, seed=20261003):
    sys.path.insert(0, str(PC.engine("z4_3464")))
    RTH = importlib.import_module(PC.fname("RTH_numerical_verification"))
    build_33344 = RTH.build_33344
    rng = np.random.default_rng(seed)
    t = build_33344(L, M)
    sel, remap = _patch_from_torus(t, fx=fx, fy=fy, u0=u0, v0=v0)
    rows = _float_rows(t, sel, remap)
    for amp in (0.5, 0.45, 0.40, 0.35, 0.30):      # amplitude fallback, keeping edge lengths > 0.25
        ell, dim = _sample_lengths(rows, rng, amp=amp)
        if ell.min() > 0.25:
            break
    else:
        raise RuntimeError("no valid realisation")
    res = float(np.max(np.abs(rows @ ell)))
    assert res < 1e-8, res
    polys, ftypes = _layout(t, sel, remap, ell)
    sq_p = [p for p, ft in zip(polys, ftypes) if ft == 4]
    sq_a = [polygon_area(p) for p in sq_p]
    tri_p = [p for p, ft in zip(polys, ftypes) if ft == 3]
    allv = np.concatenate([np.asarray(p, float) for p in polys])
    cxw, cyw = float(allv[:, 0].mean()), float(allv[:, 1].mean())
    fig, ax = _fig()
    # window aspect ratio 0.85 (matches the Fig 2d inset frame 0.34x0.50); half_w within the patch
    lo, hi = render_fig1(ax, sq_p, sq_a, grey=tri_p,
                         win=(cxw, cyw, 4.5), aspect=0.85)
    print(f"  elt_faces: 3.3.3.4.4 patch {len(polys)} faces (kernel {dim}, "
          f"res {res:.1e}); square area/mean [{lo:.2f},{hi:.2f}]")
    save_autocrop(fig, OUT / "elt_faces.png")


if __name__ == "__main__":
    print("= Fig 2 inset assets (Fig 1 area-coloring language) =")
    gen_sq()
    gen_kagome()
    gen_q488()
    gen_elt()
    print("done")
