# -*- coding: utf-8 -*-
"""
rth_zint_demo.py - 3.4.6.4 random tiling demo with a finite integer alphabet.

Method:
  - Topology/constraints: build_3464(L) torus from RTH_numerical_verification.py
    (27/27 regression checks pass);
  - Integer solution construction: rref_and_param gives the pivot formula
    x_pivot = -T*f; D=1 => T is an integer kernel basis (N=T), so any integer
    vector c gives the valid integer solution 1 - N*c (all-ones shift +
    integer kernel combination; N found to lie in {0,+-1});
  - Alphabet {4,5,6} (MOF edge-color semantics: blue 4 / green 5 / red 6):
    sparse c (3-5 modes, coefficients +/-1) => spread=3 => shift into {4,5,6};
    assert the value set is exactly three values;
  - Embedding: BFS over the edge-vertex graph (directions fixed to the
    30-degree grid unit vectors, lengths = solution values); the simply
    connected fundamental domain has no holonomy; the all-ones solution is
    compared face by face against the reference instance coordinates
    (constant offset) as an anchor;
  - Drawing: 2x2 periodic tiling + central crop window ("one window onto the
    infinite structure"); edges colored by value in three colors (discrete,
    MOF semantics), faces lightly filled by type (triangle/square/hexagon).
Output: RTH_integer_alphabet_random_tiling_demo.png/.pdf (this directory)
"""
import os
import sys
import math
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from RTH_numerical_verification import build_3464, rref_and_param, QD, qd_from_component  # noqa: E402

EDGE_COL = {4: "#1f77b4", 5: "#2ca02c", 6: "#d62728"}   # MOF semantics: blue 4 / green 5 / red 6
FACE_FILL = {3: "#dbe6f5", 4: "#fbe3c9", 6: "#d9ead3"}  # light fills for triangle/square/hexagon


# ------------- planar (open boundary) 3.4.6.4 patch: simply connected => BFS has no holonomy ----
class PlaneRTH:
    """Open-boundary 3.4.6.4 patch of W x L cells (reuses the build_3464 cell
    geometry, no periodic reduction). After finalize: faces[(keys, pts, ftype)],
    face_edges, xrows/yrows (QD), E."""

    @staticmethod
    def _rk(p):
        return (round(float(p[0]), 6), round(float(p[1]), 6))

    def __init__(self, W, L):
        import math as _m
        s = 1.0
        D = s * (1 + _m.sqrt(3))
        B1 = D * np.array((_m.cos(_m.radians(30)), _m.sin(_m.radians(30))))
        B2 = D * np.array((_m.cos(_m.radians(90)), _m.sin(_m.radians(90))))
        u60 = [(_m.cos(_m.radians(60 * m)), _m.sin(_m.radians(60 * m)))
               for m in range(6)]
        u30 = [(_m.cos(_m.radians(30 + 60 * m)), _m.sin(_m.radians(30 + 60 * m)))
               for m in range(6)]
        self.pos = {}
        self._raw = []
        for i in range(W):
            for j in range(L):
                C = i * B1 + j * B2
                v = [C + np.array(u60[m]) for m in range(6)]
                n = [np.array(u30[m]) for m in range(6)]
                self.add_face([tuple(p) for p in v], 6)
                for m in range(6):
                    self.add_face([tuple(v[m]), tuple(v[(m + 1) % 6]),
                                   tuple(v[(m + 1) % 6] + s * n[m]),
                                   tuple(v[m] + s * n[m])], 4)
                for m in range(6):
                    self.add_face([tuple(v[m]), tuple(v[m] + s * n[(m - 1) % 6]),
                                   tuple(v[m] + s * n[m])], 3)
        self.finalize()

    def add_face(self, pts, ftype):
        pts = [np.asarray(p, float) for p in pts]
        area = sum(float(pts[i][0] * pts[(i + 1) % len(pts)][1]
                         - pts[(i + 1) % len(pts)][0] * pts[i][1])
                   for i in range(len(pts)))
        assert abs(area) > 1e-9
        if area < 0:
            pts = pts[::-1]
        keys = []
        for p in pts:
            k = self._rk(p)
            if k not in self.pos:
                self.pos[k] = p
            keys.append(k)
        self._raw.append((keys, pts, ftype))

    def finalize(self):
        seen, faces = set(), []
        for keys, pts, ftype in self._raw:
            cent = np.mean(pts, axis=0)
            sig = (frozenset(keys), ftype, self._rk(cent))
            if sig in seen:
                continue
            seen.add(sig)
            faces.append((keys, pts, ftype))
        self.faces = faces
        self.V = len(self.pos)
        self.F = len(faces)
        self.edge_ids, self.edge_dirf, self.edge_cs = {}, {}, []
        self.face_edges = []
        for keys, pts, ftype in faces:
            segs = []
            n = len(keys)
            for i in range(n):
                a, b = pts[i], pts[(i + 1) % n]
                ek = self._rk((a + b) / 2)
                dv = b - a
                L = float(np.hypot(*dv))
                dvu = dv / L
                if ek not in self.edge_ids:
                    self.edge_ids[ek] = len(self.edge_ids)
                    self.edge_dirf[ek] = dvu
                    ang = math.degrees(math.atan2(dvu[1], dvu[0]))
                    ar = round(ang / 30) * 30 % 360
                    assert abs((ang - ar + 180) % 360 - 180) < 1e-6, \
                        f"direction {ang} not on the 30-degree grid"
                    self.edge_cs.append(
                        (qd_from_component(math.cos(math.radians(ar)), 3),
                         qd_from_component(math.sin(math.radians(ar)), 3)))
                dot = float(np.dot(self.edge_dirf[ek], dvu))
                sgn = 1 if dot > 0.5 else (-1 if dot < -0.5 else 0)
                assert sgn != 0
                segs.append((self.edge_ids[ek], sgn))
            self.face_edges.append(segs)
        self.E = len(self.edge_ids)
        deg = {k: 0 for k in self.pos}
        for keys, _, _ in faces:
            for k in keys:
                deg[k] += 1
        self.degs = sorted(set(deg.values()))
        self.xrows, self.yrows = [], []
        for segs in self.face_edges:
            rx = [QD(0, 0, 3) for _ in range(self.E)]
            ry = [QD(0, 0, 3) for _ in range(self.E)]
            for eid, sgn in segs:
                cx, sy = self.edge_cs[eid]
                if sgn > 0:
                    rx[eid] = rx[eid] + cx
                    ry[eid] = ry[eid] + sy
                else:
                    rx[eid] = rx[eid] - cx
                    ry[eid] = ry[eid] - sy
            self.xrows.append(rx)
            self.yrows.append(ry)
        for rx, ry in zip(self.xrows, self.yrows):
            tx = QD(0, 0, 3)
            ty = QD(0, 0, 3)
            for x in rx:
                tx = tx + x
            for y in ry:
                ty = ty + y
            assert tx.iszero() and ty.iszero(), "all-ones solution fails to close"


def sample_int_solution(Tm, piv_cols, free_cols, E, seed, n_modes=4,
                        alphabet=(4, 5, 6), max_try=400):
    """Integer kernel combination => full-edge-value solution over the
    {4,5,6} alphabet. Parameterization: x[piv] = -Tm*f, x[free] = f (f
    integer) => any sparse f generates an integer kernel solution."""
    rng = np.random.default_rng(seed)
    lo = min(alphabet)
    nfree = len(free_cols)
    for _ in range(max_try):
        cols = rng.choice(nfree, size=n_modes, replace=False)
        f = np.zeros(nfree, dtype=np.int64)
        f[cols] = rng.choice((-1, 1), size=n_modes)
        s = np.zeros(E, dtype=np.int64)
        s[free_cols] = f
        s[piv_cols] = -(Tm @ f)
        vals = set(s.tolist())
        if len(vals) == len(alphabet) and s.max() - s.min() == len(alphabet) - 1:
            return (s - s.min() + lo).astype(np.int64)
    raise RuntimeError("no solution with full alphabet coverage")


def build_embed(t):
    """Edge endpoint table + BFS embedder. Returns (eid2end, adj, start)."""
    eid2end = {}          # eid -> (va, vb, ux, uy)  unit direction a->b
    for (keys, pts, ftype), segs in zip(t.faces, t.face_edges):
        n = len(keys)
        for i, (eid, sgn) in enumerate(segs):
            a, b = keys[i], keys[(i + 1) % n]
            dv = np.asarray(pts[(i + 1) % n], float) - np.asarray(pts[i], float)
            L = float(np.hypot(*dv))
            u_ab = dv / L                      # actual direction a->b (reference instance)
            if eid in eid2end:
                (pa, pb, ux, uy) = eid2end[eid]
                assert (pa, pb) == (a, b) or (pa, pb) == (b, a)
                dold = np.array([ux, uy])
                if (pa, pb) == (a, b):
                    assert np.allclose(dold, u_ab, atol=1e-9)
                else:
                    assert np.allclose(dold, -u_ab, atol=1e-9)
                continue
            eid2end[eid] = (a, b, float(u_ab[0]), float(u_ab[1]))
    adj = {v: [] for v in t.pos}
    for eid, (a, b, ux, uy) in eid2end.items():
        adj[a].append((b, eid, ux, uy))
        adj[b].append((a, eid, -ux, -uy))
    start = next(v for v in t.pos if adj[v])
    return eid2end, adj, start


def embed(t, adj, start, s):
    """BFS unfolding (periodic solution => no holonomy); returns vkey -> (x, y)."""
    pos = {start: (0.0, 0.0)}
    stack = [start]
    while stack:
        cur = stack.pop()
        px, py = pos[cur]
        for nxt, eid, ux, uy in adj[cur]:
            if nxt in pos:
                continue
            pos[nxt] = (px + s[eid] * ux, py + s[eid] * uy)
            stack.append(nxt)
    return pos


def verify_all_ones(t, adj, start):
    """Anchor: embedding of the all-ones solution; each face's vertex ring
    differs from the reference instance coordinates by a constant vector."""
    pos = embed(t, adj, start, np.ones(t.E, dtype=np.int64))
    worst = 0.0
    for (keys, pts, ftype) in t.faces:
        if any(k not in pos for k in keys):
            continue
        p0 = np.asarray(pts[0], float)
        shift = np.array(pos[keys[0]]) - p0
        for k, p in zip(keys, pts):
            d = np.array(pos[k]) - (np.asarray(p, float) + shift)
            worst = max(worst, float(np.abs(d).max()))
    assert worst < 1e-6, f"all-ones embed mismatch {worst}"
    return worst


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.lines import Line2D

    W, L = 5, 5
    t0 = time.time()
    t = PlaneRTH(W, L)
    E = t.E
    T, piv, free, dim = rref_and_param(t.xrows, t.yrows, E)
    D = 1
    for row in T:
        for v in row:
            D = D * v.denominator // math.gcd(D, v.denominator)
    assert D == 1, f"common denominator D={D} != 1 (this script assumes an integer kernel basis)"
    Tm = np.array([[int(v) for v in row] for row in T], dtype=np.int64)
    print(f"patch {W}x{L}: V={t.V} E={E} F={t.F} dim={dim} D={D} "
          f"({time.time()-t0:.1f}s)", flush=True)

    sols = [("a", None),
            ("b", sample_int_solution(Tm, piv, free, E, 20260927)),
            ("c", sample_int_solution(Tm, piv, free, E, 20260928))]
    eid2end, adj, start = build_embed(t)
    w = verify_all_ones(t, adj, start)
    print(f"all-ones anchor worst dev = {w:.2e}", flush=True)

    MM = 1 / 25.4
    fig = plt.figure(figsize=(183 * MM, 66 * MM))
    ax_w = 0.235
    xs0 = [0.045, 0.045 + (ax_w + 0.03), 0.045 + 2 * (ax_w + 0.03)]

    for (tag, s), x0 in zip(sols, xs0):
        ax = fig.add_axes([x0, 0.12, ax_w, 0.80])
        ax.set_aspect("equal")
        ax.axis("off")
        pos = embed(t, adj, start, s if s is not None
                    else np.ones(E, dtype=np.int64))
        for keys, pts, ftype in t.faces:
            if any(k not in pos for k in keys):
                continue
            ax.add_patch(MplPolygon([pos[k] for k in keys], closed=True,
                                    facecolor=FACE_FILL[ftype],
                                    edgecolor="none", alpha=0.9, zorder=1))
        segs, cols = [], []
        for eid, (a, b, ux, uy) in eid2end.items():
            if a not in pos or b not in pos:
                continue
            segs.append([pos[a], pos[b]])
            v = 1 if s is None else int(s[eid])
            cols.append("#555555" if s is None else EDGE_COL[v])
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.5,
                                         capstyle="round", zorder=3))
        allx = [p[0] for p in pos.values()]
        ally = [p[1] for p in pos.values()]
        cxm, cym = (min(allx) + max(allx)) / 2, (min(ally) + max(ally)) / 2
        half = 0.60 * max(max(allx) - min(allx), max(ally) - min(ally)) / 2
        ax.set_xlim(cxm - half, cxm + half)
        ax.set_ylim(cym - half * 0.86, cym + half * 0.86)
        ax.text(0.02, 0.985, tag, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top", ha="left", zorder=5)

    # Discrete legend (edge values 4/5/6 + three face types), horizontal at figure bottom
    handles = [Line2D([0], [0], color=EDGE_COL[v], lw=3,
                      label=f"edge length {v}") for v in (4, 5, 6)]
    face_names = {3: "triangle", 4: "square", 6: "hexagon"}
    for ft in (3, 4, 6):
        handles.append(MplPolygon([(0, 0), (1, 0), (1, 1)], closed=True,
                                  facecolor=FACE_FILL[ft], edgecolor="none",
                                  label=face_names[ft]))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.005))

    for ext in ("png", "pdf"):
        out = os.path.join(HERE,
                           f"RTH_integer_alphabet_random_tiling_demo.{ext}")
        fig.savefig(out, dpi=300)
        if ext == "png":
            print("fig ->", out, flush=True)
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    draw()
    print(f"--- done ({time.time()-t0:.1f}s) ---")
