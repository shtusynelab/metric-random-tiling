# -*- coding: utf-8 -*-
"""
snub_lib.py - (3,3,3,3,6) snub hexagonal torus construction and nullity.
Last member of the all-rational-face families.

**Result: dim_R = dim_Q = 1 exact (verified for k=1..5) => rigid, Omega <= k,
s = 0.**
Mechanism: edge-sharing triangles lock each other (each triangle has three
equal edges; sharing an edge forces the two triangles to take the same value),
the triangle co-edge adjacency graph is connected => all triangle edges carry
one single scale; every hexagon edge borders a triangle => all edges share one
value. The unique kernel mode is the global scale.

Construction (triangular-lattice perfect 1-error-correcting code):
  - Triangular lattice a1=(1,0), a2=(1/2,sqrt(3)/2); code = ker phi with
    phi(i,j) = (i+3j) mod 7 (the six neighbor directions +a1, +a2,
    +(a2-a1) and their negatives map onto all 6 nonzero cosets => perfect
    code; code lattice Lambda_c = <(-3,1),(7,0)>, det = 7, density 1/7,
    center spacing sqrt(7));
  - Centers = code points (deleted); spokes = center-neighbor edges (deleted);
  - Remaining faces = one hexagon per center (its 6-neighbor ring) plus all
    unit triangles whose three vertices are non-centers; vertex figure =
    4 triangles + 1 hexagon = (3,3,3,3,6) (combinatorially equivalent to the
    Wythoff s{6,3} construction);
  - Torus = k*Lambda_c (compatible with the code lattice => 7k^2 lattice
    points, k^2 centers, k^2 hexagons, 8k^2 triangles, 6k^2 vertices,
    15k^2 edges, z=5).

Self-check chain (built-in asserts): perfect-code property / face and Euler
counts / all-ones closure / cyclic vertex-figure normalization =
(3,3,3,3,6) / dim_R = dim_Q (Niven: faces in {3,6}, no split rows).
Implementation note: faces are oriented by their raw lifted paths (unit steps);
edge keys = doubled midpoints reduced mod the **2 x torus lattice**
(half-lattice-point identity; lattice-point reduction would wrongly merge
parallel-edge midpoints).
"""
import math
from fractions import Fraction as Fr
import numpy as np

try:
    from RTH_numerical_verification import (field_rank_qd, _rank_fraction, QD,
                                            qd_from_component)
except ImportError:                       # when run directly as a script
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from RTH_numerical_verification import (field_rank_qd, _rank_fraction, QD,
                                            qd_from_component)


def build_snub(k):
    """Snub hexagonal torus scaled by k; returns dict(V/E/F/xrows/yrows)."""
    g1 = np.array([7 * k, 0])        # torus lattice vectors = k x code lattice (triangular coords)
    g2 = np.array([-3 * k, k])
    M = np.column_stack([g1, g2])    # M = [[7k,-3k],[0,k]], det = 7k^2 > 0
    det = 7 * k * k
    adj = np.array([[k, 3 * k], [0, 7 * k]])    # M*adj = det*I
    assert np.array_equal(M @ adj, det * np.eye(2, dtype=int))

    def reduce_lat(p):
        c = adj @ np.array(p)
        fl = c // det
        return tuple(int(x) for x in (np.array(p) - M @ fl))

    M2, adj2, det2 = 2 * M, 2 * adj, 4 * det

    def reduce_mid2(p):              # half-lattice points: mod 2 x torus lattice
        c = adj2 @ np.array(p)
        fl = c // det2
        return tuple(int(x) for x in (np.array(p) - M2 @ fl))

    def in_code(p):
        return (p[0] + 3 * p[1]) % 7 == 0

    DIRS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    pl = [reduce_lat((i, j)) for i in range(8 * k) for j in range(8 * k)]
    pl = sorted(set(pl))
    for p in pl:
        n = sum(1 for d in DIRS if in_code(reduce_lat((p[0] + d[0], p[1] + d[1]))))
        if in_code(p):
            assert n == 0, f"center {p} is adjacent to a center (code distance too small)"
        else:
            assert n == 1, f"non-center {p} has {n} center neighbors (not a perfect code)"
    verts = sorted(set(p for p in pl if not in_code(p)))
    vset = set(verts)

    def red(p):
        return reduce_lat(p)

    def mid2_key(a, b):
        return reduce_mid2((a[0] + b[0], a[1] + b[1]))

    edge_dir = {}
    face_specs = []
    seen_face = set()
    for c in sorted(set(p for p in pl if in_code(p))):
        ring = [(c[0] + d[0], c[1] + d[1]) for d in DIRS]
        segs = []
        for i in range(6):
            a, b = ring[i], ring[(i + 1) % 6]
            mk = mid2_key(a, b)
            edge_dir[mk] = (b[0] - a[0], b[1] - a[1])
            segs.append((mk, b[0] - a[0], b[1] - a[1]))
        key = (frozenset(red(q) for q in ring), 6)
        if key not in seen_face:
            seen_face.add(key)
            face_specs.append((segs, 6, key))
    tri_seen = set()
    for p in verts:
        for i in range(6):
            d1, d2 = DIRS[i], DIRS[(i + 1) % 6]
            a = (p[0] + d1[0], p[1] + d1[1])
            b = (p[0] + d2[0], p[1] + d2[1])
            if not (red(a) in vset and red(b) in vset):
                continue
            key = (frozenset((red(p), red(a), red(b))), 3)
            if key in tri_seen or key in seen_face:
                continue
            tri_seen.add(key)
            segs = []
            for (u, v) in ((p, a), (a, b), (b, p)):
                mk = mid2_key(u, v)
                edge_dir[mk] = (v[0] - u[0], v[1] - u[1])
                segs.append((mk, v[0] - u[0], v[1] - u[1]))
            face_specs.append((segs, 3, key))
    faces = face_specs
    edge_ids = {mk: i for i, mk in enumerate(sorted(edge_dir))}
    E = len(edge_ids)

    def xy_lat(m):
        return (m[0] + 0.5 * m[1], math.sqrt(3) / 2 * m[1])

    xrows, yrows = [], []
    for segs, ft, key in faces:
        rx = [QD(0, 0, 3)] * E
        ry = [QD(0, 0, 3)] * E
        rx = list(rx)
        ry = list(ry)
        for (mk, dx, dy) in segs:
            ux, uy = xy_lat((dx, dy))
            cs = (qd_from_component(ux, 3), qd_from_component(uy, 3))
            e = edge_ids[mk]
            rx[e] = rx[e] + cs[0]
            ry[e] = ry[e] + cs[1]
        xrows.append(rx)
        yrows.append(ry)

    n_hex = sum(1 for _, ft, _ in faces if ft == 6)
    n_tri = sum(1 for _, ft, _ in faces if ft == 3)
    Vt, Ft = len(verts), len(faces)
    assert (n_hex, n_tri) == (k * k, 8 * k * k), \
        f"face counts {n_hex},{n_tri} != {k*k},{8*k*k}"
    assert Vt == 6 * k * k and E == 15 * k * k, \
        f"V/E {Vt},{E} != {6*k*k},{15*k*k}"
    assert Vt - E + Ft == 0, "Euler check failed"
    for rx, ry in zip(xrows, yrows):
        tx = QD(0, 0, 3)
        ty = QD(0, 0, 3)
        for x in rx:
            tx = tx + x
        for y in ry:
            ty = ty + y
        assert tx.iszero() and ty.iszero(), "all-ones solution fails to close"
    from collections import defaultdict
    v2f = defaultdict(list)
    for segs, ft, key in faces:
        vr = sorted(key[0])
        P = [xy_lat(q) for q in vr]
        cent = (sum(q[0] for q in P) / len(P), sum(q[1] for q in P) / len(P))
        for q in vr:
            pq = xy_lat(q)
            ang = math.atan2(pq[1] - cent[1], pq[0] - cent[0])
            v2f[q].append((ang, ft))
    figs = set()
    for p, recs in v2f.items():
        recs.sort(key=lambda t: t[0])
        seq = tuple(t for _, t in recs)
        figs.add(min(seq[i:] + seq[:i] for i in range(len(seq))))
    assert figs == {(3, 3, 3, 3, 6)}, f"vertex figures {figs} != (3,3,3,3,6)"
    return dict(k=k, V=Vt, E=E, F=Ft, xrows=xrows, yrows=yrows)


def dims_snub(k):
    """Returns (dim_R, dim_Q), both exact over Fraction / Q(sqrt(3))."""
    r = build_snub(k)
    dR = r["E"] - field_rank_qd(r["xrows"] + r["yrows"])
    rows_q = []
    for rrow in r["xrows"] + r["yrows"]:
        rows_q.append([Fr(2 * x.a) for x in rrow])
        rows_q.append([Fr(2 * x.b) for x in rrow])
    dQ = r["E"] - _rank_fraction([x for x in rows_q if any(v != 0 for v in x)],
                                 r["E"])
    return dR, dQ


if __name__ == "__main__":
    for k in (1, 2, 3, 4):
        dR, dQ = dims_snub(k)
        print(f"k={k}: dim_R={dR} dim_Q={dQ} {'PASS' if dR == dQ == 1 else 'FAIL'}")
