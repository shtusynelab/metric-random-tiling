# -*- coding: utf-8 -*-
"""
snub33336_dim.py -- nullity of the (3,3,3,3,6) snub hexagonal torus.
Last branch of the all-rational-face family (project card WP-D; open
item 1 of z4_3464/README).

Construction (triangular-lattice perfect code / perfect
1-error-correcting code):
  - triangular lattice a1=(1,0), a2=(1/2,sqrt(3)/2); code lattice
    Lambda_c = <(2,1), (-1,3)> (det=7, density 1/7);
  - centers = code-lattice vertices (deleted); spokes = center-neighbor
    edges (deleted);
  - remaining faces = one hexagon per center (the ring of its 6
    neighbors) + all unit triangles with three non-center vertices;
  - vertex figure = 4 triangles + 1 hexagon = (3,3,3,3,6) (combinatorial
    equivalent of the literature snub-hexagonal Wythoff s{6,3}
    construction);
  - torus = k*(2,1) x k*(-1,3) (compatible with the code lattice =>
    7k^2 lattice sites, k^2 centers, k^2 hexagons, 8k^2 triangles,
    6k^2 vertices, 15k^2 edges).

Self-check chain: perfect-code property (each non-center has exactly 1
center neighbor) / Euler / face counts / all-ones closure / cyclic vertex
figure = (3,3,3,3,6) / dim_R = dim_Q (Niven: faces in {3,6} carry no
split rows).
Output: the dim(k) sequence; verdict = sub-extensive (zero entropy
expected) or extensive (a second 3.4.6.4-type counterexample).
"""
import os, sys, math
import numpy as np
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "z4_3464"),):
    sys.path.insert(0, p)
import verify_dsh_audit as V           # field_rank_qd / rref_and_param / QD
from RTH_numerical_verification import _rank_fraction  # noqa: E402


def build_snub(k):
    """k-scaled snub hexagonal torus; returns a simple object matching the
    Torus interface."""
    # perfect code = ker phi, phi(i,j) = (i + 3j) mod 7 (the six neighbor
    # directions map onto all 6 nonzero cosets)
    g1 = np.array([7 * k, 0])        # torus lattice vector = k times the code lattice (triangular coords)
    g2 = np.array([-3 * k, k])
    M = np.column_stack([g1, g2])    # M = [[7k,-3k],[0,k]], det = 7k^2 > 0
    det = 7 * k * k
    adj = np.array([[k, 3 * k], [0, 7 * k]])    # M @ adj = det * I
    assert np.array_equal(M @ adj, det * np.eye(2, dtype=int))

    def reduce_lat(p):
        c = adj @ np.array(p)                    # = numerator of det*(M^-1 p)
        fl = c // det                            # Python floor division = floor
        return tuple(int(x) for x in (np.array(p) - M @ fl))

    # half-lattice-point (doubled edge midpoint) reduction: identity is
    # taken mod 2x the torus lattice (otherwise parallel-edge midpoints merge)
    M2, adj2, det2 = 2 * M, 2 * adj, 4 * det

    def reduce_mid2(p):
        c = adj2 @ np.array(p)
        fl = c // det2
        return tuple(int(x) for x in (np.array(p) - M2 @ fl))

    def in_code(p):
        return (p[0] + 3 * p[1]) % 7 == 0

    DIRS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]  # 6 neighbors on the triangular lattice
    # ---- self-check 1: perfect code (each non-center has exactly 1 center neighbor) ----
    pl = [reduce_lat((i, j)) for i in range(8 * k) for j in range(8 * k)]
    pl = sorted(set(pl))
    for p in pl:
        n = sum(1 for d in DIRS if in_code(reduce_lat((p[0] + d[0], p[1] + d[1]))))
        if in_code(p):
            assert n == 0, f"center {p} adjacent to a center (code distance too small)"
        else:
            assert n == 1, f"non-center {p} has {n} center neighbors (not a perfect code)"
    verts = sorted(set(p for p in pl if not in_code(p)))
    vset = set(verts)
    # ---- faces (orientation from the raw lifted paths; edge keys = reduced doubled midpoints) ----
    def red(p):
        return reduce_lat(p)

    def mid2_key(a, b):
        return reduce_mid2((a[0] + b[0], a[1] + b[1]))

    edge_dir = {}       # mid2_key -> raw unit direction (lattice coordinates)
    face_specs = []     # ([(mid2_key, dx, dy), ...] CCW, ftype, key)
    seen_face = set()
    centers = sorted(set(p for p in pl if in_code(p)))
    for c in centers:
        ring = [(c[0] + d[0], c[1] + d[1]) for d in DIRS]     # raw lift
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
            if not (red(a) in vset and red(b) in vset and red(p) in vset):
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
    # ---- edge numbering and rows (exact over Q(sqrt(3))) ----
    edge_ids = {mk: i for i, mk in enumerate(sorted(edge_dir))}
    E = len(edge_ids)

    def xy_lat(m):                      # lattice coords -> Cartesian
        return (m[0] + 0.5 * m[1], math.sqrt(3) / 2 * m[1])

    xrows, yrows = [], []
    for segs, ft, key in faces:
        # CCW orientation: sign decided by the raw path area (vertex-lift reconstruction)
        rx = [V.QD(0, 0, 3)] * E
        ry = [V.QD(0, 0, 3)] * E
        rx = list(rx)
        ry = list(ry)
        for (mk, dx, dy) in segs:
            ux, uy = xy_lat((dx, dy))
            cs = (V.qd_from_component(ux, 3), V.qd_from_component(uy, 3))
            e = edge_ids[mk]
            rx[e] = rx[e] + cs[0]
            ry[e] = ry[e] + cs[1]
        xrows.append(rx)
        yrows.append(ry)
    # ---- self-check 2: counts and Euler ----
    n_hex = sum(1 for _, ft, _ in faces if ft == 6)
    n_tri = sum(1 for _, ft, _ in faces if ft == 3)
    Vt, Et, Ft = len(verts), E, len(faces)
    assert (n_hex, n_tri) == (k * k, 8 * k * k), \
        f"face counts {n_hex},{n_tri} != {k*k},{8*k*k}"
    assert Vt == 6 * k * k and Et == 15 * k * k, \
        f"V/E {Vt},{Et} != {6*k*k},{15*k*k}"
    assert Vt - Et + Ft == 0, "Euler check failed"
    # ---- self-check 3: all-ones closure ----
    for rx, ry in zip(xrows, yrows):
        tx = V.QD(0, 0, 3); ty = V.QD(0, 0, 3)
        for x in rx:
            tx = tx + x
        for y in ry:
            ty = ty + y
        assert tx.iszero() and ty.iszero(), "all-ones vector does not close"
    # ---- self-check 4: vertex figures ----
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
        figs.add(min(seq[i:] + seq[:i] for i in range(len(seq))))  # cyclic normalization
    assert figs == {(3, 3, 3, 3, 6)}, f"vertex figures {figs} != (3,3,3,3,6)"
    return dict(k=k, V=Vt, E=Et, F=Ft, xrows=xrows, yrows=yrows)


def main():
    print("= (3,3,3,3,6) snub hexagonal torus nullities =", flush=True)
    print("anchors: perfect code / Euler / face counts / all-ones / vertex "
          "figure built into build_snub", flush=True)
    out = []
    for k in (1, 2, 3, 4):
        try:
            r = build_snub(k)
        except AssertionError as e:
            print(f"  k={k}: construction self-check failed: {e}", flush=True)
            continue
        dR = r["E"] - V.field_rank_qd(r["xrows"] + r["yrows"])
        # Q rank (split rows: rational part + sqrt(3) part of each row)
        rows_q = []
        for rrow in r["xrows"] + r["yrows"]:
            rows_q.append([Fr(2 * x.a) for x in rrow])
            rows_q.append([Fr(2 * x.b) for x in rrow])
        dQ = r["E"] - _rank_fraction([x for x in rows_q if any(v != 0 for v in x)],
                                     r["E"])
        ext = dR / (r["F"])
        out.append((k, r["V"], r["E"], r["F"], dR, dQ, ext))
        print(f"  k={k}: V={r['V']} E={r['E']} F={r['F']} "
              f"dim_R={dR} dim_Q={dQ} ({'equal' if dR == dQ else 'split rows!'}) "
              f"dim/F={ext:.3f}", flush=True)
    print("-" * 80)
    if len(out) >= 3:
        ks = [o[0] for o in out]
        ds = [o[4] for o in out]
        print(f"dim sequence (k={ks}) = {ds}", flush=True)
        if len(set(ds)) == 1:
            print(f"verdict: dim constant = {ds[0]} => rigid (unique mode = "
                  f"global scale; mechanism = shared-edge triangle locking "
                  f"propagation; Omega <= k, s = 0)", flush=True)
        else:
            import numpy as np
            A = np.column_stack([ks, np.ones(len(ks))])
            c1 = np.linalg.lstsq(A, np.array(ds, float), rcond=None)[0]
            print(f"linear fit dim ~= {c1[0]:.2f}k + {c1[1]:.2f}", flush=True)
            verdict = "sub-extensive (zero entropy expected)" if c1[0] < 0.5 * max(ks) else \
                      "extensive (a second 3.4.6.4-type case!)"
            print(f"verdict: {verdict}", flush=True)


if __name__ == "__main__":
    main()
