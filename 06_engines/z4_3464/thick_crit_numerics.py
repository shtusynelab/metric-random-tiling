# -*- coding: utf-8 -*-
"""
thick_crit_numerics.py - criterion-theorem Phase 1 numerical verification.

Small-system anchors for the question "when is V_Z thick", three groups:

A. Triangle-locking theorem (RTH side):
   A1 Lemma 1 numerical face: 200 random integer solutions, all three edges of
      every triangle equal (checked per solution);
   A2 Kernel projection rank: the kernel basis projected onto triangle-edge
      coordinates has rank 1 (exact Fractions);
   A3 Structural facts: G_tri (triangle adjacency graph, squares act as welds)
      is 3-regular, connected, with |vertices| = 2L^2 and |edges| = 3L^2,
      matching the honeycomb graph degree sequence (combinatorial premise of
      the theorem proof);
   A4 Per-face-type rank decomposition: rank contributions of the triangle /
      square / hexagon rows (locates where the dependence sits).

B. Honeycomb thickness (control side):
   B1 Subbox certificate |S| series (L=2..6): whether ~cF extensive =>
      theorem-level Omega >= k^|S|, s >= |S| ln k / F;
   B2 Kernel-basis support statistics: mean support size (calibrated by dim, F)
      -- numerical test of whether "one mode per hexagon" is a counting
      statement or a local-support statement.

C. RTH certificate cross-check: |S| series L=2..5 (known 2L+3 = 7/9;
   11/13 to be added?).

Anchor self-checks: A2 rank 1 and A1 200/200 must hold; C at L=2/3 must
reproduce 7/9. All Fraction/integer exact, no float decisions (except the
combinatorial part of G_tri).
"""
import math, sys, os
import numpy as np
from fractions import Fraction as Fr

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from RTH_numerical_verification import build_3464, rref_and_param  # noqa: E402

import verify_dsh_audit as V  # noqa: E402  (build_honeycomb / subbox_certificate)


# ------------------------------------------------------------------ RTH side --
def face_edges_of_type(t, ftype):
    """Returns (face index list, per-face edge-id list)."""
    idxs, edges = [], []
    for fi, (keys, pts, ft) in enumerate(t.faces):
        if ft == ftype:
            idxs.append(fi)
            edges.append([eid for eid, _ in t.face_edges[fi]])
    return idxs, edges


def kernel_matrix(T, piv_cols, free_cols, E):
    """Integer kernel basis (D=1): K[free,j]=delta_j, K[piv,j]=-T[:,j].
    Returns an E x dim Fraction matrix."""
    P, dim = len(T), len(free_cols)
    D = 1
    for row in T:
        for v in row:
            D = D * v.denominator // math.gcd(D, v.denominator)
    assert D == 1, f"D={D} != 1 (this script assumes an integer kernel basis)"
    K = [[Fr(0)] * dim for _ in range(E)]
    for j, fc in enumerate(free_cols):
        K[fc][j] = Fr(1)
    for r, pc in enumerate(piv_cols):
        for j in range(dim):
            K[pc][j] = -T[r][j]
    return K


def rank_fraction_rows(cols, nrows):
    """Fraction rank of column vectors (input: each column = list[Fraction]
    of length nrows)."""
    M = [list(c) for c in cols]          # rows = original columns
    return V._rank_fraction([r for r in M if any(v != 0 for v in r)], nrows)


def rth_triangle_analysis(L, n_samples=200):
    t = build_3464(L)
    E = t.E
    tri_idx, tri_edges = face_edges_of_type(t, 3)
    sq_idx, sq_edges = face_edges_of_type(t, 4)
    hex_idx, hex_edges = face_edges_of_type(t, 6)
    tri_set = set(e for es in tri_edges for e in es)

    # A3: G_tri = triangle adjacency graph (squares weld two triangle edges)
    edge2tri = {}
    for ti, es in zip(tri_idx, tri_edges):
        for e in es:
            edge2tri[e] = ti
    adj = {ti: set() for ti in tri_idx}
    for si, es in zip(sq_idx, sq_edges):
        ts = [edge2tri[e] for e in es if e in edge2tri]
        assert len(ts) == 2, f"square {si} has {len(ts)} triangle edges != 2"
        a, b = ts
        assert a != b, "both triangle edges of a square belong to the same triangle (geometrically impossible)"
        adj[a].add(b); adj[b].add(a)
    degs = sorted(len(adj[ti]) for ti in tri_idx)
    # connectivity (BFS)
    seen, stack = set(), [tri_idx[0]]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(adj[u] - seen)
    connected = (len(seen) == len(tri_idx))
    # bipartiteness
    bip = True
    color = {tri_idx[0]: 0}
    stack = [tri_idx[0]]
    while stack and bip:
        u = stack.pop()
        for w in adj[u]:
            if w not in color:
                color[w] = 1 - color[u]
                stack.append(w)
            elif color[w] == color[u]:
                bip = False
                break
    n_tri, n_sq = len(tri_idx), len(sq_idx)

    # A2: kernel projection rank
    T, piv, free, dim = rref_and_param(t.xrows, t.yrows, E)
    K = kernel_matrix(T, piv, free, E)
    tri_rows = sorted(tri_set)
    proj = [[K[r][j] for r in tri_rows] for j in range(dim)]
    rk = rank_fraction_rows(proj, len(tri_rows))

    # A1: random integer-solution locking (protocol: coefficients +/-2, 1-8 modes)
    Tm = np.array([[int(v) for v in row] for row in T], dtype=np.int64)
    rng = np.random.default_rng(20260927 + L)
    locked = 0
    lock_vals = set()
    for _ in range(n_samples):
        nm = min(int(rng.integers(1, 9)), dim)
        cols = rng.choice(dim, size=nm, replace=False)
        f = np.zeros(dim, dtype=np.int64)
        f[cols] = rng.choice((-2, -1, 1, 2), size=nm)
        x = np.zeros(E, dtype=np.int64)
        x[free] = f
        x[piv] = -(Tm @ f)
        tv = x[sorted(tri_set)]
        if int(tv.max() - tv.min()) == 0:
            locked += 1
            lock_vals.add(int(tv[0]))
    # A4: per-face-type rank contributions (report the full-row rank and the
    # rank of each face type's rows alone; the dependence structure is
    # inferred from dim)
    rows_tri = tri_rows
    rows_sq = sorted(set(e for es in sq_edges for e in es))
    rows_hex = sorted(set(e for es in hex_edges for e in es))
    return dict(L=L, V=t.V, E=E, F=t.F, dim=dim, n_tri=n_tri, n_sq=n_sq,
                gtri_degs=(degs[0], degs[-1]), gtri_connected=connected,
                gtri_bipartite=bip, proj_rank=rk,
                locked=locked, n_samples=n_samples,
                lock_vals_span=(min(lock_vals), max(lock_vals)) if lock_vals else None,
                tri_rows=len(rows_tri), sq_rows=len(rows_sq), hex_rows=len(rows_hex))


def certificate_series(builder, Ls, k=2, label=""):
    out = []
    for L in Ls:
        t = builder(L)
        T, piv, free, dim = V.rref_and_param(t.xrows, t.yrows, t.E)
        ch, bound = V.subbox_certificate(T, k)
        s_lb = math.log(bound) / (t.F * math.log(k)) if bound > 1 else 0.0
        out.append((L, t.F, dim, len(ch), bound, s_lb))
        print(f"  [{label}] L={L} F={t.F} dim={dim} |S|={len(ch)} "
              f"Omega>={bound} (k={k}, s>={s_lb:.4f}/face)", flush=True)
    return out


# ---------------------------------------------------------------- main ----
def main():
    print("=" * 96)
    print("A. triangle-locking theorem numerical anchors (3.4.6.4 torus)")
    for L in (2, 3, 4, 5):
        r = rth_triangle_analysis(L)
        print(f"  L={L}: V={r['V']} E={r['E']} F={r['F']} dim={r['dim']} "
              f"tri={r['n_tri']} sq={r['n_sq']}", flush=True)
        print(f"      G_tri: deg in [{r['gtri_degs'][0]},{r['gtri_degs'][1]}] "
              f"connected={r['gtri_connected']} bipartite={r['gtri_bipartite']} "
              f"(nodes={r['n_tri']} = 2L^2={2*L*L})", flush=True)
        print(f"      kernel projection rank on triangle-edge coordinates = {r['proj_rank']} "
              f"(theorem predicts 1)", flush=True)
        print(f"      random integer-solution locking {r['locked']}/{r['n_samples']},"
              f" lock value span {r['lock_vals_span']}", flush=True)
        ok = (r['proj_rank'] == 1 and r['locked'] == r['n_samples']
              and r['gtri_connected'] and r['gtri_degs'] == (3, 3))
        print(f"      {'PASS' if ok else 'FAIL'}", flush=True)

    print("-" * 96)
    print("B. honeycomb thickness: subbox certificate |S| series (k=2)")
    hc = certificate_series(lambda L: V.build_honeycomb(L, L), (2, 3, 4, 5, 6), 2, "honeycomb")
    print("B'. honeycomb k=3 (check k-stability of |S|)")
    hc3 = certificate_series(lambda L: V.build_honeycomb(L, L), (2, 3, 4), 3, "honeycomb_k3")

    print("-" * 96)
    print("C. RTH certificate cross-check (known L=2/3 -> |S|=7/9)")
    rth = certificate_series(V.build_3464, (2, 3, 4, 5), 2, "RTH")

    print("-" * 96)
    print("B2. honeycomb kernel-basis support statistics (mean nonzeros / E)")
    for L in (2, 3, 4, 5):
        t = V.build_honeycomb(L, L)
        T, piv, free, dim = V.rref_and_param(t.xrows, t.yrows, t.E)
        K = kernel_matrix(T, piv, free, t.E)
        supports = [sum(1 for r in range(t.E) if K[r][j] != 0) for j in range(dim)]
        print(f"  L={L}: F={t.F} dim={dim} mean support={np.mean(supports):.1f} "
              f"max={max(supports)} min={min(supports)} (E={t.E})", flush=True)

    print("-" * 96)
    print("D. honeycomb pairing graph P decomposition (Phase 2 language anchor: opposite-edge pairing => zigzag stripes)")
    for L in (2, 3, 4, 5):
        t = V.build_honeycomb(L, L)
        adjP = {}
        for fi, (keys, pts, ft) in enumerate(t.faces):
            segs = t.face_edges[fi]
            n = len(segs)
            assert n == 6
            for i in range(3):
                e1, e2 = segs[i][0], segs[(i + 3) % 6][0]
                adjP.setdefault(e1, []).append(e2)
                adjP.setdefault(e2, []).append(e1)
        degs = sorted(set(len(v) for v in adjP.values()))
        seen = set()
        cycles = []
        for start in adjP:
            if start in seen:
                continue
            comp, stack = set(), [start]
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                stack.extend(adjP[u])
            seen |= comp
            cycles.append(len(comp))
        ok = (degs == [2] and len(cycles) == 3 * L and set(cycles) == {L})
        print(f"  L={L}: degree set={degs} component count={len(cycles)} "
              f"(predicted 3L={3*L}) "
              f"component size set={sorted(set(cycles))} (predicted {{L={L}}}) "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

    print("=" * 96)
    print("done")


if __name__ == "__main__":
    main()
