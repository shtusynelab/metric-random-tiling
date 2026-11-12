# -*- coding: utf-8 -*-
"""4.6.12 smoke v4 (numeric-geometry route): p6m incenter orbit -> real
geometry -> direction buckets -> three ranks.

Key fix (v3 hand-derived direction buckets failed): a 4.6.12 (omnitruncated
{6,3}) vertex is the orbit of the **incenter** of the p6m fundamental
triangle. The incenter is equidistant from the three mirrors, hence its
three mirror images are equidistant, so the Archimedean equal-edge
condition holds automatically (a generic point does not work; this was the
root cause of the v3 non-equidistant orbit neighbors).

Mirrors: A: y=0; B: through the origin at 30 deg; C: x=1. Fundamental
triangle (0,0),(1,0),(1,tan(30 deg)); incenter = ((3+sqrt(3))/6,
(3-sqrt(3))/6), exact in Q(sqrt(3))^2; inradius r = (3-sqrt(3))/6,
edge length = 2r. Translation lattice t1=(2,0), t2=(1,sqrt(3)) (= 2x the
rot180 center difference; verified). The three neighbor directions are the
three mirror normals: 0/120/270 deg => vertex angular gaps 90/120/150 =
faces 4/6/12 (checked).

Pipeline: orbit BFS -> nearest-neighbor edges (=2r, degree 3) -> torus
quotient (lattice site + local residual) -> half-edge face walk (face
types 4/6/12 with counts 3:2:1 + Euler check) -> direction buckets
(atan2 / 30 deg, in-face walk step +-12/n self-check) -> closure matrix
(entries (alpha + beta*sqrt(3))/2, scaled 2x to integers) -> three ranks:
  dim_R (float SVD) / dim_Q(sqrt(3)) (2x2 rational block embedding +
  Fraction elimination) / dim_split (alpha/beta row split).
Read: dim_R = F+2 extensive? dim_split = O(L) => Z entropy dead (s=0),
realizing the Niven-type prediction.

Memory budget: L=3 orbit ~600 points; block matrix 4F x 2E = 216 x 324
Fractions < 25 MB; whole script < 120 MB locally (L=4 ~1 h pure-Python
elimination, optional).
"""
import math
import sys
import time
from collections import defaultdict
from fractions import Fraction

import numpy as np

SQ3 = math.sqrt(3.0)
T1 = np.array([2.0, 0.0])
T2 = np.array([1.0, SQ3])
# incenter (exact): ((3+sqrt(3))/6, (3-sqrt(3))/6)
P_STAR = np.array([(3 + SQ3) / 6, (3 - SQ3) / 6])
EDGE = (3 - SQ3) / 3          # 2r

M_A = np.array([[1.0, 0.0], [0.0, -1.0]])
M_B = np.array([[0.5, SQ3 / 2], [SQ3 / 2, -0.5]])


def reflect(p):
    return [M_A @ p, M_B @ p, np.array([2.0 - p[0], p[1]])]



# exact (alpha, beta) pairs for the 30-degree buckets
# (value = (alpha + beta*sqrt(3))/2; hard-coded table, do not extract
# numerically: round(2c) on a sqrt(3)/2-type value gives (2,0) instead of
# (0,1), destroying the sqrt(3) information)
PAIR_COS = [(2, 0), (0, 1), (1, 0), (0, 0), (-1, 0), (0, -1),
            (-2, 0), (0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]
PAIR_SIN = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 1), (1, 0),
            (0, 0), (-1, 0), (0, -1), (-2, 0), (0, -1), (-1, 0)]


def fund_reduce(p, Lx, Ly, T1p, T2p):
    """Fundamental-domain floor reduction (gauge fixed, representation
    independent): p = mm*T1p + nn*T2p + rm, mm=floor(alpha+1e-9),
    nn=floor(beta+1e-9); returns (mm, nn, rm)."""
    A = np.array([[T1p[0], T2p[0]], [T1p[1], T2p[1]]])
    ab = np.linalg.solve(A, p)
    mm = math.floor(ab[0] + 1e-9)
    nn = math.floor(ab[1] + 1e-9)
    rm = p - mm * T1p - nn * T2p
    return mm, nn, rm


def build_orbit(Lx, Ly, margin=1.5):
    xmax = Lx * 2 + Ly * 1 + margin
    ymax = Ly * SQ3 + margin
    seeds = [P_STAR + m * T1 + n * T2
             for m in range(-2, Lx + 3) for n in range(-2, Ly + 3)]
    seen, pts = set(), {}
    frontier = list(seeds)
    for _ in range(300):
        if not frontier:
            break
        new = []
        for p in frontier:
            key = (round(p[0], 9), round(p[1], 9))
            if key in seen:
                continue
            seen.add(key)
            if -margin <= p[0] <= xmax and -margin <= p[1] <= ymax:
                pts[key] = p
            new.extend(reflect(p))
        frontier = new
    return pts


def build_torus(Lx, Ly):
    pts = build_orbit(Lx, Ly)
    keys = list(pts.keys())
    pl = np.array([pts[k] for k in keys])
    n = len(pl)
    # nearest neighbor = 2r (equal-edge; known a priori, threshold directly)
    thr = EDGE + 1e-6
    edges_raw = []
    for i in range(n):
        d = np.sqrt(((pl[i + 1:] - pl[i]) ** 2).sum(-1))
        for jj in np.flatnonzero(d < thr):
            edges_raw.append((keys[i], keys[i + 1 + jj]))
    # vertex gauge identity: lattice site + local residual
    # (invariant across translation images)
    T1p, T2p = Lx * T1, Ly * T2
    def vid_of(key):
        p = pts[key]
        mm, nn, r = fund_reduce(p, Lx, Ly, T1p, T2p)
        # torus quotient: vertex identity = fundamental-domain residual
        # alone (each vertex has a copy in every cell)
        return (round(r[0], 6), round(r[1], 6))
    vid = {k: vid_of(k) for k in keys}
    V = len(set(vid.values()))
    edges, eset = [], set()
    for ka, kb in edges_raw:
        u, v = vid[ka], vid[kb]
        if u == v:
            continue
        dvec = pts[kb] - pts[ka]
        ang = math.degrees(math.atan2(dvec[1], dvec[0])) % 360
        b = int(round(ang / 30)) % 12
        diff = abs(ang - 30 * b) % 360
        assert min(diff, 360 - diff) < 1e-4, (ka, kb, ang)
        # edge orbit key: endpoint vid pair + midpoint torus translation
        # class (canonical minimal-image reduction; candidate window +-3
        # covers lattice shifts of window ghosts; ties broken
        # lexicographically on (mm0, nn0))
        mid = (pts[ka] + pts[kb]) / 2
        mm0, nn0, rm = fund_reduce(mid, Lx, Ly, Lx * T1, Ly * T2)
        mc = (round(rm[0], 6), round(rm[1], 6))   # midpoint fundamental-domain residual = edge orbit class
        k1 = (u, v, b, mc)
        k2 = (v, u, (b + 6) % 12, mc)
        if k1 not in eset and k2 not in eset:
            eset.add(k1)
            eset.add(k2)
            edges.append((u, v, b))
    # degree self-check: exactly 3 per vertex
    deg = defaultdict(int)
    for (u, v, b) in edges:
        deg[u] += 1
        deg[v] += 1
    assert set(deg.values()) == {3} and len(deg) == V, \
        (Counter(deg.values()) if False else sorted(set(deg.values())), V, len(deg))
    return vid, edges, V, len(edges), pts


from collections import Counter


def extract_faces(edges):
    adj = {}
    for i, (u, v, b) in enumerate(edges):
        adj.setdefault(u, []).append(((b + 6) % 12, v, i))
        adj.setdefault(v, []).append((b, u, i))
    for v, lst in adj.items():
        assert len(lst) == 3
        lst.sort()
    visited, faces = set(), []
    for i, (u, v, b) in enumerate(edges):
        for start in [(u, v, (b + 6) % 12, i), (v, u, b, i)]:
            if start in visited:
                continue
            face = []
            cur = start
            while cur not in visited:
                visited.add(cur)
                face.append(cur)
                uu, vv, bdir, eid = cur
                rev = (bdir + 6) % 12
                lst = adj[vv]
                idx = [k for k, (bb, _, _) in enumerate(lst) if bb == rev][0]
                nb = lst[(idx - 1) % 3]          # left-hand face
                cur = (vv, nb[1], nb[0], nb[2])
            faces.append(face)
    return faces


def rank_fraction(rows, ncols):
    rows = [r[:] for r in rows]
    rk = 0
    for col in range(ncols):
        piv = next((i for i in range(rk, len(rows)) if rows[i][col] != 0), None)
        if piv is None:
            continue
        rows[rk], rows[piv] = rows[piv], rows[rk]
        pv = rows[rk][col]
        if pv != 1:
            rows[rk] = [x / pv for x in rows[rk]]
        for i in range(len(rows)):
            if i != rk and rows[i][col] != 0:
                fac = rows[i][col]
                rows[i] = [a - fac * b for a, b in zip(rows[i], rows[rk])]
        rk += 1
        if rk == len(rows):
            break
    return rk


def analyze(Lx, Ly):
    t0 = time.time()
    vid, edges, V, E, pts = build_torus(Lx, Ly)
    faces = extract_faces(edges)
    F = len(faces)
    sizes = [len(f) for f in faces]
    assert V - E + F == 0, (V, E, F)
    N = Lx * Ly
    assert sizes.count(4) == 3 * N and sizes.count(6) == 2 * N \
        and sizes.count(12) == N, (sizes.count(4), sizes.count(6), sizes.count(12))
    # per-face walk-step self-check + direction-bucket direct read
    # (half-edges already carry buckets; walk step = +-12/n)
    for f in faces:
        bs = [e[2] for e in f]
        n = len(bs)
        ext = {4: 3, 6: 2, 12: 1}[n]
        steps = {(bs[(k + 1) % n] - bs[k]) % 12 for k in range(n)}
        assert steps in ({ext % 12}, {(12 - ext) % 12}), (n, steps)
    eindex = {}
    for i, (u, v, b) in enumerate(edges):
        eindex[(u, v)] = i
        eindex[(v, u)] = i
    # closure matrix: face = half-edge sequence; the half-edge direction
    # bucket bdir is the directed walk direction
    M_pair = []
    for f in faces:
        rc = [[0, 0] for _ in range(E)]
        rs = [[0, 0] for _ in range(E)]
        for (uu, vv, bdir, eid) in f:
            ac, bc = PAIR_COS[bdir]
            as_, bs = PAIR_SIN[bdir]
            rc[eid][0] += ac
            rc[eid][1] += bc
            rs[eid][0] += as_
            rs[eid][1] += bs
        M_pair.append(rc)
        M_pair.append(rs)
    # float matrix uses true cos/sin (not the (alpha, beta) pairs)
    Mf_rows = []
    for f in faces:
        rc = [0.0] * E
        rs = [0.0] * E
        for (uu, vv, bdir, eid) in f:
            rc[eid] += math.cos(math.radians(30 * bdir))
            rs[eid] += math.sin(math.radians(30 * bdir))
        Mf_rows.append(rc)
        Mf_rows.append(rs)
    Mf = np.array(Mf_rows, float)
    sv = np.linalg.svd(Mf, compute_uv=False)
    rank_R = int((sv > sv.max() * max(Mf.shape) * 1e-10).sum())
    block = []
    for row in M_pair:
        ra, rb = [], []
        for (a, b) in row:
            ra += [a, 3 * b]
            rb += [b, a]
        block.append(ra)
        block.append(rb)
    t1 = time.time()
    rank_block = rank_fraction([[Fraction(x, 2) for x in r] for r in block], 2 * E)
    t2 = time.time()
    split = []
    for row in M_pair:
        split.append([Fraction(a, 2) for (a, b) in row])
        split.append([Fraction(b, 2) for (a, b) in row])
    rank_split = rank_fraction(split, E)
    # float split-rank crosscheck (fast path for large E)
    Aa = np.array([[float(a) for (a, b) in row] for row in M_pair], float)
    Ab = np.array([[float(b) for (a, b) in row] for row in M_pair], float)
    sv2 = np.linalg.svd(np.vstack([Aa, Ab]), compute_uv=False)
    rank_split_fl = int((sv2 > sv2.max() * 4 * E * 1e-10).sum())
    assert rank_split_fl == rank_split, (rank_split_fl, rank_split)
    t3 = time.time()
    return dict(Lx=Lx, Ly=Ly, N=N, V=V, E=E, F=F,
                dim_R=E - rank_R, dim_Qs3=E - rank_block // 2,
                dim_split=E - rank_split,
                t_build=round(t1 - t0, 1), t_block=round(t2 - t1, 1),
                t_split=round(t3 - t2, 1))


if __name__ == '__main__':
    out = []
    for L in [2, 3, 4]:
        r = analyze(L, L)
        out.append(r)
        print(f"L={L}: V={r['V']} E={r['E']} F={r['F']}  "
              f"dim_R={r['dim_R']}  dim_Qs3={r['dim_Qs3']}  dim_split={r['dim_split']}  "
              f"(build {r['t_build']}s, block {r['t_block']}s, split {r['t_split']}s)",
              flush=True)
    print()
    for r in out:
        print(f"L={r['Lx']}: dim_R/F = {r['dim_R']/r['F']:.4f}   "
              f"dim_Qs3/dim_R = {r['dim_Qs3']/max(r['dim_R'],1):.3f}   "
              f"dim_split = {r['dim_split']} (E={r['E']}, F={r['F']})")
    import csv
    with open('_temp/L1_review/RTH4612_smoke_dims.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print('saved _temp/L1_review/RTH4612_smoke_dims.csv')
