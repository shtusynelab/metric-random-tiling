# -*- coding: utf-8 -*-
"""3.12.12 control v2 (pipeline validation): truncated honeycomb t = 2-sqrt(3),
true geometry without wrap-around.

Edge generation: each honeycomb edge is uniquely represented by (cell (i,j), k in
{0,1,2}); TRUE hole centers c(i,j) = sqrt(3)*(i + j/2, (sqrt(3)/2)*j), neighbor true
position = c + sqrt(3)*axial2cart(D_k) (no wrap-around); truncation points =
m +/- (1/2 - t)*v. Corner cuts: the 6 corners of hole (i,j) (between directions k and
k+1); each of the three adjacent honeycomb edges contributes the truncation point
closest to the corner, forming a triangle (duplicate generation deduplicated by eset
via (u,v,b,mc)).
Torus lattice T1' = sqrt(3)*Lx*(1,0), T2' = sqrt(3)*Ly*(1/2, sqrt(3)/2);
minimum-image canonical reduction.
"""
import math
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "z3_4612"))
from roundD_4612_numeric import rank_fraction, extract_faces, fund_reduce, \
    PAIR_COS, PAIR_SIN   # noqa

SQ3 = math.sqrt(3.0)
T = 2.0 - SQ3
DIRECTIONS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
AX2CART = [(1.0, 0.0), (0.5, SQ3 / 2), (-0.5, SQ3 / 2),
           (-1.0, 0.0), (-0.5, -SQ3 / 2), (0.5, -SQ3 / 2)]


def hole_center(i, j):
    return np.array([SQ3 * (i + j / 2), 1.5 * j])


def make_canon(Lx, Ly):
    T1p = SQ3 * Lx * np.array([1.0, 0.0])
    T2p = SQ3 * Ly * np.array([0.5, SQ3 / 2])

    def canon(p):
        mm, nn, rm = fund_reduce(p, Lx, Ly, T1p, T2p)
        # torus quotient: identity = fundamental-domain residual alone
        return (round(rm[0], 6), round(rm[1], 6))
    return canon


def build_31212(Lx, Ly):
    canon = make_canon(Lx, Ly)
    N = Lx * Ly
    vid, vlist = {}, []

    def vid_of(p):
        key = canon(p)
        if key not in vid:
            vid[key] = len(vlist)
            vlist.append(p)
        return vid[key]

    edges, eset = [], set()

    def edge_of(pa, pb):
        u, w = vid_of(pa), vid_of(pb)
        if u == w:
            return
        dvec = pb - pa
        ang = math.degrees(math.atan2(dvec[1], dvec[0])) % 360
        b = int(round(ang / 30)) % 12
        diff = abs(ang - 30 * b) % 360
        assert min(diff, 360 - diff) < 1e-4, ang
        mid = (pa + pb) / 2
        _, _, mc = _min_img(mid, Lx, Ly)
        k1 = (u, w, b, mc)
        k2 = (w, u, (b + 6) % 12, mc)
        if k1 not in eset and k2 not in eset:
            eset.add(k1)
            eset.add(k2)
            edges.append((u, w, b))

    def _min_img(p, Lx, Ly):
        T1p = SQ3 * Lx * np.array([1.0, 0.0])
        T2p = SQ3 * Ly * np.array([0.5, SQ3 / 2])
        mm, nn, rm = fund_reduce(p, Lx, Ly, T1p, T2p)
        return mm, nn, (round(rm[0], 6), round(rm[1], 6))

    off = 0.5 - T
    cuts = {}
    # honeycomb edges (cell, k<3): truncation point pair (true geometry, no wrap-around)
    for i in range(Lx):
        for j in range(Ly):
            for k in range(3):
                ca = hole_center(i, j)
                cb = ca + SQ3 * np.array(AX2CART[k])
                m = (ca + cb) / 2
                vc = (cb - ca) / np.linalg.norm(cb - ca)
                vs = np.array([-vc[1], vc[0]])             # shared-edge direction (perp. to center line)
                edge_of(m + off * vs, m - off * vs)        # middle segment (on the shared edge)
                cuts[(i, j, k)] = (m + off * vs, m - off * vs)
    for i in range(Lx):
        for j in range(Ly):
            c0 = hole_center(i, j)
            for k in range(6):
                # corner position = c0 + normalized unit angle bisector
                # (cart(D_k)+cart(D_{k+1}); hexagon circumradius = 1)
                bis = np.array(AX2CART[k % 6]) + np.array(AX2CART[(k + 1) % 6])
                corner = c0 + bis / np.linalg.norm(bis)
                # the three honeycomb edges at the corner: (h0,D_k), (h0,D_{k+1}), (hA,D_{k+2})
                trio = []
                for kk in [k % 6, (k + 1) % 6]:
                    ca = c0
                    cb = ca + SQ3 * np.array(AX2CART[kk])
                    m = (ca + cb) / 2
                    vc = (cb - ca) / np.linalg.norm(cb - ca)
                    vs = np.array([-vc[1], vc[0]])
                    best = min([(np.linalg.norm(p - corner), p) for p in
                                (m + off * vs, m - off * vs)], key=lambda x: x[0])
                    trio.append(best[1])
                # third edge: hA = c0 + sqrt(3)*cart(D_k), direction D_{k+2} (hB - hA = D_{k+1} - D_k)
                ha_cart = c0 + SQ3 * np.array(AX2CART[k % 6])
                cb3 = ha_cart + SQ3 * np.array(AX2CART[(k + 2) % 6])
                m3 = (ha_cart + cb3) / 2
                vc3 = (cb3 - ha_cart) / np.linalg.norm(cb3 - ha_cart)
                vs3 = np.array([-vc3[1], vc3[0]])
                best = min([(np.linalg.norm(p - corner), p) for p in
                            (m3 + off * vs3, m3 - off * vs3)], key=lambda x: x[0])
                trio.append(best[1])
                edge_of(trio[0], trio[1])
                edge_of(trio[1], trio[2])
                edge_of(trio[2], trio[0])
    deg = defaultdict(int)
    for (u, w, b) in edges:
        deg[u] += 1
        deg[w] += 1
    import os
    if os.environ.get('D31212_SOFT') != '1':
        assert set(deg.values()) == {3} and len(deg) == len(vid), \
            (sorted(set(deg.values())), len(deg), len(vid))
    return edges, len(vid), len(edges), vlist, dict(deg)


def analyze(Lx, Ly):
    t0 = time.time()
    edges, V, E, vlist, _deg = build_31212(Lx, Ly)
    faces = extract_faces(edges)
    F = len(faces)
    sizes = [len(f) for f in faces]
    N = Lx * Ly
    assert V - E + F == 0, (V, E, F)
    assert sizes.count(3) == 2 * N and sizes.count(12) == N, Counter(sizes)
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
    rank_block = rank_fraction([[Fraction(x, 2) for x in r] for r in block], 2 * E)
    split = []
    for row in M_pair:
        split.append([Fraction(a, 2) for (a, b) in row])
        split.append([Fraction(b, 2) for (a, b) in row])
    rank_split = rank_fraction(split, E)
    return dict(Lx=Lx, N=N, V=V, E=E, F=F, dim_R=E - rank_R,
                dim_Qs3=E - rank_block // 2, dim_split=E - rank_split,
                seconds=round(time.time() - t0, 1))


if __name__ == '__main__':
    for L in [2, 3]:
        r = analyze(L, L)
        print(f"3.12.12 L={L}: V={r['V']} E={r['E']} F={r['F']}  "
              f"dim_R={r['dim_R']} (F+2={r['F']+2})  dim_Qs3={r['dim_Qs3']}  "
              f"dim_split={r['dim_split']}  ({r['seconds']}s)", flush=True)
