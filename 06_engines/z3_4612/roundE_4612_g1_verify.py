# -*- coding: utf-8 -*-
"""Geometry extraction for the flip-pair existence argument (alpha) and
the G1 multi-circle verification (gap 1).

(alpha) For 2x2, T2p-pinned: classes 18,26 form a Lemma-F pair. Print
  their edges, adjacent faces (must be exactly two duodecagons with
  opposite coefficients), the pinned-circle classes (must be disjoint).
(G1) For 4x2: find two parallel T2p-homotopic edge circles separated by
  2 columns; check disjointness and that deleting both splits the
  quotient graph into 2 components of the right sizes (two cylinders).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from collections import deque, defaultdict
import roundE_4612_pinned_dp as P
import roundD_4612_numeric as D
from roundE_4612_frontier_dp import build_face_system


def build_quotient_big(Lx, Ly, margin):
    pts = D.build_orbit(Lx, Ly, margin=margin)
    mpts = {}
    for k, v in pts.items():
        mpts.setdefault((round(v[0], 7), round(v[1], 7)),
                        (float(v[0]), float(v[1])))
    pts = {k: np.array(v) for k, v in mpts.items()}
    keys = list(pts.keys())
    pl = np.array([pts[k] for k in keys])
    raw = []
    thr = D.EDGE + 1e-6
    for i in range(len(pl)):
        d = np.sqrt(((pl[i + 1:] - pl[i]) ** 2).sum(-1))
        for jj in np.flatnonzero(d < thr):
            raw.append((keys[i], keys[i + 1 + jj]))
    T1p, T2p = Lx * D.T1, Ly * D.T2

    def vid_of(key):
        _, _, r = D.fund_reduce(pts[key], Lx, Ly, T1p, T2p)
        return (round(r[0], 6), round(r[1], 6))

    vid = {k: vid_of(k) for k in keys}
    edges, raw2q, ekeys = [], {}, {}
    for ka, kb in raw:
        u, v = vid[ka], vid[kb]
        if u == v:
            continue
        dv = pts[kb] - pts[ka]
        b = int(round((np.degrees(np.arctan2(dv[1], dv[0])) % 360) / 30)) % 12
        mid = (pts[ka] + pts[kb]) / 2
        _, _, rm = D.fund_reduce(mid, Lx, Ly, Lx * D.T1, Ly * D.T2)
        mc = (round(rm[0], 6), round(rm[1], 6))
        k1, k2 = (u, v, b, mc), (v, u, (b + 6) % 12, mc)
        if k1 in ekeys:
            q = ekeys[k1]
        elif k2 in ekeys:
            q = ekeys[k2]
        else:
            q = len(edges)
            ekeys[k1] = q
            ekeys[k2] = q
            edges.append((u, v, b))
        raw2q[(ka, kb)] = q
        raw2q[(kb, ka)] = q
    nbr = {}
    for (ka, kb), qi in raw2q.items():
        nbr.setdefault(ka, []).append((kb, qi))
    return vid, edges, len(edges), pts, raw2q, nbr


def bfs_circle(edges, pts, vid, nbr, shift, margin):
    """Find one T2p-homotopic simple circle via cover BFS p0 -> p0+shift."""
    pts_arr = {k: np.array(v) for k, v in pts.items()}
    key_of = {tuple(np.round(v, 9)): k for k, v in pts_arr.items()}
    per = shift
    good = [k for k in pts_arr
            if tuple(np.round(pts_arr[k] + per, 9)) in key_of]
    if not good:
        return None, None
    p0 = good[len(good) // 2]
    goal = key_of.get(tuple(np.round(pts_arr[p0] + per, 9)))
    prev = {p0: None}
    queue = deque([p0])
    found = False
    while queue:
        p = queue.popleft()
        if p == goal:
            found = True
            break
        for (p2, qi) in nbr.get(p, ()):
            if p2 not in prev:
                prev[p2] = (p, qi)
                queue.append(p2)
    if not found:
        return None, None
    path = []
    p = goal
    while prev[p] is not None:
        pp, qi = prev[p]
        path.append((pp, qi))
        p = pp
    path.reverse()
    seq = [pp for (pp, _) in path] + [goal]
    qes = [qi for (_, qi) in path]
    vseq = [vid[p] for p in seq]
    n = len(seq)
    for i in range(n - 1):
        for j in range(i + 2, n):
            if vseq[i] != vseq[j]:
                continue
            if len(set(vseq[i + 1:j])) != j - i - 1:
                continue
            disp = np.array(pts[seq[j]]) - np.array(pts[seq[i]])
            if np.allclose(disp, 0, atol=1e-9):
                continue
            cand = list(qes[i:j])
            ok, comps = P.separation_ok(edges, cand)
            if ok:
                return cand, seq
    return None, None


def alpha_part():
    print('=== (alpha) flip-pair geometry, 2x2 T2p pinning ===', flush=True)
    vid, edges, E, pts, raw2q, nbr = build_quotient_big(2, 2, margin=1.5)
    (face_eqs, n_cls, n_free, N, dim, all_eq_cls, face_nbrs,
     edge_cls) = build_face_system(2, 2, rank_check=False)
    # face sizes (which faces are duodecagons)
    # rebuild face->edges to size faces
    from roundD_4612_numeric import extract_faces
    faces = extract_faces(edges)
    size_of = [len(f) for f in faces]
    circ, seq = bfs_circle(edges, pts, vid, nbr, 2 * D.T2, 1.5)
    assert circ, 'no T2p circle'
    P_cls = sorted({edge_cls[qi] for qi in circ})
    print(f'T2p circle: {len(circ)} edges, pinned classes P ({len(P_cls)}): '
          f'{P_cls}', flush=True)
    for c in (18, 26):
        es = [qi for qi in range(E) if edge_cls[qi] == c]
        print(f'class {c}: edges {es}', flush=True)
        adj_faces = set()
        for qi in es:
            for fi, f in enumerate(faces):
                if any(h[3] == qi for h in f):
                    adj_faces.add((fi, size_of[fi]))
        print(f'   adjacent faces: {sorted(adj_faces)}', flush=True)
        print(f'   in pinned P? {c in set(P_cls)}', flush=True)
    # equations containing 18/26
    for fi, eqs_f in enumerate(face_eqs):
        for ki, eq in enumerate(eqs_f):
            if 18 in eq or 26 in eq:
                print(f'face {fi} (size {size_of[fi]}) eq {ki}: '
                      f'{dict(sorted(eq.items()))}', flush=True)
    # T2-translate: classes of edges shifted by +T2
    pts_arr = {k: np.array(v) for k, v in pts.items()}
    key_of = {tuple(np.round(v, 9)): k for k, v in pts_arr.items()}
    shifted_cls = {}
    for qi in range(E):
        u, v, b = edges[qi]
        # recover a raw pair for this qedge: search nbr for qi
        raws = [(ka, kb) for (ka, kb), q in raw2q.items() if q == qi]
        ka, kb = raws[0]
        if ka in key_of and tuple(np.round(pts_arr[ka] + D.T2, 9)) in key_of \
                and tuple(np.round(pts_arr[kb] + D.T2, 9)) in key_of:
            ka2 = key_of[tuple(np.round(pts_arr[ka] + D.T2, 9))]
            kb2 = key_of[tuple(np.round(pts_arr[kb] + D.T2, 9))]
            shifted_cls[qi] = raw2q[(ka2, kb2)]
    print(f'class 18 edges {sorted(qi for qi in range(E) if edge_cls[qi]==18)}'
          f' -> T2-shifted qedges '
          f'{sorted(shifted_cls.get(qi, -1) for qi in range(E) if edge_cls[qi]==18)}',
          flush=True)
    print(f'  shifted classes: '
          f'{sorted({edge_cls[v] for k, v in shifted_cls.items() if edge_cls[k]==18})}',
          flush=True)


def g1_part():
    print('=== (G1) parallel circle pair, 4x2 ===', flush=True)
    vid, edges, E, pts, raw2q, nbr = build_quotient_big(4, 2, margin=5.0)
    from roundD_4612_numeric import extract_faces
    faces = extract_faces(edges)
    V = len({v for (u, v, b) in edges} | {u for (u, v, b) in edges})
    print(f'E={E} V={V} faces={len(faces)} (expect V=96 E=144)', flush=True)
    C0, seq = bfs_circle(edges, pts, vid, nbr, 2 * D.T2, 5.0)
    assert C0, 'no C0'
    ok0, comps0 = P.separation_ok(edges, C0)
    print(f'C0: {len(C0)} edges, non-separating={ok0} (comps {comps0})',
          flush=True)
    # C1 = C0 translated by 2*T1 (two columns) in the cover
    pts_arr = {k: np.array(v) for k, v in pts.items()}
    key_of = {tuple(np.round(v, 9)): k for k, v in pts_arr.items()}
    path_pts = seq[:-1]
    shift_ok = all(tuple(np.round(pts_arr[p] + 2 * D.T1, 9)) in key_of
                   for p in path_pts)
    if not shift_ok:
        print('C1 translation out of window', flush=True)
        return
    C1 = []
    for i in range(len(path_pts)):
        ka = path_pts[i]
        kb = seq[i + 1]
        ka2 = key_of[tuple(np.round(pts_arr[ka] + 2 * D.T1, 9))]
        kb2 = key_of[tuple(np.round(pts_arr[kb] + 2 * D.T1, 9))]
        C1.append(raw2q[(ka2, kb2)])
    disjoint = not (set(C0) & set(C1))
    print(f'C1: {len(C1)} edges, disjoint from C0: {disjoint}', flush=True)
    # delete both circles: expect 2 components (two cylinders), sizes
    cut = set(C0) | set(C1)
    adj = defaultdict(list)
    for i, (u, v, b) in enumerate(edges):
        if i in cut:
            continue
        adj[u].append(v)
        adj[v].append(u)
    seen = set()
    comps = []
    for v0 in adj:
        if v0 in seen:
            continue
        queue = deque([v0])
        seen.add(v0)
        comp = [v0]
        while queue:
            x = queue.popleft()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    queue.append(y)
                    comp.append(y)
        comps.append(len(comp))
    print(f'delete both: components={len(comps)} sizes={sorted(comps)} '
          f'(expect 2 comps of V/2={V // 2})', flush=True)


if __name__ == '__main__':
    g1_part()
