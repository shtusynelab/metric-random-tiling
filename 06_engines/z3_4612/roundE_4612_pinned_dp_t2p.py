# -*- coding: utf-8 -*-
"""4.6.12 T2p-pinned n0(2,L) series (binary {0,1}), self-contained.

Direction-corrected pinning: the circle is T2p-homotopic (shift L*T2,
wrapping the LONG direction) -- this matches the stitching theorem's
cylinder-band boundary. Everything (quotient graph, face system, circle
search) is built from ONE cover realization at a single enlarged margin,
so class ids are consistent by construction (no cross-margin bridging).

Checks per L (fail-closed):
  X-dim: face-system dim = 4N+4 (Fraction RREF, L<=4 only)
  X-sep: circle non-separating (delete -> 1 component)
  X-mono: n0(2,L) <= Omega(2,L) from strips CSV
  X-exp: n0(2,L) >= 2^L  (the constructive-bound target; reported, and
         asserted from L=2..14 since the theorem predicts it)

Output: RTH4612_binary_pinned_t2p.csv
"""
import csv
import math
import os
import sys
import time
from collections import defaultdict, deque
from fractions import Fraction
from itertools import product

import numpy as np

import roundD_4612_numeric as D
from roundD_4612_numeric import extract_faces, PAIR_COS, PAIR_SIN

HERE = os.path.dirname(os.path.abspath(__file__))
STRIPS_CSV = os.path.join(HERE, 'RTH4612_binary_strips.csv')
OUT_CSV = os.path.join(HERE, 'RTH4612_binary_pinned_t2p.csv')
PEAK_GATE = 2 * 10 ** 8


def build_cover(Lx, Ly, margin):
    """Cover realization: orbit points (merged), quotient graph with
    full raw2q bookkeeping, cover adjacency."""
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


def build_face_system_from(edges, Lx, Ly, rank_check):
    """Face-attributed reduced system from the given quotient edges."""
    faces = extract_faces(edges)
    N = Lx * Ly
    E = len(edges)
    assert len(faces) == 6 * N and E == 18 * N, (len(faces), E, 6 * N, 18 * N)
    edge_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for h in f:
            edge_faces[h[3]].append(fi)
    size_of = [len(f) for f in faces]
    etype = {}
    for eid, (f1, f2) in edge_faces.items():
        s1, s2 = sorted((size_of[f1], size_of[f2]))
        etype[eid] = {(4, 12): 'SD', (4, 6): 'SH', (6, 12): 'HD'}[(s1, s2)]
    parent = list(range(E))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for f in faces:
        if len(f) == 4:
            for i in (0, 1):
                union(f[i][3], f[(i + 2) % 4][3])
    cls_of = {}
    for eid in range(E):
        r = find(eid)
        if r not in cls_of:
            cls_of[r] = len(cls_of)
    n_cls = len(cls_of)
    assert n_cls == 12 * N, (n_cls, 12 * N)
    edge_cls = [cls_of[find(eid)] for eid in range(E)]
    face_eqs = []
    for f in faces:
        rows = [defaultdict(int) for _ in range(4)]
        for (uu, vv, bdir, eid) in f:
            c = cls_of[find(eid)]
            ac, bc = PAIR_COS[bdir]
            as_, bs = PAIR_SIN[bdir]
            rows[0][c] += ac
            rows[1][c] += bc
            rows[2][c] += as_
            rows[3][c] += bs
        eqs_f = []
        for r in rows:
            r = {k: v for k, v in r.items() if v != 0}
            if r:
                eqs_f.append(dict(r))
        face_eqs.append(eqs_f)
    n_eqs = sum(len(e) for e in face_eqs)
    if Lx >= 2:
        assert n_eqs == 8 * N, (n_eqs, 8 * N)
    all_eq_cls = sorted({c for eqs_f in face_eqs for eq in eqs_f for c in eq})
    dim = None
    if rank_check:
        rows_f = [[Fraction(eq.get(c, 0)) for c in all_eq_cls]
                  for eqs_f in face_eqs for eq in eqs_f]
        rk = 0
        for col in range(len(all_eq_cls)):
            piv = next((i for i in range(rk, len(rows_f))
                        if rows_f[i][col] != 0), None)
            if piv is None:
                continue
            rows_f[rk], rows_f[piv] = rows_f[piv], rows_f[rk]
            pv = rows_f[rk][col]
            if pv != 1:
                rows_f[rk] = [x / pv for x in rows_f[rk]]
            for i in range(len(rows_f)):
                if i != rk and rows_f[i][col] != 0:
                    fac = rows_f[i][col]
                    rows_f[i] = [a - fac * b
                                 for a, b in zip(rows_f[i], rows_f[rk])]
            rk += 1
            if rk == len(rows_f):
                break
        dim = n_cls - rk
        assert dim == 4 * N + 4, (dim, 4 * N + 4)
    n_free = n_cls - len(all_eq_cls)
    face_nbrs = defaultdict(set)
    for eid, fl in edge_faces.items():
        f1, f2 = fl
        face_nbrs[f1].add(f2)
        face_nbrs[f2].add(f1)
    return (face_eqs, n_cls, n_free, N, all_eq_cls, edge_cls, face_nbrs)


def find_t2p_circle(vid, edges, pts, nbr, L, margin):
    pts_arr = {k: np.array(v) for k, v in pts.items()}
    key_of = {tuple(np.round(v, 9)): k for k, v in pts_arr.items()}
    T2p = L * D.T2
    good = [k for k in pts_arr
            if tuple(np.round(pts_arr[k] + T2p, 9)) in key_of]
    if not good:
        return None
    for p0 in good[::max(1, len(good) // 40)]:
        goal = key_of.get(tuple(np.round(pts_arr[p0] + T2p, 9)))
        if goal is None:
            continue
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
            continue
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
                cut = set(cand)
                adj = defaultdict(list)
                for ei, (u, v, b) in enumerate(edges):
                    if ei in cut:
                        continue
                    adj[u].append(v)
                    adj[v].append(u)
                seen2 = set()
                comps = 0
                for v0 in adj:
                    if v0 in seen2:
                        continue
                    comps += 1
                    q2 = deque([v0])
                    seen2.add(v0)
                    while q2:
                        x = q2.popleft()
                        for y in adj[x]:
                            if y not in seen2:
                                seen2.add(y)
                                q2.append(y)
                if comps == 1:
                    return cand
                if comps == 1:
                    return cand
    return None


def pinned_dp(face_eqs, n_cls, all_eq_cls, pinned, face_nbrs):
    cls_faces = defaultdict(list)
    for fi, eqs_f in enumerate(face_eqs):
        for eq in eqs_f:
            for c in eq:
                cls_faces[c].append(fi)
    # BFS on the TRUE face graph (edge-sharing adjacency): equation
    # co-occurrence would isolate square faces (no equations)
    order = []
    seen = {0}
    queue = deque([0])
    while queue:
        f = queue.popleft()
        order.append(f)
        for g in sorted(face_nbrs.get(f, ())):
            if g not in seen:
                seen.add(g)
                queue.append(g)
    assert len(order) == len(face_eqs)
    # plan (same as frontier_plan)
    pos = {f: i for i, f in enumerate(order)}
    last_pos = {c: max(pos[f] for f in fl) for c, fl in cls_faces.items()}
    frontier = []
    assigned = set()
    plan = []
    for step, f in enumerate(order):
        f_cls = sorted({c for eq in face_eqs[f] for c in eq})
        new = [c for c in f_cls if c not in assigned]
        new.sort(key=lambda c: -last_pos[c])
        out = [c for c in frontier + new if last_pos[c] > step]
        covered = set(frontier) | set(new)
        for eq in face_eqs[f]:
            assert set(eq) <= covered
        plan.append((list(frontier), new, out, f, face_eqs[f]))
        assigned.update(new)
        frontier = out
    assert not frontier
    state = {0: 1}
    peak = 1
    for step, (fr_in, new, out, f, eqs_f) in enumerate(plan):
        if not new and fr_in == out:
            continue
        k = len(new)
        choices = [((0,) if c in pinned else (0, 1)) for c in new]
        nstate = defaultdict(int)
        for key, cnt in state.items():
            val = {c: (key >> i) & 1 for i, c in enumerate(fr_in)}
            for combo in (product(*choices) if k else [()]):
                if k:
                    for c, v in zip(new, combo):
                        val[c] = v
                ok = True
                for eq in eqs_f:
                    if sum(co * val[c] for c, co in eq.items()) != 0:
                        ok = False
                        break
                if not ok:
                    continue
                nk = 0
                for i2, c in enumerate(out):
                    nk |= val[c] << i2
                nstate[nk] += cnt
        state = nstate
        peak = max(peak, len(state))
        if peak > PEAK_GATE:
            print('FAIL-CLOSED: peak states', peak, flush=True)
            sys.exit(2)
    assert set(state) == {0}
    return state[0], peak


def main():
    strips = {}
    if os.path.exists(STRIPS_CSV):
        with open(STRIPS_CSV) as f:
            for row in csv.DictReader(f):
                if row['W'] == '2' and row['Omega_01']:
                    strips[int(row['L'])] = int(row['Omega_01'])
    new = not os.path.exists(OUT_CSV)
    f = open(OUT_CSV, 'a', newline='')
    w = csv.writer(f)
    if new:
        w.writerow(['L', 'N', 'circle_len', 'pinned_cls', 'peak', 'DP_count',
                    'n0_full', 'ln_n0', 'ln_n0_over_L', 'Omega_ref',
                    'ge2L', 'time_s'])
    for L in range(2, 15):
        t0 = time.time()
        margin = max(6.0, L * 2.5)
        vid, edges, E, pts, raw2q, nbr = build_cover(2, L, margin)
        circ = find_t2p_circle(vid, edges, pts, nbr, L, margin)
        if circ is None:
            print(f'L={L}: FAIL no T2p circle', flush=True)
            w.writerow([L, 2 * L, '', '', '', 'FAIL', '', '', '', '', '',
                        f'{time.time() - t0:.0f}'])
            f.flush()
            continue
        (face_eqs, n_cls, n_free, N, all_eq_cls,
         edge_cls, face_nbrs) = build_face_system_from(edges, 2, L,
                                                       rank_check=(L <= 4))
        pinned = frozenset(edge_cls[qi] for qi in circ)
        dp, peak = pinned_dp(face_eqs, n_cls, all_eq_cls, pinned, face_nbrs)
        all_eq = set(all_eq_cls)
        pinned_free = len([c for c in pinned if c not in all_eq])
        n0 = dp * (2 ** (n_free - pinned_free))
        lam = math.log(n0) / L
        ref = strips.get(L, '')
        ge2L = n0 >= 2 ** L
        print(f'L={L}: circle={len(circ)} pinned={len(pinned)} '
              f'(free {pinned_free}) peak={peak} n0={n0} ln/L={lam:.4f} '
              f'ge2L={ge2L} (ref={ref}) [{time.time() - t0:.0f}s]',
              flush=True)
        assert not ref or n0 <= ref, ('pinned > unpinned', L)
        w.writerow([L, 2 * L, len(circ), len(pinned), peak, dp, n0,
                    f'{math.log(n0):.4f}', f'{lam:.6f}', ref,
                    'Y' if ge2L else 'N', f'{time.time() - t0:.0f}'])
        f.flush()
    f.close()
    print('T2P PINNED SERIES DONE', flush=True)


if __name__ == '__main__':
    main()
