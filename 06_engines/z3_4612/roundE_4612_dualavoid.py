# -*- coding: utf-8 -*-
"""Decisive experiment for the dual-avoiding construction: enumerate all
binary solutions of the 2xL torus (L=2,3) and count how many touch
neither the T1p circle classes nor the T2p circle classes. If the count
grows like 2^{cL}, the wide-torus construction (independent strips each
taking an avoiding solution) gives s_01(inf) >= c ln2 / 12 without any
pinning theorem.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import math
from collections import deque, defaultdict
from itertools import product

import numpy as np

import roundE_4612_pinned_dp_t2p as T
import roundE_4612_pinned_dp as P1
import roundD_4612_numeric as D
from roundE_4612_frontier_dp import build_face_system, face_bfs_order


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


def find_circle(vid, edges, pts, nbr, shift):
    pts_arr = {k: np.array(v) for k, v in pts.items()}
    key_of = {tuple(np.round(v, 9)): k for k, v in pts_arr.items()}
    good = [k for k in pts_arr
            if tuple(np.round(pts_arr[k] + shift, 9)) in key_of]
    for p0 in good[::max(1, len(good) // 40)]:
        goal = key_of.get(tuple(np.round(pts_arr[p0] + shift, 9)))
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
                ok, comps = P1.separation_ok(edges, cand)
                if ok:
                    return cand
    return None


def collect_solutions(face_eqs, n_cls, face_nbrs):
    """Greedy-order DFS collecting all {0,1} solutions (per-equation)."""
    eqs = [(fi, ki, dict(eq)) for fi, eqs_f in enumerate(face_eqs)
           for ki, eq in enumerate(eqs_f)]
    eq_of = defaultdict(list)
    for fi, ki, eq in eqs:
        for c, co in eq.items():
            eq_of[c].append((fi, ki, eq, co))
    # greedy plan order (massive pruning); naive range order made the
    # 2x2 enumeration run for hours without finishing
    cls_faces = defaultdict(list)
    for fi, eqs_f in enumerate(face_eqs):
        for eq in eqs_f:
            for c in eq:
                cls_faces[c].append(fi)
    face_order = face_bfs_order(face_nbrs, len(face_eqs))
    pos = {f: i for i, f in enumerate(face_order)}
    last_pos = {c: max(pos[f] for f in fl) for c, fl in cls_faces.items()}
    frontier, assigned, plan = [], set(), []
    for step, f in enumerate(face_order):
        f_cls = sorted({c for eq in face_eqs[f] for c in eq})
        new = [c for c in f_cls if c not in assigned]
        new.sort(key=lambda c: -last_pos[c])
        out = [c for c in frontier + new if last_pos[c] > step]
        plan.append((frontier, new, out, f, face_eqs[f]))
        assigned.update(new)
        frontier = out
    order = [c for (_, new, _, _, _) in plan for c in new]
    remaining = {(fi, ki): set(eq.keys()) for fi, ki, eq in eqs}
    parts = {}
    sols = []
    sys.setrecursionlimit(100000)

    def dfs(d):
        if d == n_cls:
            sols.append(frozenset(c for c, v in sols_d.items() if v))
            return
        c = order[d]
        for v in (0, 1):
            sols_d[c] = v
            touched = eq_of.get(c, [])
            ok = True
            done = []
            for fi, ki, eq, co in touched:
                key = (fi, ki)
                parts[key] = parts.get(key, 0) + co * v
                remaining[key].discard(c)
                if not remaining[key]:
                    done.append(key)
            for key in done:
                if parts[key] != 0:
                    ok = False
                    break
            if ok:
                dfs(d + 1)
            for fi, ki, eq, co in touched:
                key = (fi, ki)
                parts[key] = parts.get(key, 0) - co * v
                remaining[key].add(c)
            del sols_d[c]

    sols_d = {}
    parts = {}
    dfs(0)
    return sols


def main():
    for L in (2, 3):
        margin = max(6.0, L * 2.5)
        vid, edges, E, pts, raw2q, nbr = build_quotient_big(2, L, margin)
        (face_eqs, n_cls, n_free, N, dim, all_eq_cls, face_nbrs,
         edge_cls) = build_face_system(2, L, rank_check=False)
        C1p = find_circle(vid, edges, pts, nbr, 2 * D.T1)
        C2p = find_circle(vid, edges, pts, nbr, L * D.T2)
        P1c = set(edge_cls[qi] for qi in C1p)
        P2c = set(edge_cls[qi] for qi in C2p)
        print(f'=== 2x{L}: T1p circle {len(C1p)}e/{len(P1c)}c, '
              f'T2p circle {len(C2p)}e/{len(P2c)}c, overlap '
              f'{len(P1c & P2c)} ===', flush=True)
        sols = collect_solutions(face_eqs, n_cls, face_nbrs)
        print(f'    solutions (eq vars): {len(sols)}', flush=True)
        cnt = {'none': 0, 't1only': 0, 't2only': 0, 'both': 0}
        for s in sols:
            v1 = bool(s & P1c)
            v2 = bool(s & P2c)
            key = 'both' if (v1 and v2) else ('t1only' if v1 else
                                              ('t2only' if v2 else 'none'))
            cnt[key] += 1
        print(f'    avoidance: {cnt}', flush=True)
        avoid = cnt['none']
        if avoid:
            print(f'    ln(avoid)/L = {math.log(avoid) / L:.4f} '
                  f'(c ln2 target needs this to persist)', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
