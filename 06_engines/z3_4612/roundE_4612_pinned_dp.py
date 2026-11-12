# -*- coding: utf-8 -*-
"""4.6.12 pinned stitching: n0(2,L) series (binary {0,1}).

Theorem (pinned stitching, RTH4612_pinning_splice_theorem_derivation.md):
    Omega(W, L) >= n0(2, L)^floor(W/2)
hence  liminf_W lim_L ln Omega(W,L)/(6WL) >= lambda0*(2)/12,
    lambda0*(2) = limsup_L ln n0(2,L)/L.
Decisive quantity: lambda0*(2) > 0?

Pinned circle: a T1-homotopic NATIVE-EDGE circle (faces not crossed)
found by BFS on the universal cover from p0 to p0 + T1. Pinning: every
union-find class touching the circle image is forced to 0.

Fail-closed checks per L:
  X-sep: circle quotient image has pairwise distinct edges AND the
    quotient graph minus the circle has exactly 2 components (annulus).
  X-mono: n0(2,L) <= Omega(2,L) (from strips CSV).

Engine: frontier DP with pinned classes' enumeration domain {0}.
Output: RTH4612_binary_pinned.csv
"""
import csv
import math
import os
import sys
import time
from collections import defaultdict, deque
from itertools import product

import numpy as np

from roundE_4612_frontier_dp import (build_face_system, face_bfs_order,
                                     frontier_plan, PEAK_STATE_GATE as PEAK_GATE)
from roundD_4612_numeric import build_orbit, fund_reduce, extract_faces, \
    T1, T2, EDGE

HERE = os.path.dirname(os.path.abspath(__file__))
STRIPS_CSV = os.path.join(HERE, 'RTH4612_binary_strips.csv')
OUT_CSV = os.path.join(HERE, 'RTH4612_binary_pinned.csv')


# ---------------------------------------------------------------- geometry
def build_quotient(Lx, Ly):
    """Rebuild build_torus with raw-edge -> quotient-edge bookkeeping.

    Returns (vid, edges, V, E, pts, cls_of_edges, raw2q, nbr)
      edges      : list of (u, v, b)              quotient edges
      cls_of_edges: list of class id per quotient edge
      raw2q      : dict (ka, kb) -> quotient edge index
      nbr        : dict p -> list of (p2, qedge)  cover adjacency
    """
    pts = build_orbit(Lx, Ly, margin=1.5)
    # merge near-duplicate orbit points (fp noise from the orbit BFS):
    # two keys whose coordinates agree to <1e-7 are the same cover point;
    # nearest-neighbour spacing is EDGE ~ 1.155 so rounding at 7 digits
    # is safe and kills the zero-displacement ghost edges that otherwise
    # poison the cover BFS (diagnosed on 2x2: first BFS step was a ghost)
    mpts = {}
    for k, v in pts.items():
        mpts.setdefault((round(v[0], 7), round(v[1], 7)),
                        (float(v[0]), float(v[1])))
    pts = {k: np.array(v) for k, v in mpts.items()}
    keys = list(pts.keys())
    pl = np.array([pts[k] for k in keys])
    n = len(pl)
    thr = EDGE + 1e-6
    raw = []
    for i in range(n):
        d = np.sqrt(((pl[i + 1:] - pl[i]) ** 2).sum(-1))
        for jj in np.flatnonzero(d < thr):
            raw.append((keys[i], keys[i + 1 + jj]))
    T1p, T2p = Lx * T1, Ly * T2

    def vid_of(key):
        p = pts[key]
        mm, nn, r = fund_reduce(p, Lx, Ly, T1p, T2p)
        return (round(r[0], 6), round(r[1], 6))

    vid = {k: vid_of(k) for k in keys}
    edges, raw2q, edge_keys = [], {}, {}
    for ka, kb in raw:
        u, v = vid[ka], vid[kb]
        if u == v:
            continue
        dvec = pts[kb] - pts[ka]
        ang = math.degrees(math.atan2(dvec[1], dvec[0])) % 360
        b = int(round(ang / 30)) % 12
        mid = (pts[ka] + pts[kb]) / 2
        mm0, nn0, rm = fund_reduce(mid, Lx, Ly, Lx * T1, Ly * T2)
        mc = (round(rm[0], 6), round(rm[1], 6))
        k1 = (u, v, b, mc)
        k2 = (v, u, (b + 6) % 12, mc)
        if k1 in edge_keys:
            q = edge_keys[k1]
        elif k2 in edge_keys:
            q = edge_keys[k2]
        else:
            q = len(edges)
            edge_keys[k1] = q
            edge_keys[k2] = q
            edges.append((u, v, b))
        raw2q[(ka, kb)] = q
        raw2q[(kb, ka)] = q
    # union-find equalities (square opposite edges) -> classes per q-edge
    parent = list(range(len(edges)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # need faces to know edge types; reuse extract_faces on quotient edges
    faces = extract_faces(edges)
    edge_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for h in f:
            edge_faces[h[3]].append(fi)
    size_of = [len(f) for f in faces]
    for f in faces:
        if len(f) == 4:
            for i in (0, 1):
                a, bb = f[i][3], f[(i + 2) % 4][3]
                ra, rb = find(a), find(bb)
                if ra != rb:
                    parent[rb] = ra
    cls_of_edges = [find(e) for e in range(len(edges))]
    nbr = defaultdict(list)
    for (ka, kb), qi in raw2q.items():
        nbr[ka].append((kb, qi))
    return vid, edges, len(edges), pts, cls_of_edges, raw2q, nbr


def find_circle_candidates(nbr, pts, vids, T1p):
    """Cover BFS p0 -> p0+T1p (T1p = Lx*T1: the FULL torus period -- T1
    itself is half a period and the 'circle' to it is an open path!);
    segment at first repeated vid; return simple-loop candidates with
    non-zero lattice displacement."""
    cands = []
    # pre-filter starts whose +T1p translate is inside the cover window
    pts_arr = {k: np.array(v) for k, v in pts.items()}
    keys = list(pts_arr.keys())
    key_of = {tuple(np.round(v, 9)): k for k, v in pts_arr.items()}
    good = [k for k in keys
            if tuple(np.round(pts_arr[k] + T1p, 9)) in key_of]
    starts = good[:40]
    for p0 in starts:
        goal = key_of.get(tuple(np.round(pts_arr[p0] + T1p, 9)))
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
        # path entries are (predecessor, quotient edge); the predecessor
        # sequence already starts at p0, so appending p0 again created a
        # phantom duplicate head that poisoned every window
        seq = [pp for (pp, _) in path] + [goal]
        qes = [qi for (_, qi) in path]
        vseq = [vids[p] for p in seq]
        n = len(seq)
        for i in range(n - 1):
            for j in range(i + 2, n):
                if vseq[i] != vseq[j]:
                    continue
                interior = vseq[i + 1:j]
                if len(set(interior)) != j - i - 1:
                    continue           # segment not simple (vertex self-cross)
                # NOTE: repeated quotient edges within the segment are
                # harmless (pinning is per class; separation check decides)
                disp = np.array(pts[seq[j]]) - np.array(pts[seq[i]])
                if np.allclose(disp, 0, atol=1e-9):
                    continue           # contractible segment
                cands.append(list(qes[i:j]))
        if cands:
            break
    return cands


def separation_ok(edges, circle):
    """Quotient graph minus the circle edges must stay CONNECTED with
    exactly 1 component: cutting the torus along ONE non-separating
    (T1p-homotopic) circle yields a single cylinder. components==2 would
    mean a separating circle bounding a finite patch -- not usable."""
    cut = set(circle)
    adj = defaultdict(list)
    for i, (u, v, b) in enumerate(edges):
        if i in cut:
            continue
        adj[u].append(v)
        adj[v].append(u)
    seen = set()
    comps = 0
    for v0 in adj:
        if v0 in seen:
            continue
        comps += 1
        queue = deque([v0])
        seen.add(v0)
        while queue:
            x = queue.popleft()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    queue.append(y)
    return comps == 1, comps


# ---------------------------------------------------------------- pinning
def run_pinned(L, verbose=True):
    """n0(2, L) via pinned frontier DP. Returns (n0_full, n_free_eff, info)."""
    # dp-side geometry: canonical class ids (edge_cls), all_eq_cls, ...
    (face_eqs, n_cls, n_free, N, dim, all_eq_cls, face_nbrs,
     edge_cls) = build_face_system(2, L, rank_check=False)
    # cover-side geometry for the circle search (independent rebuild)
    vid, edges_q, E_q, pts, cls_of_edges_q, raw2q, nbr = build_quotient(2, L)
    # crosscheck: the two independent rebuilds must agree on scale and
    # the equivalence-class PARTITION (ids may differ; canonical form =
    # smallest member edge id per class must match edge-by-edge)
    assert E_q == 18 * 2 * L, ('quotient edge count mismatch', E_q)

    def canon(cls_list):
        groups = defaultdict(list)
        for e, c in enumerate(cls_list):
            groups[c].append(e)
        sig = {}
        for members in groups.values():
            m = min(members)
            for e in members:
                sig[e] = m
        return [sig[e] for e in range(len(cls_list))]

    assert canon(cls_of_edges_q) == canon(edge_cls), \
        'class partition mismatch (independent union-finds disagree)'
    cands = find_circle_candidates(nbr, pts, vid, 2 * T1)
    if not cands:
        return None, None, 'no T1 edge circle candidate found'
    circle = None
    comps_seen = []
    for cand in cands:
        ok, comps = separation_ok(edges_q, cand)
        comps_seen.append(comps)
        if ok:
            circle = cand
            break
    if circle is None:
        return None, None, (f'separation failed for all '
                            f'{len(comps_seen)} candidates '
                            f'(components seen: {sorted(set(comps_seen))})')
    pinned_cls = sorted({edge_cls[qi] for qi in circle})
    if verbose:
        print(f'  circle: {len(circle)} q-edges, {len(pinned_cls)} classes '
              f'pinned, separation comps={comps_seen[0]}', flush=True)
    dp, tw, peak, dt = _dp_pinned(face_eqs, n_cls, N, face_nbrs,
                                  all_eq_cls, frozenset(pinned_cls))
    all_eq = set(all_eq_cls)
    pinned_free = [c for c in pinned_cls if c not in all_eq]
    n_free_eff = n_free - len(pinned_free)
    n0_full = dp * (2 ** n_free_eff)
    return n0_full, n_free_eff, {'tree_w': tw, 'peak': peak, 'dt': dt,
                                 'circle_len': len(circle),
                                 'pinned_cls': len(pinned_cls),
                                 'pinned_free': len(pinned_free)}


def _dp_pinned(face_eqs, n_cls, N, face_nbrs, all_eq_cls, pinned):
    """Pinned frontier DP over a prebuilt system (no rebuild)."""
    import time as _t
    t0 = _t.time()
    cls_faces = defaultdict(list)
    for f, eqs_f in enumerate(face_eqs):
        for eq in eqs_f:
            for c in eq:
                cls_faces[c].append(f)
    face_order = face_bfs_order(face_nbrs, len(face_eqs))
    plan = frontier_plan(face_eqs, face_order, cls_faces)
    vals = (0, 1)
    state = {0: 1}
    tree_w = 0
    peak = 1
    for step, (fr_in, new, out, f, eqs_f) in enumerate(plan):
        if not new and fr_in == out:
            continue
        k = len(new)
        choices = [((0,) if c in pinned else vals) for c in new]
        nstate = defaultdict(int)
        for key, cnt in state.items():
            val = {c: (key >> i) & 1 for i, c in enumerate(fr_in)}
            for combo in product(*choices) if k else [()]:
                if k:
                    for c, v in zip(new, combo):
                        val[c] = v
                ok = True
                for eq in eqs_f:
                    s = 0
                    for c, co in eq.items():
                        s += co * val[c]
                    if s != 0:
                        ok = False
                        break
                if not ok:
                    continue
                nk = 0
                for i, c in enumerate(out):
                    nk |= val[c] << i
                nstate[nk] += cnt
        state = nstate
        tree_w = max(tree_w, len(fr_in) + k, len(out))
        peak = max(peak, len(state))
        if peak > PEAK_GATE:
            print(f'FAIL-CLOSED: peak {peak} in pinned DP (2 x L series)',
                  flush=True)
            sys.exit(2)
    assert set(state) == {0}, 'nonzero terminal keys remain'
    return state[0], tree_w, peak, _t.time() - t0


def load_strips():
    out = {}
    if os.path.exists(STRIPS_CSV):
        with open(STRIPS_CSV) as f:
            for row in csv.DictReader(f):
                if row['W'] == '2' and row['Omega_01']:
                    out[int(row['L'])] = int(row['Omega_01'])
    return out


def main():
    strips = load_strips()
    new = not os.path.exists(OUT_CSV)
    f = open(OUT_CSV, 'a', newline='')
    w = csv.writer(f)
    if new:
        w.writerow(['L', 'N', 'circle_len', 'pinned_cls', 'pinned_free',
                    'tree_w', 'peak', 'DP_count', 'n0_full', 'ln_n0',
                    'ln_n0_over_L', 'Omega_01_ref', 'time_s'])
    for L in range(2, 15):
        t0 = time.time()
        n0_full, _nfe, info = run_pinned(L)
        if n0_full is None:
            print(f'L={L}: FAIL-CLOSED {info}', flush=True)
            w.writerow([L, 2 * L, '', '', '', '', '', 'FAIL', '', '', '', '',
                        f'{time.time() - t0:.0f}'])
            f.flush()
            continue
        lam0 = math.log(n0_full) / L
        ref = strips.get(L, '')
        msg = ''
        if ref:
            msg = f'  mono {"OK" if n0_full <= ref else "VIOLATED"}'
        print(f'L={L}: circle={info["circle_len"]} pinned={info["pinned_cls"]} '
              f'(free {info["pinned_free"]}) tree_w={info["tree_w"]} '
              f'peak={info["peak"]} n0={n0_full} ln/L={lam0:.4f} '
              f'(Omega_ref={ref}){msg} [{info["dt"]:.0f}s]', flush=True)
        assert not ref or n0_full <= ref, ('pinned > unpinned', L)
        w.writerow([L, 2 * L, info['circle_len'], info['pinned_cls'],
                    info['pinned_free'], info['tree_w'], info['peak'],
                    '', n0_full, f'{math.log(n0_full):.4f}',
                    f'{lam0:.6f}', ref, f'{info["dt"]:.0f}'])
        f.flush()
    f.close()
    print('PINNED SERIES DONE', flush=True)


if __name__ == '__main__':
    main()
